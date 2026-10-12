using UnityEngine;
using UnityEngine.Rendering;
#if UDON
using MonoBehaviour = VRC.Udon.Common.Interfaces.IUdonEventReceiver;
using Graphics = VRC.SDKBase.VRCGraphics;
using AsyncGPUReadbackRequest = VRC.SDK3.Rendering.VRCAsyncGPUReadbackRequest;
#endif

namespace ShaderGPT.Udon {
public class GPTGenerator : UdonMonoBehaviour {
	public GameObject modelPrefab;
	public int maxLength = 2048;
	public float temperature = 0;
	public float repetitionPenalty = 1f;
	public int idleFrames = 0; // negative means extra steps per frame
	[Min(1)] public int prepFrames = 1;
	public int decoderAuxiliary = 0;
	public MonoBehaviour eventTarget;
	public string eventMethod;

	// ---- WorldLLM patch
	// passBudgetPerFrame > 0 spreads one decoder step over several frames, at most that many Blits per frame.
	// Steps only start and stop at step boundaries, so the KV cache never sees a half-applied step.
	public int passBudgetPerFrame = 0;
	// At most this many readbacks may be outstanding; the decoder waits at a step boundary otherwise.
	public int maxInFlight = 2;
	public const int StateIdle = 0, StatePrefill = 1, StateGenerating = 2, StateError = 3, StateFatalError = 4;
	[System.NonSerialized] public int state;
	// Incremented by Play/Stop. Readbacks requested under an older epoch are dropped, so results of a stopped
	// sequence cannot leak into the next one. Unity completes readbacks in request order, so a FIFO of epochs
	// identifies which request a completion belongs to without touching the GPU input layout.
	[System.NonSerialized] public int playEpoch;
	[System.NonSerialized] public int inFlight;
	[System.NonSerialized] public int readbackErrors;
	// Test hook: the next readback completion is handled as if request.hasError were true.
	[System.NonSerialized] public bool debugInjectReadbackError;
	private readonly int[] epochQueue = new int[16];
	private int epochHead, epochCount;
	private int decodeCursor;
	private bool stopRequested;

	private Material[] matEncoders;
	private Material matInput;
	private Texture2D texInput;
	private float[] costs;
	private float totalCost;
	private int maxEncoder;
	void LoadEncoder() {
		totalCost = 0f;
		costs = new float[matEncoders.Length];
		var maxScatter = 0;
		for(int i=0; i<matEncoders.Length; i++) {
			var mat = matEncoders[i];
			var name = mat.shader.name;
			var rt = (RenderTexture)mat.GetTexture("TexO");
			if(name.Contains("Scatter"))
				maxScatter = Mathf.Max(maxScatter, ScatterCapacity(mat, out var peek));
			float cost = Mathf.Max(4, rt.width) * Mathf.Max(4, rt.height);
			if(name.Contains("Conv"))
				cost *= mat.GetVector("DimX").x;
			if(name.Contains("Linear"))
				cost *= mat.GetVector("DimW").x;
			costs[i] = cost;
			totalCost += cost;
		}
		maxEncoder = maxScatter;

		var tex0 = matEncoders[0].GetTexture("TexX");
		texInput = tex0 && tex0.GetType() == typeof(Texture2D) ? (Texture2D)tex0 : null; // polyfill "as" operator
		foreach(var mat in matEncoders)
			if(mat.shaderKeywords.Length == 0) {
				matInput = mat;
				break;
			}
	}
	static int ScatterCapacity(Material mat, out int logPeek) {
		var height = (int)mat.GetVector("DimO").x;
		logPeek = mat.HasInt("LogPeek") ? mat.GetInt("LogPeek") : -1;
		if(logPeek < 0)
			return height;
		var bucketCount = Mathf.Max(1, height >> logPeek);
		var logCapacity = bucketCount - 1 + logPeek;
		return logCapacity >= 30 ? int.MaxValue : 1 << logCapacity;
	}

	private Material[] matDecoders;
	private Material matRepeat;
	private Material matGumbel;
	private Material matOutput;
	private RenderTexture bufOutput;
	void LoadDecoder() {
		var maxScatter = 0;
		foreach(var mat in matDecoders) {
			var name = mat.shader.name;
			if(name.Contains("Scatter")) {
				maxScatter = Mathf.Max(maxScatter, ScatterCapacity(mat, out var peek));
				if(peek >= 0) {
					maxLogPeek = Mathf.Max(maxLogPeek, peek);
					minLogPeek = Mathf.Min(minLogPeek, peek);
				}
			}
		}
		if(maxScatter > 0)
			maxLength = Mathf.Min(maxLength, maxScatter);

		matOutput = matDecoders[matDecoders.Length-1];
		bufOutput = (RenderTexture)matOutput.GetTexture("TexO");
		foreach(var mat in matDecoders)
			if(mat.shaderKeywords.Length == 1) {
				var keyword = mat.shaderKeywords[0];
				if(keyword == "RELU")
					matRepeat = mat;
				if(keyword == "REDUCE_MINMAX")
					matGumbel = mat;
			}
	}

	// you need to set encoderTokens or encoderInputs before Play()
	[System.NonSerialized] public int[] encoderTokens;
	[System.NonSerialized] public float[] encoderInputs;
	[System.NonSerialized] public Texture encoderImage;
	[System.NonSerialized] public int encoderIndex;
	[System.NonSerialized] public int encoderStep;
	void PrepEncoder() {
		if(encoderTokens == null && encoderInputs == null && encoderImage == null) {
			blitIndex = matEncoders.Length; // skip encoder
			return;
		}
		if(encoderInputs != null)
			SetPixelData(texInput, encoderInputs);
		else if(encoderImage != null) {
			matEncoders[0].SetTexture("TexX", encoderImage);
			matInput.SetVector("VecAdd", new Vector4(0,0,0, encoderIndex));
			encoderIndex += encoderStep;
			if(encoderIndex + encoderStep > maxEncoder)
				encoderIndex = 0;
		} else {
			var maxTokens = texInput.height;
			var numTokens = Mathf.Min(maxTokens, encoderTokens.Length);
			var tokenData = new float[maxTokens*4];
			for(int i=0; i<maxTokens; i++) {
				if(i < numTokens)
					tokenData[i*4+0] = encoderTokens[i];
				tokenData[i*4+1] = i;
				tokenData[i*4+2] = numTokens;
			}
			SetPixelData(texInput, tokenData);
		}
		blitIndex = 0;
		reservoir = 0;
	}
	private byte[] pixelData;
	void SetPixelData(Texture2D tex, float[] data) {
		Resize(ref pixelData, data.Length*4);
		System.Buffer.BlockCopy(data, 0, pixelData, 0, pixelData.Length);
		tex.LoadRawTextureData(pixelData);
		tex.Apply(false, false);
	}
	private int blitIndex;
	private float reservoir;
	void StepEncoder() {
		reservoir += totalCost / prepFrames;
		var executed = false;
		while(blitIndex < matEncoders.Length) {
			var cost = costs[blitIndex];
			if(cost > reservoir && executed)
				break;
			reservoir -= cost;
			Blit(matEncoders[blitIndex++]);
			executed = true;
		}
	}

	// you need to set inputIndex and inputTokens before Play()
	[System.NonSerialized] public int inputIndex;
	[System.NonSerialized] public int[] inputTokens;
	[System.NonSerialized] public int outputIndex;
	[System.NonSerialized] public int outputToken;
	private int maxLogPeek = -1, minLogPeek = int.MaxValue;
	private readonly int[] logNext = new int[32];
	private int recentIndexFirst, recentIndexLast;
	void PrepDecoder() {
		if(matRepeat) {
			matRepeat.SetFloat("Bias", repetitionPenalty);
			matRepeat.SetFloat("Scale", 1f/repetitionPenalty);
		}
		if(matGumbel) {
			matGumbel.SetFloat("Bias", temperature);
		}

		// detect external changes to inputIndex
		if(maxLogPeek >= 0 && recentIndexLast != inputIndex) {
			if(inputIndex < recentIndexFirst || recentIndexLast < inputIndex) {
				// rewind to the last usable block in log cache
				var best = 0;
				foreach(var x in logNext)
					if(x <= inputIndex)
						best = Mathf.Max(best, x);
				recentIndexFirst = inputIndex = best;
			}
			recentIndexLast = inputIndex;
		}
	}
	// First half of a decoder step: point the output material at the next input token.
	void BeginStep() {
		// Several steps can run in one frame; the sampling noise is seeded by the frame time, so each step also gets its position.
		if(matGumbel)
			matGumbel.SetFloat("Seed", inputIndex);
		var chan2 = encoderTokens == null ? decoderAuxiliary : encoderTokens.Length;
		// make sure the buffer stores the last token
		if(0 < inputIndex && inputIndex <= inputTokens.Length) {
			matOutput.SetVector("VecMul", Vector4.zero);
			matOutput.SetVector("VecAdd", new Vector4(inputTokens[inputIndex-1],inputIndex-1,chan2,0));
			Graphics.Blit(null, bufOutput, matOutput, 0);
		}
		if(inputIndex < inputTokens.Length) {
			matOutput.SetVector("VecMul", Vector4.zero);
			matOutput.SetVector("VecAdd", new Vector4(inputTokens[inputIndex],inputIndex,chan2,0));
		} else {
			matOutput.SetVector("VecMul", new Vector4(1,0,0,0));
			matOutput.SetVector("VecAdd", new Vector4(0,inputIndex,chan2,0));
		}
		state = inputIndex < inputTokens.Length ? StatePrefill : StateGenerating;
	}
	// Second half: bookkeeping after the last pass of the step, then one readback of the output texel.
	void EndStep() {
		if(maxLogPeek >= 0 && inputIndex > 0) {
			// (inputIndex+1)-th token generation needs to process inputIndex-th token,
			// which conservatively requires the (1<<logPeek)-wide block ending at (inputIndex-1) to remain intact,
			// whose block index in log cache is ctz(1+((inputIndex-1)>>logPeek)).
			// a block is unusable if it's partially updated, i.e. (inputIndex-1) % (1<<logPeek) < (1<<logPeek)-1
			var x = 1+((inputIndex-1)>>maxLogPeek);
			logNext[Mathf.RoundToInt(Mathf.Log(x & -x, 2))] = inputIndex % (1<<maxLogPeek) == 0 ? inputIndex+1 : 0;
		}
		recentIndexLast = ++inputIndex;
		recentIndexFirst = Mathf.Max(recentIndexFirst, recentIndexLast-(1<<minLogPeek));
		RequestReadback();
	}
	void StepDecoder() {
		BeginStep();
		foreach(var mat in matDecoders)
			Blit(mat);
		EndStep();
	}
	static void Blit(Material mat) {
		var rt = (RenderTexture)mat.GetTexture("TexO");
		Graphics.Blit(null, rt, mat, -1);
	}
	void RequestReadback() {
		epochQueue[(epochHead + epochCount) % epochQueue.Length] = playEpoch;
		epochCount = Mathf.Min(epochCount + 1, epochQueue.Length);
		inFlight++;
		AsyncGPUReadback_Request(bufOutput, 0, TextureFormat.RGBAFloat);
	}

	private int frameIndex;
	public void Play() {
		if(matDecoders == null) {
#if UNITY_EDITOR // avoid modifying asset in editor
			if(modelPrefab.GetInstanceID() >= 0) {
				modelPrefab = Object.Instantiate(modelPrefab);
				modelPrefab.SetActive(false);
			}
#endif
			var renderers = modelPrefab.GetComponentsInChildren<MeshRenderer>();
			Debug.Assert(renderers.Length == 1 || renderers.Length == 2);
			if(renderers.Length == 2) {
				var renderer = renderers[0];
				matEncoders = renderer.GetInstanceID() < 0 ? renderer.materials : renderer.sharedMaterials;
				LoadEncoder();
			}
			if(renderers.Length >= 1) {
				var renderer = renderers[renderers.Length-1];
				// if renderer is instanced, use ".materials" to avoid saving modification
				matDecoders = renderer.GetInstanceID() < 0 ? renderer.materials : renderer.sharedMaterials;
				LoadDecoder();
			}
		}
		if(matEncoders != null)
			PrepEncoder();
		PrepDecoder();
		frameIndex = 0;
		decodeCursor = 0;
		stopRequested = false;
		playEpoch++;
		state = inputIndex < inputTokens.Length ? StatePrefill : StateGenerating;
		this.enabled = true;
	}
	// Stops at the next step boundary; a step already in progress is completed first.
	public void Stop() {
		playEpoch++;
		if(decodeCursor > 0) {
			stopRequested = true;
			return;
		}
		this.enabled = false;
		if(state != StateError && state != StateFatalError)
			state = StateIdle;
	}
	public bool IsAtStepBoundary() {
		return decodeCursor == 0;
	}
	public void Update() {
		if(matEncoders != null && blitIndex < matEncoders.Length) {
			StepEncoder();
			return;
		}
		if(passBudgetPerFrame > 0) {
			UpdateBudgeted();
			return;
		}
		if(frameIndex != 0) {
			frameIndex--;
			return;
		}
		frameIndex = Mathf.Max(0, idleFrames);
		for(int i=Mathf.Min(0, idleFrames); i<=0; i++) {
			if(inputIndex >= maxLength) {
				this.enabled = false;
				state = StateIdle;
				return;
			}
			if(inFlight >= maxInFlight)
				return;
			StepDecoder();
		}
	}
	// Several steps may run in one frame while the budget lasts: the next step reads its input token on the GPU,
	// so only maxInFlight (outstanding readbacks) and the budget bound how far one frame goes.
	void UpdateBudgeted() {
		var budget = passBudgetPerFrame;
		while(budget > 0) {
			if(decodeCursor == 0) {
				if(stopRequested) {
					FinishStop();
					return;
				}
				if(frameIndex != 0) {
					frameIndex--;
					return;
				}
				if(inputIndex >= maxLength) {
					this.enabled = false;
					state = StateIdle;
					return;
				}
				if(inFlight >= maxInFlight)
					return;
				frameIndex = Mathf.Max(0, idleFrames);
				BeginStep();
			}
			while(budget > 0 && decodeCursor < matDecoders.Length) {
				Blit(matDecoders[decodeCursor++]);
				budget--;
			}
			if(decodeCursor < matDecoders.Length)
				return;
			decodeCursor = 0;
			EndStep();
			if(stopRequested) {
				FinishStop();
				return;
			}
			if(frameIndex != 0)
				return;
		}
	}
	void FinishStop() {
		stopRequested = false;
		this.enabled = false;
		if(state != StateError && state != StateFatalError)
			state = StateIdle;
	}
	void OnGPUReadback(float[] data) {
		outputToken = Mathf.RoundToInt(data[0]);
		outputIndex = Mathf.RoundToInt(data[1]);
		if(outputIndex >= inputIndex)
			return; // skip tokens from last sequence
		if(eventTarget != null)
			eventTarget.SendMessage(eventMethod);
	}

	private float[] readbackData;
	public override void OnAsyncGpuReadbackComplete(AsyncGPUReadbackRequest request) {
		var epoch = playEpoch;
		if(epochCount > 0) {
			epoch = epochQueue[epochHead];
			epochHead = (epochHead + 1) % epochQueue.Length;
			epochCount--;
		}
		inFlight = Mathf.Max(0, inFlight - 1);
		if(!this)
			return;
		var failed = debugInjectReadbackError || request.hasError || !request.done;
		debugInjectReadbackError = false;
		if(epoch != playEpoch)
			return; // result of a stopped sequence
		if(failed) {
			// The output texel has already been overwritten by later steps, so the token cannot be re-read;
			// stop and let the event target restart from the last confirmed token.
			readbackErrors++;
			playEpoch++;
			this.enabled = false;
			decodeCursor = 0;
			stopRequested = false;
			state = StateError;
			if(eventTarget != null)
				eventTarget.SendMessage(eventMethod);
			return;
		}
		if(!this.enabled)
			return;
		Resize(ref readbackData, request.width*request.height*4);
		request.GetData<float>().CopyTo(readbackData);
		OnGPUReadback(readbackData);
	}
	static void Resize<T>(ref T[] array, int size) {
		if(array == null || array.Length != size)
			array = new T[size];
	}
}
}
