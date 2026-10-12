using UdonSharp;
using UnityEngine;
using VRC.SDK3.Rendering;
using VRC.SDKBase;
using VRC.Udon.Common.Interfaces;

namespace WorldLLM {
// Local GPU self-test (Task 05/06): one Blit of a shader that writes a known float4, read back as float[].
// A client that cannot Blit to float render targets or read them back cannot be the worker.
[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class LlmSelfTest : UdonSharpBehaviour {
	public Material constantMaterial;   // WorldLLM/SelfTestConstant, writes _Value = (262, 7, run tag, 5)
	public RenderTexture target;        // 1x1 ARGBFloat
	public UdonSharpBehaviour listener;
	public string listenerEvent = "OnSelfTestFinished";

	public float timeoutSeconds = 5f;

	[System.NonSerialized] public bool running;
	[System.NonSerialized] public bool completed;
	[System.NonSerialized] public bool passed;
	// The readback result arrives through UdonBehaviour.RunEvent, which drops events while the behaviour is not
	// ready yet (ClientSim before Start, for example). A dropped result would leave running set forever and block
	// the worker, so a run that gets no result in timeoutSeconds ends with timedOut set and can be started again.
	[System.NonSerialized] public bool timedOut;
	private bool errorSeen;   // a readback error arrived during this run (its run cannot be told from the data)
	private int runSerial;
	private int expectedTag;
	private float[] data = new float[4];

	public void Run() {
		if(running)
			return;
		running = true;
		completed = false;
		passed = false;
		timedOut = false;
		errorSeen = false;
		runSerial++;
		expectedTag = runSerial % 1000 + 1;
		constantMaterial.SetVector("_Value", new Vector4(262, 7, expectedTag, 5));
		VRCGraphics.Blit(null, target, constantMaterial, -1);
		VRCAsyncGPUReadback.Request(target, 0, TextureFormat.RGBAFloat, (IUdonEventReceiver)this);
		SendCustomEventDelayedSeconds(nameof(_CheckTimeout), timeoutSeconds);
	}

	private int checkedSerial;
	public void _CheckTimeout() {
		// Delayed events are not cancelled, so only the newest run may time out.
		checkedSerial++;
		if(checkedSerial != runSerial || !running)
			return;
		running = false;
		if(errorSeen) {
			completed = true;   // only errors arrived: the GPU cannot do the Blit + float readback
			passed = false;
		} else {
			timedOut = true;
		}
		if(listener != null)
			listener.SendCustomEvent(listenerEvent);
	}

	public override void OnAsyncGpuReadbackComplete(VRCAsyncGPUReadbackRequest request) {
		if(!running)
			return;   // result of a run that already timed out
		// An error may belong to an earlier run: note it and let the timeout decide. A late result of an earlier run
		// carries that run's tag: ignore it and keep waiting.
		if(request.hasError || !request.TryGetData(data)) {
			errorSeen = true;
			return;
		}
		if(Mathf.RoundToInt(data[2]) != expectedTag)
			return;
		running = false;
		completed = true;
		passed = Mathf.RoundToInt(data[0]) == 262 && Mathf.RoundToInt(data[1]) == 7 && Mathf.RoundToInt(data[3]) == 5;
		if(listener != null)
			listener.SendCustomEvent(listenerEvent);
	}
}
}
