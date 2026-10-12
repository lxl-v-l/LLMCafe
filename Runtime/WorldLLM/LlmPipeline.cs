using ShaderGPT.Udon;
using UdonSharp;
using UnityEngine;
using VRC.Udon;

namespace WorldLLM {
// Single-user chat pipeline over a baked ShaderGPT decoder (Task 05).
// The conversation is kept as token ids so that every new prompt shares its prefix with what the generator has
// already processed and the KV cache is reused up to the first differing token. The template is LFM2's ChatML:
//   <|startoftext|>[<|im_start|>system\n{s}<|im_end|>\n]{<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n}*
// The generator is addressed through UdonBehaviour (SetProgramVariable / SendCustomEvent) so that GPTGenerator and
// the camera-chain driver of Task 10 are interchangeable.
[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class LlmPipeline : UdonSharpBehaviour {
	public GPTTokenizer tokenizer;
	public UdonBehaviour generator;
	public UdonBehaviour listener;
	public string listenerEvent = "OnPipelineUpdate";
	[TextArea] public string systemPrompt = "";
	public LlmOutputFilter outputFilter;   // optional (Task 08)
	public string blockedMessage = "（この応答は出力フィルタにより非表示にしました）";
	public int maxNewTokens = 256;
	public int maxContextTokens = 2048;
	public int maxReplyChars = 1000;
	public int maxRetries = 3;
	public int maxTurns = 8;  // same as LlmHistory.Capacity

	public const int StatusLoading = 0, StatusReady = 1, StatusGenerating = 2, StatusDone = 3, StatusError = 4, StatusFatal = 5;
	[System.NonSerialized] public int status;
	[System.NonSerialized] public string replyText = "";
	[System.NonSerialized] public string lastUserText = "";
	[System.NonSerialized] public bool lastBlocked;   // the last response was stopped by the output filter
	[System.NonSerialized] public int generatedTokens;
	[System.NonSerialized] public int promptTokens;
	[System.NonSerialized] public int reusedTokens;
	[System.NonSerialized] public int retries;
	[System.NonSerialized] public float submitTime, firstTokenTime, lastTokenTime;
	[System.NonSerialized] public int progressTick;
	[System.NonSerialized] public int droppedTurns;

	// History (completed turns). Ids of all completed turns are concatenated in convIds; turnOffsets[i] is where turn i starts.
	[System.NonSerialized] public string[] turnUser;
	[System.NonSerialized] public string[] turnReply;
	[System.NonSerialized] public int turnCount;
	private int[] convIds = new int[0];
	private int[] turnOffsets;

	private int[] prefixIds;     // <|startoftext|> + optional system segment
	private int[] promptIds;     // prompt of the response in progress
	private int[] userSegIds;    // user segment of the response in progress
	private int[] genIds;
	private int eosId, newlineId;
	private bool initialized;

	void Start() {
		turnUser = new string[maxTurns];
		turnReply = new string[maxTurns];
		turnOffsets = new int[maxTurns];
		genIds = new int[maxNewTokens + 8];
		SendCustomEventDelayedFrames(nameof(_WaitForTokenizer), 1);
	}

	public void _WaitForTokenizer() {
		if(!tokenizer.isReady) {
			SendCustomEventDelayedFrames(nameof(_WaitForTokenizer), 1);
			return;
		}
		var bos = tokenizer.Encode("<|startoftext|>");
		prefixIds = string.IsNullOrEmpty(systemPrompt) ? bos
			: Concat(bos, tokenizer.Encode("<|im_start|>system\n" + Sanitize(systemPrompt, 2000) + "<|im_end|>\n"));
		eosId = tokenizer.eos_token_id;
		var nl = tokenizer.Encode("\n");
		newlineId = nl.Length > 0 ? nl[0] : -1;
		initialized = true;
		status = StatusReady;
		Notify();
	}

	// ---- prompt variants (cafe NPC)
	// A caller with several jobs builds one prefix per job once and switches between them, instead of replacing the
	// system prompt text (which re-encoded it every time). A prompt text is the system prompt, optionally followed by
	// examples separated by a line of "===", each ending in a line "→ answer"; the examples become finished
	// user/assistant turns, which a small model follows more closely than examples inside the system text.
	public int[] BuildPrefix(string prompt) {
		var parts = prompt.Replace("\r\n", "\n").Split(new string[] { "\n===\n" }, System.StringSplitOptions.None);
		var ids = Concat(tokenizer.Encode("<|startoftext|>"),
			tokenizer.Encode("<|im_start|>system\n" + Sanitize(parts[0], 2000) + "<|im_end|>\n"));
		for(int i = 1; i < parts.Length; i++) {
			var example = parts[i].Trim();
			var arrow = example.LastIndexOf("\n→ ");
			if(arrow < 0)
				continue;
			ids = Concat(ids, tokenizer.Encode("<|im_start|>user\n" + Sanitize(example.Substring(0, arrow), 600)
				+ "<|im_end|>\n<|im_start|>assistant\n" + Sanitize(example.Substring(arrow + 3), 300) + "<|im_end|>\n"));
		}
		return ids;
	}

	public void SetPrefix(int[] ids) {
		if(ids != null && ids.Length > 0)
			prefixIds = ids;
	}

	// ---- public API (UI and Task 06 call these)
	public bool IsBusy() {
		return status == StatusGenerating || status == StatusLoading;
	}

	public bool Submit(string userText) {
		return SubmitContent(SanitizePrompt(userText, 200));
	}

	public bool SubmitObservation(string observation) {
		return SubmitContent(Sanitize(observation, 1200));
	}

	// Shared conversation (Task 06): the user turn is "{speaker}: {text}" so the model can tell people apart.
	public bool SubmitAs(string speaker, string userText) {
		var text = SanitizeSharedPrompt(userText, 200);
		if(text.Length == 0)
			return false;
		return SubmitContent(UserContent(speaker, text));
	}

	public static string UserContent(string speaker, string text) {
		return Sanitize(speaker, 32) + ": " + text;
	}

	bool SubmitContent(string text) {
		if(!initialized || IsBusy() || status == StatusFatal)
			return false;
		if(text.Length == 0)
			return false;
		lastUserText = text;
		userSegIds = tokenizer.Encode("<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n");
		// Token budget: prompt + reply must fit in the KV cache. Drop the oldest complete turns first.
		while(turnCount > 0 && prefixIds.Length + convIds.Length + userSegIds.Length + maxNewTokens > maxContextTokens)
			DropOldestTurn();
		if(prefixIds.Length + convIds.Length + userSegIds.Length + maxNewTokens > maxContextTokens)
			return false;
		promptIds = Concat(Concat(prefixIds, convIds), userSegIds);
		tokenizer.decodeState = 0;
		replyText = "";
		lastBlocked = false;
		if(outputFilter != null)
			outputFilter.BeginResponse();
		generatedTokens = 0;
		retries = 0;
		submitTime = Time.realtimeSinceStartup;
		firstTokenTime = 0;
		lastTokenTime = 0;
		StartGenerator(promptIds);
		return true;
	}

	public void StopGeneration() {
		if(status != StatusGenerating)
			return;
		generator.SendCustomEvent("Stop");
		FinishTurn(false);
	}

	// Abnormal stop (watchdog, lost worker role): the response is dropped, nothing is added to the conversation, and
	// the prompt can be answered again later. The KV of the prompt prefix stays reusable.
	public void AbortGeneration() {
		if(status != StatusGenerating)
			return;
		generationSerial++;   // late stop/error handling of this generation is ignored
		generator.SendCustomEvent("Stop");
		generator.SetProgramVariable("inputTokens", promptIds);
		replyText = "";
		status = initialized ? StatusReady : StatusLoading;
		Notify();
	}

	public void ClearHistory() {
		if(status == StatusGenerating)
			generator.SendCustomEvent("Stop");
		convIds = new int[0];
		turnCount = 0;
		replyText = "";
		tokenizer.decodeState = 0;
		if(status != StatusFatal)
			status = initialized ? StatusReady : StatusLoading;
		Notify();
	}

	// Worker handoff (Task 06): rebuild the conversation ids from the synced history texts. The KV cache is
	// forgotten, so the next prompt is prefilled from position 0.
	public void ImportHistory(string[] speakers, string[] prompts, string[] replies) {
		convIds = new int[0];
		turnCount = 0;
		var n = speakers == null ? 0 : speakers.Length;
		for(int i = 0; i < n; i++) {
			var user = tokenizer.Encode("<|im_start|>user\n" + UserContent(speakers[i], Sanitize(prompts[i], 200)) + "<|im_end|>\n<|im_start|>assistant\n");
			var reply = tokenizer.Encode(Sanitize(replies[i], maxReplyChars) + "<|im_end|>\n");
			if(turnCount >= maxTurns)
				DropOldestTurn();
			turnOffsets[turnCount] = convIds.Length;
			turnUser[turnCount] = UserContent(speakers[i], prompts[i]);
			turnReply[turnCount] = replies[i];
			turnCount++;
			convIds = Concat(Concat(convIds, user), reply);
		}
		// Leave room for a 200-character prompt (about 256 tokens) and the reply.
		while(turnCount > 0 && prefixIds.Length + convIds.Length + maxNewTokens + 256 > maxContextTokens)
			DropOldestTurn();
		ForgetCache();
		Notify();
	}

	public void ResetAfterFatal() {
		retries = 0;
		generator.SetProgramVariable("state", GPTGenerator.StateIdle);
		status = initialized ? StatusReady : StatusLoading;
		Notify();
	}

	// Rebuilds the KV cache from the history after a worker handoff (Task 06) or for the cache-consistency checks:
	// forget what the generator holds so that the next prompt is prefilled from position 0.
	public void ForgetCache() {
		generator.SetProgramVariable("inputTokens", new int[0]);
		generator.SetProgramVariable("inputIndex", 0);
	}

	// Last finished response, readable with GetProgramVariable by the editor checks.
	[System.NonSerialized] public int[] lastResponseIds = new int[0];
	[System.NonSerialized] public string lastResponseText = "";
	private bool debugReplay;

	// Test hook (cache-consistency checks): regenerate the last prompt without touching the history.
	public void _DebugRegenerate() {
		if(!initialized || IsBusy() || promptIds == null)
			return;
		debugReplay = true;
		tokenizer.decodeState = 0;
		replyText = "";
		if(outputFilter != null)
			outputFilter.BeginResponse();
		generatedTokens = 0;
		retries = 0;
		submitTime = Time.realtimeSinceStartup;
		StartGenerator(promptIds);
	}

	public int[] GetConversationIds() {
		return convIds;
	}

	public int[] GetLastResponseIds() {
		var ids = new int[generatedTokens];
		System.Array.Copy(genIds, ids, generatedTokens);
		return ids;
	}

	public int[] GetLastPromptIds() {
		return promptIds;
	}

	// ---- generator callback
	public void OnOutputToken() {
		if(status != StatusGenerating)
			return;
		var genState = (int)generator.GetProgramVariable("state");
		if(genState == GPTGenerator.StateError) {
			ScheduleErrorHandling();
			return;
		}
		var token = (int)generator.GetProgramVariable("outputToken");
		var index = (int)generator.GetProgramVariable("outputIndex");
		progressTick++;
		if(index < promptIds.Length + generatedTokens)
			return; // prompt token or a token already confirmed before a restart
		if(index > promptIds.Length + generatedTokens) {
			ScheduleErrorHandling(); // a step result went missing
			return;
		}
		genIds[generatedTokens++] = token;
		lastTokenTime = Time.realtimeSinceStartup;
		if(generatedTokens == 1)
			firstTokenTime = lastTokenTime;
		if(token == eosId) {
			ScheduleGeneratorStop();
			FinishTurn(true);
			return;
		}
		var piece = tokenizer.Decode(token);
		// The reply is capped at maxReplyChars. A token that would cross the cap is not part of the reply, so the
		// conversation ids, the streamed text and the shared history all hold the same canonical reply.
		if(replyText.Length + piece.Length > maxReplyChars) {
			generatedTokens--;
			ScheduleGeneratorStop();
			FinishTurn(false);
			return;
		}
		replyText += piece;
		if(outputFilter != null && outputFilter.Append(piece)) {
			ScheduleGeneratorStop();
			FinishBlocked();
			return;
		}
		if(generatedTokens >= maxNewTokens || replyText.Length >= maxReplyChars) {
			ScheduleGeneratorStop();
			FinishTurn(false);
			return;
		}
		Notify();
	}

	// ---- internals
	// Stop requested from inside the generator's callback is applied one frame later (see ScheduleErrorHandling).
	// generationSerial keeps a late stop from hitting a generation started in the meantime.
	private int generationSerial, pendingStopSerial = -1;
	void ScheduleGeneratorStop() {
		pendingStopSerial = generationSerial;
		SendCustomEventDelayedFrames(nameof(_StopGenerator), 1);
	}

	public void _StopGenerator() {
		if(pendingStopSerial == generationSerial)
			generator.SendCustomEvent("Stop");
		pendingStopSerial = -1;
	}

	void StartGenerator(int[] ids) {
		generationSerial++;
		var previous = (int[])generator.GetProgramVariable("inputTokens");
		var prefix = 0;
		if(previous != null)
			for(; prefix < previous.Length && prefix < ids.Length; prefix++)
				if(previous[prefix] != ids[prefix])
					break;
		reusedTokens = prefix;
		promptTokens = ids.Length;
		generator.SetProgramVariable("inputTokens", ids);
		generator.SetProgramVariable("inputIndex", prefix);
		generator.SetProgramVariable("eventTarget", (UdonBehaviour)(Component)this);
		generator.SetProgramVariable("eventMethod", nameof(OnOutputToken));
		status = StatusGenerating;
		generator.SendCustomEvent("Play");
		Notify();
	}

	// OnOutputToken runs inside the generator's readback callback. Calling back into the generator (Play/Stop)
	// from there re-enters a running UdonBehaviour, which the Udon VM does not support reliably, so error
	// recovery runs one frame later.
	private bool errorPending;
	private int errorSerial;
	void ScheduleErrorHandling() {
		if(errorPending)
			return;
		errorPending = true;
		errorSerial = generationSerial;
		SendCustomEventDelayedFrames(nameof(_HandleReadbackError), 1);
	}

	public void _HandleReadbackError() {
		errorPending = false;
		if(status != StatusGenerating || errorSerial != generationSerial)
			return;
		retries++;
		if(retries > maxRetries) {
			generator.SendCustomEvent("Stop");
			generator.SetProgramVariable("state", GPTGenerator.StateFatalError);
			status = StatusFatal;
			Notify();
			return;
		}
		// Restart from the last confirmed token; the confirmed reply tokens are fed back as input.
		var confirmed = new int[promptIds.Length + generatedTokens];
		System.Array.Copy(promptIds, confirmed, promptIds.Length);
		System.Array.Copy(genIds, 0, confirmed, promptIds.Length, generatedTokens);
		StartGenerator(confirmed);
	}

	void FinishTurn(bool endedWithEos) {
		var reply = new int[generatedTokens];
		System.Array.Copy(genIds, reply, generatedTokens);
		lastResponseIds = reply;
		lastResponseText = replyText;
		if(debugReplay) {
			debugReplay = false;
			generator.SetProgramVariable("inputTokens", Concat(promptIds, reply));
			status = StatusDone;
			Notify();
			return;
		}
		// The turn as the model saw it: user segment + reply ids, closed by <|im_end|>\n.
		int[] closing;
		if(endedWithEos)
			closing = newlineId >= 0 ? new int[] { newlineId } : new int[0];
		else
			closing = newlineId >= 0 ? new int[] { eosId, newlineId } : new int[] { eosId };
		if(turnCount >= maxTurns)
			DropOldestTurn();
		turnOffsets[turnCount] = convIds.Length;
		turnUser[turnCount] = lastUserText;
		turnReply[turnCount] = replyText;
		turnCount++;
		convIds = Concat(Concat(convIds, userSegIds), Concat(reply, closing));
		// Let the next prompt reuse the KV of the reply: the generator has processed prompt + reply tokens.
		generator.SetProgramVariable("inputTokens", Concat(promptIds, reply));
		status = StatusDone;
		Notify();
	}

	// A response stopped by the output filter is not added to the conversation: the next prompt continues from
	// the conversation before it (the KV of the shared prefix is reused) and the reply text is replaced.
	void FinishBlocked() {
		var reply = new int[generatedTokens];
		System.Array.Copy(genIds, reply, generatedTokens);
		generator.SetProgramVariable("inputTokens", Concat(promptIds, reply));
		lastResponseIds = new int[0];
		replyText = blockedMessage;
		lastResponseText = replyText;
		lastBlocked = true;
		debugReplay = false;
		status = StatusDone;
		Notify();
	}

	void DropOldestTurn() {
		if(turnCount == 0)
			return;
		var cut = turnCount > 1 ? turnOffsets[1] : convIds.Length;
		var rest = new int[convIds.Length - cut];
		System.Array.Copy(convIds, cut, rest, 0, rest.Length);
		convIds = rest;
		for(int i = 1; i < turnCount; i++) {
			turnOffsets[i - 1] = turnOffsets[i] - cut;
			turnUser[i - 1] = turnUser[i];
			turnReply[i - 1] = turnReply[i];
		}
		turnCount--;
		droppedTurns++;
	}

	void Notify() {
		if(listener != null)
			listener.SendCustomEvent(listenerEvent);
	}

	// Keeps user text from injecting chat-template control tokens and strips control characters.
	public static string Sanitize(string text, int maxChars) {
		if(text == null)
			return "";
		var t = text.Replace("\r\n", "\n").Replace("<|", "＜|").Replace("|>", "|＞");
		if(t.IndexOf('\udb40') >= 0)
			t = RemoveSupplementaryVariationSelectors(t);
		var chars = new char[Mathf.Min(t.Length, maxChars)];
		var n = 0;
		for(int i = 0; i < t.Length && n < chars.Length; i++) {
			var c = t[i];
			if(c == '\n' || c == '\t' || (c >= ' ' && !IsControlOrFormat(c)))
				chars[n++] = c;
		}
		return new string(chars, 0, n).Trim();
	}

	// U+E0100-E01EF (variation selectors supplement) as UTF-16 pairs: high surrogate U+DB40, low U+DD00-U+DDEF.
	public static string RemoveSupplementaryVariationSelectors(string text) {
		var chars = new char[text.Length];
		var n = 0;
		for(int i = 0; i < text.Length; i++) {
			if(text[i] == '\udb40' && i + 1 < text.Length && text[i + 1] >= '\udd00' && text[i + 1] <= '\uddef') {
				i++;
				continue;
			}
			chars[n++] = text[i];
		}
		return new string(chars, 0, n);
	}

	// DEL and C1 controls, zero-width characters and bidirectional overrides.
	public static bool IsControlOrFormat(char c) {
		return (c >= '\u007f' && c <= '\u009f') || (c >= '\u200b' && c <= '\u200f') || (c >= '\u202a' && c <= '\u202e')
			|| (c >= '\u2060' && c <= '\u206f') || c == '\ufeff' || c == '\u00ad' || c == '\u061c' || c == '\u180e'
			|| c == '\u034f' || (c >= '\ufe00' && c <= '\ufe0f');
	}

	// Shared prompts are one line: a line break could push the real "name: " label out of view and make the text
	// look like another player's message (also to the model).
	public static string SanitizeSharedPrompt(string text, int maxChars) {
		return SanitizePrompt(text, maxChars).Replace("\n", " ").Replace("\t", " ");
	}

	// User prompts (Task 08): control characters as above, URLs removed, then the length limit.
	public static string SanitizePrompt(string text, int maxChars) {
		return Sanitize(RemoveUrls(Sanitize(text, maxChars * 4)), maxChars);
	}

	// Removes http(s):// and www. addresses, also in full-width letters. Only a short window at a possible URL start is
	// normalized for the check, so the rest of the text keeps its characters (full-width punctuation stays as typed).
	public static string RemoveUrls(string text) {
		if(string.IsNullOrEmpty(text))
			return "";
		var chars = new char[text.Length];
		var n = 0;
		var i = 0;
		while(i < text.Length) {
			var c = text[i];
			if(c == 'h' || c == 'H' || c == 'w' || c == 'W' || c == 'ｈ' || c == 'Ｈ' || c == 'ｗ' || c == 'Ｗ') {
				var head = text.Substring(i, Mathf.Min(8, text.Length - i)).Normalize(System.Text.NormalizationForm.FormKC).ToLowerInvariant();
				if(head.StartsWith("http://") || head.StartsWith("https://") || head.StartsWith("www.")) {
					while(i < text.Length && IsUrlChar(text[i]))
						i++;
					continue;
				}
			}
			chars[n++] = c;
			i++;
		}
		return new string(chars, 0, n);
	}

	// Printable ASCII, and the full-width letters, digits and URL symbols (． ／ ： ＝ ＆ ％ ＃ － ＿ ～).
	static bool IsUrlChar(char c) {
		if(c > ' ' && c < '\u007f')
			return true;
		return (c >= '０' && c <= '９') || (c >= 'Ａ' && c <= 'Ｚ') || (c >= 'ａ' && c <= 'ｚ')
			|| c == '．' || c == '／' || c == '：' || c == '＝' || c == '＆' || c == '％' || c == '＃' || c == '－' || c == '＿' || c == '～';
	}

	static int[] Concat(int[] a, int[] b) {
		var r = new int[a.Length + b.Length];
		System.Array.Copy(a, r, a.Length);
		System.Array.Copy(b, 0, r, a.Length, b.Length);
		return r;
	}
}
}
