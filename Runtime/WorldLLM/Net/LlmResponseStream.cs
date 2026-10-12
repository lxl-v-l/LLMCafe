using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon.Common;

namespace WorldLLM {
// The response being generated, synced as the full text (Task 06). Late joiners get the current state from Manual
// sync without replaying deltas. The worker marks it dirty; LlmBandwidth decides when it is actually sent.
[UdonBehaviourSyncMode(BehaviourSyncMode.Manual)]
public class LlmResponseStream : UdonSharpBehaviour {
	public LlmBandwidth bandwidth;
	public UdonSharpBehaviour listener;
	public string listenerEvent = "OnNetworkStateChanged";
	public int maxChars = 1000;

	[UdonSynced] public int responseId;
	[UdonSynced] public int requesterId;
	[UdonSynced] public string requesterName = "";
	[UdonSynced] public string prompt = "";
	[UdonSynced] public string text = "";
	[UdonSynced] public bool done = true;
	[UdonSynced] public bool regenerating;

	public void Begin(int playerId, string playerName, string promptText, bool isRegeneration) {
		responseId++;
		requesterId = playerId;
		requesterName = playerName;
		prompt = promptText;
		text = "";
		done = false;
		regenerating = isRegeneration;
		Changed();
	}

	public void SetText(string value) {
		var t = value ?? "";
		if(t.Length > maxChars)
			t = t.Substring(0, maxChars);
		if(t == text)
			return;
		text = t;
		Changed();
	}

	public void Finish(string finalText) {
		SetText(finalText);
		done = true;
		regenerating = false;
		Changed();
	}

	public int EstimateBytes() {
		return LlmBandwidth.StringBytes(requesterName) + LlmBandwidth.StringBytes(prompt) + LlmBandwidth.StringBytes(text) + 4 * 3 + 2 + 64;
	}

	void Changed() {
		bandwidth.RequestSend(LlmBandwidth.ChannelStream);
		Notify();
	}

	public override void OnDeserialization() {
		Notify();
	}

	public override void OnPostSerialization(SerializationResult result) {
		bandwidth.OnSent(LlmBandwidth.ChannelStream, result.success, result.byteCount);
	}

	void Notify() {
		if(listener != null)
			listener.SendCustomEvent(listenerEvent);
	}
}
}
