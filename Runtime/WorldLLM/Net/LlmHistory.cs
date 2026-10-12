using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon.Common;

namespace WorldLLM {
// Completed turns of the shared conversation (Task 06): up to 8 turns, oldest first. Sent only when a turn completes
// or the history is cleared. A new worker rebuilds its KV cache from these texts.
[UdonBehaviourSyncMode(BehaviourSyncMode.Manual)]
public class LlmHistory : UdonSharpBehaviour {
	public const int Capacity = 8;
	public LlmBandwidth bandwidth;
	public UdonSharpBehaviour listener;
	public string listenerEvent = "OnNetworkStateChanged";

	[UdonSynced] public string[] speakers = new string[0];
	[UdonSynced] public string[] prompts = new string[0];
	[UdonSynced] public string[] replies = new string[0];
	// Prompt ID (playerId, joinNonce, clientSeq) of each turn: a new worker drops a queue head that is already here.
	[UdonSynced] public int[] playerIds = new int[0];
	[UdonSynced] public int[] nonces = new int[0];
	[UdonSynced] public int[] seqs = new int[0];
	// Prompt IDs of the last LedgerCapacity answered prompts, including answers hidden by the output filter (which are
	// not turns). A new worker drops a queue head found here.
	public const int LedgerCapacity = 16;
	[UdonSynced] public int[] answeredIds = new int[0];
	[UdonSynced] public int[] answeredNonces = new int[0];
	[UdonSynced] public int[] answeredSeqs = new int[0];
	[UdonSynced] public int revision;

	public int Count() {
		return speakers == null ? 0 : speakers.Length;
	}

	public void Append(string speaker, string prompt, string reply, int playerId, int nonce, int seq) {
		var n = Count();
		var keep = n >= Capacity ? Capacity - 1 : n;
		var s = new string[keep + 1];
		var p = new string[keep + 1];
		var r = new string[keep + 1];
		var ids = new int[keep + 1];
		var ns = new int[keep + 1];
		var qs = new int[keep + 1];
		for(int i = 0; i < keep; i++) {
			s[i] = speakers[n - keep + i];
			p[i] = prompts[n - keep + i];
			r[i] = replies[n - keep + i];
			ids[i] = playerIds[n - keep + i];
			ns[i] = nonces[n - keep + i];
			qs[i] = seqs[n - keep + i];
		}
		s[keep] = speaker;
		p[keep] = prompt;
		r[keep] = reply;
		ids[keep] = playerId;
		ns[keep] = nonce;
		qs[keep] = seq;
		speakers = s;
		prompts = p;
		replies = r;
		playerIds = ids;
		nonces = ns;
		seqs = qs;
		Changed();
	}

	public bool HasPrompt(int playerId, int nonce, int seq) {
		for(int i = 0; i < Count(); i++)
			if(playerIds[i] == playerId && nonces[i] == nonce && seqs[i] == seq)
				return true;
		for(int i = 0; i < answeredIds.Length; i++)
			if(answeredIds[i] == playerId && answeredNonces[i] == nonce && answeredSeqs[i] == seq)
				return true;
		return false;
	}

	public void MarkAnswered(int playerId, int nonce, int seq) {
		answeredIds = AppendKeep(answeredIds, playerId);
		answeredNonces = AppendKeep(answeredNonces, nonce);
		answeredSeqs = AppendKeep(answeredSeqs, seq);
		Changed();
	}

	static int[] AppendKeep(int[] a, int v) {
		var n = Mathf.Min(a.Length, LedgerCapacity - 1);
		var r = new int[n + 1];
		System.Array.Copy(a, a.Length - n, r, 0, n);
		r[n] = v;
		return r;
	}

	// Keeps the newest `count` turns (used when the worker's token budget drops old turns).
	public void TrimTo(int count) {
		var n = Count();
		if(count >= n)
			return;
		Keep(count);
		Changed();
	}

	public void Clear() {
		speakers = new string[0];
		prompts = new string[0];
		replies = new string[0];
		playerIds = new int[0];
		nonces = new int[0];
		seqs = new int[0];
		answeredIds = new int[0];
		answeredNonces = new int[0];
		answeredSeqs = new int[0];
		Changed();
	}

	// Keeps the newest n turns of all six arrays (n must not exceed any array's length).
	void Keep(int n) {
		var s = new string[n];
		var p = new string[n];
		var r = new string[n];
		var ids = new int[n];
		var ns = new int[n];
		var qs = new int[n];
		for(int i = 0; i < n; i++) {
			s[i] = speakers[speakers.Length - n + i] == null ? "" : speakers[speakers.Length - n + i];
			p[i] = prompts[prompts.Length - n + i] == null ? "" : prompts[prompts.Length - n + i];
			r[i] = replies[replies.Length - n + i] == null ? "" : replies[replies.Length - n + i];
			ids[i] = playerIds[playerIds.Length - n + i];
			ns[i] = nonces[nonces.Length - n + i];
			qs[i] = seqs[seqs.Length - n + i];
		}
		speakers = s;
		prompts = p;
		replies = r;
		playerIds = ids;
		nonces = ns;
		seqs = qs;
	}

	public int EstimateBytes() {
		var bytes = 64 + 4;
		for(int i = 0; i < Count(); i++)
			bytes += LlmBandwidth.StringBytes(speakers[i]) + LlmBandwidth.StringBytes(prompts[i]) + LlmBandwidth.StringBytes(replies[i]) + 12 + 12;
		bytes += 12 * answeredIds.Length;
		return bytes;
	}

	void Changed() {
		revision++;
		bandwidth.RequestSend(LlmBandwidth.ChannelHistory);
		Notify();
	}

	// Synced from another client and not trusted: same length for the three arrays, at most Capacity, no nulls.
	public void Normalize() {
		if(speakers == null) speakers = new string[0];
		if(prompts == null) prompts = new string[0];
		if(replies == null) replies = new string[0];
		if(playerIds == null) playerIds = new int[0];
		if(nonces == null) nonces = new int[0];
		if(seqs == null) seqs = new int[0];
		if(answeredIds == null) answeredIds = new int[0];
		if(answeredNonces == null) answeredNonces = new int[0];
		if(answeredSeqs == null) answeredSeqs = new int[0];
		var m = Mathf.Min(Mathf.Min(answeredIds.Length, answeredNonces.Length), Mathf.Min(answeredSeqs.Length, LedgerCapacity));
		if(answeredIds.Length != m || answeredNonces.Length != m || answeredSeqs.Length != m) {
			answeredIds = new int[0];
			answeredNonces = new int[0];
			answeredSeqs = new int[0];
		}
		var n = Mathf.Min(Mathf.Min(Mathf.Min(speakers.Length, prompts.Length), Mathf.Min(replies.Length, Capacity)),
			Mathf.Min(playerIds.Length, Mathf.Min(nonces.Length, seqs.Length)));
		Keep(n);
	}

	public override void OnDeserialization() {
		Normalize();
		Notify();
	}

	public override void OnPostSerialization(SerializationResult result) {
		bandwidth.OnSent(LlmBandwidth.ChannelHistory, result.success, result.byteCount);
	}

	void Notify() {
		if(listener != null)
			listener.SendCustomEvent(listenerEvent);
	}
}
}
