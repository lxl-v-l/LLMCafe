using UdonSharp;
using UnityEngine;
using VRC.SDKBase;

namespace WorldLLM {
// Local send scheduler for the four synced behaviours on LlmNet (Task 06). Nothing calls RequestSerialization
// directly; behaviours mark themselves dirty here and this script decides when each may send so that this
// client's Udon output stays within budgetBytes per windowSeconds (VRChat's limit is per sending client).
//  - 6,000 B of the budget are reserved for Heartbeat and Controller, so liveness and the prompt queue always go out.
//  - Every send is recorded in a per-channel FIFO with the revision it carried and a reservation of its estimate.
//    OnPostSerialization callbacks arrive in send order, so each callback is matched to the oldest send of its channel:
//    its byteCount enters the 10 s window and, on success, only that send's revision counts as acknowledged.
//  - A send without a callback for 5 s expires: its reservation is released and the channel may send again.
//    Callbacks carry no send identity, so after the first expiry the channel cannot match callbacks to sends any more.
//    It switches to degraded mode until this client stops being the owner: callbacks only count bytes and flag a
//    failure on the oldest send still in flight (conservative), and each send counts as acknowledged when it expires
//    without a failure flag (state sync sends the full state each time, so a
//    lost update is repaired by the next change). Degraded mode never acknowledges a send by someone else's callback
//    and never leaves a channel pending forever (which would starve the Stream). ClientSim, which sends no callbacks
//    for world objects, runs in degraded mode. By design degraded mode treats "no failure report" as success: a retry
//    whose callback is lost is acknowledged by its timeout, like any other send in this mode.
//  - A send that expires without a callback is counted in the 10 s window at its estimate (from its send time), so
//    lost callbacks cannot hide bytes. In degraded mode callbacks add no bytes (the estimate is already counted).
//  - A client that loses ownership with sends still in flight starts its next ownership term in degraded mode: their
//    late callbacks cannot be told from callbacks of the new term.
//  - Requests are coalesced per channel; the estimate is taken from the latest state right before sending.
//  - Priority: Heartbeat > Controller > History > Stream. Stream also waits while a higher channel is pending.
[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class LlmBandwidth : UdonSharpBehaviour {
	public const int ChannelHeartbeat = 0, ChannelController = 1, ChannelHistory = 2, ChannelStream = 3;
	const int Channels = 4;
	const int FlightCapacity = 4;   // sends per channel awaiting a callback

	public LlmHeartbeat heartbeat;
	public LlmWorkerController controller;
	public LlmHistory history;
	public LlmResponseStream stream;
	public int budgetBytes = 30000;
	public int reservedBytes = 6000;
	public float windowSeconds = 10f;
	public float streamMinInterval = 0.3f;
	public float estimateFactor = 1.5f;
	public float reservationTimeout = 5f;

	[System.NonSerialized] public int windowBytes;
	[System.NonSerialized] public int sentTotal;
	[System.NonSerialized] public int maxRatioPercent;   // max(actual / raw estimate) observed, x100
	[System.NonSerialized] public int failures;
	[System.NonSerialized] public int schedulerResets;
	[System.NonSerialized] public int onSentCalls, onSentSuccesses, lastByteCount, lateCallbacks;

	private readonly int[] dirtyRevision = new int[Channels];
	private readonly int[] sentRevision = new int[Channels];
	private readonly int[] ackRevision = new int[Channels];
	private readonly int[] consecutiveTimeouts = new int[Channels];
	private readonly bool[] degraded = new bool[Channels];
	private readonly int[] failedRevision = new int[Channels];   // newest revision reported with success == false
	private bool wasOwner;
	private float lastStreamSend = -100f;
	// In-flight sends: channel c uses slots [c * FlightCapacity, (c + 1) * FlightCapacity) as a ring.
	private readonly int[] flightHead = new int[Channels];
	private readonly int[] flightCount = new int[Channels];
	private readonly int[] flightRevision = new int[Channels * FlightCapacity];
	private readonly int[] flightReserve = new int[Channels * FlightCapacity];   // 0 once expired
	private readonly int[] flightRaw = new int[Channels * FlightCapacity];
	private readonly float[] flightTime = new float[Channels * FlightCapacity];
	private readonly bool[] flightFailed = new bool[Channels * FlightCapacity];   // degraded mode: a failure was reported
	private readonly bool[] outstandingAtLoss = new bool[Channels];
	// Window entries: time and bytes of actual sends.
	private readonly float[] entryTime = new float[256];
	private readonly int[] entryBytes = new int[256];
	private int entryHead, entryCount;

	public void RequestSend(int channel) {
		dirtyRevision[channel]++;
	}

	// Revision of the latest request on a channel, and of the latest state handed to RequestSerialization.
	// The controller uses these as a commit barrier: a prompt leaves the queue only after its answer and history
	// entry have been sent.
	public int DirtyRevisionOf(int channel) {
		return dirtyRevision[channel];
	}

	public int SentRevisionOf(int channel) {
		return sentRevision[channel];
	}

	// Latest revision confirmed by a successful OnPostSerialization (or, in degraded mode, by an expiry).
	public int AckRevisionOf(int channel) {
		return ackRevision[channel];
	}

	// Newest revision whose send was reported as failed.
	public int FailedRevisionOf(int channel) {
		return failedRevision[channel];
	}

	public void OnSent(int channel, bool success, int byteCount) {
		onSentCalls++;
		if(success)
			onSentSuccesses++;
		lastByteCount = byteCount;
		sentTotal += byteCount;
		if(degraded[channel]) {
			// The callback cannot be matched to a send: count the bytes, and attribute a failure to the oldest send
			// still in flight (so it is not acknowledged by its timeout). Reservations are released only by timeouts,
			// which keeps one send per channel at a time.
			lateCallbacks++;
			if(!success) {
				failures++;
				for(int k = 0; k < flightCount[channel]; k++) {
					var s = channel * FlightCapacity + (flightHead[channel] + k) % FlightCapacity;
					if(flightRevision[s] > 0) {
						flightFailed[s] = true;
						failedRevision[channel] = Mathf.Max(failedRevision[channel], flightRevision[s]);
						break;
					}
				}
			}
			return;
		}
		if(flightCount[channel] == 0) {
			AddEntry(byteCount, Time.realtimeSinceStartup);
			lateCallbacks++;   // no record: counted in the window, nothing acknowledged
			return;
		}
		var slot = channel * FlightCapacity + flightHead[channel];
		flightHead[channel] = (flightHead[channel] + 1) % FlightCapacity;
		flightCount[channel]--;
		if(flightReserve[slot] > 0) {
			consecutiveTimeouts[channel] = 0;
			flightReserve[slot] = 0;
			AddEntry(byteCount, Time.realtimeSinceStartup);
		} else {
			lateCallbacks++;   // its estimate entered the window when it expired
		}
		if(flightRaw[slot] > 0)
			maxRatioPercent = Mathf.Max(maxRatioPercent, byteCount * 100 / flightRaw[slot]);
		if(success) {
			ackRevision[channel] = Mathf.Max(ackRevision[channel], flightRevision[slot]);
		} else {
			failures++;  // the latest state stays unacknowledged and is sent again
			failedRevision[channel] = Mathf.Max(failedRevision[channel], flightRevision[slot]);
		}
	}

	void Update() {
		var now = Time.realtimeSinceStartup;
		ExpireEntries(now);
		var owner = Networking.IsOwner(heartbeat.gameObject);
		if(!owner && wasOwner)
			for(int c = 0; c < Channels; c++)
				outstandingAtLoss[c] = flightCount[c] > 0;
		if(owner && !wasOwner) {
			ResetScheduler();
			for(int c = 0; c < Channels; c++) {
				degraded[c] = outstandingAtLoss[c];   // late callbacks of the previous term may still arrive
				outstandingAtLoss[c] = false;
			}
		}
		wasOwner = owner;
		for(int c = 0; c < Channels; c++)
			for(int k = 0; k < flightCount[c]; k++) {
				var slot = c * FlightCapacity + (flightHead[c] + k) % FlightCapacity;
				if(flightRevision[slot] > 0 && now - flightTime[slot] > reservationTimeout) {
					var wasActive = flightReserve[slot] > 0;
					if(wasActive)
						AddEntry(flightReserve[slot], flightTime[slot]);   // no callback: count the estimate
					flightReserve[slot] = 0;
					if(!degraded[c] && wasActive) {
						degraded[c] = true;
						failures++;
						Debug.LogWarning("[WorldLLM] serialization callback missing on channel " + c + "; acknowledging by timeout from now on");
					}
					if(degraded[c]) {
						if(!flightFailed[slot])
							ackRevision[c] = Mathf.Max(ackRevision[c], flightRevision[slot]);
						flightRevision[slot] = 0;   // handled
					}
				}
			}
		// Drop handled entries from the head of each ring.
		for(int c = 0; c < Channels; c++)
			while(flightCount[c] > 0 && flightRevision[c * FlightCapacity + flightHead[c]] == 0 && degraded[c]) {
				flightHead[c] = (flightHead[c] + 1) % FlightCapacity;
				flightCount[c]--;
			}
		var higherPending = false;
		for(int c = 0; c < Channels; c++) {
			var pending = dirtyRevision[c] > ackRevision[c];
			if(!pending || ActiveFlights(c) > 0) {
				higherPending |= pending;
				continue;
			}
			if(!IsOwnerOf(c))
				continue;
			if(Networking.IsClogged && c != ChannelHeartbeat) {
				higherPending = true;
				continue;
			}
			if(c == ChannelStream && (higherPending || now - lastStreamSend < streamMinInterval))
				continue;
			var raw = EstimateBytes(c);
			var estimate = Mathf.Max(1, Mathf.CeilToInt(raw * estimateFactor));
			var limit = c <= ChannelController ? budgetBytes : budgetBytes - reservedBytes;
			if(windowBytes + Reserved() + estimate > limit) {
				higherPending = true;
				continue;
			}
			if(flightCount[c] == FlightCapacity) {
				// Only expired sends are left (an active one blocks above): forget the oldest.
				flightHead[c] = (flightHead[c] + 1) % FlightCapacity;
				flightCount[c]--;
			}
			var newSlot = c * FlightCapacity + (flightHead[c] + flightCount[c]) % FlightCapacity;
			flightCount[c]++;
			flightRevision[newSlot] = dirtyRevision[c];
			flightReserve[newSlot] = estimate;
			flightRaw[newSlot] = raw;
			flightTime[newSlot] = now;
			flightFailed[newSlot] = false;
			sentRevision[c] = dirtyRevision[c];
			if(c == ChannelStream)
				lastStreamSend = now;
			Serialize(c);
			higherPending = true;
		}
	}

	int ActiveFlights(int c) {
		var n = 0;
		for(int k = 0; k < flightCount[c]; k++)
			if(flightReserve[c * FlightCapacity + (flightHead[c] + k) % FlightCapacity] > 0)
				n++;
		return n;
	}

	bool IsOwnerOf(int c) {
		return Networking.IsOwner(heartbeat.gameObject);  // all four live on the LlmNet GameObject
	}

	int Reserved() {
		var sum = 0;
		for(int i = 0; i < flightReserve.Length; i++)
			sum += flightReserve[i];
		return sum;
	}

	int EstimateBytes(int c) {
		if(c == ChannelHeartbeat)
			return 16 + 64;
		if(c == ChannelController)
			return controller.EstimateBytes();
		if(c == ChannelHistory)
			return history.EstimateBytes();
		return stream.EstimateBytes();
	}

	void Serialize(int c) {
		if(c == ChannelHeartbeat)
			heartbeat.RequestSerialization();
		else if(c == ChannelController)
			controller.RequestSerialization();
		else if(c == ChannelHistory)
			history.RequestSerialization();
		else
			stream.RequestSerialization();
	}

	void AddEntry(int bytes, float time) {
		if(entryCount == entryTime.Length) {
			windowBytes -= entryBytes[entryHead];
			entryHead = (entryHead + 1) % entryTime.Length;
			entryCount--;
		}
		var i = (entryHead + entryCount) % entryTime.Length;
		entryTime[i] = time;
		entryBytes[i] = bytes;
		entryCount++;
		windowBytes += bytes;
	}

	void ExpireEntries(float now) {
		while(entryCount > 0 && now - entryTime[entryHead] > windowSeconds) {
			windowBytes -= entryBytes[entryHead];
			entryHead = (entryHead + 1) % entryTime.Length;
			entryCount--;
		}
	}

	void ResetScheduler() {
		schedulerResets++;
		for(int c = 0; c < Channels; c++) {
			flightHead[c] = 0;
			flightCount[c] = 0;
			consecutiveTimeouts[c] = 0;
			degraded[c] = false;
		}
		for(int i = 0; i < flightReserve.Length; i++) {
			flightReserve[i] = 0;
			flightRevision[i] = 0;
			flightFailed[i] = false;
		}
	}

	// UTF-16 chars x 2 + 4 bytes per array element + 64 bytes overhead (Task 06 estimate formula).
	public static int StringBytes(string s) {
		return s == null ? 0 : s.Length * 2;
	}
}
}
