using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon.Common;

namespace WorldLLM {
// Small synced liveness record of the worker (Task 06). Kept apart from the controller so that the 2-second
// heartbeat never re-sends the prompt queue.
[UdonBehaviourSyncMode(BehaviourSyncMode.Manual)]
public class LlmHeartbeat : UdonSharpBehaviour {
	public const int PhaseNone = 0, PhaseInitializing = 1, PhaseLoadingWeights = 2, PhaseReady = 3, PhaseFailed = 4;

	public LlmBandwidth bandwidth;
	public float beatSeconds = 2f;

	[UdonSynced] public int heartbeat;
	[UdonSynced] public int progressTick;
	[UdonSynced] public int state;        // generator state contract (0..4)
	[UdonSynced] public int workerPhase;

	// Local time of the last observed change, for "worker absent" detection on every client.
	[System.NonSerialized] public float lastBeatTime;
	[System.NonSerialized] public float lastProgressTime;
	private int seenHeartbeat = -1, seenProgress = -1;

	void Start() {
		lastBeatTime = Time.realtimeSinceStartup;
		lastProgressTime = lastBeatTime;
		SendCustomEventDelayedSeconds(nameof(_Beat), beatSeconds);
	}

	public void _Beat() {
		SendCustomEventDelayedSeconds(nameof(_Beat), beatSeconds);
		if(!Networking.IsOwner(gameObject))
			return;
		heartbeat++;
		Observe();
		bandwidth.RequestSend(LlmBandwidth.ChannelHeartbeat);
	}

	public void SetWorkerStatus(int generatorState, int phase, int progress) {
		var changed = state != generatorState || workerPhase != phase;
		state = generatorState;
		workerPhase = phase;
		progressTick = progress;
		Observe();
		if(changed)
			bandwidth.RequestSend(LlmBandwidth.ChannelHeartbeat);
	}

	public override void OnDeserialization() {
		Observe();
	}

	public override void OnPostSerialization(SerializationResult result) {
		bandwidth.OnSent(LlmBandwidth.ChannelHeartbeat, result.success, result.byteCount);
	}

	void Observe() {
		var now = Time.realtimeSinceStartup;
		if(heartbeat != seenHeartbeat) {
			seenHeartbeat = heartbeat;
			lastBeatTime = now;
		}
		if(progressTick != seenProgress) {
			seenProgress = progressTick;
			lastProgressTime = now;
		}
	}

	public float SecondsSinceBeat() {
		return Time.realtimeSinceStartup - lastBeatTime;
	}

	public float SecondsSinceProgress() {
		return Time.realtimeSinceStartup - lastProgressTime;
	}
}
}
