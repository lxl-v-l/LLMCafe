using UdonSharp;
using UnityEngine;
using VRC.SDK3.Data;
using VRC.SDK3.UdonNetworkCalling;
using VRC.SDKBase;
using VRC.Udon.Common;
using VRC.Udon.Common.Interfaces;

namespace WorldLLM {
// Worker management and the shared prompt queue (Task 06). The worker is the owner of the LlmNet GameObject
// (this behaviour, LlmResponseStream, LlmHistory and LlmHeartbeat live together, so one SetOwner moves all four).
// "Owner" alone never means "can run the model": workerReady is set only after the new owner set itself up.
[UdonBehaviourSyncMode(BehaviourSyncMode.Manual)]
public class LlmWorkerController : UdonSharpBehaviour {
	public const int QueueCapacity = 8, ProcessedCapacity = 16;

	public LlmBandwidth bandwidth;
	public LlmHeartbeat heartbeat;
	public LlmResponseStream stream;
	public LlmHistory history;
	public LlmPipeline pipeline;
	public LlmSelfTest selfTest;
	public GameObject[] workerOnlyObjects;     // e.g. the tokenizer, activated when this client becomes the worker
	public UdonSharpBehaviour listener;
	public string listenerEvent = "OnNetworkStateChanged";
	public float cooldownSeconds = 10f;
	// Android viewer scene (Task 08): no model assets, this client can never become the worker.
	public bool viewerOnly;
	public float tickSeconds = 0.25f;
	public float absentAfterSeconds = 6f;      // display "no worker"
	public float takeoverAfterSeconds = 20f;   // allow ownership transfer on a silent worker (congestion allowance)
	public float failedTakeoverSeconds = 10f;
	public float progressStallSeconds = 15f;
	public float initStallSeconds = 20f;

	[UdonSynced] public bool workerReady;
	[UdonSynced] public int workerPlayerId = -1;
	[UdonSynced] public string workerName = "";
	[UdonSynced] public int currentRequesterId = -1;
	[UdonSynced] public int queueRevision;
	[UdonSynced] public int[] queuePlayerIds = new int[0];
	[UdonSynced] public int[] queueNonces = new int[0];
	[UdonSynced] public int[] queueSeqs = new int[0];
	[UdonSynced] public string[] queuePrompts = new string[0];
	[UdonSynced] public string[] queueNames = new string[0];
	[UdonSynced] public int[] processedPlayerIds = new int[0];
	[UdonSynced] public int[] processedNonces = new int[0];
	[UdonSynced] public int[] processedSeqs = new int[0];

	[System.NonSerialized] public int joinNonce;
	[System.NonSerialized] public string localMessage = "";
	[System.NonSerialized] public bool localWorkerActive;
	[System.NonSerialized] public bool workerLeft;
	private int clientSeq;
	private bool becomeRequested;
	private bool headInProgress;
	private int headPlayerId = -1, headNonce, headSeq;
	private string headName = "", headPrompt = "";
	private readonly DataDictionary lastSubmitTime = new DataDictionary();
	private int lastProgress = -1;
	private float lastProgressChange;
	private float failedObservedSince = -1f;
	private float initStarted;
	// Commit barrier: a finished prompt leaves the queue only after the history revision holding its answer was
	// acknowledged (OnPostSerialization success). Without an acknowledgement for commitTimeoutSeconds (ClientSim never
	// sends one; a callback may be lost) it leaves once that revision has at least been sent. The history carries the
	// prompt ID, so a worker that takes over while the head is still queued drops it instead of answering it twice.
	private bool commitPending;
	private int commitHistoryRevision;
	private float commitSince;
	public float commitTimeoutSeconds = 10f;
	// Sender side: a prompt sent to the worker waits for its ID to appear in the synced queue (or processed list).
	private int pendingSeq;
	private string pendingText = "";
	private float pendingSince;
	public float acceptTimeoutSeconds = 25f;   // longer than the 20 s congestion allowance used for takeovers
	[System.NonSerialized] public string restoreText = "";   // the UI puts a rejected prompt back into the field

	void Start() {
		joinNonce = Random.Range(1, int.MaxValue);
		SendCustomEventDelayedSeconds(nameof(_Tick), tickSeconds);
	}

	// ---- local entry points (UI)
	public bool SubmitLocal(string rawText) {
		var text = LlmPipeline.SanitizeSharedPrompt(rawText, 200);
		if(text.Length == 0)
			return false;
		if(WorkerAbsent()) {
			localMessage = "ワーカーがいません";
			Notify();
			return false;
		}
		if(!Utilities.IsValid(Networking.LocalPlayer))
			return false;
		// Sender-side hints from synced state; the worker enforces the same rules (Enqueue).
		var now = Time.realtimeSinceStartup;
		if(clientSeq > 0 && now - lastLocalSubmit < cooldownSeconds) {
			localMessage = "送信は " + Mathf.CeilToInt(cooldownSeconds) + " 秒に1回までです";
			Notify();
			return false;
		}
		if(queuePlayerIds.Length >= QueueCapacity && !HasWaitingEntry(Networking.LocalPlayer.playerId)) {
			localMessage = "待ち行列が満杯です（" + QueueCapacity + "件）。しばらくしてから送ってください";
			Notify();
			return false;
		}
		lastLocalSubmit = now;
		clientSeq++;
		pendingSeq = clientSeq;
		pendingText = rawText.Length > 200 ? rawText.Substring(0, 200) : rawText;   // restored as typed
		pendingSince = now;
		if(Networking.IsOwner(gameObject))
			Enqueue(Networking.LocalPlayer, text, joinNonce, clientSeq);
		else
			SendCustomNetworkEvent(NetworkEventTarget.Owner, nameof(SubmitPrompt), text, joinNonce, clientSeq);
		localMessage = "";
		return true;
	}

	// The worker does not report rejections (full queue reached by someone else first, cooldown, worker change), so
	// the sender watches the synced queue: if its prompt never shows up, the text goes back into the input field.
	void CheckPending() {
		if(pendingSeq == 0)
			return;
		var local = Networking.LocalPlayer;
		if(!Utilities.IsValid(local))
			return;
		var id = local.playerId;
		if(Contains(queuePlayerIds, queueNonces, queueSeqs, id, joinNonce, pendingSeq) >= 0
			|| Contains(processedPlayerIds, processedNonces, processedSeqs, id, joinNonce, pendingSeq) >= 0) {
			pendingSeq = 0;
			return;
		}
		if(Time.realtimeSinceStartup - pendingSince < acceptTimeoutSeconds)
			return;
		pendingSeq = 0;
		// An older prompt still waiting to be restored is kept (only a preset, which can be sent again, could reach here
		// while one waits: a typed prompt empties the field, which restores the older one first).
		if(restoreText.Length == 0)
			restoreText = pendingText;
		localMessage = "送信が受け付けられませんでした。入力欄に戻したので、もう一度送ってください";
		Notify();
	}

	private float lastLocalSubmit;

	// A waiting (not yet answered) entry of this player, which a new submission replaces.
	bool HasWaitingEntry(int playerId) {
		for(int i = currentRequesterId >= 0 ? 1 : 0; i < queuePlayerIds.Length; i++)
			if(queuePlayerIds[i] == playerId)
				return true;
		return false;
	}

	public void RequestStopLocal() {
		if(!Utilities.IsValid(Networking.LocalPlayer))
			return;
		if(Networking.IsOwner(gameObject))
			StopCurrent(Networking.LocalPlayer);
		else
			SendCustomNetworkEvent(NetworkEventTarget.Owner, nameof(RequestStop));
	}

	// The worker clears the shared conversation; on other clients Clear only hides the local view (UI handles it).
	public bool ClearShared() {
		if(!Networking.IsOwner(gameObject) || !localWorkerActive)
			return false;
		if(pipeline.status == LlmPipeline.StatusGenerating)
			pipeline.StopGeneration();
		pipeline.ClearHistory();
		history.Clear();
		queuePlayerIds = new int[0]; queueNonces = new int[0]; queueSeqs = new int[0]; queuePrompts = new string[0]; queueNames = new string[0];
		commitPending = false;
		headInProgress = false;
		currentRequesterId = -1;
		stream.Finish("");
		MarkQueueChanged();
		return true;
	}

	public void RequestBecomeWorker() {
#if UNITY_ANDROID || UNITY_IOS
		localMessage = "このプラットフォームではワーカーになれません";
		Notify();
		return;
#else
		if(viewerOnly) {
			localMessage = "閲覧専用版ではワーカーになれません";
			Notify();
			return;
		}
		if(!Utilities.IsValid(Networking.LocalPlayer))
			return;
		if(localWorkerActive && workerReady && Networking.IsOwner(gameObject)) {
			localMessage = "すでにワーカーです";
			Notify();
			return;
		}
		becomeRequested = true;
		if(!selfTest.completed) {
			selfTest.Run();   // continues in OnSelfTestFinished
			return;
		}
		if(!selfTest.passed) {
			becomeRequested = false;
			localMessage = "GPU 自己診断に失敗したため、ワーカーになれません";
			Notify();
			return;
		}
		if(!CanTakeOver(Networking.LocalPlayer)) {
			becomeRequested = false;
			localMessage = "ワーカーが稼働中のため交代できません";
			Notify();
			return;
		}
		if(Networking.IsOwner(gameObject))
			BeginWorker();
		else
			Networking.SetOwner(Networking.LocalPlayer, gameObject);
#endif
	}

	public void OnSelfTestFinished() {
		if(selfTest.timedOut) {
			becomeRequested = false;
			recoverRequested = false;
			localMessage = "GPU 自己診断が応答しませんでした。もう一度押してください";
			Notify();
			return;
		}
		if(becomeRequested)
			RequestBecomeWorker();
		else if(recoverRequested)
			Recover();
	}

	private bool recoverRequested;
	// After FatalError: re-run the self-test and, if it passes, accept prompts again.
	public void Recover() {
		if(!Networking.IsOwner(gameObject) || heartbeat.workerPhase != LlmHeartbeat.PhaseFailed)
			return;
		if(!recoverRequested) {
			recoverRequested = true;
			selfTest.Run();
			return;
		}
		recoverRequested = false;
		if(!selfTest.passed) {
			localMessage = "GPU 自己診断に失敗しました";
			Notify();
			return;
		}
		pipeline.ResetAfterFatal();
		workerReady = true;
		localWorkerActive = true;
		heartbeat.SetWorkerStatus(0, LlmHeartbeat.PhaseReady, pipeline.progressTick);
		MarkQueueChanged();
	}

	// ---- network events
	[NetworkCallable(1)]
	public void SubmitPrompt(string text, int nonce, int seq) {
		if(!Networking.IsOwner(gameObject))
			return;
		var sender = NetworkCalling.CallingPlayer;
		if(!Utilities.IsValid(sender))
			return;
		Enqueue(sender, LlmPipeline.SanitizeSharedPrompt(text, 200), nonce, seq);
	}

	[NetworkCallable(1)]
	public void RequestStop() {
		if(!Networking.IsOwner(gameObject))
			return;
		var sender = NetworkCalling.CallingPlayer;
		if(Utilities.IsValid(sender))
			StopCurrent(sender);
	}

	void StopCurrent(VRCPlayerApi requester) {
		if(!headInProgress)
			return;
		if(requester.playerId != headPlayerId && !requester.isLocal)
			return; // only the requester of the current response or the worker may stop it
		pipeline.StopGeneration();
	}

	// ---- queue (worker side)
	void Enqueue(VRCPlayerApi player, string text, int nonce, int seq) {
		if(!workerReady || text.Length == 0)
			return;
		var id = player.playerId;
		if(System.Array.IndexOf(blockedPlayerIds, id) >= 0)
			return; // blocked locally by this worker
		if(Contains(processedPlayerIds, processedNonces, processedSeqs, id, nonce, seq) >= 0
			|| Contains(queuePlayerIds, queueNonces, queueSeqs, id, nonce, seq) >= 0)
			return; // duplicate
		var now = Time.realtimeSinceStartup;
		if(lastSubmitTime.TryGetValue(id, out var last) && now - last.Float < cooldownSeconds)
			return;
		lastSubmitTime[id] = now;
		var name = LlmPipeline.Sanitize(player.displayName, 32);
		// One pending prompt per player: a new submission replaces a waiting one (not the one being answered).
		for(int i = HeadBusy() ? 1 : 0; i < queuePlayerIds.Length; i++)
			if(queuePlayerIds[i] == id) {
				queueNonces[i] = nonce; queueSeqs[i] = seq; queuePrompts[i] = text; queueNames[i] = name;
				MarkQueueChanged();
				return;
			}
		if(queuePlayerIds.Length >= QueueCapacity)
			return;
		queuePlayerIds = AppendInt(queuePlayerIds, id);
		queueNonces = AppendInt(queueNonces, nonce);
		queueSeqs = AppendInt(queueSeqs, seq);
		queuePrompts = AppendString(queuePrompts, text);
		queueNames = AppendString(queueNames, name);
		MarkQueueChanged();
	}

	// The head entry belongs to the response in progress, or to a finished one waiting for its commit: it must not be
	// replaced or removed as a waiting entry.
	bool HeadBusy() {
		return headInProgress || commitPending;
	}

	void PopHead(bool markProcessed) {
		if(queuePlayerIds.Length == 0)
			return;
		if(markProcessed) {
			processedPlayerIds = KeepLast(AppendInt(processedPlayerIds, queuePlayerIds[0]), ProcessedCapacity);
			processedNonces = KeepLast(AppendInt(processedNonces, queueNonces[0]), ProcessedCapacity);
			processedSeqs = KeepLast(AppendInt(processedSeqs, queueSeqs[0]), ProcessedCapacity);
		}
		queuePlayerIds = RemoveAtInt(queuePlayerIds, 0);
		queueNonces = RemoveAtInt(queueNonces, 0);
		queueSeqs = RemoveAtInt(queueSeqs, 0);
		queuePrompts = RemoveAtString(queuePrompts, 0);
		queueNames = RemoveAtString(queueNames, 0);
	}

	void MarkQueueChanged() {
		queueRevision++;
		bandwidth.RequestSend(LlmBandwidth.ChannelController);
		Notify();
	}

	// ---- worker loop
	public void _Tick() {
		SendCustomEventDelayedSeconds(nameof(_Tick), tickSeconds);
		ObserveFailure();
		CheckPending();
		if(Networking.IsOwner(gameObject) && localWorkerActive)
			WorkerLoop();
		Notify();
	}

	void WorkerLoop() {
		if(!workerReady)
			return;
		if(pipeline.status == LlmPipeline.StatusFatal) {
			EnterFailed("（GPU 読み戻し障害で中断しました）");
			return;
		}
		// Watchdog: a generation whose progress counter stops is treated as a failed worker.
		// The clock only runs while generating. The cafe brain submits to the pipeline directly, so without this the
		// first generation after 15 idle seconds was aborted at once and the worker marked failed.
		var now = Time.realtimeSinceStartup;
		if(pipeline.progressTick != lastProgress || pipeline.status != LlmPipeline.StatusGenerating) {
			lastProgress = pipeline.progressTick;
			lastProgressChange = now;
		}
		if(pipeline.status == LlmPipeline.StatusGenerating && now - lastProgressChange > progressStallSeconds) {
			// Abort, not stop: the partial reply is not committed and the prompt stays at the head of the queue, so
			// it is answered again after recovery (or by the next worker).
			pipeline.AbortGeneration();
			EnterFailed("（応答が止まったため中断しました）");
			return;
		}
		heartbeat.SetWorkerStatus(GeneratorState(), LlmHeartbeat.PhaseReady, pipeline.progressTick);
		if(commitPending) {
			// The fallback applies only while no failure is known for this revision; after success == false the
			// history is sent again and the prompt waits for a successful send.
			var acked = bandwidth.AckRevisionOf(LlmBandwidth.ChannelHistory) >= commitHistoryRevision;
			var sent = bandwidth.SentRevisionOf(LlmBandwidth.ChannelHistory) >= commitHistoryRevision;
			var failed = bandwidth.FailedRevisionOf(LlmBandwidth.ChannelHistory) >= commitHistoryRevision;
			if(!acked && !(sent && !failed && now - commitSince >= commitTimeoutSeconds))
				return;
			commitPending = false;
			currentRequesterId = -1;
			PopHead(true);
			MarkQueueChanged();
		}
		if(headInProgress || queuePlayerIds.Length == 0 || pipeline.IsBusy())
			return;
		if(history.HasPrompt(queuePlayerIds[0], queueNonces[0], queueSeqs[0])) {
			PopHead(true);   // answered by the previous worker, whose queue update did not arrive
			MarkQueueChanged();
			return;
		}
		headPlayerId = queuePlayerIds[0];
		headNonce = queueNonces[0];
		headSeq = queueSeqs[0];
		headName = queueNames[0];
		headPrompt = queuePrompts[0];
		var regenerating = !stream.done && stream.requesterId == headPlayerId && stream.prompt == headPrompt;
		if(!pipeline.SubmitAs(headName, headPrompt)) {
			PopHead(true);
			MarkQueueChanged();
			return;
		}
		headInProgress = true;
		currentRequesterId = headPlayerId;
		lastProgressChange = now;
		stream.Begin(headPlayerId, headName, headPrompt, regenerating);
		MarkQueueChanged();
	}

	// LlmPipeline.listener on the worker. The pipeline may notify while this behaviour is running (the controller
	// calls StopGeneration/AbortGeneration/SubmitAs), so the update is handled one frame later, coalesced.
	private bool pipelineUpdateScheduled;
	public void OnPipelineUpdate() {
		if(pipelineUpdateScheduled)
			return;
		pipelineUpdateScheduled = true;
		SendCustomEventDelayedFrames(nameof(_HandlePipelineUpdate), 1);
	}

	public void _HandlePipelineUpdate() {
		pipelineUpdateScheduled = false;
		if(!Networking.IsOwner(gameObject) || !localWorkerActive || !headInProgress) {
			Notify();
			return;
		}
		if(pipeline.status == LlmPipeline.StatusGenerating) {
			stream.SetText(pipeline.replyText);
		} else if(pipeline.status == LlmPipeline.StatusDone) {
			stream.Finish(pipeline.replyText);
			// A response stopped by the output filter is shown masked but kept out of the shared history, as it is
			// kept out of the model's conversation.
			if(!pipeline.lastBlocked)
				history.Append(headName, headPrompt, stream.text, headPlayerId, headNonce, headSeq);  // text capped like the stream
			history.MarkAnswered(headPlayerId, headNonce, headSeq);   // also for hidden answers: no second answer after a takeover
			if(pipeline.turnCount < history.Count())
				history.TrimTo(pipeline.turnCount);  // the worker's token budget dropped old turns
			headInProgress = false;   // currentRequesterId stays until the commit removes the head
			commitPending = true;
			commitHistoryRevision = bandwidth.DirtyRevisionOf(LlmBandwidth.ChannelHistory);
			commitSince = Time.realtimeSinceStartup;
		}
		Notify();
	}

	void EnterFailed(string message) {
		if(commitPending) {
			// The answer was complete; only the queue update was waiting for the send.
			commitPending = false;
			PopHead(true);
		}
		headInProgress = false;
		currentRequesterId = -1;
		stream.Finish(message);
		workerReady = false;
		heartbeat.SetWorkerStatus(GeneratorState(), LlmHeartbeat.PhaseFailed, pipeline.progressTick);
		MarkQueueChanged();
		selfTest.Run();
	}

	int GeneratorState() {
		return pipeline.status == LlmPipeline.StatusFatal ? ShaderGPT.Udon.GPTGenerator.StateFatalError
			: pipeline.status == LlmPipeline.StatusGenerating ? ShaderGPT.Udon.GPTGenerator.StateGenerating
			: ShaderGPT.Udon.GPTGenerator.StateIdle;
	}

	// ---- ownership
	void BeginWorker() {
		becomeRequested = false;
		localWorkerActive = false;   // set again by the last init step
		NormalizeQueue();
		history.Normalize();
		commitPending = false;
		workerReady = false;
		workerLeft = false;
		workerPlayerId = Networking.LocalPlayer.playerId;
		workerName = LlmPipeline.Sanitize(Networking.LocalPlayer.displayName, 32);
		headInProgress = false;
		currentRequesterId = -1;
		foreach(var go in workerOnlyObjects)
			if(go != null)
				go.SetActive(true);
		initStarted = Time.realtimeSinceStartup;
		heartbeat.SetWorkerStatus(0, LlmHeartbeat.PhaseInitializing, 0);
		MarkQueueChanged();
		ScheduleInitStep();
	}

	// One pending init step at most: a step left over from an earlier term continues the current one.
	private bool initStepScheduled;
	void ScheduleInitStep() {
		if(initStepScheduled)
			return;
		initStepScheduled = true;
		SendCustomEventDelayedFrames(nameof(_InitWorkerStep), 1);
	}

	public void _InitWorkerStep() {
		initStepScheduled = false;
		if(!Networking.IsOwner(gameObject) || localWorkerActive)
			return;
		if(pipeline.status == LlmPipeline.StatusGenerating)
			pipeline.AbortGeneration();   // left over from an earlier term as worker
		if(pipeline.status == LlmPipeline.StatusLoading) {
			// Loading counts as progress for the takeover lease.
			heartbeat.SetWorkerStatus(0, LlmHeartbeat.PhaseInitializing, Mathf.FloorToInt((Time.realtimeSinceStartup - initStarted) * 4f));
			ScheduleInitStep();
			return;
		}
		// The KV cache is rebuilt lazily: the next prompt prefills the imported history from position 0.
		pipeline.ImportHistory(history.speakers, history.prompts, history.replies);
		workerReady = true;
		localWorkerActive = true;
		lastProgressChange = Time.realtimeSinceStartup;
		heartbeat.SetWorkerStatus(0, LlmHeartbeat.PhaseReady, pipeline.progressTick);
		MarkQueueChanged();
	}

	// Only self-requests are valid: nobody may hand the worker role to a third player, who did not volunteer.
	public override bool OnOwnershipRequest(VRCPlayerApi requestingPlayer, VRCPlayerApi requestedOwner) {
		if(!Utilities.IsValid(requestingPlayer) || !Utilities.IsValid(requestedOwner) || requestingPlayer.playerId != requestedOwner.playerId)
			return false;
		return CanTakeOver(requestedOwner);
	}

	public override void OnOwnershipTransferred(VRCPlayerApi player) {
		// Every client stops trusting the previous worker immediately; the new owner sets workerReady itself.
		var wasActive = localWorkerActive;
		localWorkerActive = player.isLocal && localWorkerActive;
		if(!player.isLocal) {
			// This client lost (or never had) the role: stop generating and forget any pending volunteer request.
			becomeRequested = false;
			commitPending = false;
			headInProgress = false;
			if(wasActive)
				pipeline.AbortGeneration();
			Notify();
			return;
		}
		if(becomeRequested) {
			BeginWorker();
			return;
		}
		// Automatic transfer after the owner left: this client did not volunteer, so it only clears the flag.
		workerReady = false;
		workerPlayerId = -1;
		workerName = "";
		headInProgress = false;
		heartbeat.SetWorkerStatus(0, LlmHeartbeat.PhaseNone, 0);
		MarkQueueChanged();
	}

	// ---- local block list (worker only, not synced; Task 08)
	[System.NonSerialized] public int[] blockedPlayerIds = new int[0];

	// Blocks the player whose prompt is being answered now.
	public void BlockCurrentRequester() {
		if(!Networking.IsOwner(gameObject) || !localWorkerActive || !headInProgress)
			return;
		var local = Networking.LocalPlayer;
		if(Utilities.IsValid(local) && headPlayerId == local.playerId)
			return;
		BlockPlayer(headPlayerId);
	}

	// The player's prompts are ignored on this worker until it leaves the instance: queued ones are removed and a
	// response in progress for them is stopped.
	public void BlockPlayer(int playerId) {
		if(System.Array.IndexOf(blockedPlayerIds, playerId) < 0)
			blockedPlayerIds = AppendInt(blockedPlayerIds, playerId);
		RemoveWaiting(playerId);
		localMessage = "ブロックしました（このワーカーの間だけ有効）";
		if(headInProgress && headPlayerId == playerId)
			pipeline.StopGeneration();
		Notify();
	}

	void RemoveWaiting(int playerId) {
		for(int i = queuePlayerIds.Length - 1; i >= (HeadBusy() ? 1 : 0); i--)
			if(queuePlayerIds[i] == playerId) {
				queuePlayerIds = RemoveAtInt(queuePlayerIds, i);
				queueNonces = RemoveAtInt(queueNonces, i);
				queueSeqs = RemoveAtInt(queueSeqs, i);
				queuePrompts = RemoveAtString(queuePrompts, i);
				queueNames = RemoveAtString(queueNames, i);
				MarkQueueChanged();
			}
	}

	public override void OnPlayerLeft(VRCPlayerApi player) {
		if(!Utilities.IsValid(player))
			return;
		if(player.playerId == workerPlayerId)
			workerLeft = true;
		if(!Networking.IsOwner(gameObject))
			return;
		RemoveWaiting(player.playerId);
	}

	// Ownership rule, evaluated identically on
	// every client from synced state:
	// (0) the current worker may always volunteer again (handoff on the same client);
	// (1) a healthy worker that is generating, or whose answer waits for its commit, keeps the role against everyone;
	// (2) with no worker, or a replaceable one (left, silent heartbeat, Failed/FatalError for failedTakeoverSeconds,
	//     stalled initialization), any PC player may take it;
	// (3) a healthy idle or initializing worker may be replaced only by the instance owner (in Group/Public instances
	//     nobody is the instance owner, so a healthy worker keeps the role there).
	public bool CanTakeOver(VRCPlayerApi requester) {
		if(!Utilities.IsValid(requester))
			return false;
		if(requester.playerId == workerPlayerId)
			return true;
		var silent = heartbeat.SecondsSinceBeat() > takeoverAfterSeconds;
		var generating = heartbeat.state == ShaderGPT.Udon.GPTGenerator.StatePrefill || heartbeat.state == ShaderGPT.Udon.GPTGenerator.StateGenerating;
		// currentRequesterId stays set until the answered prompt leaves the queue (commit barrier).
		if(workerReady && (generating || currentRequesterId >= 0) && !silent && !workerLeft)
			return false;
		var noWorker = !workerReady && (heartbeat.workerPhase == LlmHeartbeat.PhaseNone || workerPlayerId < 0);
		var failedLongEnough = failedObservedSince >= 0f && Time.realtimeSinceStartup - failedObservedSince >= failedTakeoverSeconds;
		var initStalled = (heartbeat.workerPhase == LlmHeartbeat.PhaseInitializing || heartbeat.workerPhase == LlmHeartbeat.PhaseLoadingWeights)
			&& heartbeat.SecondsSinceProgress() > initStallSeconds;
		if(noWorker || workerLeft || silent || failedLongEnough || initStalled)
			return true;
		return requester.isInstanceOwner;
	}

	void ObserveFailure() {
		var failed = heartbeat.workerPhase == LlmHeartbeat.PhaseFailed || heartbeat.state == ShaderGPT.Udon.GPTGenerator.StateFatalError;
		if(!failed)
			failedObservedSince = -1f;
		else if(failedObservedSince < 0f)
			failedObservedSince = Time.realtimeSinceStartup;
	}

	// ---- state for the UI
	public bool WorkerAbsent() {
		return !workerReady || workerLeft || heartbeat.SecondsSinceBeat() > absentAfterSeconds;
	}

	public int LocalQueuePosition() {
		var localPlayer = Networking.LocalPlayer;
		if(!Utilities.IsValid(localPlayer))
			return -1;  // the local player may not exist yet right after the world loads
		var id = localPlayer.playerId;
		for(int i = 0; i < queuePlayerIds.Length; i++)
			if(queuePlayerIds[i] == id)
				return i;
		return -1;
	}

	public int EstimateBytes() {
		var bytes = 64 + 4 * 5 + LlmBandwidth.StringBytes(workerName);
		for(int i = 0; i < queuePlayerIds.Length; i++)
			bytes += 12 + LlmBandwidth.StringBytes(queuePrompts[i]) + LlmBandwidth.StringBytes(queueNames[i]);
		return bytes + 12 * processedPlayerIds.Length;
	}

	// Synced arrays come from another client and are not trusted: null arrays become empty, parallel arrays are cut
	// to their common length and to the capacity, null strings become empty. A malformed queue would otherwise make
	// the next worker's loop index out of range and halt this behaviour.
	void NormalizeQueue() {
		if(queuePlayerIds == null) queuePlayerIds = new int[0];
		if(queueNonces == null) queueNonces = new int[0];
		if(queueSeqs == null) queueSeqs = new int[0];
		if(queuePrompts == null) queuePrompts = new string[0];
		if(queueNames == null) queueNames = new string[0];
		var n = Mathf.Min(Mathf.Min(Mathf.Min(queuePlayerIds.Length, queueNonces.Length), Mathf.Min(queueSeqs.Length, queuePrompts.Length)), Mathf.Min(queueNames.Length, QueueCapacity));
		queuePlayerIds = CutInt(queuePlayerIds, n);
		queueNonces = CutInt(queueNonces, n);
		queueSeqs = CutInt(queueSeqs, n);
		queuePrompts = CutString(queuePrompts, n);
		queueNames = CutString(queueNames, n);
		if(processedPlayerIds == null) processedPlayerIds = new int[0];
		if(processedNonces == null) processedNonces = new int[0];
		if(processedSeqs == null) processedSeqs = new int[0];
		var m = Mathf.Min(Mathf.Min(processedPlayerIds.Length, processedNonces.Length), Mathf.Min(processedSeqs.Length, ProcessedCapacity));
		processedPlayerIds = CutInt(processedPlayerIds, m);
		processedNonces = CutInt(processedNonces, m);
		processedSeqs = CutInt(processedSeqs, m);
		if(workerName == null) workerName = "";
	}

	static int[] CutInt(int[] a, int n) {
		if(a.Length == n)
			return a;
		var r = new int[n];
		System.Array.Copy(a, r, n);
		return r;
	}

	static string[] CutString(string[] a, int n) {
		var r = a.Length == n ? a : new string[n];
		if(r != a)
			System.Array.Copy(a, r, n);
		for(int i = 0; i < n; i++)
			if(r[i] == null)
				r[i] = "";
		return r;
	}

	private int seenWorkerPlayerId = -1;
	public override void OnDeserialization() {
		NormalizeQueue();
		if(workerPlayerId != seenWorkerPlayerId) {
			seenWorkerPlayerId = workerPlayerId;
			workerLeft = false;  // a different worker took over
		}
		Notify();
	}

	public override void OnPostSerialization(SerializationResult result) {
		bandwidth.OnSent(LlmBandwidth.ChannelController, result.success, result.byteCount);
	}

	void Notify() {
		if(listener != null)
			listener.SendCustomEvent(listenerEvent);
	}

	// ---- array helpers (Udon has no List<T>)
	static int Contains(int[] ids, int[] nonces, int[] seqs, int id, int nonce, int seq) {
		for(int i = 0; i < ids.Length; i++)
			if(ids[i] == id && nonces[i] == nonce && seqs[i] == seq)
				return i;
		return -1;
	}
	static int[] AppendInt(int[] a, int v) {
		var r = new int[a.Length + 1];
		System.Array.Copy(a, r, a.Length);
		r[a.Length] = v;
		return r;
	}
	static string[] AppendString(string[] a, string v) {
		var r = new string[a.Length + 1];
		System.Array.Copy(a, r, a.Length);
		r[a.Length] = v;
		return r;
	}
	static int[] RemoveAtInt(int[] a, int index) {
		var r = new int[a.Length - 1];
		System.Array.Copy(a, 0, r, 0, index);
		System.Array.Copy(a, index + 1, r, index, a.Length - index - 1);
		return r;
	}
	static string[] RemoveAtString(string[] a, int index) {
		var r = new string[a.Length - 1];
		System.Array.Copy(a, 0, r, 0, index);
		System.Array.Copy(a, index + 1, r, index, a.Length - index - 1);
		return r;
	}
	static int[] KeepLast(int[] a, int n) {
		if(a.Length <= n)
			return a;
		var r = new int[n];
		System.Array.Copy(a, a.Length - n, r, 0, n);
		return r;
	}
}
}
