using UdonSharp;
using UnityEngine;
using UnityEngine.UI;
using VRC.SDK3.Data;
using VRC.SDK3.UdonNetworkCalling;
using VRC.SDKBase;
using VRC.Udon.Common.Interfaces;
using WorldLLM;

namespace CafeNPC
{
    [UdonBehaviourSyncMode(BehaviourSyncMode.Manual)]
    public class CafeNpcBrain : UdonSharpBehaviour
    {
        public CafeNpcController actor;
        public LlmWorkerController worker;
        public LlmPipeline pipeline;
        public InputField input;
        public Text statusText;
        public GameObject conversationPanel;
        public GameObject settingsPanel;
        public InputField instructionsInput;
        public Text settingsStatus;
        public bool useJobChoices;
        public CafeDecider decider;
        public bool useDecider;

        [UdonSynced]
        public string behaviorInstructions =
            "カフェの店員として働く。お客様の注文を聞き、必要な物を準備して届ける。日本語で丁寧に接する。";

        [UdonSynced]
        public int configurationRevision;

        [UdonSynced]
        public string guestMessage = "";

        [UdonSynced]
        public int messageRevision;

        [UdonSynced]
        public string lastDecision = "";

        [UdonSynced]
        public int servedMessageRevision;

        [UdonSynced]
        public int observedServedCount;

        [UdonSynced]
        public string recentEvents = "";

        [UdonSynced]
        public int observedJobCount;

        [UdonSynced]
        public int requestGuestId = -1;
        private int decisionMessageRevision;
        private int servingMessageRevision;

        public void OpenConversation()
        {
            var local = Networking.LocalPlayer;
            if (!Utilities.IsValid(local) || conversationPanel == null)
                return;
            conversationPanel.transform.SetParent(null, true);
            conversationPanel.SetActive(true);
            PlaceConversation(local);
        }

        public void BecomeWorker()
        {
            worker.RequestBecomeWorker();
        }

        public void CloseConversation()
        {
            conversationPanel.SetActive(false);
            SetTypingFreeze(false);
        }

        public void OpenSettings()
        {
            instructionsInput.text = behaviorInstructions;
            settingsPanel.SetActive(true);
            settingsStatus.text = "設定変更は本人確認方式の決定後に有効になります";
        }

        public void CloseSettings()
        {
            settingsPanel.SetActive(false);
        }

        public void ApplyInstructions()
        {
            // Authorization is intentionally closed until the user chooses an identity mechanism.
            // Instance ownership and volunteering for inference do not authorize configuration edits.
            settingsStatus.text = "設定変更は本人確認方式の決定後に有効になります";
        }

        private string shownStatus;

        // Written only when it changes: assigning Text.text every frame rebuilds the canvas every frame.
        void Status(string text)
        {
            if (text == shownStatus)
                return;
            shownStatus = text;
            statusText.text = text;
        }

        public string lastObservation = "";
        public string lastModelOutput = "";
        public int acceptedDecisions;
        private VRCPlayerApi[] players = new VRCPlayerApi[80];
        private bool pending;
        private float nextDecision;
        private float lastMessageAt = -10;

        public string GuestLocation()
        {
            var guest = VRCPlayerApi.GetPlayerById(requestGuestId);
            if (!Utilities.IsValid(guest))
                return "発言者の場所は未確認";
            Vector3 position = guest.GetPosition();
            int nearest = -1;
            float distance = 1.6f;
            if (actor.placementAnchors != null)
                for (int i = 0; i < actor.route.Length; i++)
                {
                    if (actor.placementAnchors[i] == null)
                        continue;
                    float candidate = Vector3.Distance(position, actor.route[i].position);
                    if (candidate < distance)
                    {
                        distance = candidate;
                        nearest = i;
                    }
                }
            return nearest >= 0
                ? "発言者は" + actor.StationName(nearest) + "の近く"
                : "発言者は席を特定できない場所にいる";
        }

        void Update()
        {
            UpdateLocalConversation();
            if (!worker.localWorkerActive || !Networking.IsOwner(worker.gameObject))
            {
                pending = false;
                Status("ワーカーがいません。入口の壁のボードで「ワーカーになる」を押してください");
                // Walking to a table needs no model: without a worker, whoever owns the NPC still answers the bell.
                if (useDecider && Networking.IsOwner(gameObject) && Networking.IsOwner(actor.gameObject))
                    UpdateBell();
                return;
            }
            if (!Networking.IsOwner(gameObject))
            {
                Networking.SetOwner(Networking.LocalPlayer, gameObject);
                return;
            }
            if (!Networking.IsOwner(actor.gameObject))
            {
                Networking.SetOwner(Networking.LocalPlayer, actor.gameObject);
                return;
            }
            if (actor.servedCount > observedServedCount)
            {
                observedServedCount = actor.servedCount;
                servedMessageRevision = servingMessageRevision;
                recentEvents = "注文" + servedMessageRevision + "のコーヒーを配膳済み。";
                RequestSerialization();
            }
            if (actor.completedJobCount > observedJobCount)
            {
                observedJobCount = actor.completedJobCount;
                recentEvents = LlmPipeline.Sanitize(
                    "依頼" + actor.completedJobRevision + ": " + actor.completedJobResult + "。\n" + recentEvents,
                    300
                );
                if (useDecider)
                    OnJobCompleted();
                RequestSerialization();
            }
            if (useDecider)
            {
                UpdateService();
                return;
            }
            if (pending)
            {
                Status("茜犬が考えています…");
                if (pipeline.status == LlmPipeline.StatusGenerating && FirstObject(pipeline.replyText).Length > 0)
                    pipeline.StopGeneration();
                if (pipeline.status == LlmPipeline.StatusDone)
                {
                    pending = false;
                    lastModelOutput = pipeline.replyText;
                    if (decisionMessageRevision == messageRevision)
                        ApplyModelDecision();
                    nextDecision = Time.time + 5;
                }
                else if (pipeline.status == LlmPipeline.StatusError || pipeline.status == LlmPipeline.StatusFatal)
                {
                    pending = false;
                    Status("推論エラー。自律行動は待機中です");
                    nextDecision = Time.time + 15;
                }
                return;
            }
            Status("茜犬: " + lastDecision + "\n" + actor.actionResult);
            if (!actor.actionComplete || Time.time < nextDecision || pipeline.IsBusy())
                return;
            string nearby = "";
            int count = 0;
            VRCPlayerApi.GetPlayers(players);
            for (int i = 0; i < players.Length; i++)
            {
                var p = players[i];
                if (!Utilities.IsValid(p))
                    continue;
                var pos = p.GetPosition();
                if (Mathf.Abs(pos.x) > 6 || pos.z < -5 || pos.z > 5 || Mathf.Abs(pos.y) > 3)
                    continue;
                if (count < 4)
                    nearby += p.playerId + ":" + LlmPipeline.Sanitize(p.displayName, 20) + " ";
                if (count == 0 && requestGuestId < 0)
                    actor.guestId = p.playerId;
                count++;
            }
            if (requestGuestId >= 0)
                actor.guestId = requestGuestId;
            string available = actor.AvailableActions();
            lastObservation =
                "OPERATOR INSTRUCTIONS: "
                + behaviorInstructions
                + "\nCURRENT STATE: location="
                + actor.station
                + "("
                + actor.StationName(actor.station)
                + "); holding_coffee="
                + actor.holdingDrink
                + "; guests="
                + count
                + " ["
                + nearby
                + "].\n"
                + (
                    messageRevision > servedMessageRevision
                        ? "ACTIVE GUEST REQUEST #" + messageRevision + ": " + guestMessage
                        : "NO ACTIVE REQUEST. Previous request completed; do not prepare or deliver again."
                )
                + "\nLAST DELIVERED REQUEST #"
                + servedMessageRevision
                + ". MEMORY: "
                + recentEvents
                + "\nLAST ACTION RESULT: "
                + actor.actionResult
                + "\nAVAILABLE NOW: "
                + available
                + "\nChoose one useful action to advance the guest request. Do not repeat completed speech or deliveries. JSON only.";
            if (useJobChoices)
            {
                lastObservation =
                    "仕事内容: "
                    + LlmPipeline.Sanitize(behaviorInstructions, 500)
                    + "\n最新の発言 #"
                    + messageRevision
                    + ": "
                    + guestMessage
                    + "\n"
                    + GuestLocation()
                    + "\n現在地: "
                    + actor.StationName(actor.station)
                    + "。持ち物: "
                    + (
                        actor.holdingDrink ? "コーヒー"
                        : actor.heldItem >= 0 ? actor.ItemName(actor.heldItem)
                        : "なし"
                    )
                    + "\n物: "
                    + actor.ItemState()
                    + "\n結果: "
                    + LlmPipeline.Sanitize(actor.actionResult, 100)
                    + "\n実行済み: "
                    + LlmPipeline.Sanitize(recentEvents, 180)
                    + "\n今できる仕事:\n"
                    + actor.BuildJobChoices();
                if (lastObservation.Length > 1200)
                {
                    Status("候補が入力上限を超えました。自律行動は待機中です");
                    nextDecision = Time.time + 15;
                    return;
                }
            }
            decisionMessageRevision = messageRevision;
            pipeline.ClearHistory();
            if (pipeline.SubmitObservation(lastObservation))
            {
                pending = true;
                Debug.Log("[CafeNPC observation] " + lastObservation);
            }
            else
                nextDecision = Time.time + 5;
        }

        private bool typingFrozen;
        private int frozenSeat = -1;
        public VRC.SDK3.Components.VRCStation[] seatObjects; // per seat index (set by Register Tables And Seats)

        // Keys typed into the order field also reach locomotion: W/A/S/D walk and Space (also the IME conversion key) jumps,
        // which leaves a seat. A seated guest keeps the seat (exit disabled) instead of being immobilized: Immobilize(false)
        // would also lift the seat's own immobilization.
        void SetTypingFreeze(bool typing)
        {
            var local = Networking.LocalPlayer;
            if (typing == typingFrozen || !Utilities.IsValid(local))
                return;
            typingFrozen = typing;
            if (typing)
            {
                frozenSeat = -1;
                if (seatPlayers != null && seatObjects != null)
                    for (int i = 0; i < seatPlayers.Length && i < seatObjects.Length; i++)
                        if (seatPlayers[i] == local.playerId && seatObjects[i] != null)
                            frozenSeat = i;
                if (frozenSeat >= 0)
                    seatObjects[frozenSeat].disableStationExit = true;
                else
                    local.Immobilize(true);
            }
            else
            {
                if (frozenSeat >= 0)
                    seatObjects[frozenSeat].disableStationExit = false;
                else
                    local.Immobilize(false);
                frozenSeat = -1;
            }
        }

        void UpdateLocalConversation()
        {
            var local = Networking.LocalPlayer;
            if (!Utilities.IsValid(local))
                return;
            // Typing W/A/S/D into the order field also walked the guest away from the table; hold them still while it has focus.
            bool typing = conversationPanel != null && conversationPanel.activeSelf && input != null && input.isFocused;
            SetTypingFreeze(typing);
            if (conversationPanel == null || !conversationPanel.activeSelf)
                return;
            // In VR the panel stays where it was opened, so that either hand can point at it; it only comes along when
            // the guest has walked away from it.
            if (local.IsUserInVR())
            {
                var head = local.GetTrackingData(VRCPlayerApi.TrackingDataType.Head).position;
                if (Vector3.Distance(head, conversationPanel.transform.position) > 1.3f)
                    PlaceConversation(local);
            }
        }

        void PlaceConversation(VRCPlayerApi local)
        {
            var head = local.GetTrackingData(VRCPlayerApi.TrackingDataType.Head);
            Vector3 position;
            if (local.IsUserInVR())
            {
                // In front of the guest at arm's length and a little below the eyes, fixed in the room (it used to ride on
                // the left hand, which made it hard to aim at and to type on).
                var ahead = head.rotation * Vector3.forward;
                ahead.y = 0;
                if (ahead.sqrMagnitude < .01f)
                    ahead = local.GetRotation() * Vector3.forward;
                position = head.position + ahead.normalized * .6f + Vector3.down * .2f;
                conversationPanel.transform.localScale = Vector3.one * .0008f;
            }
            else
            {
                position = head.position + head.rotation * new Vector3(0, -.25f, .72f);
                conversationPanel.transform.localScale = Vector3.one * .001f;
            }
            conversationPanel.transform.position = position;
            conversationPanel.transform.rotation = Quaternion.LookRotation(position - head.position);
        }

        void ApplyModelDecision()
        {
            Debug.Log("[CafeNPC model] " + lastModelOutput);
            DataToken json;
            if (
                !VRCJson.TryDeserializeFromJson(FirstObject(lastModelOutput), out json)
                || json.TokenType != TokenType.DataDictionary
            )
            {
                actor.actionResult = "失敗: JSON形式の行動が必要";
                return;
            }
            var data = json.DataDictionary;
            DataToken actionToken,
                targetToken,
                speechToken;
            if (useJobChoices)
            {
                DataToken choiceToken;
                if (
                    !data.TryGetValue("choice", TokenType.Double, out choiceToken)
                    || !data.TryGetValue("speech", TokenType.String, out speechToken)
                )
                {
                    actor.actionResult = "失敗: choiceとspeechが必要";
                    return;
                }
                double choice = choiceToken.Double;
                if (choice < 0 || choice >= actor.choiceCount || choice != Mathf.Floor((float)choice))
                {
                    actor.actionResult = "失敗: 候補にある番号が必要";
                    return;
                }
                string spoken = LlmPipeline.Sanitize(speechToken.String, 120);
                if (actor.ExecuteJobChoice((int)choice, spoken, decisionMessageRevision))
                {
                    if (actor.activeAction == "serve")
                        servingMessageRevision = decisionMessageRevision;
                    if (actor.activeAction == "speak")
                        recentEvents = LlmPipeline.Sanitize("直前の発言: " + spoken + "。\n" + recentEvents, 300);
                    lastDecision = "候補" + (int)choice + ": " + actor.activeAction;
                    acceptedDecisions++;
                    Debug.Log("[CafeNPC accepted] " + lastDecision);
                    RequestSerialization();
                }
                return;
            }
            if (
                !data.TryGetValue("action", TokenType.String, out actionToken)
                || !data.TryGetValue("target", TokenType.Double, out targetToken)
                || !data.TryGetValue("speech", TokenType.String, out speechToken)
            )
            {
                actor.actionResult = "失敗: action,target,speechが必要";
                return;
            }
            double targetNumber = targetToken.Double;
            if (
                targetNumber < 0
                || targetNumber >= actor.route.Length
                || targetNumber != Mathf.Floor((float)targetNumber)
            )
            {
                actor.actionResult = "失敗: targetは有効なstation番号が必要";
                return;
            }
            var action = actionToken.String;
            var speech = LlmPipeline.Sanitize(speechToken.String, 120);
            if (actor.ExecuteDecision(action, (int)targetNumber, speech))
            {
                if (action == "serve")
                    servingMessageRevision = decisionMessageRevision;
                if (action == "speak")
                    recentEvents = "あなたの直前の発言: " + speech + "。配膳済みの注文番号=" + servedMessageRevision;
                lastDecision = action + " → " + (int)targetNumber;
                acceptedDecisions++;
                Debug.Log("[CafeNPC accepted] " + lastDecision);
                RequestSerialization();
            }
        }

        public string FirstObject(string text)
        {
            if (string.IsNullOrEmpty(text))
                return "";
            int start = text.IndexOf('{');
            if (start < 0)
                return "";
            string prefix = text.Substring(0, start).Trim();
            if (prefix.Length > 0 && prefix != "```json" && prefix != "```")
                return "";
            int depth = 0;
            bool quoted = false;
            bool escaped = false;
            for (int i = start; i < text.Length; i++)
            {
                char c = text[i];
                if (quoted)
                {
                    if (escaped)
                        escaped = false;
                    else if (c == '\\')
                        escaped = true;
                    else if (c == '\"')
                        quoted = false;
                }
                else
                {
                    if (c == '\"')
                        quoted = true;
                    else if (c == '{')
                        depth++;
                    else if (c == '}')
                    {
                        depth--;
                        if (depth == 0)
                            return text.Substring(start, i - start + 1);
                    }
                }
            }
            return "";
        }

        public void SendGuestMessage()
        {
            string text = LlmPipeline.SanitizeSharedPrompt(input.text, 200);
            if (text.Length == 0)
                return;
            if (Networking.IsOwner(gameObject))
                AcceptMessage(Networking.LocalPlayer, text);
            else
                SendCustomNetworkEvent(NetworkEventTarget.Owner, nameof(ReceiveGuestMessage), text);
            input.text = "";
        }

        [NetworkCallable(1)]
        public void ReceiveGuestMessage(string text)
        {
            if (Networking.IsOwner(gameObject))
                AcceptMessage(NetworkCalling.CallingPlayer, text);
        }

        void AcceptMessage(VRCPlayerApi player, string text)
        {
            if (!Utilities.IsValid(player))
                return;
            var pos = player.GetPosition();
            if (Mathf.Abs(pos.x) > 6 || Mathf.Abs(pos.z) > 5)
                return;
            // The guest being served may correct or stop the NPC at once; anyone else waits for their turn.
            if (useDecider && player.playerId != requestGuestId && Serving())
            {
                Wait(player.playerId, text);
                return;
            }
            if (Time.time - lastMessageAt < 2)
                return;
            lastMessageAt = Time.time;
            string said = LlmPipeline.SanitizeSharedPrompt(text, 200);
            guestMessage = LlmPipeline.Sanitize(player.displayName, 20) + ": " + said;
            if (pending)
            {
                pipeline.AbortGeneration();
                pending = false;
            }
            if (player.playerId != requestGuestId)
            {
                memoryCount = 0;
                serviceNote = null;
                offeredItems = null;
            }
            if (useDecider)
                BeginRequest(said);
            actor.InterruptJob();
            requestGuestId = player.playerId;
            messageRevision++;
            nextDecision = Time.time;
            RequestSerialization();
        }

        // ---- Bills: delivered priced items are charged to the guest who ordered them, until the guest pays.
        // A guest with an unpaid bill who says goodbye, or walks to the exit, is stopped by the NPC.
        public Transform exitPoint;
        private int[] billPlayers;
        private int[] billAmounts;
        private bool[] billChased;
        private float nextExitCheck;

        int BillIndex(int playerId, bool create)
        {
            if (billPlayers == null)
            {
                billPlayers = new int[16];
                billAmounts = new int[16];
                billChased = new bool[16];
                for (int i = 0; i < 16; i++)
                    billPlayers[i] = -1;
            }
            int free = -1;
            for (int i = 0; i < billPlayers.Length; i++)
            {
                if (billPlayers[i] == playerId && playerId >= 0)
                    return i;
                if (free < 0 && (billPlayers[i] < 0 || billAmounts[i] == 0))
                    free = i;
            }
            if (!create || free < 0)
                return -1;
            billPlayers[free] = playerId;
            billAmounts[free] = 0;
            billChased[free] = false;
            return free;
        }

        public int BillOf(int playerId)
        {
            int i = BillIndex(playerId, false);
            return i < 0 ? 0 : billAmounts[i];
        }

        void AddBill(int playerId, int yen)
        {
            if (yen <= 0 || playerId < 0)
                return;
            int i = BillIndex(playerId, true);
            if (i >= 0)
                billAmounts[i] += yen;
        }

        void ClearBill(int playerId)
        {
            int i = BillIndex(playerId, false);
            if (i >= 0)
            {
                billAmounts[i] = 0;
                billPlayers[i] = -1;
            }
        }

        public override void OnPlayerLeft(VRCPlayerApi player)
        {
            if (Utilities.IsValid(player))
                ClearBill(player.playerId);
        }

        // A world without a script that sets it has no jump at all. 3 is VRChat's usual jump.
        public override void OnPlayerJoined(VRCPlayerApi player)
        {
            if (Utilities.IsValid(player) && player.isLocal)
                player.SetJumpImpulse(3);
        }

        void StopUnpaid(int playerId)
        {
            stage = StageIdle;
            int i = BillIndex(playerId, false);
            if (i >= 0)
                billChased[i] = true;
            int total = BillOf(playerId);
            lastDecision = "お会計がまだのお客様を呼び止める";
            RequestSerialization();
            Say("あっ、お客様！お会計がまだなんです。合計" + total + "円になります。", EmotionSurprise);
            // Walk up to the guest: the route point nearest to them.
            var guest = VRCPlayerApi.GetPlayerById(playerId);
            if (!Utilities.IsValid(guest))
                return;
            int nearest = -1;
            float best = float.MaxValue;
            for (int r = 0; r < actor.route.Length; r++)
            {
                float d = Vector3.Distance(guest.GetPosition(), actor.route[r].position);
                if (d < best)
                {
                    best = d;
                    nearest = r;
                }
            }
            requestGuestId = playerId;
            actor.guestId = playerId;
            if (nearest >= 0 && nearest != actor.station)
                actor.ExecuteDecision("move", nearest, "");
        }

        // Owner side: an unpaid guest near the exit is stopped once; walking away from it re-arms the check.
        void WatchExit()
        {
            if (exitPoint == null || billPlayers == null || Time.time < nextExitCheck)
                return;
            nextExitCheck = Time.time + .5f;
            for (int i = 0; i < billPlayers.Length; i++)
            {
                if (billPlayers[i] < 0 || billAmounts[i] == 0)
                    continue;
                var player = VRCPlayerApi.GetPlayerById(billPlayers[i]);
                if (!Utilities.IsValid(player))
                    continue;
                float d = Vector3.Distance(player.GetPosition(), exitPoint.position);
                if (d > 3)
                    billChased[i] = false;
                else if (
                    d < 1.1f
                    && !billChased[i]
                    && stage == StageIdle
                    && !replying
                    && !replyQueued
                    && !checkQueued
                    && !awaitingCheck
                )
                {
                    if (!actor.actionComplete && actor.heldItem < 0)
                        actor.InterruptJob();
                    if (actor.actionComplete)
                        StopUnpaid(billPlayers[i]);
                }
            }
        }

        // ---- Waiting line: the NPC serves one guest at a time. What another guest says meanwhile is kept, one message a
        // guest (a second one replaces the first), and answered in order once the NPC is free. Worker only, not synced.
        private const int LineCapacity = 6;
        private int[] lineGuests = new int[LineCapacity];
        private string[] lineTexts = new string[LineCapacity];
        private int lineCount;

        bool Serving()
        {
            return stage != StageIdle
                || !actor.actionComplete
                || replying
                || replyQueued
                || checkQueued
                || awaitingCheck
                || voiceLine != null;
        }

        void Wait(int playerId, string text)
        {
            for (int i = 0; i < lineCount; i++)
                if (lineGuests[i] == playerId)
                {
                    lineTexts[i] = text;
                    return;
                }
            if (lineCount == LineCapacity)
                return;
            lineGuests[lineCount] = playerId;
            lineTexts[lineCount] = text;
            lineCount++;
            Debug.Log("[CafeNPC line] " + lineCount + " waiting");
        }

        // Returns true when a waiting guest's message was taken up this frame.
        bool UpdateLine()
        {
            if (lineCount == 0 || Serving())
                return false;
            int guest = lineGuests[0];
            string text = lineTexts[0];
            lineCount--;
            for (int i = 0; i < lineCount; i++)
            {
                lineGuests[i] = lineGuests[i + 1];
                lineTexts[i] = lineTexts[i + 1];
            }
            var player = VRCPlayerApi.GetPlayerById(guest);
            if (!Utilities.IsValid(player))
                return false; // left while waiting
            lastMessageAt = -100;
            AcceptMessage(player, text);
            return true;
        }

        // ---- Table bells: an idle NPC walks to the table that rang; a busy one goes after finishing its current work.
        private bool bellPending;
        private int bellStation = -1;
        private int bellCallerId = -1;
        private float bellAt = -100;
        private bool walkingToBell;

        public void CallFromBell(int station)
        {
            if (Networking.IsOwner(gameObject))
                AcceptBell(Networking.LocalPlayer, station);
            else
                SendCustomNetworkEvent(NetworkEventTarget.Owner, nameof(ReceiveBell), station);
        }

        [NetworkCallable(1)]
        public void ReceiveBell(int station)
        {
            if (Networking.IsOwner(gameObject))
                AcceptBell(NetworkCalling.CallingPlayer, station);
        }

        void AcceptBell(VRCPlayerApi player, int station)
        {
            if (!Utilities.IsValid(player) || !actor.IsPlacementStation(station) || station == actor.homeStation)
                return;
            bellPending = true;
            bellStation = station;
            bellCallerId = player.playerId;
            bellAt = Time.time;
        }

        // Returns true while the bell is being answered this frame.
        bool UpdateBell()
        {
            if (walkingToBell && actor.actionComplete)
            {
                walkingToBell = false;
                // A new guest message interrupts the walk (activeAction becomes "wait"); only a finished walk greets.
                if (actor.activeAction == "move" && actor.actionResult.StartsWith("成功"))
                {
                    Show("お待たせしました！ご注文をお伺いします。", EmotionSmile);
                }
            }
            if (
                !bellPending
                || stage != StageIdle
                || !actor.actionComplete
                || replying
                || replyQueued
                || checkQueued
                || awaitingCheck
            )
                return false;
            bellPending = false;
            requestGuestId = bellCallerId;
            actor.guestId = bellCallerId;
            lastDecision = "ベルで呼ばれた" + actor.StationName(bellStation) + "へ";
            RequestSerialization();
            if (actor.station == bellStation)
            {
                Show("はーい、ご注文をお伺いします！", EmotionSmile);
                return true;
            }
            walkingToBell = actor.ExecuteDecision("move", bellStation, "はーい、ただいま！");
            return true;
        }

        // ---- Decider service: the decider chooses, deterministic code only executes and records facts.
        // Protocol: one yes/no question per registered
        // item, plus the request kind only while an earlier request is still in progress; then one job choice
        // at a time. A completed delivery removes the item from the pending set and the job is chosen again.
        private const int StageIdle = 0,
            StageRecognize = 1,
            StageDecide = 2,
            StageExecute = 3,
            StageUnavailable = 4,
            StageKind = 5,
            StageAgree = 6,
            StageDestination = 7,
            StageExtract = 8,
            StageMany = 9;
        public const int EmotionNeutral = 0,
            EmotionSmile = 1,
            EmotionTrouble = 2,
            EmotionSurprise = 3,
            EmotionShy = 4;
        private int stage;
        private int questionIndex;
        private bool awaitingAnswer;
        private string requestText = "";
        private string previousRequestText;
        private bool[] previousItems;
        private bool[] itemAsked;
        private bool[] pendingItems;
        private string requestKind;
        private string[] serviceEvents = new string[8]; // native keeps history[-8:]
        private int serviceEventCount;
        private int currentJobItem = -1;
        private string jobPickupPlace;
        private bool jobPickupLogged;
        private bool replied;
        private bool replying;
        private bool replyQueued;
        private bool checkQueued;
        private bool awaitingCheck;
        private string replyPlan,
            replyPlanEnglish,
            replyFallback,
            replyCandidate;
        private int replyEmotion,
            candidateEmotion;
        private float replyStartedAt;
        private int replyRetries;
        private string[] jobKeys;
        private int[] jobItems;
        private int[] jobTargets;

        [System.NonSerialized]
        public string lastDeciderTrace = "";

        void BeginRequest(string text)
        {
            if (decider != null)
                decider.Cancel();
            if (replying)
            {
                pipeline.AbortGeneration();
                replying = false;
            }
            replyQueued = false;
            checkQueued = false;
            awaitingCheck = false;
            TrackPickup(); // a request can arrive in the same frame the item was picked up
            int count = actor.itemNames == null ? 0 : actor.itemNames.Length;
            bool inProgress = actor.heldItem >= 0 || AnyPending();
            if (inProgress)
            {
                previousRequestText = requestText;
                previousItems = pendingItems;
                AddServiceEvent(
                    "{\"event\": \"guest_changed_request\", \"previous_utterance\": "
                        + CafeDecider.Json(requestText)
                        + "}"
                );
            }
            else
            {
                previousRequestText = null;
                previousItems = null;
                serviceEventCount = 0;
            }
            requestText = text;
            requestKind = null;
            pendingItems = new bool[count];
            itemAsked = new bool[count];
            questionIndex = 0;
            awaitingAnswer = false;
            replied = false;
            currentJobItem = -1;
            namedDestination = -1;
            previousOrder = inProgress ? extractedOrder : null;
            extractedItems = new bool[count];
            extractedOrder = null;
            extractedPlace = -1;
            extractedOther = false;
            extractSubmitted = false;
            extractAttempts = 0;
            legacyItems = false;
            askingReverse = false;
            manyRequested = false;
            kindKnown = false;
            kindIsOrder = false;
            kindRecheck = false;
            askingLeave = false;
            agreeRecheck = false;
            recognized = false;
            if (inProgress)
                offeredItems = null;
            // An answer to the NPC's own question is checked first. A correction to work in progress is read against the
            // previous request as before. Anything else is first sorted into order / pay / leave / chat.
            stage =
                offeredItems != null ? StageAgree
                : inProgress ? StageExtract
                : StageKind;
        }

        void UpdateService()
        {
            UpdateVoice();
            UpdateReply();
            TrackPickup();
            if (stage == StageIdle && !awaitingAnswer && UpdateLine())
                return;
            if (stage == StageIdle && !awaitingAnswer && UpdateBell())
                return; // before the decider check: it loads for a minute
            if (decider == null || !decider.IsReady())
            {
                Status("茜犬の判断モデルを読み込み中です");
                return;
            }
            Status("茜犬: " + lastDecision + "\n" + actor.actionResult);
            if (stage == StageExtract)
            {
                UpdateExtract();
                return;
            }
            if (awaitingCheck)
            {
                if (decider.phase == CafeDecider.PhaseRunning)
                    return;
                awaitingCheck = false;
                Debug.Log("[CafeNPC decider answer] " + (decider.phase == CafeDecider.PhaseDone ? decider.answer : -1));
                PublishReply(decider.phase == CafeDecider.PhaseDone && decider.answer == 0);
                return;
            }
            if (awaitingAnswer)
            {
                if (decider.phase == CafeDecider.PhaseRunning)
                {
                    Status("茜犬が考えています…");
                    return;
                }
                awaitingAnswer = false;
                Debug.Log("[CafeNPC decider answer] " + (decider.phase == CafeDecider.PhaseDone ? decider.answer : -1));
                if (decider.phase != CafeDecider.PhaseDone)
                {
                    FailService("判断モデルの失敗: " + decider.error);
                    return;
                }
                if (stage == StageRecognize)
                    OnRecognitionAnswer(decider.answer);
                else if (stage == StageDecide)
                    OnJobAnswer(decider.answer);
                else if (stage == StageUnavailable)
                    OnUnavailableAnswer(decider.answer);
                else if (stage == StageKind)
                    OnKindAnswer(decider.answer);
                else if (stage == StageAgree)
                    OnAgreeAnswer(decider.answer);
                else if (stage == StageMany)
                    OnManyAnswer(decider.answer);
                else if (stage == StageDestination)
                    OnDestinationAnswer(decider.answer);
                return;
            }
            if (checkQueued)
            {
                AskReplyCheck();
                return;
            }
            WatchExit();
            if (stage == StageRecognize)
                AskRecognition();
            else if (stage == StageUnavailable)
                AskUnavailable();
            else if (stage == StageKind)
                AskKind();
            else if (stage == StageAgree)
                AskAgree();
            else if (stage == StageMany)
                AskMany();
            else if (stage == StageDestination)
                AskDestination();
            else if (stage == StageDecide && actor.actionComplete)
                AskJob();
        }

        public void OnDeciderAnswer() { } // the service polls decider.phase; this keeps the listener event harmless

        string RecognitionState()
        {
            return "{\"latest_guest_utterance\": "
                + CafeDecider.Json(requestText)
                + ", \"held_item\": "
                + (actor.heldItem >= 0 ? CafeDecider.Json(actor.itemKeys[actor.heldItem]) : "null")
                + ", \"previous_guest_utterance\": "
                + CafeDecider.Json(previousRequestText)
                + ", \"previous_requested_items\": "
                + ItemList(previousItems)
                + "}";
        }

        void AskItem(string label)
        {
            AskItem(label, false);
        }

        void AskItem(string label, bool reversed)
        {
            string no = "no: The current request does not require providing or transporting " + label + ".";
            string yes = "yes: The current request requires providing or transporting " + label + ".";
            Ask(
                RecognitionState(),
                "Is "
                    + label
                    + " one of the items requested for the guest task? Include every item in a multi-item request, even when verbs are omitted. For references, use the previous request and its recognized items.",
                reversed ? new[] { yes, no } : new[] { no, yes }
            );
        }

        // ---- Order reading. The 1.2B Japanese model proposes the order (items in the order asked, a named place, anything
        // not on the menu) and the decider confirms each proposed item. Natively the 1.2B read every test order
        // correctly but invented items for chat, paying and leaving; the 0.8b
        // decider (the size limit's choice) missed ordered items and took npc_location for the destination. A proposed item
        // stays if the decider says yes in either option order (it refuses short orders like "お水ください" in one order);
        // items nobody proposed are never asked about, which kept the re-asked "yes" from adding unordered items.
        public string orderPrompt;
        public string replyPrompt;
        private bool[] extractedItems;
        private int[] extractedOrder,
            previousOrder;
        private int extractedPlace = -1;
        private bool extractedOther,
            extractSubmitted,
            legacyItems;
        private int extractAttempts;
        private float extractStartedAt;

        private int[] orderPrefix,
            replyPrefix;

        void UsePrefix(bool order)
        {
            if (orderPrefix == null)
            {
                orderPrefix = pipeline.BuildPrefix(orderPrompt);
                replyPrefix = pipeline.BuildPrefix(replyPrompt);
            }
            pipeline.SetPrefix(order ? orderPrefix : replyPrefix);
        }

        void UpdateExtract()
        {
            if (!extractSubmitted)
            {
                if (pipeline.IsBusy() || replying)
                    return;
                UsePrefix(true);
                pipeline.ClearHistory();
                string user =
                    "前の発言: "
                    + (previousRequestText != null ? "「" + previousRequestText + "」" : "なし")
                    + "\n今の発言: 「"
                    + requestText
                    + "」";
                if (pipeline.SubmitObservation(user))
                {
                    extractSubmitted = true;
                    extractStartedAt = Time.time;
                    extractAttempts++;
                }
                return;
            }
            bool aborted = pipeline.status == LlmPipeline.StatusReady && Time.time - extractStartedAt > .5f;
            bool failed =
                pipeline.status == LlmPipeline.StatusError
                || pipeline.status == LlmPipeline.StatusFatal
                || aborted
                || Time.time - extractStartedAt > 30;
            if (pipeline.status == LlmPipeline.StatusDone)
            {
                if (ParseOrder(pipeline.replyText))
                {
                    stage = StageRecognize;
                    return;
                }
                failed = true;
            }
            if (!failed)
                return;
            if (pipeline.status == LlmPipeline.StatusGenerating)
                pipeline.AbortGeneration();
            extractSubmitted = false;
            if (extractAttempts < 2)
                return;
            // The conversation model is unavailable or keeps writing non-JSON: ask the decider item by item instead.
            Debug.LogWarning("[CafeNPC order] reading failed; asking the decider per item");
            legacyItems = true;
            stage = StageRecognize;
        }

        bool ParseOrder(string text)
        {
            DataToken json;
            if (
                !VRCJson.TryDeserializeFromJson(FirstObject(text), out json)
                || json.TokenType != TokenType.DataDictionary
            )
                return false;
            var data = json.DataDictionary;
            DataToken items,
                place,
                other;
            if (!data.TryGetValue("items", TokenType.DataList, out items))
                return false;
            int count = extractedItems.Length;
            var order = new int[count];
            int ordered = 0;
            for (int i = 0; i < items.DataList.Count; i++)
            {
                if (items.DataList[i].TokenType != TokenType.String)
                    continue;
                int index = -1;
                for (int k = 0; k < count; k++)
                    if (actor.itemKeys[k] == items.DataList[i].String)
                        index = k;
                if (index < 0)
                {
                    extractedOther = true;
                    continue;
                } // something not on the menu (e.g. "juice")
                if (!extractedItems[index] && ordered < count)
                {
                    extractedItems[index] = true;
                    order[ordered++] = index;
                }
            }
            extractedOrder = new int[ordered];
            System.Array.Copy(order, extractedOrder, ordered);
            if (data.TryGetValue("place", TokenType.String, out place))
                for (int i = 0; i < actor.route.Length; i++)
                    if (actor.IsPlacementStation(i) && i != actor.homeStation && actor.StationKey(i) == place.String)
                        extractedPlace = i;
            if (data.TryGetValue("other", TokenType.Boolean, out other) && other.Boolean)
                extractedOther = true;
            Debug.Log(
                "[CafeNPC order] "
                    + text.Trim()
                    + " -> items "
                    + ItemList(extractedItems)
                    + " place "
                    + (extractedPlace >= 0 ? actor.StationKey(extractedPlace) : "none")
                    + " other "
                    + extractedOther
            );
            return true;
        }

        private bool askingReverse;

        void AskRecognition()
        {
            int count = pendingItems.Length;
            string state = RecognitionState();
            if (!legacyItems)
                while (questionIndex < count && !extractedItems[questionIndex])
                    questionIndex++;
            if (questionIndex < count)
            {
                AskItem(actor.itemLabels[questionIndex], askingReverse);
                return;
            }
            if (previousRequestText == null)
            {
                FinishRecognition();
                return;
            }
            Ask(
                state,
                "Classify the latest guest request. Is it a new task, only a destination correction to the existing task, stopping with the currently held item, or no transport task? A destination correction keeps the previous requested items.",
                new[]
                {
                    "new_task: Provide or transport requested items. The request states which items are wanted.",
                    "destination_change: Continue the existing item task but change where the item should be delivered. The previous items remain requested.",
                    "stop_holding: Stop transporting; keep the currently held item without moving.",
                    "no_transport: Conversation, a question, waiting, or a request that requires no item transport.",
                }
            );
        }

        void OnRecognitionAnswer(int choice)
        {
            int count = pendingItems.Length;
            if (questionIndex < count)
            {
                bool yes = askingReverse ? choice == 0 : choice == 1;
                if (!legacyItems && !yes && !askingReverse)
                {
                    askingReverse = true;
                    return;
                } // the same item again, reversed
                askingReverse = false;
                itemAsked[questionIndex] = yes;
                questionIndex++;
                return; // AskRecognition asks the next item, the request kind, or finishes
            }
            string[] kinds = { "new_task", "destination_change", "stop_holding", "no_transport" };
            requestKind = kinds[choice];
            FinishRecognition();
        }

        void FinishRecognition()
        {
            int count = pendingItems.Length;
            for (int i = 0; i < count; i++)
                pendingItems[i] = itemAsked[i];
            if (requestKind == null)
                requestKind = AnyPending() ? "new_task" : "no_transport";
            if (requestKind == "destination_change")
                extractedOrder = previousOrder;
            if (requestKind == "new_task" || requestKind == "destination_change")
                namedDestination = extractedPlace;
            if (requestKind == "destination_change" && previousItems != null)
                for (int i = 0; i < count; i++)
                    pendingItems[i] = previousItems[i];
            else if (requestKind == "stop_holding" || requestKind == "no_transport")
                for (int i = 0; i < count; i++)
                    pendingItems[i] = false;
            Restock();
            recognized = true;
            lastDeciderTrace =
                (kindKnown ? lastDeciderTrace + " / " : "")
                + "認識: "
                + ItemList(pendingItems)
                + (requestKind != null ? " / " + requestKind : "");
            Debug.Log("[CafeNPC decider] " + lastDeciderTrace);
            if (AnyPending())
            {
                stage =
                    requestKind == "new_task" ? StageMany
                    : legacyItems ? StageDestination
                    : StageDecide;
                return;
            }
            if (requestKind == "stop_holding")
            {
                stage = StageIdle;
                lastDecision = "その場で待つ";
                RequestSerialization();
                // The 1.2B model kept turning this into "please hold it yourself"; a fixed line is used.
                Say("はーい、このままお持ちしておきますね。", EmotionNeutral);
                return;
            }
            // Nothing registered is requested. Something not on the menu was asked for (read by the conversation model;
            // the decider's question in the per-item fallback): the reply apologises instead of promising it.
            if (legacyItems)
            {
                stage = StageUnavailable;
                return;
            }
            if (extractedOther)
            {
                OnUnavailableAnswer(1);
                return;
            }
            // The conversation model read an order the decider would not confirm (a misspelt item, a question about
            // one): the NPC asks the guest instead of guessing either way.
            if (kindIsOrder && extractedOrder != null && extractedOrder.Length > 0)
            {
                Offer();
                return;
            }
            OnUnavailableAnswer(0);
        }

        private bool[] offeredItems;
        private int[] offeredOrder;
        private string offeredQuestion;
        private bool agreeRecheck,
            recognized;

        void Offer()
        {
            offeredItems = new bool[extractedItems.Length];
            offeredOrder = extractedOrder;
            string names = "";
            for (int k = 0; k < extractedOrder.Length; k++)
            {
                offeredItems[extractedOrder[k]] = true;
                names += (k > 0 ? "と" : "") + actor.ItemName(extractedOrder[k]);
            }
            offeredQuestion = names + "のご注文でよろしいですか？";
            stage = StageIdle;
            lastDecision = "注文を確かめる";
            RequestSerialization();
            // Asked in fixed words: the next utterance is judged as an answer to exactly this question.
            Say(offeredQuestion, EmotionNeutral);
        }

        // The guest's answer to that question; a "yes" in both option orders takes the order as offered.
        void AskAgree()
        {
            string no = "no: The guest declines, corrects the staff, or says something else.";
            string yes = "yes: The guest agrees or confirms, for example yes, that is right, or please.";
            Ask(
                "{\"staff_question\": "
                    + CafeDecider.Json(offeredQuestion)
                    + ", \"latest_guest_utterance\": "
                    + CafeDecider.Json(requestText)
                    + "}",
                "The staff asked staff_question. Does latest_guest_utterance agree to it?",
                agreeRecheck ? new[] { yes, no } : new[] { no, yes }
            );
        }

        void OnAgreeAnswer(int choice)
        {
            bool yes = agreeRecheck ? choice == 0 : choice == 1;
            if (yes && !agreeRecheck)
            {
                agreeRecheck = true;
                return;
            } // AskAgree asks again in the other order
            agreeRecheck = false;
            var items = offeredItems;
            var order = offeredOrder;
            offeredItems = null;
            offeredOrder = null;
            Debug.Log("[CafeNPC decider] 聞き返しへの返事: " + (yes ? "はい" : "それ以外"));
            if (!yes)
            {
                stage = StageKind;
                return;
            }
            for (int i = 0; i < pendingItems.Length && i < items.Length; i++)
                pendingItems[i] = items[i];
            extractedOrder = order;
            requestKind = "new_task";
            kindKnown = true;
            kindIsOrder = true;
            Restock();
            lastDeciderTrace = "認識: " + ItemList(pendingItems) + " / 聞き返して確定";
            stage = StageDecide;
        }

        // Each item is one object, so a request for a hundred muffins still brings one: the reply jokes about the number.
        private bool manyRequested;

        void AskMany()
        {
            Ask(
                "{\"latest_guest_utterance\": " + CafeDecider.Json(requestText) + "}",
                "Does the guest ask for ten or more of one item, or an unlimited or impossible amount?",
                new[]
                {
                    "no: An ordinary amount: one or a few, or no number at all.",
                    "yes: Ten or more of an item, or an unlimited amount such as infinite, all of them, or as many as possible.",
                }
            );
        }

        void OnManyAnswer(int choice)
        {
            manyRequested = choice == 1;
            lastDeciderTrace += " / 数=" + (manyRequested ? "多すぎ" : "ふつう");
            Debug.Log("[CafeNPC decider] " + lastDeciderTrace);
            stage = legacyItems ? StageDestination : StageDecide;
        }

        // Each item is a single object. Ordering one that was eaten, drunk or taken by a guest again means a new serving:
        // the used one is replaced by a fresh one on the kitchen worktop, which the NPC then carries as usual.
        void Restock()
        {
            bool changed = false;
            for (int i = 0; i < pendingItems.Length; i++)
            {
                var edible = actor.Edible(i);
                if (!pendingItems[i] || edible == null || !(edible.Consumed() || edible.free) || edible.HeldByPlayer())
                    continue;
                if (i == actor.heldItem)
                    continue;
                edible.Restock();
                actor.itemStations[i] = actor.homeStation;
                if (actor.itemSlots != null && i < actor.itemSlots.Length)
                    actor.itemSlots[i] = -1;
                changed = true;
            }
            if (changed)
                actor.RequestSerialization();
        }

        // Same validated question shape as the registered items, asked about everything else.
        void AskUnavailable()
        {
            AskItem(
                "another food or drink (anything except "
                    + ItemList(AllItems())
                    + ", e.g. juice, beer, sandwich, pasta)"
            );
        }

        void OnUnavailableAnswer(int choice)
        {
            lastDeciderTrace += " / 未提供品の依頼=" + (choice == 1 ? "あり" : "なし");
            Debug.Log("[CafeNPC decider] " + lastDeciderTrace);
            if (choice == 0)
            {
                // No order after all. A correction to work in progress has not been sorted yet: ask the kind now.
                if (!kindKnown)
                {
                    stage = StageKind;
                    return;
                }
                stage = StageIdle;
                if (kindIsOrder)
                {
                    lastDecision = "注文を聞き返す";
                    RequestSerialization();
                    Say("すみません、ご注文をもう一度うかがえますか？", EmotionTrouble);
                }
                else
                    Chat();
                return;
            }
            stage = StageIdle;
            lastDecision = "会話で応える";
            RequestSerialization();
            StartReply(
                "何も運ばない。頼まれた品は店に置いていないので謝る。",
                "No transport now. Apologize that the requested item is not available. Do not promise or offer to bring or serve anything now.",
                "ごめんなさい、それは置いてないんです。メニューにあるものならお持ちしますよ！",
                EmotionTrouble
            );
        }

        // ---- What the guest is doing: order / pay / problem / chat, then for a chat whether it is a goodbye.
        // The old chain of yes/no questions (unavailable? pay? leaving?) answered "leaving" to "only one came" and then
        // stopped the guest for the bill. "leave" is not offered among the kinds either: next to the others the decider
        // chose it for misspelt orders and for complaints in both option orders. As a question of its own, asked in both
        // orders, it was right on 43 of 48 utterances.
        private bool kindKnown,
            kindIsOrder,
            kindRecheck,
            askingLeave;

        void AskKind()
        {
            string state =
                "{\"latest_guest_utterance\": "
                + CafeDecider.Json(requestText)
                + ", \"previous_guest_utterance\": "
                + CafeDecider.Json(LastGuestLine())
                + "}";
            if (askingLeave)
            {
                string no =
                    "no: The guest is staying: a remark, a question, thanks for something served, or an answer.";
                string yes = "yes: The guest says goodbye, or says they are leaving or going home.";
                Ask(
                    state,
                    "Is the guest saying goodbye or telling the staff that they are going home now?",
                    kindRecheck ? new[] { yes, no } : new[] { no, yes }
                );
                return;
            }
            Ask(
                state,
                "What is the guest doing in latest_guest_utterance, said to the cafe staff?",
                new[]
                {
                    "chat: Anything else: small talk, a goodbye, a question about prices, recommendations, the staff or the cafe, thanks, a comment on the food, or a short answer such as yes.",
                    "problem: Says something is wrong with the service: too few came, an item is missing or late, or it is not what was ordered.",
                    "pay: Asks for the bill, to pay, or to check out.",
                    "order: Wants food, a drink, water or the menu brought (misspelled item names included), asks whether an item is available, or changes an order or where it goes.",
                }
            );
        }

        void OnKindAnswer(int choice)
        {
            if (askingLeave)
            {
                bool yes = kindRecheck ? choice == 0 : choice == 1;
                if (yes && !kindRecheck)
                {
                    kindRecheck = true;
                    return;
                } // AskKind asks again in the other order
                askingLeave = false;
                kindRecheck = false;
                lastDeciderTrace += yes ? " / 退店" : "";
                Debug.Log("[CafeNPC decider] " + lastDeciderTrace);
                stage = StageIdle;
                if (yes)
                    Leave();
                else
                    Chat();
                return;
            }
            int kind = choice; // 0 chat, 1 problem, 2 pay, 3 order
            if (recognized && kind == 3)
                kind = 0; // the items were already read and none was requested
            lastDeciderTrace =
                (recognized ? lastDeciderTrace + " / " : "")
                + "種別: "
                + (
                    kind == 3 ? "注文"
                    : kind == 2 ? "会計"
                    : kind == 1 ? "指摘"
                    : "雑談"
                );
            Debug.Log("[CafeNPC decider] " + lastDeciderTrace);
            kindKnown = true;
            kindIsOrder = kind == 3;
            if (kind == 3)
            {
                stage = StageExtract;
                return;
            }
            // "Leaving without paying" reads as pay; with nothing on the bill there is nothing to pay, so it is a chat or a goodbye.
            if (kind == 0 || (kind == 2 && BillOf(requestGuestId) == 0))
            {
                askingLeave = true;
                return;
            } // AskKind asks whether it is a goodbye
            stage = StageIdle;
            if (kind == 2)
                Pay();
            else
                Problem();
        }

        // A complaint about the service: apologise, with what the NPC remembers about it (e.g. why only one came).
        void Problem()
        {
            lastDecision = "指摘に応える";
            RequestSerialization();
            StartReply(
                "何も運ばない。行き届かなかったことを謝る。"
                    + (serviceNote != null ? "（覚え書き: " + serviceNote + "）" : ""),
                "No transport and no offer. Apologize for the trouble.",
                serviceNote != null
                    ? "ごめんなさい！数が多すぎて、1つだけにさせてもらったんです。"
                    : "ごめんなさい！すぐ確かめますね。",
                EmotionTrouble
            );
        }

        // Paying is role-play: the NPC states the total and thanks the guest; the bill is then settled.
        void Pay()
        {
            int total = BillOf(requestGuestId);
            if (total == 0)
            {
                lastDecision = "会話で応える";
                RequestSerialization();
                StartReply(
                    "何も運ばない。お会計するものはまだ何もないと伝える。",
                    "No transport. Tell the guest there is nothing to pay yet.",
                    "お会計はまだ何もありませんよ。ご注文があればどうぞ！",
                    EmotionSmile
                );
                return;
            }
            ClearBill(requestGuestId);
            lastDecision = "お会計 " + total + "円";
            RequestSerialization();
            Say("お会計は合計" + total + "円です。ありがとうございました！", EmotionSmile);
        }

        void Leave()
        {
            if (BillOf(requestGuestId) > 0)
            {
                StopUnpaid(requestGuestId);
                return;
            }
            lastDecision = "お見送りする";
            RequestSerialization();
            Say("ありがとうございました。またのお越しをお待ちしてます！", EmotionSmile);
            memoryCount = 0;
            serviceNote = null;
        }

        void Chat()
        {
            lastDecision = "会話で応える";
            RequestSerialization();
            StartReply(
                "何も運ばない。お客様の言葉にそのまま応える。"
                    + (serviceNote != null ? "（覚え書き: " + serviceNote + "）" : ""),
                "No transport and no offer. Only reply to the conversation.",
                "えへへ、ありがとうございます。ごゆっくりどうぞ！",
                EmotionSmile
            );
        }

        bool[] AllItems()
        {
            var all = new bool[actor.itemKeys.Length];
            for (int i = 0; i < all.Length; i++)
                all[i] = true;
            return all;
        }

        // Where the items go is asked on its own, before the job choice. Folded into the job options, the 0.8b decider
        // picked "deliver to counter" for guests at a table (it took npc_location for the destination); with no place named
        // the guest's own seat or table is a fact, not a decision.
        private int namedDestination = -1;
        private int[] destinationStations;

        void AskDestination()
        {
            int count = 0;
            for (int i = 0; i < actor.route.Length; i++)
                if (actor.IsPlacementStation(i) && i != actor.homeStation)
                    count++;
            destinationStations = new int[count + 1];
            var options = new string[count + 1];
            destinationStations[0] = -1;
            options[0] = "none: The request names no place to deliver to.";
            int n = 1;
            for (int i = 0; i < actor.route.Length; i++)
            {
                if (!actor.IsPlacementStation(i) || i == actor.homeStation)
                    continue;
                destinationStations[n] = i;
                options[n] =
                    actor.StationKey(i)
                    + ": The request names the "
                    + actor.StationKey(i).Replace("_", " ")
                    + " ("
                    + actor.StationName(i)
                    + ") as the place to deliver to.";
                n++;
            }
            Ask(
                RecognitionState(),
                "Does the latest guest request (or the previous request it changes) name a place where the items should be delivered? Choose none if no place is named.",
                options
            );
        }

        void OnDestinationAnswer(int choice)
        {
            namedDestination = choice >= 0 && choice < destinationStations.Length ? destinationStations[choice] : -1;
            lastDeciderTrace +=
                " / 届け先=" + (namedDestination >= 0 ? actor.StationKey(namedDestination) : "指定なし");
            Debug.Log("[CafeNPC decider] " + lastDeciderTrace);
            stage = StageDecide;
        }

        void AskJob()
        {
            int count = pendingItems.Length;
            // A named place wins; otherwise the guest's seat or table. Only an unknown guest position leaves every table open.
            int allowed = namedDestination >= 0 ? namedDestination : GuestStation();
            int destinations = 0;
            for (int target = 0; target < actor.route.Length; target++)
                if (
                    actor.IsPlacementStation(target)
                    && target != actor.homeStation
                    && (allowed < 0 || target == allowed)
                )
                    destinations++;
            jobKeys = new string[2 + count * actor.route.Length];
            jobItems = new int[jobKeys.Length];
            jobTargets = new int[jobKeys.Length];
            int n = 0;
            jobKeys[n] = "wait: Wait silently here; keep any held item in hand.";
            jobItems[n] = -1;
            n++;
            jobKeys[n] = "speak: Reply or ask the guest a question; do not move objects.";
            jobItems[n] = -2;
            n++;
            // The order as read: deliver the first undelivered item in the order the guest asked for (the decider
            // ignored "先にメニューを、それからお水を"). Without a reading (fallback) the decider chooses among all.
            int next = -1;
            if (extractedOrder != null)
                for (int k = 0; k < extractedOrder.Length && next < 0; k++)
                    if (pendingItems[extractedOrder[k]] && (actor.heldItem < 0 || actor.heldItem == extractedOrder[k]))
                        next = extractedOrder[k];
            for (int item = 0; item < count; item++)
            {
                if (actor.heldItem >= 0 && item != actor.heldItem)
                    continue;
                if (next >= 0 && item != next)
                    continue;
                // Only items the decider recognized as requested are offered; it then chooses destination and order.
                // Offering the other physically possible moves made it carry delivered items back (verification).
                if (!pendingItems[item])
                    continue;
                // The decider renders at most 26 letter options; later items wait for the next choice.
                if (n + destinations > 26)
                    break;
                for (int target = 0; target < actor.route.Length; target++)
                {
                    if (
                        !actor.IsPlacementStation(target)
                        || target == actor.homeStation
                        || !actor.CanTransport(item, target)
                    )
                        continue;
                    if (allowed >= 0 && target != allowed)
                        continue;
                    string key = actor.itemKeys[item],
                        place = actor.StationKey(target);
                    jobKeys[n] =
                        "deliver_"
                        + key
                        + "_"
                        + place
                        + ": Carry "
                        + key
                        + " to "
                        + place
                        + " and put it down. Includes going to the item, picking it up and transport.";
                    jobItems[n] = item;
                    jobTargets[n] = target;
                    n++;
                }
            }
            if (n > 26)
            {
                FailService("仕事の候補が多すぎます: " + n);
                return;
            }
            // One requested item with a known destination: nothing left to decide.
            if (n == 3)
            {
                Debug.Log("[CafeNPC decider] 仕事: 候補が1つ");
                OnJobAnswer(2);
                return;
            }
            var options = new string[n];
            System.Array.Copy(jobKeys, options, n);
            string objects = "";
            for (int i = 0; i < count; i++)
                objects +=
                    (i > 0 ? ", " : "")
                    + CafeDecider.Json(actor.itemKeys[i])
                    + ": "
                    + CafeDecider.Json(actor.ItemPlaceKey(i));
            string events = "";
            for (int i = 0; i < serviceEventCount; i++)
                events +=
                    (i > 0 ? ", " : "")
                    + (
                        serviceEventCount >= 8
                            ? "{\"_index\": " + i + ", " + serviceEvents[i].Substring(1)
                            : serviceEvents[i]
                    );
            string state =
                "{\"latest_guest_utterance\": "
                + CafeDecider.Json(requestText)
                + ", \"guest_location\": "
                + CafeDecider.Json(GuestLocationKey())
                + ", \"npc_location\": "
                + CafeDecider.Json(actor.StationKey(NearestPlacementStation(actor.transform.position)))
                + ", \"held_item\": "
                + (actor.heldItem >= 0 ? CafeDecider.Json(actor.itemKeys[actor.heldItem]) : "null")
                + ", \"objects\": {"
                + objects
                + "}, \"recent_events\": ["
                + events
                + "], \"unfulfilled_requested_items\": "
                + ItemList(pendingItems)
                + ", \"previous_guest_utterance\": "
                + CafeDecider.Json(previousRequestText)
                + (requestKind != null ? ", \"recognized_task_modification\": " + CafeDecider.Json(requestKind) : "")
                + "}";
            Ask(
                state,
                "Select the next useful job for the latest guest request and actual state. Do not repeat completed deliveries. Prioritize new changes or cancellations. Fulfill an unfulfilled requested item, using the guest location if no destination is named, and respect any delivery order in the original request. Physically possible jobs for other items are not requested jobs.",
                options
            );
        }

        void OnJobAnswer(int choice)
        {
            int item = jobItems[choice];
            lastDeciderTrace = "仕事: " + jobKeys[choice].Substring(0, jobKeys[choice].IndexOf(':'));
            Debug.Log(
                "[CafeNPC decider] "
                    + lastDeciderTrace
                    + " ("
                    + decider.tokenCount
                    + " tokens, "
                    + decider.seconds.ToString("F1")
                    + " s)"
            );
            acceptedDecisions++;
            if (item == -1)
            {
                stage = StageIdle;
                lastDecision = "待機";
                RequestSerialization();
                if (!replied)
                    StartReply(
                        "何も運ばない。今はその場で待つ。",
                        "Wait here; no transport.",
                        "はーい、ここで待ってますね。",
                        EmotionNeutral
                    );
                return;
            }
            if (item == -2)
            {
                stage = StageIdle;
                lastDecision = "お客様に返事をする";
                RequestSerialization();
                StartReply(
                    "何も運ばない。どこへお持ちすればいいか聞く。",
                    "No transport. Ask the guest where to bring it.",
                    "どちらへお持ちしましょうか？",
                    EmotionNeutral
                );
                return;
            }
            int target = jobTargets[choice];
            if (
                !actor.ExecuteTransportAt(
                    item,
                    target,
                    target == actor.counterStation ? CounterSlotForGuest() : -1,
                    "",
                    messageRevision
                )
            )
            {
                FailService("運搬を開始できません");
                return;
            }
            currentJobItem = item;
            stage = StageExecute;
            jobPickupLogged = actor.heldItem == item;
            jobPickupPlace = actor.station == actor.itemStations[item] ? null : actor.ItemPlaceKey(item);
            lastDecision = actor.ItemName(item) + "を" + actor.StationName(target) + "へ";
            RequestSerialization();
            if (!replied)
            {
                string name = actor.ItemName(item),
                    place = actor.StationName(target);
                string plan = name + "を" + place + "へ運ぶ。";
                string english = "Carry " + actor.itemKeys[item] + " to " + actor.StationKey(target) + ".";
                string fallback = "はーい、" + name + "ですね。" + place + "へお持ちします！";
                int emotion = EmotionSmile;
                if (manyRequested)
                {
                    plan = name + "を1つだけ" + place + "へ運ぶ。頼まれた数は多すぎるので冗談で返す。";
                    english =
                        "Carry one "
                        + actor.itemKeys[item]
                        + " to "
                        + actor.StationKey(target)
                        + ". The amount asked for is far too many, so joke about it; only one is brought.";
                    int line = messageRevision % 3;
                    fallback =
                        line == 0
                            ? "そんなに運んだら腕がもげちゃいます！"
                                + name
                                + "はまず1つ、"
                                + place
                                + "へお持ちしますね。"
                        : line == 1 ? "お店が空っぽになっちゃいますよ！" + name + "は1つでがまんしてくださいね。"
                        : "数えてるうちに日が暮れちゃいます！とりあえず" + name + "を1つお持ちしますね。";
                    emotion = EmotionSurprise;
                    serviceNote = "さっき頼まれた数が多すぎたので、" + name + "は1つだけにした。";
                }
                else
                    serviceNote = null;
                for (int i = 0; i < pendingItems.Length; i++)
                    if (pendingItems[i] && i != item)
                    {
                        plan += "そのあと" + actor.ItemName(i) + "も運ぶ。";
                        english += " Then carry " + actor.itemKeys[i] + ".";
                    }
                StartReply(plan, english, fallback, emotion);
                replyTarget = target;
            }
        }

        void OnJobCompleted()
        {
            if (stage != StageExecute || currentJobItem < 0 || actor.completedJobRevision != messageRevision)
                return;
            pendingItems[currentJobItem] = false;
            if (actor.itemPrices != null && currentJobItem < actor.itemPrices.Length)
                AddBill(requestGuestId, actor.itemPrices[currentJobItem]);
            string item = CafeDecider.Json(actor.itemKeys[currentJobItem]),
                place = CafeDecider.Json(actor.StationKey(actor.station));
            TrackPickup();
            AddServiceEvent("{\"event\": \"moved_with_item\", \"location\": " + place + ", \"item\": " + item + "}");
            AddServiceEvent(
                "{\"event\": \"delivery_completed\", \"item\": " + item + ", \"destination\": " + place + "}"
            );
            currentJobItem = -1;
            stage = AnyPending() ? StageDecide : StageIdle;
            if (stage == StageIdle)
            {
                lastDecision = "ご注文の品をお届けしました";
                RequestSerialization();
            }
        }

        void Ask(string state, string question, string[] options)
        {
            LogAsk(state, question, options);
            if (decider.Ask(state, question, options))
                awaitingAnswer = true;
            else
                FailService("判断モデルに質問できません: " + decider.error);
        }

        void FailService(string reason)
        {
            Debug.LogWarning("[CafeNPC decider] " + reason);
            stage = StageIdle;
            awaitingAnswer = false;
            lastDecision = reason;
            RequestSerialization();
        }

        // The conversation model only words the reply; the plan comes from the decider's choice so speech and action agree.
        // The decider then checks the wording against the plan; a reply promising anything else is replaced by the
        // plan's fixed wording. The submission waits while the conversation model is still busy instead of dropping it.
        void StartReply(string plan, string english, string fallback, int emotion)
        {
            replied = true;
            replyPlan = plan;
            replyPlanEnglish = english;
            replyFallback = fallback;
            replyEmotion = emotion;
            replyRetries = 0;
            replyTarget = -1;
            if (replying)
            {
                pipeline.AbortGeneration();
                replying = false;
            }
            checkQueued = false;
            replyQueued = true;
        }

        // A line that needs no wording by the model (totals, goodbyes): shown at once.
        void Say(string line, int emotion)
        {
            StartReply("", "", line, emotion);
            replyQueued = false;
            replyCandidate = "";
            PublishReply(false);
        }

        // The balloon, the expression, the gesture and the voice start together: one serialization carries the line,
        // its expression and its speech tokens. With a voice the line waits until the tokens are written (about a second).
        public CafeVoiceGenerator voiceGenerator; // worker only; null or not ready: lines appear at once, silent
        private string voiceLine;
        private int voiceEmotion;

        void Show(string line, int emotion)
        {
            if (voiceLine != null)
            {
                voiceGenerator.Cancel();
                voiceLine = null;
            }
            if (
                voiceGenerator != null
                && voiceGenerator.gameObject.activeInHierarchy
                && line.Length > 0
                && voiceGenerator.Speak(line)
            )
            {
                voiceLine = line;
                voiceEmotion = emotion;
                return;
            }
            Publish(line, emotion, null);
        }

        void Publish(string line, int emotion, ushort[] speech)
        {
            actor.message = line;
            actor.emotion = emotion;
            actor.messageSerial++;
            actor.speechTokens = speech != null ? speech : new ushort[0];
            if (speech != null && speech.Length > 0)
                actor.voiceSerial++;
            actor.RequestSerialization();
        }

        void UpdateVoice()
        {
            if (voiceLine == null || voiceGenerator.phase == CafeVoiceGenerator.PhaseRunning)
                return;
            bool spoken = voiceGenerator.phase == CafeVoiceGenerator.PhaseDone;
            Debug.Log(
                "[CafeNPC voice] "
                    + (
                        spoken
                            ? voiceGenerator.speech.Length
                                + " tokens in "
                                + voiceGenerator.seconds.ToString("F1")
                                + " s"
                            : "no voice"
                    )
                    + ": "
                    + voiceLine
            );
            string line = voiceLine;
            voiceLine = null;
            Publish(line, voiceEmotion, spoken ? voiceGenerator.speech : null);
        }

        // The last exchanges with this guest, and what the NPC should remember about the last service. Without them
        // every reply was written as if nothing had been said before.
        private string[] memoryGuest = new string[2],
            memoryReply = new string[2];
        private int memoryCount;
        private string serviceNote;

        void Remember(string guest, string reply)
        {
            if (string.IsNullOrEmpty(guest) || string.IsNullOrEmpty(reply))
                return;
            if (memoryCount == memoryGuest.Length)
            {
                for (int i = 1; i < memoryCount; i++)
                {
                    memoryGuest[i - 1] = memoryGuest[i];
                    memoryReply[i - 1] = memoryReply[i];
                }
                memoryCount--;
            }
            memoryGuest[memoryCount] = LlmPipeline.Sanitize(guest, 60);
            memoryReply[memoryCount] = reply;
            memoryCount++;
        }

        string LastGuestLine()
        {
            return memoryCount > 0 ? memoryGuest[memoryCount - 1] : null;
        }

        void UpdateReply()
        {
            if (replyQueued && !pipeline.IsBusy() && stage != StageExtract)
            {
                UsePrefix(false);
                pipeline.ClearHistory();
                string lines = "";
                for (int i = 0; i < memoryCount; i++)
                    lines += "客「" + memoryGuest[i] + "」\n茜犬「" + memoryReply[i] + "」\n";
                // SubmitObservation, not Submit: Submit cuts the text at 200 characters, which dropped the plan after a long utterance.
                if (pipeline.SubmitObservation(lines + "客「" + requestText + "」\nすること: " + replyPlan))
                {
                    replyQueued = false;
                    replying = true;
                    replyStartedAt = Time.time;
                }
                return;
            }
            if (!replying)
                return;
            // The worker's stall watchdog aborts a slow generation back to Ready without Done/Error: retry once, then
            // fall back to the plan wording. A generation that never reports back must not leave the guest unanswered.
            if (pipeline.status == LlmPipeline.StatusReady && Time.time - replyStartedAt > .5f)
            {
                replying = false;
                if (replyRetries++ == 0)
                {
                    replyQueued = true;
                    Debug.LogWarning("[CafeNPC reply] generation was aborted; retrying");
                }
                else
                    PublishReply(false);
                return;
            }
            if (Time.time - replyStartedAt > 30)
            {
                Debug.LogWarning("[CafeNPC reply] conversation model did not finish; status=" + pipeline.status);
                pipeline.AbortGeneration();
                replying = false;
                PublishReply(false);
                return;
            }
            if (pipeline.status == LlmPipeline.StatusDone)
            {
                replying = false;
                ReadReply(pipeline.replyText);
                if (replyCandidate.Length == 0 || Misstates(replyCandidate))
                    PublishReply(false);
                else
                    checkQueued = true;
            }
            else if (pipeline.status == LlmPipeline.StatusError || pipeline.status == LlmPipeline.StatusFatal)
            {
                replying = false;
                PublishReply(false);
            }
        }

        // The model writes "expression｜line"; an unknown or missing tag keeps the plan's own expression.
        void ReadReply(string raw)
        {
            raw = raw.Replace("\n", "").Replace("  ", "").Trim();
            if (raw.StartsWith("→"))
                raw = raw.Substring(1).Trim();
            candidateEmotion = replyEmotion;
            int bar = raw.IndexOf('｜');
            if (bar < 0)
                bar = raw.IndexOf('|');
            if (bar >= 0 && bar <= 4)
            {
                string tag = raw.Substring(0, bar).Trim();
                if (tag == "笑顔")
                    candidateEmotion = EmotionSmile;
                else if (tag == "困り")
                    candidateEmotion = EmotionTrouble;
                else if (tag == "驚き")
                    candidateEmotion = EmotionSurprise;
                else if (tag == "照れ")
                    candidateEmotion = EmotionShy;
                else if (tag == "普通")
                    candidateEmotion = EmotionNeutral;
                raw = raw.Substring(bar + 1).Trim();
            }
            replyCandidate = LlmPipeline.Sanitize(raw.Replace("「", "").Replace("」", ""), 80);
        }

        // Slips the decider's check lets through, because it reads the plan in English and the reply in Japanese:
        // naming another table than the one the item goes to, promising a delivery when nothing is carried, and
        // "waiting" written for "bringing".
        // This reads the NPC's own wording only, never the guest's.
        private int replyTarget = -1;

        bool Misstates(string reply)
        {
            // The 1.2B model writes 「お待ちしています」 (I am waiting) where it means 「お持ちします」 (I will bring it), and
            // 「お待たせしました」 before anything was brought. Telling it not to in the prompt made it say so more often.
            if (reply.Contains("お待ちし") || reply.Contains("お待たせ"))
                return true;
            if (replyTarget < 0)
                return reply.Contains("お届け")
                    || reply.Contains("お持ちし")
                    || reply.Contains("運び")
                    || reply.Contains("持って行")
                    || reply.Contains("持ってい");
            string own = PlaceWord(replyTarget);
            string rest = own.Length > 0 ? reply.Replace(own, "") : reply;
            for (int i = 0; i < actor.route.Length; i++)
            {
                if (i == replyTarget || i == actor.homeStation || !actor.IsPlacementStation(i))
                    continue;
                string other = PlaceWord(i);
                if (other.Length > 0 && rest.Contains(other))
                    return true;
            }
            return false;
        }

        string PlaceWord(int station)
        {
            return actor.StationName(station).Replace("のテーブル", "");
        }

        void AskReplyCheck()
        {
            checkQueued = false;
            string state =
                "{\"guest_utterance\": "
                + CafeDecider.Json(requestText)
                + ", \"npc_plan\": "
                + CafeDecider.Json(replyPlanEnglish)
                + ", \"npc_reply\": "
                + CafeDecider.Json(replyCandidate)
                + "}";
            string question =
                "Does npc_reply promise, offer, or claim to bring, prepare, or serve anything that npc_plan does not include, name a different place than npc_plan, ask the guest to move, carry, or hold something, or otherwise contradict npc_plan?";
            string[] options = new[]
            {
                "no: npc_reply is consistent with npc_plan.",
                "yes: npc_reply promises or claims something outside npc_plan.",
            };
            LogAsk(state, question, options);
            if (decider.Ask(state, question, options))
                awaitingCheck = true;
            else
                PublishReply(false);
        }

        // One line per decider question with its full text, so native runs can re-ask exactly the same rows
        // outside Unity; the answer follows as "[CafeNPC decider answer]".
        void LogAsk(string state, string question, string[] options)
        {
            Debug.Log("[CafeNPC decider state] " + state + " || " + question + " || " + string.Join(" | ", options));
        }

        void PublishReply(bool useCandidate)
        {
            string line = useCandidate ? replyCandidate : replyFallback;
            Show(line, useCandidate ? candidateEmotion : replyEmotion);
            Remember(requestText, line);
            Debug.Log(
                "[CafeNPC reply] "
                    + (useCandidate ? "" : "(plan wording; model reply was: " + replyCandidate + ") ")
                    + line
            );
        }

        // Native history records the pickup as separate events; derive them from the actor's synced hand state.
        void TrackPickup()
        {
            if (stage != StageExecute || currentJobItem < 0 || jobPickupLogged || actor.heldItem != currentJobItem)
                return;
            jobPickupLogged = true;
            if (jobPickupPlace != null)
                AddServiceEvent(
                    "{\"event\": \"moved_to_pickup\", \"location\": " + CafeDecider.Json(jobPickupPlace) + "}"
                );
            AddServiceEvent(
                "{\"event\": \"picked_up\", \"item\": " + CafeDecider.Json(actor.itemKeys[currentJobItem]) + "}"
            );
        }

        void AddServiceEvent(string json)
        {
            if (serviceEventCount == serviceEvents.Length)
            {
                for (int i = 1; i < serviceEvents.Length; i++)
                    serviceEvents[i - 1] = serviceEvents[i];
                serviceEventCount--;
            }
            serviceEvents[serviceEventCount++] = json;
        }

        bool AnyPending()
        {
            if (pendingItems == null)
                return false;
            for (int i = 0; i < pendingItems.Length; i++)
                if (pendingItems[i])
                    return true;
            return false;
        }

        // Sorted item keys like Python sorted(set), or null.
        string ItemList(bool[] items)
        {
            if (items == null)
                return "null";
            int count = 0;
            for (int i = 0; i < items.Length; i++)
                if (items[i])
                    count++;
            var keys = new string[count];
            count = 0;
            for (int i = 0; i < items.Length; i++)
                if (items[i])
                    keys[count++] = actor.itemKeys[i];
            for (int i = 1; i < keys.Length; i++)
            for (int j = i; j > 0 && string.CompareOrdinal(keys[j - 1], keys[j]) > 0; j--)
            {
                var t = keys[j];
                keys[j] = keys[j - 1];
                keys[j - 1] = t;
            }
            string result = "[";
            for (int i = 0; i < keys.Length; i++)
                result += (i > 0 ? ", " : "") + CafeDecider.Json(keys[i]);
            return result + "]";
        }

        // The decider was evaluated with counter/table locations only, never the aisle waypoints between them.
        int NearestPlacementStation(Vector3 position)
        {
            int nearest = 0;
            float best = float.MaxValue;
            for (int i = 0; i < actor.route.Length; i++)
            {
                if (!actor.IsPlacementStation(i))
                    continue;
                float d = Vector3.Distance(position, actor.route[i].position);
                if (d < best)
                {
                    best = d;
                    nearest = i;
                }
            }
            return nearest;
        }

        // Seat index -> route station of the table it belongs to, and -> counter slot for stools (set by the seat installer).
        public int[] seatStations;
        public int[] seatCounterSlots;
        public Transform[] seatPoints; // station enter point per seat index

        // The seat a player sits on, or -1. Station events fill seatPlayers on the clients that saw them; a worker who
        // joined or took over after the guest sat down saw none, so a player standing on a seat point also counts.
        int SeatOf(int playerId)
        {
            if (playerId < 0 || seatStations == null)
                return -1;
            if (seatPlayers != null)
                for (int i = 0; i < seatPlayers.Length; i++)
                    if (seatPlayers[i] == playerId)
                        return i;
            var player = VRCPlayerApi.GetPlayerById(playerId);
            if (!Utilities.IsValid(player) || seatPoints == null)
                return -1;
            Vector3 position = player.GetPosition();
            for (int i = 0; i < seatPoints.Length && i < seatStations.Length; i++)
                if (seatPoints[i] != null && Vector3.Distance(position, seatPoints[i].position) < .15f)
                    return i;
            return -1;
        }

        // The stool the requesting guest sits on; a standing guest gets the nearest counter spot.
        int CounterSlotForGuest()
        {
            int seat = SeatOf(requestGuestId);
            if (seat >= 0 && seatCounterSlots != null && seat < seatCounterSlots.Length && seatCounterSlots[seat] >= 0)
                return seatCounterSlots[seat];
            var guest = VRCPlayerApi.GetPlayerById(requestGuestId);
            if (!Utilities.IsValid(guest) || actor.counterSlots == null)
                return -1;
            int nearest = -1;
            float best = float.MaxValue;
            for (int i = 0; i < actor.counterSlots.Length; i++)
            {
                float d = Mathf.Abs(guest.GetPosition().x - actor.counterSlots[i].position.x);
                if (d < best)
                {
                    best = d;
                    nearest = i;
                }
            }
            return nearest;
        }

        private int[] seatPlayers;

        public void SeatEntered(int seat, int playerId)
        {
            if (seatStations == null || seat < 0 || seat >= seatStations.Length)
                return;
            if (seatPlayers == null || seatPlayers.Length != seatStations.Length)
            {
                seatPlayers = new int[seatStations.Length];
                for (int i = 0; i < seatPlayers.Length; i++)
                    seatPlayers[i] = -1;
            }
            seatPlayers[seat] = playerId;
        }

        public void SeatExited(int seat, int playerId)
        {
            if (seatPlayers != null && seat >= 0 && seat < seatPlayers.Length && seatPlayers[seat] == playerId)
                seatPlayers[seat] = -1;
        }

        string GuestLocationKey()
        {
            int station = GuestStation();
            return station >= 0 ? actor.StationKey(station) : "unknown";
        }

        // A seated guest's table is a fact; standing guests fall back to the nearest table approach point. -1: unknown.
        int GuestStation()
        {
            int seat = SeatOf(requestGuestId);
            if (seat >= 0)
                return seatStations[seat];
            var guest = VRCPlayerApi.GetPlayerById(requestGuestId);
            if (!Utilities.IsValid(guest))
                return -1;
            Vector3 position = guest.GetPosition();
            int nearest = -1;
            float distance = float.MaxValue;
            for (int i = 0; i < actor.route.Length; i++)
            {
                if (!actor.IsPlacementStation(i))
                    continue;
                float candidate = Vector3.Distance(position, actor.route[i].position);
                if (candidate < distance)
                {
                    distance = candidate;
                    nearest = i;
                }
            }
            if (nearest < 0)
                return -1;
            if (distance < 1.6f)
                return nearest;
            // A guest who just rang the bell of their nearest table belongs to it even when standing a little farther off.
            // Only a tie-break for the nearest table: a bell never pulls a guest to another table (tables are 2.65 m apart),
            // and it expires so a later order from the same guest is placed by where they stand.
            bool recentBell = requestGuestId == bellCallerId && bellStation == nearest && Time.time - bellAt < 90;
            return recentBell ? nearest : -1;
        }
    }
}
