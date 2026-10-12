using UdonSharp;
using UnityEngine;
using UnityEngine.UI;
using VRC.SDKBase;

namespace CafeNPC
{
    [UdonBehaviourSyncMode(BehaviourSyncMode.Manual)]
    public class CafeNpcController : UdonSharpBehaviour
    {
        public Transform[] route;
        public string[] stationNames;
        public int[] routeParents;
        public GameObject[] transportItems;
        public string[] itemNames;
        public string[] itemKeys; // decider identifiers, e.g. "water"
        public string[] itemLabels; // decider question labels, e.g. "water (水, お水)"
        public string[] stationKeys; // decider identifiers per route index, e.g. "window_table"
        public float[] itemPlacementHeights;
        public int[] itemPrices; // yen per item; 0 for water and the menu
        public CafeEdible[] edibles; // per item: pickup / eat / drink state (a guest-held item is not placed by the NPC)
        public Transform[] placementAnchors;
        public Transform[] counterSlots; // countertop spot in front of each counter stool
        public int counterStation; // station whose deliveries go to a counter slot
        public int homeStation = -1; // where items are kept; not a delivery destination
        public Transform[] limbs; // left thigh, right thigh, left upper arm, right upper arm
        public Transform[] knees; // left shin, right shin
        public Transform[] feet; // left, right
        public Transform[] toes; // left, right (optional)
        public Transform leftForearm;
        public Transform leftHand;
        public Transform hips;
        public Transform spine;
        public Transform chest;
        public Transform head;
        public SkinnedMeshRenderer face; // body mesh carrying the blink and viseme shape keys
        public int blinkShape = -1;
        public int[] talkShapes; // vowel visemes (aa, oh, ih) cycled while a new line is shown
        public int[] faceShapes; // blend shapes that make the expressions
        public float[] faceWeights; // per face shape, one weight per emotion (CafeNpcBrain.Emotion*: neutral, smile, trouble, surprise, shy)
        public bool[] faceMouth; // per face shape: a mouth shape, held back while the talking mouth moves
        public Text speech;
        public Transform speechBubble; // the balloon canvas that faces the local viewer (the Text sits inside it)
        public GameObject cup;
        public Transform carryForearm;
        public Transform carryHand;
        public int uprightItem = -1; // the item carried standing up instead of flat on the hand (the menu card)
        public CafeNpcBrain brain;
        public float walkSpeed = 1.1f;

        [UdonSynced]
        public int phase;

        [UdonSynced]
        public int station;

        [UdonSynced]
        public int guestId = -1;

        [UdonSynced]
        public Vector3 from;

        [UdonSynced]
        public Vector3 destination;

        [UdonSynced]
        public double beganAt;

        [UdonSynced]
        public float duration;

        [UdonSynced]
        public string message = "いらっしゃいませ。";

        [UdonSynced]
        public int emotion = 1; // expression shown with the line

        [UdonSynced]
        public int messageSerial; // counts lines, so the same wording twice still plays its expression and gesture

        [UdonSynced]
        public ushort[] speechTokens = new ushort[0]; // the current line as MioTTS speech tokens, empty without a voice

        [UdonSynced]
        public int voiceSerial; // counts spoken lines
        public CafeVoicePlayer voice;

        [UdonSynced]
        public bool holdingDrink;

        [UdonSynced]
        public int targetStation;

        [UdonSynced]
        public bool actionComplete = true;

        [UdonSynced]
        public string actionResult = "待機中";

        [UdonSynced]
        public int servedCount;

        [UdonSynced]
        public string activeAction = "wait";

        [UdonSynced]
        public int movingTo;

        [UdonSynced]
        public int[] itemStations;

        [UdonSynced]
        public int[] itemSlots; // counter slot of an item placed at counterStation, -1 otherwise

        [UdonSynced]
        public int jobSlot = -1;

        [UdonSynced]
        public int heldItem = -1;

        [UdonSynced]
        public int jobItem = -1;

        [UdonSynced]
        public int jobDestination = -1;

        [UdonSynced]
        public int jobRequestRevision;

        [UdonSynced]
        public int completedJobRevision;

        [UdonSynced]
        public int completedJobCount;

        [UdonSynced]
        public string completedJobResult = "";
        public string[] choiceActions;
        public int[] choiceItems;
        public int[] choiceTargets;
        public int choiceCount;
        private Quaternion[] rest;
        private string shown;
        private Quaternion[] kneeRest,
            footRest,
            toeRest;
        private Quaternion leftForearmRest,
            rightForearmRest,
            hipsRest,
            spineRest,
            chestRest,
            headRest;
        private float walkBlend,
            gaitPhase,
            bob;
        private float nextBlink = 1,
            blinkAt = -1,
            talkUntil;
        private float[] faceShown;
        private float emotionUntil,
            gestureAt = -10;
        private int shownSerial = -1,
            gesture,
            handGesture;
        private int heardSerial;

        void Start()
        {
            // Rest poses relative to the NPC root, so swings can use the body's own right axis
            // (the imported bones have arbitrary local axes).
            if (limbs == null || limbs.Length < 4)
            {
                Debug.LogError("[CafeNpc] limbs must hold 4 bones; run CafeNPC/Configure Carry Pose");
                return;
            }
            var inv = Quaternion.Inverse(transform.rotation);
            rest = new Quaternion[limbs.Length];
            for (int i = 0; i < limbs.Length; i++)
                rest[i] = inv * limbs[i].rotation;
            kneeRest = RestOf(knees, inv);
            footRest = RestOf(feet, inv);
            toeRest = RestOf(toes, inv);
            if (leftForearm != null)
                leftForearmRest = inv * leftForearm.rotation;
            if (carryForearm != null)
                rightForearmRest = inv * carryForearm.rotation;
            if (hips != null)
                hipsRest = inv * hips.rotation;
            if (spine != null)
                spineRest = inv * spine.rotation;
            if (chest != null)
                chestRest = inv * chest.rotation;
            if (head != null)
                headRest = inv * head.rotation;
            // The saved scene pose has elbows and hands converging in front of the crotch.
            // Re-aim the arm rest poses so they hang beside the thighs with slightly bent elbows.
            if (leftForearm != null && leftHand != null)
            {
                rest[2] = AimRest(rest[2], limbs[2].position, leftForearm.position, new Vector3(-.17f, -1, 0), inv);
                leftForearmRest = AimRest(
                    leftForearmRest,
                    leftForearm.position,
                    leftHand.position,
                    new Vector3(-.1f, -1, .14f),
                    inv
                );
            }
            if (carryForearm != null && carryHand != null)
            {
                rest[3] = AimRest(rest[3], limbs[3].position, carryForearm.position, new Vector3(.17f, -1, 0), inv);
                rightForearmRest = AimRest(
                    rightForearmRest,
                    carryForearm.position,
                    carryHand.position,
                    new Vector3(.1f, -1, .14f),
                    inv
                );
            }
            if (Networking.IsOwner(gameObject))
            {
                from = route[0].position;
                destination = from;
                station = 0;
                movingTo = 0;
                beganAt = Networking.GetServerTimeInSeconds();
                duration = 3;
                RequestSerialization();
            }
        }

        void Update()
        {
            if (rest == null)
                return;
            var elapsed = (float)(Networking.GetServerTimeInSeconds() - beganAt);
            var moving = Vector3.Distance(from, destination) > .01f;
            if (duration > 0)
            {
                transform.position = Vector3.Lerp(from, destination, Mathf.Clamp01(elapsed / duration));
                bob = 0;
            }
            var direction = destination - from;
            if (moving && elapsed < duration)
            {
                transform.rotation = Quaternion.Slerp(
                    transform.rotation,
                    Quaternion.LookRotation(direction),
                    Time.deltaTime * 7
                );
            }
            else
            {
                var look = transform.position + route[station].forward;
                var guest = VRCPlayerApi.GetPlayerById(guestId);
                if (jobItem >= 0 && phase == 5 && transportItems[jobItem] != null)
                    look = transportItems[jobItem].transform.position;
                else if (jobItem >= 0 && phase == 7)
                {
                    var anchor = PlacementAnchor(jobDestination, jobSlot);
                    if (anchor != null)
                        look = anchor.position;
                }
                else if (Utilities.IsValid(guest) && Vector3.Distance(guest.GetPosition(), transform.position) < 3.5f)
                    look = guest.GetPosition();
                var delta = look - transform.position;
                delta.y = 0;
                if (delta.sqrMagnitude > .0025f)
                    transform.rotation = Quaternion.Slerp(
                        transform.rotation,
                        Quaternion.LookRotation(delta),
                        Time.deltaTime * 4
                    );
            }
            Animate(moving && elapsed < duration);
            AnimateFace();
            if (holdingDrink && carryHand != null)
            {
                cup.transform.position = carryHand.TransformPoint(new Vector3(0, .08f, 0)) + Vector3.up * .035f;
                cup.transform.rotation = Quaternion.Euler(0, transform.eulerAngles.y, 0);
            }
            cup.SetActive(holdingDrink);
            // A new line starts its balloon, mouth movement, expression and gesture in the same frame.
            if (shown != message || shownSerial != messageSerial)
            {
                speech.text = message;
                shown = message;
                shownSerial = messageSerial;
                talkUntil =
                    message != null && message.Length > 0
                        ? Time.time + Mathf.Clamp(message.Length * .09f, .6f, 4.5f)
                        : 0;
                emotionUntil = talkUntil + 2.5f;
                if (talkUntil > 0)
                {
                    gestureAt = Time.time;
                    gesture = emotion;
                    handGesture = HandGestureOf(message, emotion);
                }
                if (speechBubble != null)
                    speechBubble.gameObject.SetActive(message != null && message.Length > 0);
            }
            // The same serialization that brings the line brings its speech tokens; each client decodes them itself.
            if (heardSerial != voiceSerial)
            {
                heardSerial = voiceSerial;
                if (voice != null && speechTokens != null && speechTokens.Length > 0)
                {
                    voice.Play(speechTokens);
                    // The mouth moves for as long as the voice lasts (25 tokens a second), not by the length of the text.
                    talkUntil = Time.time + speechTokens.Length / 25f + .3f;
                    emotionUntil = talkUntil + 2.5f;
                }
            }
            var local = Networking.LocalPlayer;
            if (Utilities.IsValid(local))
            {
                var eye = local.GetTrackingData(VRCPlayerApi.TrackingDataType.Head).position;
                var bubble = speechBubble != null ? speechBubble : speech.transform;
                bubble.rotation = Quaternion.LookRotation(bubble.position - eye);
            }
            if (!Networking.IsOwner(gameObject))
                return;
            TrackFreeItems();
            if (actionComplete || elapsed < duration)
                return;
            if (phase == 1)
                station = movingTo;
            if (jobItem >= 0)
            {
                if (phase == 5)
                {
                    heldItem = jobItem;
                    itemStations[jobItem] = -1;
                    var edible = Edible(jobItem);
                    if (edible != null)
                        edible.Reclaim();
                }
                if (phase == 7)
                {
                    itemStations[jobItem] = station;
                    heldItem = -1;
                    if (itemSlots != null && jobItem < itemSlots.Length)
                        itemSlots[jobItem] = station == counterStation ? jobSlot : -1;
                    completedJobRevision = jobRequestRevision;
                    completedJobCount++;
                    completedJobResult = ItemName(jobItem) + "を" + StationName(station) + "に置いた";
                    jobItem = -1;
                    jobDestination = -1;
                    CompleteAction(completedJobResult);
                    return;
                }
                AdvanceTransport();
                return;
            }
            if (phase == 1 && station != targetStation)
            {
                Move(NextRouteHop(station, targetStation));
                return;
            }
            if (phase == 3)
                holdingDrink = true;
            if (phase == 4)
            {
                holdingDrink = false;
                servedCount++;
            }
            CompleteAction(activeAction + "完了、現在地=" + StationName(station));
        }

        void LateUpdate()
        {
            if (transportItems == null || itemStations == null)
                return;
            for (int i = 0; i < transportItems.Length; i++)
            {
                var item = transportItems[i];
                if (item == null || i >= itemStations.Length)
                    continue;
                var edible = Edible(i);
                if (edible != null)
                    edible.SetCarried(i == heldItem);
                if (i != heldItem && IsFree(i))
                    continue; // a guest has it; VRCObjectSync moves it
                if (i == heldItem && carryHand != null)
                {
                    if (i == uprightItem)
                    {
                        // The menu card is carried standing, leaning back a little, its page towards whoever she walks to.
                        item.transform.position = carryHand.TransformPoint(new Vector3(0, .08f, 0)) + Vector3.up * .15f;
                        item.transform.rotation = Quaternion.Euler(-78, transform.eulerAngles.y + 180, 0);
                    }
                    else
                    {
                        item.transform.position =
                            carryHand.TransformPoint(new Vector3(0, .08f, 0)) + Vector3.up * .035f;
                        item.transform.rotation = Quaternion.Euler(0, transform.eulerAngles.y, 0);
                    }
                }
                else
                {
                    int place = itemStations[i];
                    int slot = itemSlots != null && i < itemSlots.Length ? itemSlots[i] : -1;
                    var anchor = PlacementAnchor(place, slot);
                    if (anchor != null)
                    {
                        float height =
                            itemPlacementHeights != null && i < itemPlacementHeights.Length
                                ? itemPlacementHeights[i]
                                : 0;
                        item.transform.position = anchor.position + Vector3.up * height + SpotOffset(i, place, slot);
                        item.transform.rotation = anchor.rotation;
                    }
                }
            }
        }

        public CafeEdible Edible(int item)
        {
            return edibles != null && item >= 0 && item < edibles.Length ? edibles[item] : null;
        }

        bool IsFree(int item)
        {
            var edible = Edible(item);
            return edible != null && edible.free;
        }

        // A guest may carry an item to another table; record the table it now rests on so the decider state and the
        // NPC's pickup route follow it. An item dropped away from every table keeps its last table.
        void TrackFreeItems()
        {
            if (itemStations == null || transportItems == null)
                return;
            bool changed = false;
            for (int i = 0; i < transportItems.Length && i < itemStations.Length; i++)
            {
                if (i == heldItem || !IsFree(i) || edibles[i].HeldByPlayer())
                    continue;
                int nearest = -1;
                float best = 1.2f;
                for (int place = 0; place < route.Length; place++)
                {
                    var anchor = PlacementAnchor(place, -1);
                    if (anchor == null)
                        continue;
                    var d = transportItems[i].transform.position - anchor.position;
                    d.y = 0;
                    if (d.magnitude < best)
                    {
                        best = d.magnitude;
                        nearest = place;
                    }
                }
                if (nearest >= 0 && nearest != itemStations[i])
                {
                    itemStations[i] = nearest;
                    if (itemSlots != null && i < itemSlots.Length)
                        itemSlots[i] = -1;
                    changed = true;
                }
            }
            if (changed)
                RequestSerialization();
        }

        // Items sharing a spot fill a small grid: centre, right, left along the edge the NPC faces, then a row further in.
        // The station's facing gives the axes, so the grid lies along the table edge whichever side the table is on.
        Vector3 SpotOffset(int item, int place, int slot)
        {
            int k = 0;
            for (int j = 0; j < item; j++)
                if (
                    j != heldItem
                    && itemStations[j] == place
                    && (itemSlots == null || j >= itemSlots.Length || itemSlots[j] == slot)
                )
                    k++;
            if (place < 0 || place >= route.Length || route[place] == null)
                return Vector3.zero;
            int column = k % 3;
            float side =
                column == 0 ? 0
                : column == 1 ? .18f
                : -.18f;
            return route[place].right * side + route[place].forward * ((k / 3) * .17f);
        }

        // Procedural gait (reviewed with Codex). All angles are in the NPC root frame, never the bones'
        // arbitrary local axes: pitch about root right (negative = forward), yaw about up, roll about forward.
        // Bones are posed parent-first because each Pose writes an absolute rotation.
        // Left leg phase: s>0 leg in front, c>0 leg swinging forward. Arms swing opposite to the same-side leg.
        void Animate(bool walking)
        {
            walkBlend = Mathf.MoveTowards(walkBlend, walking ? 1 : 0, Time.deltaTime * (walking ? 3.5f : 5.5f));
            float w = walkBlend * walkBlend * (3 - 2 * walkBlend);
            // Keep cycling while blending out so the stride settles instead of freezing mid-step.
            if (walkBlend > .001f)
                gaitPhase = Mathf.Repeat(
                    gaitPhase + Time.deltaTime * 2 * Mathf.PI * walkSpeed / 1.35f * Mathf.Lerp(.35f, 1, walkBlend),
                    2 * Mathf.PI
                );
            float s = Mathf.Sin(gaitPhase),
                c = Mathf.Cos(gaitPhase);
            float breathe = Mathf.Sin(Time.time * 1.6f) * 1.2f * (1 - w);
            bool carrying = holdingDrink || heldItem >= 0;

            // Pelvis turns with the forward leg and drops toward the swing side; spine and chest counter it, head stays level.
            // Standing still she shifts her weight slowly and lets her head drift and tilt a little; a fully frozen body
            // with only breathing read as a mannequin. The hips move by about a degree so the feet do not visibly slide.
            float idle = 1 - w,
                sway = Mathf.Sin(Time.time * .7f) * idle;
            float lookYaw = (Mathf.PerlinNoise(Time.time * .11f, 3.1f) - .5f) * 12 * idle,
                lookTilt = (Mathf.PerlinNoise(Time.time * .09f, 7.3f) - .5f) * 7 * idle;
            // A short gesture with each line, only while standing: a nod with a smile, a small bow with an apology,
            // a start backwards in surprise, a tilt of the head when bashful. Positive pitch leans the torso forward.
            float nod = 0,
                lean = 0,
                tilt = 0,
                sinceLine = Time.time - gestureAt;
            if (sinceLine < 1.2f)
            {
                float whole = Mathf.Sin(Mathf.Clamp01(sinceLine / 1.2f) * Mathf.PI) * idle;
                if (gesture == 1)
                    nod = 7 * Mathf.Sin(Mathf.Clamp01(sinceLine / .5f) * Mathf.PI) * idle;
                else if (gesture == 2)
                    lean = 13 * whole;
                else if (gesture == 3)
                    lean = -6 * Mathf.Sin(Mathf.Clamp01(sinceLine / .6f) * Mathf.PI) * idle;
                else if (gesture == 4)
                    tilt = 8 * whole;
            }
            Pose3(hips, hipsRest, 0, 4 * s * w + .8f * sway, 2 * c * w + 1 * sway);
            Pose3(spine, spineRest, breathe * .4f + lean, -2.5f * s * w - .6f * sway, -1.2f * c * w - 1.3f * sway);
            Pose3(chest, chestRest, 0, -1.5f * s * w, -.6f * c * w - .5f * sway);
            Pose3(head, headRest, nod + lean * .3f, -.5f * s * w + lookYaw, lookTilt + tilt);

            float leftLeg = -22 * s * w,
                rightLeg = 22 * s * w;
            Pose(limbs[0], rest[0], leftLeg);
            Pose(limbs[1], rest[1], rightLeg);
            if (kneeRest.Length == 2)
            {
                // Knee flex peaks mid-swing; a little stance flex avoids a locked knee.
                Pose(knees[0], kneeRest[0], leftLeg + (5 + 34 * Mathf.Max(0, c)) * w);
                Pose(knees[1], kneeRest[1], rightLeg + (5 + 34 * Mathf.Max(0, -c)) * w);
            }
            // Feet use root-frame pitch so soles stay near the floor instead of following the shin.
            if (footRest.Length == 2)
            {
                Pose(feet[0], footRest[0], FootPitch(gaitPhase) * w);
                Pose(feet[1], footRest[1], FootPitch(gaitPhase + Mathf.PI) * w);
            }
            if (toeRest.Length == 2)
            {
                Pose(toes[0], toeRest[0], -18 * Pulse(gaitPhase, Mathf.PI * 1.55f, .35f) * w);
                Pose(toes[1], toeRest[1], -18 * Pulse(gaitPhase + Mathf.PI, Mathf.PI * 1.55f, .35f) * w);
            }

            float leftArm = 13 * s * w + breathe,
                rightArm = carrying ? -20 : -13 * s * w + breathe;
            Pose(limbs[2], rest[2], leftArm);
            Pose(limbs[3], rest[3], rightArm);
            // Elbows bend a little more on the forward swing.
            if (leftForearm != null)
                Pose(leftForearm, leftForearmRest, leftArm - 6 - 8 * Mathf.Max(0, -s) * w);
            if (carryForearm != null)
                Pose(carryForearm, rightForearmRest, rightArm - (carrying ? 78 : 6 + 8 * Mathf.Max(0, s) * w));

            // Hands, with some lines and only while standing. The right hand joins in only when it carries nothing.
            // Directions are in the NPC's own frame: x to her right, z forward.
            float handTime = handGesture == 1 ? 2f : 1.7f;
            if (handGesture > 0 && sinceLine < handTime && leftForearm != null && carryForearm != null)
            {
                float k =
                    Mathf.SmoothStep(0, 1, Mathf.Clamp01(sinceLine / .3f))
                    * Mathf.SmoothStep(0, 1, Mathf.Clamp01((handTime - sinceLine) / .4f))
                    * idle;
                if (handGesture == 1)
                {
                    // A small wave beside the shoulder.
                    float swing = .3f * Mathf.Sin(sinceLine * 10);
                    if (carrying)
                        Reach(true, new Vector3(-.75f, -.6f, .3f), new Vector3(-.2f - swing, 1, .25f), k);
                    else
                        Reach(false, new Vector3(.75f, -.6f, .3f), new Vector3(.2f + swing, 1, .25f), k);
                }
                else if (handGesture == 2)
                {
                    // Both hands together in front, with the bow.
                    Reach(true, new Vector3(-.2f, -1, .2f), new Vector3(.8f, -.5f, .6f), k);
                    if (!carrying)
                        Reach(false, new Vector3(.2f, -1, .2f), new Vector3(-.8f, -.5f, .6f), k);
                }
                else if (handGesture == 3)
                    // One hand up to the mouth.
                    Reach(true, new Vector3(-.3f, -.8f, .55f), new Vector3(.6f, .85f, .2f), k);
                else
                    // An open hand held out to the side: this way, here you are.
                    Reach(true, new Vector3(-.5f, -.8f, .35f), new Vector3(-.65f, .1f, .75f), k);
            }

            // Apply only the change in bob: the position is not re-lerped every frame (e.g. after an interrupt).
            float nextBob = (1 - Mathf.Cos(2 * gaitPhase)) * .009f * w;
            transform.position += Vector3.up * (nextBob - bob);
            bob = nextBob;
        }

        // Blinks every few seconds (sometimes twice) and moves the mouth while a new line is being "spoken".
        void AnimateFace()
        {
            if (face == null)
                return;
            // The expression of the current line stays a little after the mouth stops, then eases back to neutral.
            int expression = Time.time < emotionUntil ? emotion : 0;
            if (faceShapes != null && faceWeights != null)
            {
                if (faceShown == null || faceShown.Length != faceShapes.Length)
                    faceShown = new float[faceShapes.Length];
                for (int i = 0; i < faceShapes.Length; i++)
                {
                    int cell = i * 5 + expression;
                    float target = expression >= 0 && cell < faceWeights.Length ? faceWeights[cell] : 0;
                    // Mouth shapes wait until the line has been spoken: added to a viseme they pull the lips out of shape.
                    if (Time.time < talkUntil && faceMouth != null && i < faceMouth.Length && faceMouth[i])
                        target = 0;
                    faceShown[i] = Mathf.MoveTowards(faceShown[i], target, Time.deltaTime * 450);
                    if (faceShapes[i] >= 0)
                        face.SetBlendShapeWeight(faceShapes[i], faceShown[i]);
                }
            }
            if (blinkShape >= 0)
            {
                if (Time.time >= nextBlink)
                {
                    blinkAt = Time.time;
                    nextBlink = Time.time + (Random.value < .2f ? .32f : Random.Range(2.5f, 6f));
                }
                float b = blinkAt < 0 ? 0 : 1 - Mathf.Abs((Time.time - blinkAt) / .075f - 1); // closes and opens over 0.15 s
                // Bashful eyes are already closed; a blink on top folds the eyelids through each other.
                if (expression == 4)
                    b = 0;
                face.SetBlendShapeWeight(blinkShape, Mathf.Clamp01(b) * 100);
            }
            if (talkShapes == null || talkShapes.Length == 0)
                return;
            bool talking = Time.time < talkUntil;
            int vowel = (int)(Time.time * 7) % talkShapes.Length;
            float open = talking ? (.45f + .55f * Mathf.Abs(Mathf.Sin(Time.time * 14))) * 70 : 0;
            for (int i = 0; i < talkShapes.Length; i++)
                if (talkShapes[i] >= 0)
                    face.SetBlendShapeWeight(talkShapes[i], i == vowel ? open : 0);
        }

        // Toes up at heel strike (leg fully forward), toes down at push-off (leg behind, starting to swing).
        float FootPitch(float phase)
        {
            return -8 * Pulse(phase, Mathf.PI * .5f, .5f) + 14 * Pulse(phase, Mathf.PI * 1.55f, .38f);
        }

        float Pulse(float phase, float center, float width)
        {
            float d = Mathf.Abs(Mathf.Repeat(phase - center + Mathf.PI, 2 * Mathf.PI) - Mathf.PI);
            return Mathf.Clamp01(1 - d / width);
        }

        void Pose(Transform bone, Quaternion restRelative, float pitch)
        {
            if (bone == null)
                return;
            bone.rotation = transform.rotation * Quaternion.AngleAxis(pitch, Vector3.right) * restRelative;
        }

        void Pose3(Transform bone, Quaternion restRelative, float pitch, float yaw, float roll)
        {
            if (bone == null)
                return;
            bone.rotation =
                transform.rotation
                * Quaternion.AngleAxis(yaw, Vector3.up)
                * Quaternion.AngleAxis(roll, Vector3.forward)
                * Quaternion.AngleAxis(pitch, Vector3.right)
                * restRelative;
        }

        // Blends one arm from where the walk left it toward a pose given by the directions of its two bones.
        void Reach(bool left, Vector3 upper, Vector3 fore, float k)
        {
            float side = left ? -1 : 1;
            var upperBone = left ? limbs[2] : limbs[3];
            var foreBone = left ? leftForearm : carryForearm;
            var upperTo =
                transform.rotation
                * Quaternion.FromToRotation(new Vector3(.17f * side, -1, 0), upper)
                * (left ? rest[2] : rest[3]);
            var foreTo =
                transform.rotation
                * Quaternion.FromToRotation(new Vector3(.1f * side, -1, .14f), fore)
                * (left ? leftForearmRest : rightForearmRest);
            upperBone.rotation = Quaternion.Slerp(upperBone.rotation, upperTo, k);
            foreBone.rotation = Quaternion.Slerp(foreBone.rotation, foreTo, k);
        }

        // Which hand gesture goes with a line: 1 wave (greeting, farewell), 2 hands together (apology),
        // 3 hand to the mouth (surprise), 4 open hand (this way, here you are), 0 none.
        int HandGestureOf(string line, int feeling)
        {
            if (line == null)
                return 0;
            if (
                line.Contains("いらっしゃいませ")
                || line.Contains("ようこそ")
                || line.Contains("こんにちは")
                || line.Contains("こんばんは")
                || line.Contains("おはよう")
                || line.Contains("お越し")
                || line.Contains("またね")
                || line.Contains("お気をつけて")
                || line.Contains("さようなら")
            )
                return 1;
            if (feeling == 2)
                return 2;
            if (feeling == 3)
                return 3;
            if (line.Contains("どうぞ") || line.Contains("こちら") || line.Contains("お待たせ"))
                return 4;
            return 0;
        }

        Quaternion[] RestOf(Transform[] bones, Quaternion inv)
        {
            var result = new Quaternion[bones == null ? 0 : bones.Length];
            for (int i = 0; i < result.Length; i++)
                if (bones[i] != null)
                    result[i] = inv * bones[i].rotation;
            return result;
        }

        // Rotate a root-relative rest so the bone (start->end) points along the desired root-space direction.
        Quaternion AimRest(Quaternion restRelative, Vector3 start, Vector3 end, Vector3 desired, Quaternion inv)
        {
            return Quaternion.FromToRotation(inv * (end - start), desired) * restRelative;
        }

        void CompleteAction(string result)
        {
            actionComplete = true;
            actionResult = "成功: " + result;
            phase = 0;
            RequestSerialization();
        }

        public string ItemName(int item)
        {
            return itemNames != null && item >= 0 && item < itemNames.Length ? itemNames[item] : "物" + item;
        }

        public string ItemState()
        {
            string result = "";
            if (itemStations == null)
                return result;
            for (int i = 0; i < itemStations.Length; i++)
                result += ItemName(i) + "=" + (i == heldItem ? "手に持っている" : StationName(itemStations[i])) + "; ";
            return result;
        }

        public bool CanTransport(int item, int target)
        {
            if (
                !actionComplete
                || holdingDrink
                || transportItems == null
                || itemStations == null
                || item < 0
                || item >= transportItems.Length
                || item >= itemStations.Length
                || transportItems[item] == null
            )
                return false;
            if (
                placementAnchors == null
                || target < 0
                || target >= placementAnchors.Length
                || target >= route.Length
                || placementAnchors[target] == null
                || route[target] == null
            )
                return false;
            if (heldItem >= 0 && heldItem != item)
                return false;
            var edible = Edible(item);
            if (edible != null && edible.HeldByPlayer())
                return false;
            int source = itemStations[item];
            return source != target
                && (
                    (heldItem == item && source == -1)
                    || (heldItem < 0 && source >= 0 && source < route.Length && route[source] != null)
                );
        }

        public bool ExecuteTransport(int item, int target, string speechText, int requestRevision)
        {
            return ExecuteTransportAt(item, target, -1, speechText, requestRevision);
        }

        // slot: counter slot for deliveries to counterStation (the requesting guest's stool), -1 for the default spot.
        public bool ExecuteTransportAt(int item, int target, int slot, string speechText, int requestRevision)
        {
            if (!Networking.IsOwner(gameObject) || !actionComplete)
                return false;
            if (!CanTransport(item, target))
                return Reject("実行できない物または行き先");
            if (!string.IsNullOrEmpty(speechText))
                message = speechText;
            jobItem = item;
            jobDestination = target;
            jobRequestRevision = requestRevision;
            jobSlot =
                target == counterStation && counterSlots != null && slot >= 0 && slot < counterSlots.Length ? slot : -1;
            activeAction = "deliver";
            actionComplete = false;
            AdvanceTransport();
            return true;
        }

        void AdvanceTransport()
        {
            if (Vector3.Distance(transform.position, StationPosition(station)) > .05f)
            {
                phase = 1;
                Move(station);
                return;
            }
            int target = heldItem == jobItem ? jobDestination : itemStations[jobItem];
            if (station != target)
            {
                phase = 1;
                Move(NextRouteHop(station, target));
                return;
            }
            phase = heldItem == jobItem ? 7 : 5;
            Hold(1, message);
        }

        public void InterruptJob()
        {
            if (!Networking.IsOwner(gameObject) || actionComplete)
                return;
            jobItem = -1;
            jobDestination = -1;
            from = transform.position - Vector3.up * bob;
            destination = from;
            duration = 0;
            phase = 0;
            actionComplete = true;
            activeAction = "wait";
            actionResult =
                "中断: 新しい発言を受け取り、実行中の仕事を止めた。持ち物="
                + (
                    heldItem >= 0 ? ItemName(heldItem)
                    : holdingDrink ? "コーヒー"
                    : "なし"
                );
            RequestSerialization();
        }

        public int NextRouteHop(int current, int target)
        {
            if (current == target)
                return target;
            if (routeParents == null || routeParents.Length != route.Length)
                return current + (target > current ? 1 : -1);
            int node = target;
            for (int i = 0; i < route.Length; i++)
            {
                int parent = routeParents[node];
                if (parent == current)
                    return node;
                if (parent < 0)
                    break;
                node = parent;
            }
            return routeParents[current];
        }

        void AddChoice(string action, int item, int target)
        {
            choiceActions[choiceCount] = action;
            choiceItems[choiceCount] = item;
            choiceTargets[choiceCount] = target;
            choiceCount++;
        }

        public string BuildJobChoices()
        {
            int count = transportItems == null ? 0 : transportItems.Length;
            choiceActions = new string[4 + (count + 1) * route.Length];
            choiceItems = new int[choiceActions.Length];
            choiceTargets = new int[choiceActions.Length];
            choiceCount = 0;
            if (!actionComplete)
                return "";
            AddChoice("wait", -1, station);
            AddChoice("speak", -1, station);
            for (int item = 0; item < count; item++)
            for (int target = 0; target < route.Length; target++)
                if (CanTransport(item, target))
                    AddChoice("deliver", item, target);
            if (IsActionAvailable("prepare", 0))
                AddChoice("prepare", -1, 0);
            if (IsActionAvailable("serve", 3))
                AddChoice("serve", -1, 3);
            for (int target = 0; target < route.Length; target++)
                if (IsActionAvailable("move", target) && (placementAnchors[target] != null || target == 3))
                    AddChoice("move", -1, target);
            string result = "";
            for (int i = 0; i < choiceCount; i++)
            {
                string action = choiceActions[i];
                string description =
                    action == "wait" ? "動かず黙って待つ"
                    : action == "speak" ? "客に話す・質問する"
                    : action == "deliver"
                        ? ItemName(choiceItems[i]) + "を" + StationName(choiceTargets[i]) + "へ運んで置く"
                    : action == "prepare" ? "コーヒーを作って持つ"
                    : action == "serve" ? "コーヒーを客に渡す"
                    : StationName(choiceTargets[i]) + "へ歩く";
                result += i + ": " + description + "\n";
            }
            return result;
        }

        public bool ExecuteJobChoice(int choice, string speechText, int requestRevision)
        {
            if (choiceActions == null || choice < 0 || choice >= choiceCount)
                return false;
            if (choiceActions[choice] == "deliver")
                return ExecuteTransport(choiceItems[choice], choiceTargets[choice], speechText, requestRevision);
            return ExecuteDecision(choiceActions[choice], choiceTargets[choice], speechText);
        }

        public string StationKey(int index)
        {
            if (
                stationKeys != null
                && index >= 0
                && index < stationKeys.Length
                && !string.IsNullOrEmpty(stationKeys[index])
            )
                return stationKeys[index];
            return "place_" + index;
        }

        // Where the decider sees an item: a station key, or "npc_hand" while carried.
        public string ItemPlaceKey(int item)
        {
            if (item == heldItem || itemStations == null || item >= itemStations.Length || itemStations[item] < 0)
                return "npc_hand";
            var edible = Edible(item);
            if (edible != null && edible.HeldByPlayer())
                return "guest_hand";
            return StationKey(itemStations[item]);
        }

        public bool IsPlacementStation(int index)
        {
            return placementAnchors != null
                && index >= 0
                && index < placementAnchors.Length
                && placementAnchors[index] != null;
        }

        public string StationName(int index)
        {
            if (
                stationNames != null
                && index >= 0
                && index < stationNames.Length
                && !string.IsNullOrEmpty(stationNames[index])
            )
                return stationNames[index];
            return "場所" + index;
        }

        public bool IsActionAvailable(string action, int target)
        {
            if (!actionComplete)
                return false;
            if (action == "move")
                return target >= 0 && target < route.Length && target != station;
            if (action == "prepare")
                return target == 0 && station == 0 && !holdingDrink && heldItem < 0;
            if (action == "serve")
                return target == 3 && station == 3 && holdingDrink;
            return (action == "speak" || action == "wait") && target == station;
        }

        public string AvailableActions()
        {
            if (!actionComplete)
                return "行動の実行中。完了を待つ。";
            string result = "";
            for (int i = 0; i < route.Length; i++)
            {
                if (IsActionAvailable("move", i))
                    result +=
                        "move target"
                        + i
                        + ": "
                        + (holdingDrink ? "コーヒーを持って" : "空手で")
                        + StationName(i)
                        + "へ歩く。置かない。\n";
            }
            if (IsActionAvailable("prepare", 0))
                result += "prepare target0: カウンターでコーヒーを作って手に持つ。\n";
            if (IsActionAvailable("serve", 3))
                result += "serve target3: 手のコーヒーを客に渡す。渡すと手が空になる。\n";
            result += "speak target" + station + ": 現在地で客に話す。物は動かさない。\n";
            result += "wait target" + station + ": 現在地で黙って待つ。持ち物は変わらない。";
            return result;
        }

        public bool ExecuteDecision(string action, int target, string speechText)
        {
            if (!Networking.IsOwner(gameObject) || !actionComplete)
                return false;
            if (!IsActionAvailable(action, target))
                return Reject("現在実行できる候補にない行動または対象");
            if (speechText.Length > 0)
                message = speechText;
            activeAction = action;
            actionComplete = false;
            if (action == "move")
            {
                phase = 1;
                targetStation = target;
                Move(NextRouteHop(station, target));
            }
            else
            {
                phase =
                    action == "prepare" ? 3
                    : action == "serve" ? 4
                    : 2;
                Hold(
                    action == "prepare" ? 5
                        : action == "wait" ? 8
                        : 3,
                    message
                );
            }
            return true;
        }

        bool Reject(string reason)
        {
            actionResult = "失敗: " + reason;
            RequestSerialization();
            return false;
        }

        void Move(int target)
        {
            from = transform.position - Vector3.up * bob;
            destination = StationPosition(target);
            movingTo = target;
            duration = Mathf.Max(.15f, Vector3.Distance(from, destination) / walkSpeed);
            beganAt = Networking.GetServerTimeInSeconds();
            RequestSerialization();
        }

        // Carrying to the counter, the NPC walks behind the counter to the guest's stool instead of its centre.
        Vector3 StationPosition(int index)
        {
            var position = route[index].position;
            if (
                index == counterStation
                && jobSlot >= 0
                && jobItem >= 0
                && heldItem == jobItem
                && jobDestination == counterStation
            )
            {
                // Step up to the counter so the reach to the slot matches a table placement (~0.7 m).
                position.x = counterSlots[jobSlot].position.x;
                position.z = Mathf.Min(position.z, counterSlots[jobSlot].position.z + .7f);
            }
            return position;
        }

        public Transform PlacementAnchor(int place, int slot)
        {
            if (place == counterStation && slot >= 0 && counterSlots != null && slot < counterSlots.Length)
                return counterSlots[slot];
            if (placementAnchors == null || place < 0 || place >= placementAnchors.Length)
                return null;
            return placementAnchors[place];
        }

        void Hold(float seconds, string text)
        {
            from = transform.position - Vector3.up * bob;
            destination = from;
            duration = seconds;
            message = text;
            beganAt = Networking.GetServerTimeInSeconds();
            RequestSerialization();
        }

        public override void OnOwnershipTransferred(VRCPlayerApi player)
        {
            if (player.isLocal)
                RequestSerialization();
        }

        public override void Interact()
        {
            if (brain != null)
                brain.OpenConversation();
        }
    }
}
