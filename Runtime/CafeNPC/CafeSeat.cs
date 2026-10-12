using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon.Common;

namespace CafeNPC
{
    // A sittable chair or stool. Station enter/exit events fire on every client, so each client's brain
    // keeps the same seat table without extra network sync; the worker reads it for the guest's location.
    //
    // The seated guest can raise or lower where they sit, to fit their avatar: Page Up / Page Down on a keyboard,
    // the right stick up / down in VR. The height is synced so that everyone sees the guest at the same place, and
    // goes back to the seat's own height when the guest stands up.
    [UdonBehaviourSyncMode(BehaviourSyncMode.Manual)]
    public class CafeSeat : UdonSharpBehaviour
    {
        public CafeNpcBrain brain;
        public int seatIndex;
        public float lowest = -.25f,
            highest = .4f,
            metresPerSecond = .25f;

        [UdonSynced]
        public float lift; // metres above the seat point
        private Transform seatPoint;
        private Vector3 basePosition;
        private bool ready,
            localSeated,
            changed;
        private float lookVertical,
            sentAt;

        void Prepare()
        {
            if (ready)
                return;
            ready = true;
            seatPoint = (
                (VRC.SDK3.Components.VRCStation)GetComponent(typeof(VRC.SDK3.Components.VRCStation))
            ).stationEnterPlayerLocation;
            basePosition = seatPoint.position;
        }

        void Apply()
        {
            Prepare();
            seatPoint.position = basePosition + Vector3.up * lift;
        }

        public override void Interact()
        {
            Networking.LocalPlayer.UseAttachedStation();
        }

        public override void OnStationEntered(VRCPlayerApi player)
        {
            if (!Utilities.IsValid(player))
                return;
            brain.SeatEntered(seatIndex, player.playerId);
            if (!player.isLocal)
                return;
            Networking.SetOwner(player, gameObject);
            localSeated = true;
            lookVertical = 0;
            lift = 0;
            Apply();
            RequestSerialization();
        }

        public override void OnStationExited(VRCPlayerApi player)
        {
            if (!Utilities.IsValid(player))
                return;
            brain.SeatExited(seatIndex, player.playerId);
            if (!player.isLocal)
                return;
            localSeated = false;
            if (Networking.IsOwner(gameObject))
            {
                lift = 0;
                Apply();
                RequestSerialization();
            }
        }

        public override void OnDeserialization()
        {
            Apply();
        }

        // The right stick in VR. On desktop this is the mouse, which must not move the seat.
        public override void InputLookVertical(float value, UdonInputEventArgs args)
        {
            lookVertical = value;
        }

        void Update()
        {
            if (!localSeated)
                return;
            float direction =
                Input.GetKey(KeyCode.PageUp) ? 1
                : Input.GetKey(KeyCode.PageDown) ? -1
                : 0;
            if (direction == 0 && Networking.LocalPlayer.IsUserInVR() && Mathf.Abs(lookVertical) > .6f)
                direction = Mathf.Sign(lookVertical);
            if (direction != 0)
            {
                lift = Mathf.Clamp(lift + direction * metresPerSecond * Time.deltaTime, lowest, highest);
                Apply();
                changed = true;
            }
            if (changed && Time.time - sentAt > .2f)
            {
                changed = false;
                sentAt = Time.time;
                RequestSerialization();
            }
        }
    }
}
