using UdonSharp;
using UnityEngine;
using VRC.SDKBase;

namespace CafeNPC
{
    // A served item a guest can pick up, eat or drink. Use (trigger / click) takes a bite or a sip; in VR, holding it
    // at the mouth does the same. The NPC places the item while `free` is false; once a guest picks it up the item
    // is theirs (VRCObjectSync carries it) until the NPC takes it back for another delivery.
    // Continuous, not Manual: VRCObjectSync on the same object rejects manually synced behaviours. The two fields are tiny.
    [UdonBehaviourSyncMode(BehaviourSyncMode.Continuous)]
    public class CafeEdible : UdonSharpBehaviour
    {
        public Transform portion; // food model that shrinks, or liquid column that drains; null for the menu
        public bool liquid; // drains from the top with its bottom fixed, instead of shrinking
        public int maxBites = 4;

        [UdonSynced]
        public int bites;

        [UdonSynced]
        public bool free;
        private VRC_Pickup pickup;
        private Rigidbody body;
        private Vector3 fullScale,
            fullPosition;
        private int shownBites = -1;
        private bool shownFree;
        private float atMouthSince = -1;

        void Start()
        {
            pickup = (VRC_Pickup)GetComponent(typeof(VRC_Pickup));
            body = GetComponent<Rigidbody>();
            if (portion != null)
            {
                fullScale = portion.localScale;
                fullPosition = portion.localPosition;
            }
            Apply();
        }

        public bool HeldByPlayer()
        {
            return pickup != null && pickup.IsHeld;
        }

        public bool Consumed()
        {
            return bites > 0;
        }

        public override void OnPickup()
        {
            Networking.SetOwner(Networking.LocalPlayer, gameObject);
            if (!free)
            {
                free = true;
                Apply();
            }
        }

        public override void OnPickupUseDown()
        {
            Bite();
        }

        void Update()
        {
            if (shownBites != bites || shownFree != free)
                Apply();
            if (pickup == null || !pickup.IsHeld || portion == null)
            {
                atMouthSince = -1;
                return;
            }
            var holder = pickup.currentPlayer;
            if (!Utilities.IsValid(holder) || !holder.isLocal || !holder.IsUserInVR())
            {
                atMouthSince = -1;
                return;
            }
            var head = holder.GetTrackingData(VRCPlayerApi.TrackingDataType.Head).position;
            if (Vector3.Distance(head, transform.position) > .2f)
            {
                atMouthSince = -1;
                return;
            }
            if (atMouthSince < 0)
                atMouthSince = Time.time;
            else if (Time.time - atMouthSince > .8f)
            {
                Bite();
                atMouthSince = Time.time;
            }
        }

        public void Bite()
        {
            if (portion == null || bites >= maxBites)
                return;
            Networking.SetOwner(Networking.LocalPlayer, gameObject);
            bites++;
            Apply();
        }

        // A fresh one from the kitchen: the NPC's owner calls this before carrying the item again.
        public void Restock()
        {
            Networking.SetOwner(Networking.LocalPlayer, gameObject);
            if (pickup != null)
                pickup.Drop();
            bites = 0;
            free = false;
            Apply();
        }

        // Guests cannot take an item out of the NPC's hand mid-delivery.
        public void SetCarried(bool carried)
        {
            if (pickup != null && pickup.pickupable == carried)
                pickup.pickupable = !carried;
        }

        public void Reclaim()
        {
            if (!free)
                return;
            Networking.SetOwner(Networking.LocalPlayer, gameObject);
            if (pickup != null)
                pickup.Drop();
            free = false;
            Apply();
        }

        public override void OnDeserialization()
        {
            Apply();
        }

        void Apply()
        {
            shownBites = bites;
            shownFree = free;
            // Placed by the NPC: kinematic. Free: falls onto the table when a guest lets go.
            if (body != null)
            {
                body.isKinematic = !free;
                body.useGravity = free;
            }
            if (portion == null)
                return;
            float left = maxBites > 0 ? 1f - (float)bites / maxBites : 1;
            portion.gameObject.SetActive(left > .001f);
            if (left <= .001f)
                return;
            if (liquid)
            {
                // A primitive cylinder is 2 units tall around its centre: keep the bottom, lower the top.
                float bottom = fullPosition.y - fullScale.y;
                portion.localScale = new Vector3(fullScale.x, fullScale.y * left, fullScale.z);
                portion.localPosition = new Vector3(fullPosition.x, bottom + fullScale.y * left, fullPosition.z);
            }
            else
            {
                portion.localScale = fullScale * Mathf.Lerp(.4f, 1f, left);
            }
        }
    }
}
