using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon.Common.Interfaces;

namespace CafeNPC
{
    // A service bell on a table. Pressing it chimes for everyone and asks the NPC to come to that table.
    // NoVariableSync, not None: a behaviour with sync mode None cannot receive the network event that rings it for everyone.
    [UdonBehaviourSyncMode(BehaviourSyncMode.NoVariableSync)]
    public class CafeBell : UdonSharpBehaviour
    {
        public CafeNpcBrain brain;
        public int station; // route index of the table this bell calls to
        public AudioSource chime;
        public Transform plunger;
        private Vector3 plungerRest;
        private float rungAt = -10;

        void Start()
        {
            if (plunger != null)
                plungerRest = plunger.localPosition;
        }

        public override void Interact()
        {
            SendCustomNetworkEvent(NetworkEventTarget.All, nameof(Ring));
            brain.CallFromBell(station);
        }

        public void Ring()
        {
            rungAt = Time.time;
            if (chime != null)
                chime.Play();
        }

        void Update()
        {
            if (plunger == null)
                return;
            // The plunger dips and springs back over a fifth of a second.
            float t = Mathf.Clamp01((Time.time - rungAt) / .2f);
            plunger.localPosition = plungerRest - Vector3.up * .006f * Mathf.Sin(t * Mathf.PI);
        }
    }
}
