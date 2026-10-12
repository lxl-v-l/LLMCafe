using UdonSharp;
using UnityEngine;
using UnityEngine.UI;

namespace CafeNPC
{
    // A wall switch that turns the chairs' "Sit" prompt off and on, for the guest who presses it only: with it off the
    // seats cannot be aimed at, which helps when walking around or picking things up from a table. A guest who is
    // already seated stays seated.
    [UdonBehaviourSyncMode(BehaviourSyncMode.None)]
    public class CafeSeatSwitch : UdonSharpBehaviour
    {
        public Collider[] seats;
        public Text label;
        public Image plate;
        public Color onColor = new Color(.3f, .55f, .35f),
            offColor = new Color(.5f, .32f, .28f);
        private bool off;

        void Start()
        {
            Show();
        }

        public override void Interact()
        {
            off = !off;
            for (int i = 0; i < seats.Length; i++)
                if (seats[i] != null)
                    seats[i].enabled = !off;
            Show();
        }

        void Show()
        {
            if (label != null)
                label.text = off ? "sit判定 OFF" : "sit判定 ON";
            if (plate != null)
                plate.color = off ? offColor : onColor;
            InteractionText = ""; // the plate says it all: no hover text
        }
    }
}
