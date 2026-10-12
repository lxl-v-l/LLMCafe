using UdonSharp;
using UnityEngine;
using UnityEngine.UI;

namespace CafeNPC
{
    // Two sliders on the wall: how loud the NPC's voice and the background music are, for the guest who moves them
    // only. Nothing is synced; everyone sets their own.
    [UdonBehaviourSyncMode(BehaviourSyncMode.None)]
    public class CafeVolume : UdonSharpBehaviour
    {
        public AudioSource voice;
        public AudioSource music;
        public Slider voiceSlider;
        public Slider musicSlider;
        public Text voiceLabel;
        public Text musicLabel;

        // The music file is mastered loud: at a volume of 0.01 it already sits under the NPC's voice. The slider is
        // therefore squared and scaled, so that its middle is that level and its top is only four times as loud.
        public float musicTop = .04f;

        void Start()
        {
            // The sliders start where the scene has the two sources.
            if (voice != null && voiceSlider != null)
                voiceSlider.SetValueWithoutNotify(voice.volume);
            if (music != null && musicSlider != null)
                musicSlider.SetValueWithoutNotify(Mathf.Sqrt(Mathf.Clamp01(music.volume / musicTop)));
            Show();
        }

        // Called by both sliders when they move.
        public void Changed()
        {
            if (voice != null && voiceSlider != null)
                voice.volume = voiceSlider.value;
            if (music != null && musicSlider != null)
                music.volume = musicSlider.value * musicSlider.value * musicTop;
            Show();
        }

        void Show()
        {
            if (voiceLabel != null && voiceSlider != null)
                voiceLabel.text = "店員の声  " + Mathf.RoundToInt(voiceSlider.value * 100);
            if (musicLabel != null && musicSlider != null)
                musicLabel.text = "BGM  " + Mathf.RoundToInt(musicSlider.value * 100);
        }
    }
}
