using ShaderGPT.Udon;
using UdonSharp;
using UnityEngine;
using VRC.Udon;

namespace CafeNPC
{
    // Worker only. Turns one line of text into speech tokens with a MioTTS language model (a baked ShaderGPT graph):
    // the prompt is the line as a user turn, the reply is a run of <|s_N|> tokens, 25 per second of speech.
    // The tokens are what gets synced; every client decodes them to audio itself (CafeVoicePlayer).
    [UdonBehaviourSyncMode(BehaviourSyncMode.None)]
    public class CafeVoiceGenerator : UdonSharpBehaviour
    {
        public GPTTokenizer tokenizer; // the conversation model's: MioTTS-0.4B shares its text vocabulary
        public UdonBehaviour generator;
        public int maxTokens = 256; // 10.2 s; the player's largest graph (the installer sets it)
        public float timeoutSeconds = 8;

        public const int PhaseIdle = 0,
            PhaseRunning = 1,
            PhaseDone = 2,
            PhaseFailed = -1;

        [System.NonSerialized]
        public int phase;

        [System.NonSerialized]
        public ushort[] speech; // token numbers N of <|s_N|>, valid in PhaseDone

        [System.NonSerialized]
        public float seconds;

        // Token numbers of the voice model. MioTTS-0.4B: ends with 7, <|s_0|> is 64400, takes a BOS (0.1B: 229, 65536, none).
        public int endToken = 7,
            speechBase = 64400,
            speechTokens = 12800;
        public bool prependBos = true;
        private int[] prompt;
        private ushort[] buffer;
        private int count,
            received;
        private float startedAt;

        public bool IsReady()
        {
            return tokenizer != null && tokenizer.isReady && generator != null;
        }

        public bool Speak(string text)
        {
            if (!IsReady())
                return false;
            Cancel();
            // The NPC's name is written 茜犬 but said "Akane": the model reads 茜 that way.
            text = text.Replace("茜犬", "茜");
            // The line goes in as written. MioTTS's own normalisation (half-width ! and ?, no closing 。) was tried in the
            // world and the voice was judged better without it.
            var body = tokenizer.Encode("<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n");
            int lead = prependBos ? 1 : 0;
            prompt = new int[body.Length + lead];
            if (prependBos)
                prompt[0] = tokenizer.bos_token_id;
            System.Array.Copy(body, 0, prompt, lead, body.Length);
            if (buffer == null || buffer.Length != maxTokens)
                buffer = new ushort[maxTokens];
            count = 0;
            received = 0;
            generator.SendCustomEvent("Stop");
            generator.SetProgramVariable("inputTokens", prompt);
            generator.SetProgramVariable("inputIndex", 0);
            generator.SetProgramVariable("eventTarget", (UdonBehaviour)(Component)this);
            generator.SetProgramVariable("eventMethod", nameof(OnOutput));
            phase = PhaseRunning;
            startedAt = Time.realtimeSinceStartup;
            generator.SendCustomEvent("Play");
            return true;
        }

        public void Cancel()
        {
            if (phase != PhaseRunning)
                return;
            phase = PhaseIdle;
            generator.SendCustomEvent("Stop");
        }

        public void OnOutput()
        {
            if (phase != PhaseRunning)
                return;
            int state = (int)generator.GetProgramVariable("state");
            if (state == GPTGenerator.StateError || state == GPTGenerator.StateFatalError)
            {
                Finish(PhaseFailed);
                return;
            }
            int index = (int)generator.GetProgramVariable("outputIndex");
            if (index < prompt.Length + received)
                return; // a prompt position, or a token already taken
            int token = (int)generator.GetProgramVariable("outputToken");
            received++;
            if (token >= speechBase && token < speechBase + speechTokens)
                buffer[count++] = (ushort)(token - speechBase);
            if (token == endToken || count >= maxTokens || received >= maxTokens + 8)
                Finish(count > 0 ? PhaseDone : PhaseFailed);
        }

        void Finish(int result)
        {
            seconds = Time.realtimeSinceStartup - startedAt;
            speech = new ushort[count];
            System.Array.Copy(buffer, speech, count);
            phase = result;
            // Stopping inside the generator's own callback would re-enter it; stop on the next frame.
            SendCustomEventDelayedFrames(nameof(StopGenerator), 1);
        }

        public void StopGenerator()
        {
            if (phase != PhaseRunning)
                generator.SendCustomEvent("Stop");
        }

        void Update()
        {
            if (phase == PhaseRunning && Time.realtimeSinceStartup - startedAt > timeoutSeconds)
            {
                generator.SendCustomEvent("Stop");
                count = 0;
                Finish(PhaseFailed);
            }
        }
    }
}
