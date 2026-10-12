using ShaderGPT.Udon;
using UdonSharp;
using UnityEngine;
using VRC.Udon;

namespace CafeNPC
{
    // Asks the 2B decider one multiple-choice question in its native "plain, state-first" layout
    // (decider/prompt.py in the model's repository) and reports the restricted argmax option. The baked graph
    // compares the letter logits A..J at the final "Answer: (" token, which is the native decision
    // for both choice and yes/no questions because temperature softmax preserves the argmax.
    [UdonBehaviourSyncMode(BehaviourSyncMode.None)]
    public class CafeDecider : UdonSharpBehaviour
    {
        public GPTTokenizer tokenizer;
        public UdonBehaviour generator;
        public UdonSharpBehaviour listener;
        public string listenerEvent = "OnDeciderAnswer";
        public int maxTokens = 500; // must stay below the baked decoder cache length (installer sets it)
        public float timeoutSeconds = 60;

        // Only the embedding rows the cafe needs are in the world (prune_vocabulary.py). The tokenizer still numbers
        // tokens as the original model does: vocabularyMap[id] is the row, or -(n+1) for a token that was dropped and is
        // written as the pieces at vocabularySplits[n] (a count, then that many rows). Empty: the full table is there.
        public int[] vocabularyMap,
            vocabularySplits;

        public const int PhaseIdle = 0,
            PhaseRunning = 1,
            PhaseDone = 2,
            PhaseFailed = -1;

        [System.NonSerialized]
        public int phase;

        [System.NonSerialized]
        public int answer = -1;

        [System.NonSerialized]
        public int tokenCount;

        [System.NonSerialized]
        public float seconds;

        [System.NonSerialized]
        public string error = "";
        private int[] tokens;
        private int optionCount;
        private float startedAt;
        private const string Letters = "ABCDEFGHIJ";

        // Decider tokenizer ids: "\n(" encodes to [198, 7]; single-letter labels A..Z are 32..57 (label_table in prompt.py).
        private const int NewlineId = 198,
            OpenParenId = 7,
            FirstLabelId = 32,
            MaxWideOptions = 26;

        public bool IsReady()
        {
            return tokenizer != null && tokenizer.isReady && generator != null;
        }

        // context: the state text (JSON). Options are "name: description" strings in the native order.
        public bool Ask(string context, string question, string[] options)
        {
            if (phase == PhaseRunning)
                return false;
            answer = -1;
            error = "";
            if (!IsReady())
                return Fail("decider tokenizer is not ready");
            if (options == null || options.Length < 1 || options.Length > MaxWideOptions)
                return Fail("decider supports 1-26 options");
            // Context and question are encoded separately, exactly like the native prompt builder, so no BPE merge crosses them.
            var head = tokenizer.Encode("Context:\n" + context);
            int[] tail;
            if (options.Length <= Letters.Length)
            {
                string piece = "\n\nQuestion: " + question + "\nOptions:";
                for (int i = 0; i < options.Length; i++)
                    piece += "\n(" + Letters[i] + ") " + options[i];
                tail = tokenizer.Encode(piece + "\nAnswer: (");
            }
            else
            {
                // Native "wide" rendering (prompt.py): per option the ids of "\n(", one label token, then ") text".
                tail = tokenizer.Encode("\n\nQuestion: " + question + "\nOptions:");
                for (int i = 0; i < options.Length; i++)
                {
                    var text = tokenizer.Encode(") " + options[i]);
                    var row = new int[3 + text.Length];
                    row[0] = NewlineId;
                    row[1] = OpenParenId;
                    row[2] = FirstLabelId + i;
                    System.Array.Copy(text, 0, row, 3, text.Length);
                    tail = Concat(tail, row);
                }
                tail = Concat(tail, tokenizer.Encode("\nAnswer: ("));
            }
            tokens = Rows(Concat(head, tail));
            tokenCount = tokens.Length;
            if (tokenCount >= maxTokens)
                return Fail("decider prompt too long: " + tokenCount);
            optionCount = options.Length;
            generator.SendCustomEvent("Stop");
            generator.SetProgramVariable("inputTokens", tokens);
            generator.SetProgramVariable("inputIndex", 0);
            generator.SetProgramVariable("decoderAuxiliary", optionCount);
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
                Fail("decider generator readback failed");
                SendCustomEventDelayedFrames(nameof(StopGenerator), 1);
                Notify();
                return;
            }
            if ((int)generator.GetProgramVariable("outputIndex") != tokens.Length)
                return;
            int choice = (int)generator.GetProgramVariable("outputToken");
            seconds = Time.realtimeSinceStartup - startedAt;
            if (choice < 0 || choice >= optionCount)
                Fail("decider choice outside options: " + choice);
            else
            {
                answer = choice;
                phase = PhaseDone;
            }
            // Stopping inside the generator's own callback would re-enter it; stop on the next frame.
            SendCustomEventDelayedFrames(nameof(StopGenerator), 1);
            Notify();
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
                Fail("decider timed out");
                Notify();
            }
        }

        int[] Rows(int[] ids)
        {
            if (vocabularyMap == null || vocabularyMap.Length == 0)
                return ids;
            int length = 0;
            for (int i = 0; i < ids.Length; i++)
            {
                int row = vocabularyMap[ids[i]];
                length += row >= 0 ? 1 : vocabularySplits[-row - 1];
            }
            var rows = new int[length];
            int at = 0;
            for (int i = 0; i < ids.Length; i++)
            {
                int row = vocabularyMap[ids[i]];
                if (row >= 0)
                {
                    rows[at++] = row;
                    continue;
                }
                int start = -row - 1,
                    count = vocabularySplits[start];
                System.Array.Copy(vocabularySplits, start + 1, rows, at, count);
                at += count;
            }
            return rows;
        }

        static int[] Concat(int[] a, int[] b)
        {
            var result = new int[a.Length + b.Length];
            System.Array.Copy(a, result, a.Length);
            System.Array.Copy(b, 0, result, a.Length, b.Length);
            return result;
        }

        bool Fail(string reason)
        {
            phase = PhaseFailed;
            answer = -1;
            error = reason;
            Debug.LogWarning("[CafeDecider] " + reason);
            return false;
        }

        void Notify()
        {
            if (listener != null)
                listener.SendCustomEvent(listenerEvent);
        }

        // Python json.dumps(..., ensure_ascii=False) string literal, the format the decider was evaluated with.
        public static string Json(string value)
        {
            if (value == null)
                return "null";
            string result = "\"";
            for (int i = 0; i < value.Length; i++)
            {
                char c = value[i];
                if (c == '"')
                    result += "\\\"";
                else if (c == '\\')
                    result += "\\\\";
                else if (c == '\n')
                    result += "\\n";
                else if (c == '\r')
                    result += "\\r";
                else if (c == '\t')
                    result += "\\t";
                else if (c == '\b')
                    result += "\\b";
                else if (c == '\f')
                    result += "\\f";
                else if (c < ' ')
                    result += "\\u" + ((int)c).ToString("x4");
                else
                    result += c;
            }
            return result + "\"";
        }
    }
}
