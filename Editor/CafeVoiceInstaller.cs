#if UNITY_EDITOR
using System;
using System.Linq;
using System.Reflection;
using ShaderGPT;
using ShaderGPT.Models;
using ShaderGPT.Udon;
using UdonSharp;
using UdonSharpEditor;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace CafeNPC.EditorTools
{
    // The NPC's voice: bakes the two ShaderGPT graphs and wires them into the cafe scene.
    //   worker:  line -> MioTTS-0.4B language model -> speech tokens (CafeVoiceGenerator), synced with the line
    //   everyone: speech tokens -> MioCodec decoder -> AudioClip at the NPC (CafeVoicePlayer)
    // Order: 1 and 2 in Play Mode (baking traces a real run), then 3 outside Play Mode.
    public static class CafeVoiceInstaller
    {
        // MioTTS-0.4B shares the conversation model's text vocabulary. MioTTS-0.1B was tried and dropped (it derails on some
        // replies); it needs its own tokenizer: a folder with tokenizer-split/, EndToken 229,
        // SpeechBase 65536, PrependBos false, OwnTokenizer true.
        // Q4_0. Q8_0 sounded steadier (MioTTS-0.4B-Q8_0) but adds 0.15 GB, and the world must stay under VRChat's 1.2 GB.
        public const string LanguageModel = "Assets/CafeNPC/Models/MioTTS-0.4B-Q4_0";
        const string TokenizerSplit = LanguageModel + "/tokenizer-split/";
        const string Loader = "Assets/CafeNPC/Models/MioTTS-0.4B-Q4_0/MioTTS-0.4B-Q4_0.basiclm.prefab";

        // <|s_0|> follows the 17,013 text tokens kept by prune_text_vocabulary.py ... english (64400 with the full vocabulary).
        const int EndToken = 7,
            SpeechBase = 17013;
        const bool PrependBos = true,
            OwnTokenizer = false;
        public const string Decoder = "Assets/CafeNPC/Models/MioCodecDecoder-F1up3";

        // Lines are padded with silence up to the next of these lengths (tokens, 25 per second).
        // The longest is the limit of one line: 256 tokens are 10.2 s (replies of 40 characters reached 186).
        public static readonly int[] DecoderTokens = { 64, 128, 192, 256 };
        const int ContextLength = 512; // prompt (a line is at most 80 characters) plus 256 speech tokens

        static string LanguagePrefab => LanguageModel + "/" + System.IO.Path.GetFileName(LanguageModel) + ".prefab";

        static string DecoderPrefab(int tokens) => Decoder + "/MioCodecDecoder-" + tokens + ".prefab";

        [MenuItem("CafeNPC/Voice/1 Bake Language Model (Play Mode)", false, 1)]
        public static void BakeLanguageModel()
        {
            if (!EditorApplication.isPlaying)
                throw new InvalidOperationException("Enter Play Mode first");
            // The loader object comes from the Q4_0 folder; the weights and the config (whose folder the graph is written
            // to) are read from the folder being baked.
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(Loader);
            var go = UnityEngine.Object.Instantiate(prefab);
            var lm = go.GetComponent<BasicLM>();
            var assets = ModelAssetUtility.LoadModelAssets(LanguageModel, false);
            lm.configJson = assets.configJson;
            lm.generationConfig.max_length = ContextLength;
            lm.textures = assets.textures;
            lm.task = BasicLM.Task.Bake;
            go.SetActive(true); // OnEnable traces one step and writes <folder>/<name>.prefab
            UnityEngine.Object.Destroy(go);
        }

        [MenuItem("CafeNPC/Voice/2 Bake Decoder (Play Mode)", false, 2)]
        public static void BakeDecoder()
        {
            foreach (int tokens in DecoderTokens)
                BakeDecoder(tokens);
        }

        public static void BakeDecoder(int tokens)
        {
            if (!EditorApplication.isPlaying)
                throw new InvalidOperationException("Enter Play Mode first");
            var assets = ModelAssetUtility.LoadModelAssets(Decoder, false);
            var fn = new Functional
            {
                ctx = new TensorContext(),
                kernels = assets.shaders.ToDictionary(x => x.name.Split('/')[1], x => x),
            };
            var model = AutoModel.FromPretrained<MioCodecDecoder>(fn, assets.configJson, assets.textures);
            var ctx = fn.ctx;
            using (ctx.Tracing())
            {
                // No data is set on the token tensor, so it is exported as the graph's input texture; the first pass
                // copies from it so that the player finds it as that pass's TexX (the convention of ShaderGPT's own bakes).
                var ids = ctx.CPUTensor(tokens, 1);
                var marker = ctx.GPUTensor(1, 1);
                fn.Copy_(marker[..1, ..1], ids[..1, ..1]);
                ctx.Release(marker);
                // The frame positions are the same for every line of this length: a constant of the graph.
                int frames = tokens * 2;
                var data = new float[frames * 4];
                for (int i = 0; i < frames; i++)
                {
                    data[i * 4 + 1] = i;
                    data[i * 4 + 2] = frames;
                }
                var frameIds = ctx.CPUTensor(frames, 1);
                frameIds.SetData(data);
                var waveform = model.Decode(ids, frameIds);
                ctx.Release(waveform);
                ctx.Release(ids);
                ctx.Release(frameIds);
            }
            ctx.ExportTracing(DecoderPrefab(tokens));
            Debug.Log("[CafeVoice] baked " + DecoderPrefab(tokens));
        }

        [MenuItem("CafeNPC/Voice/3 Install Voice", false, 3)]
        public static void InstallVoice()
        {
            if (EditorApplication.isPlaying)
                throw new InvalidOperationException("Install outside Play Mode");
            foreach (var type in new[] { typeof(CafeVoiceGenerator), typeof(CafeVoicePlayer) })
            {
                var program = UdonSharpEditorUtility.GetUdonSharpProgramAsset(type);
                if (program == null || program.CompiledVersion < UdonSharpProgramVersion.CurrentVersion)
                    throw new InvalidOperationException(
                        "Compile the UdonSharp program asset of " + type.Name + " first"
                    );
            }
            var language = AssetDatabase.LoadAssetAtPath<GameObject>(LanguagePrefab);
            if (!language)
                throw new InvalidOperationException("Bake the language model first: " + LanguagePrefab);
            var graphs = DecoderTokens
                .Select(n => AssetDatabase.LoadAssetAtPath<GameObject>(DecoderPrefab(n)))
                .ToArray();
            if (graphs.Any(g => !g))
                throw new InvalidOperationException("Bake the decoder first");
            var config = JsonUtility.FromJson<MioCodecDecoderConfig>(
                AssetDatabase.LoadAssetAtPath<TextAsset>(Decoder + "/config.json").text
            );

            var brain = UnityEngine.Object.FindObjectOfType<CafeNpcBrain>();
            var actor = brain.actor;
            var worker = brain.worker;
            var setup = typeof(UdonSharpEditorUtility).GetMethod(
                "RunBehaviourSetup",
                BindingFlags.NonPublic | BindingFlags.Static,
                null,
                new[] { typeof(UdonSharpBehaviour) },
                null
            );

            // Worker side: its own generator, active only on the worker like the decider.
            var root = GameObject.Find("WorldLLM").transform;
            var old = root.Find("Voice");
            if (old)
                UnityEngine.Object.DestroyImmediate(old.gameObject);
            var service = new GameObject("Voice");
            service.transform.SetParent(root, false);
            service.SetActive(false);
            var tokenizer = brain.pipeline.tokenizer;
            if (OwnTokenizer)
            {
                // A voice model with its own text vocabulary; like the decider's, it is loaded only on the worker.
                var body = AssetDatabase.LoadAssetAtPath<TextAsset>(TokenizerSplit + "tokenizer.body.json");
                if (!body)
                    throw new InvalidOperationException("Split the voice tokenizer into " + TokenizerSplit);
                var chunks = Newtonsoft
                    .Json.Linq.JObject.Parse(body.text)["chunks"]
                    .Select(c => AssetDatabase.LoadAssetAtPath<TextAsset>(TokenizerSplit + (string)c["file"]))
                    .ToArray();
                if (chunks.Any(c => !c))
                    throw new InvalidOperationException("Missing tokenizer chunk in " + TokenizerSplit);
                var tokenizerObject = new GameObject("VoiceTokenizer");
                tokenizerObject.transform.SetParent(service.transform, false);
                tokenizer = tokenizerObject.AddComponent<GPTTokenizer>();
                tokenizer.tokenizerBodyJson = body;
                tokenizer.tokenizerChunkJsons = chunks;
                setup.Invoke(null, new object[] { tokenizer });
            }
            var generator = service.AddComponent<GPTGenerator>();
            var voice = service.AddComponent<CafeVoiceGenerator>();
            setup.Invoke(null, new object[] { generator });
            setup.Invoke(null, new object[] { voice });
            generator.modelPrefab = language;
            generator.maxLength = ContextLength;
            generator.temperature = .8f; // MioTTS's default sampling
            generator.passBudgetPerFrame = 1000;
            generator.maxInFlight = 8;
            generator.enabled = false;
            voice.tokenizer = tokenizer;
            voice.endToken = EndToken;
            voice.speechBase = SpeechBase;
            voice.prependBos = PrependBos;
            voice.generator = UdonSharpEditorUtility.GetBackingUdonBehaviour(generator);
            voice.maxTokens = DecoderTokens[DecoderTokens.Length - 1];
            worker.workerOnlyObjects = worker
                .workerOnlyObjects.Where(o => o && o.name != "Voice")
                .Append(service)
                .ToArray();

            // Everyone: the decoder and the speaker at the NPC's head.
            var mouth = actor.transform.Find("Voice");
            if (mouth)
                UnityEngine.Object.DestroyImmediate(mouth.gameObject);
            mouth = new GameObject("Voice").transform;
            mouth.SetParent(actor.transform, false);
            mouth.localPosition = new Vector3(0, 1.2f, .1f);
            var audio = mouth.gameObject.AddComponent<AudioSource>();
            audio.playOnAwake = false;
            audio.spatialBlend = 1;
            audio.volume = 1;
            audio.rolloffMode = AudioRolloffMode.Logarithmic;
            audio.minDistance = 1.5f;
            audio.maxDistance = 14;
            var player = mouth.gameObject.AddComponent<CafeVoicePlayer>();
            setup.Invoke(null, new object[] { player });
            player.graphs = graphs;
            player.graphTokens = DecoderTokens;
            player.output = audio;
            player.padToken = config.pad_token;
            player.hop = config.hop;
            player.sampleRate = config.sampling_rate;
            player.leadingSamples = (config.n_fft - config.hop) / 2;
            player.framesPerToken = 2 * config.upsample_factors.Aggregate(1, (a, b) => a * b);

            brain.voiceGenerator = voice;
            actor.voice = player;
            foreach (
                var behaviour in new UdonSharpBehaviour[] { tokenizer, generator, voice, player, brain, actor, worker }
            )
                UdonSharpEditorUtility.CopyProxyToUdon(behaviour);
            // As with the decider: an enabled generator steps with no input when the worker activates it.
            UdonSharpEditorUtility.GetBackingUdonBehaviour(generator).enabled = false;
            EditorSceneManager.MarkSceneDirty(brain.gameObject.scene);
            EditorSceneManager.SaveOpenScenes();
        }
    }
}
#endif
