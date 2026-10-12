#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using ShaderGPT;
using ShaderGPT.Models;
using UnityEditor;
using UnityEngine;
using Cache = ShaderGPT.Cache;

namespace CafeNPC.EditorTests
{
    // The decider as the world runs it: the prompt goes in a token at a time, and only the answer letters are scored.
    public sealed class RestrictedDecider : Qwen3_5Text
    {
        public RestrictedDecider(Functional fn, Qwen3_5TextConfig config)
            : base(fn, config) { }

        public Tensor Feed(TensorView input, Cache cache)
        {
            FixSize0("model.embed_tokens.weight.T", config.hidden_size);
            return Qwen3_5TextModel("model", input, cache);
        }

        public Tensor LetterLogits(TensorView hidden, TensorView letters)
        {
            var path =
                state_dict.ContainsKey("lm_head.weight") || state_dict.ContainsKey("lm_head.weight.T")
                    ? "lm_head"
                    : "model.embed_tokens";
            FixSize0(path + ".weight.T", config.hidden_size);
            var rows = nn.Embedding(path, (letters, 0));
            var scores = fn.Linear(hidden[^1.., ..], rows);
            ctx.Release(rows);
            return scores;
        }
    }

    public static class CafeDeciderBaker
    {
        public const string Folder = "Assets/CafeNPC/Models/Decider2B-Q4"; // the first model tried; the parity harness still uses it
        public const string PrefabPath = Folder + "/Decider2B-Q4.choices.prefab";

        // The model the world uses, and the cache length its service needs.
        public const string WorldFolder = "Assets/CafeNPC/Models/Decider08B-Q4";
        public const int WorldContext = 1024;

        // The label table (the token of each answer letter, decider-label-table.json) is read from the model folder, and
        // the bake report is written there.
        public const string LabelTable = "decider-label-table.json";
        const int MaxOptions = 255;

        [Serializable]
        public class Labels
        {
            public int max_options;
            public int[] letter_ids;
        }

        [Serializable]
        public class Vocabulary
        {
            public string level;
            public int size;
            public int[] map,
                splits;
        }

        // vocabulary.json next to a model exported with only part of its embedding table; null when the table is whole.
        public static Vocabulary LoadVocabulary(string folder)
        {
            string path = folder + "/vocabulary.json";
            return File.Exists(path) ? JsonUtility.FromJson<Vocabulary>(File.ReadAllText(path)) : null;
        }

        [Serializable]
        public class BakeReport
        {
            public string scope,
                prefab;
            public int options,
                materials,
                render_textures,
                remaining_tensors;
            public int[] mask_counts,
                mask_choices;
        }

        public static BakeReport Bake() => Bake(Folder, 512);

        [MenuItem("CafeNPC/Decider/0 Bake Decider (Play Mode)", false, 0)]
        public static void BakeWorldDecider()
        {
            var report = Bake(WorldFolder, WorldContext);
            Debug.Log("[CafeDecider] baked " + report.prefab + ": " + report.materials + " passes");
        }

        // folder: an exported decider model folder; the graph is written next to it as <folder name>.choices.prefab.
        // contextLength: decoder cache positions; the service needs ~520 tokens for two items x five destinations.
        public static BakeReport Bake(string folder, int contextLength)
        {
            string prefabPath = folder + "/" + Path.GetFileName(folder) + ".choices.prefab";
            if (!Application.isPlaying || EditorApplication.isCompiling || EditorApplication.isUpdating)
                throw new InvalidOperationException("Shader tracing requires settled Play Mode");
            if (!File.Exists(folder + "/" + LabelTable))
                throw new FileNotFoundException("Put " + LabelTable + " into " + folder);
            var labels = JsonUtility.FromJson<Labels>(File.ReadAllText(folder + "/" + LabelTable));
            if (labels.max_options != MaxOptions || labels.letter_ids.Length != MaxOptions)
                throw new InvalidDataException("Unexpected decider label table");
            var vocabulary = LoadVocabulary(folder);
            if (vocabulary != null)
            {
                // The label tokens are numbered by the original vocabulary; the graph reads the rows that are stored.
                for (int i = 0; i < labels.letter_ids.Length; i++)
                {
                    int row = vocabulary.map[labels.letter_ids[i]];
                    if (row < 0)
                        throw new InvalidDataException(
                            "A label token was dropped from the vocabulary: " + labels.letter_ids[i]
                        );
                    labels.letter_ids[i] = row;
                }
            }
            var assets = ModelAssetUtility.LoadModelAssets(folder, prepare: false);
            var config = (Qwen3_5TextConfig)PretrainedConfig.FromPretrained<Qwen3_5TextConfig>(assets.configJson);
            var fn = new Functional
            {
                ctx = new TensorContext(),
                kernels = assets.shaders.ToDictionary(s => s.name.Split('/')[1], s => s),
            };
            var model = new RestrictedDecider(fn, config);
            model.LoadStateDict(assets.textures);
            model.generation_config = new GenerationConfig { max_length = contextLength, repetition_penalty = 1 };
            var cache = new Cache(fn, contextLength);
            try
            {
                var counts = new[] { 1, 2, 6, 255 };
                var choices = CheckMask(fn, counts);
                using (fn.ctx.Tracing())
                {
                    fn.ctx.defaultValue = 0;
                    var input = fn.ctx.GPUTensor(1, 1);
                    fn.ctx.defaultValue = null;
                    var hidden = model.Feed(input, cache);
                    var letters = fn.ctx.CPUTensor(256, 1);
                    var data = new float[256 * 4];
                    for (int i = 0; i < MaxOptions; i++)
                        data[4 * i] = labels.letter_ids[i];
                    letters.SetData(data);
                    var scores = model.LetterLogits(hidden, letters);
                    fn.ctx.Release(hidden);
                    fn.ctx.Release(letters);
                    var choice = fn.Argmax(scores, window: ChoiceWindow(input));
                    fn.ctx.Release(scores);
                    fn.Copy_(input, choice);
                    fn.ctx.Release(choice);
                    fn.ctx.Release(input);
                }
                var prefab = (GameObject)fn.ctx.ExportTracing(prefabPath);
                var renderers = prefab.GetComponentsInChildren<MeshRenderer>();
                if (renderers.Length != 1)
                    throw new InvalidDataException("Expected a single decoder graph");
                var materials = renderers[0].sharedMaterials;
                if (materials.Any(m => m == null || m.GetTexture("TexO") == null))
                    throw new InvalidDataException("Baked output texture missing");
                cache.Reset();
                fn.ctx.ReleaseAll();
                var report = new BakeReport
                {
                    scope =
                        "Actual 255-label restricted scorer, dynamic candidate mask, traced in Play Mode. Not NPC, Udon execution, multiplayer or performance proof.",
                    prefab = prefabPath,
                    options = MaxOptions,
                    materials = materials.Length,
                    render_textures = AssetDatabase.LoadAllAssetsAtPath(prefabPath).OfType<RenderTexture>().Count(),
                    remaining_tensors = fn.ctx.TensorCount(),
                    mask_counts = counts,
                    mask_choices = choices,
                };
                File.WriteAllText(folder + "/decider-bake.json", JsonUtility.ToJson(report, true));
                return report;
            }
            finally
            {
                RenderTexture.active = null;
                cache.Reset();
                fn.ctx.ReleaseAll();
            }
        }

        static Window ChoiceWindow(TensorView input) => new(float.NegativeInfinity, 0, 1, (input, 2), start2: 0);

        static int[] CheckMask(Functional fn, int[] counts)
        {
            var observed = new int[counts.Length];
            var input = fn.ctx.CPUTensor(1, 1);
            var scores = fn.ctx.CPUTensor(1, 64);
            var values = Enumerable.Range(0, 256).Select(i => (float)i).ToArray();
            scores.SetData(values);
            for (int i = 0; i < counts.Length; i++)
            {
                fn.ctx.Release(input);
                input = fn.ctx.CPUTensor(1, 1);
                input.SetData(new float[] { 0, 0, counts[i], 0 });
                var result = fn.Argmax(scores, window: ChoiceWindow(input));
                observed[i] = Mathf.RoundToInt(result.GetData<float>()[0]);
                fn.ctx.Release(result);
                if (observed[i] != counts[i] - 1)
                    throw new InvalidDataException("Candidate mask failed");
            }
            fn.ctx.Release(input);
            fn.ctx.Release(scores);
            return observed;
        }
    }
}
#endif
