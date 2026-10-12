using UdonSharp;
using UnityEngine;
using VRC.SDK3.Rendering;
using VRC.SDKBase;
using VRC.Udon.Common.Interfaces;

namespace CafeNPC
{
    // Every client. Turns the synced speech tokens into audio with the MioCodec decoder (baked ShaderGPT graphs) and
    // plays it at the NPC. The graphs have fixed lengths: the tokens are padded with a token that decodes to silence up
    // to the smallest graph that holds them, and only the spoken part is played. The largest graph is the longest line.
    [UdonBehaviourSyncMode(BehaviourSyncMode.None)]
    public class CafeVoicePlayer : UdonSharpBehaviour
    {
        public GameObject[] graphs; // baked decoder prefabs, shortest first
        public int[] graphTokens; // tokens each graph decodes
        public AudioSource output;
        public int padToken = 4021;
        public int hop = 98,
            leadingSamples = 147,
            framesPerToken = 18,
            sampleRate = 44100;
        public int passesPerFrame = 48;

        [System.NonSerialized]
        public int playedCount; // clips started (the scenario test reads it)

        [System.NonSerialized]
        public float lastSeconds; // length of the last clip

        [System.NonSerialized]
        public float lastDecodeSeconds;
        private Material[][] passes;
        private Texture2D[] inputs;
        private RenderTexture[] outputs;
        private int running = -1,
            cursor,
            tokenCount;
        private bool waiting;
        private float startedAt;
        private byte[] raw;
        private float[] pixels,
            readback;
        private AudioClip clip; // one clip as long as the largest graph, rewritten for every line
        private ushort[] queued;

        void Start()
        {
            passes = new Material[graphs.Length][];
            inputs = new Texture2D[graphs.Length];
            outputs = new RenderTexture[graphs.Length];
            for (int i = 0; i < graphs.Length; i++)
            {
                if (graphs[i] == null)
                    continue;
                var renderers = graphs[i].GetComponentsInChildren<MeshRenderer>(true);
                if (renderers.Length == 0)
                    continue;
                var materials = renderers[0].sharedMaterials;
                passes[i] = materials;
                // The bake copies from the token texture first, so the first pass names it; the last pass writes the waveform.
                var first = materials[0].GetTexture("TexX");
                if (first != null && first.GetType() == typeof(Texture2D))
                    inputs[i] = (Texture2D)first;
                outputs[i] = (RenderTexture)materials[materials.Length - 1].GetTexture("TexO");
            }
        }

        public bool IsBusy()
        {
            return running >= 0;
        }

        // A line that arrives while another is being decoded waits for it; a third replaces the one waiting.
        public void Play(ushort[] tokens)
        {
            if (tokens == null || tokens.Length == 0 || passes == null)
                return;
            if (running >= 0)
            {
                queued = tokens;
                return;
            }
            int graph = -1;
            for (int i = 0; i < graphTokens.Length && graph < 0; i++)
                if (graphTokens[i] >= tokens.Length && passes[i] != null && inputs[i] != null)
                    graph = i;
            if (graph < 0)
                return;
            int length = graphTokens[graph];
            if (pixels == null || pixels.Length != length * 4)
                pixels = new float[length * 4];
            for (int i = 0; i < length; i++)
            {
                pixels[i * 4] = i < tokens.Length ? tokens[i] : padToken;
                pixels[i * 4 + 1] = i;
                pixels[i * 4 + 2] = length;
                pixels[i * 4 + 3] = 0;
            }
            if (raw == null || raw.Length != pixels.Length * 4)
                raw = new byte[pixels.Length * 4];
            System.Buffer.BlockCopy(pixels, 0, raw, 0, raw.Length);
            inputs[graph].LoadRawTextureData(raw);
            inputs[graph].Apply(false, false);
            running = graph;
            cursor = 0;
            waiting = false;
            tokenCount = tokens.Length;
            startedAt = Time.realtimeSinceStartup;
        }

        void Update()
        {
            if (running < 0 || waiting)
                return;
            var materials = passes[running];
            for (int n = 0; n < passesPerFrame && cursor < materials.Length; n++)
            {
                var material = materials[cursor++];
                VRCGraphics.Blit(null, (RenderTexture)material.GetTexture("TexO"), material, -1);
            }
            if (cursor < materials.Length)
                return;
            waiting = true;
            VRCAsyncGPUReadback.Request(outputs[running], 0, TextureFormat.RGBAFloat, (IUdonEventReceiver)this);
        }

        public override void OnAsyncGpuReadbackComplete(VRCAsyncGPUReadbackRequest request)
        {
            if (running < 0 || !waiting)
                return;
            int graph = running;
            running = -1;
            waiting = false;
            if (!request.hasError && request.done)
            {
                int width = request.width * 4,
                    rows = request.height;
                if (readback == null || readback.Length != width * rows)
                    readback = new float[width * rows];
                request.TryGetData(readback);
                // One row per frame with `hop` samples; the speech starts leadingSamples in, the padding after it is dropped.
                int spoken = Mathf.Min(rows, tokenCount * framesPerToken);
                int total = spoken * hop - leadingSamples;
                if (total > 0)
                {
                    var flat = new float[spoken * hop];
                    for (int row = 0; row < spoken; row++)
                        System.Array.Copy(readback, row * width, flat, row * hop, hop);
                    int capacity = graphTokens[graphTokens.Length - 1] * framesPerToken * hop;
                    if (clip == null)
                        clip = AudioClip.Create("akane-voice", capacity, 1, sampleRate, false);
                    var samples = new float[capacity]; // silence after the line
                    System.Array.Copy(flat, leadingSamples, samples, 0, total);
                    clip.SetData(samples, 0);
                    output.Stop();
                    output.clip = clip;
                    output.Play();
                    playedCount++;
                    lastSeconds = (float)total / sampleRate;
                    lastDecodeSeconds = Time.realtimeSinceStartup - startedAt;
                }
            }
            if (queued != null)
            {
                var next = queued;
                queued = null;
                Play(next);
            }
        }
    }
}
