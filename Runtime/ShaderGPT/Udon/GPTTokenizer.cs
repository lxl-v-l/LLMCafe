using UnityEngine;
using System.Text.RegularExpressions;
#if UDON
using VRC.SDK3.Data;
#else
using DataDictionary = System.Collections.Generic.Dictionary<string,int>;
#endif

namespace ShaderGPT.Udon {
[DefaultExecutionOrder(-1)] // make it run before pipeline
public class GPTTokenizer : UdonMonoBehaviour {
	[Header("Tokenizer")]
	public TextAsset tokenizerJson;

	private string[] added_tokens;
	private string   pre_tokenizer_pattern;
	private string   pre_tokenizer_behavior;
	private bool     byte_fallback;
	private bool     ignore_merges;
	private string[] vocab;
	private string[] merges;
	private float [] scores;
	private string   bos_token;
	private string   eos_token;
	private string   pad_token;
	private string   unk_token;
#if UDON
	void LoadJson() {
		VRCJson.TryDeserializeFromJson(tokenizerJson.text, out var tokenizer);
		added_tokens  = tokenizer.GetStringArray(nameof(added_tokens));
		if(tokenizer.DataDictionary.TryGetValue("pre_tokenizer", TokenType.DataDictionary, out var pre_tokenizer)) {
			pre_tokenizer_pattern  = pre_tokenizer.GetString("pattern");
			pre_tokenizer_behavior = pre_tokenizer.GetString("behavior");
		}
		tokenizer.DataDictionary.TryGetValue("model", TokenType.DataDictionary, out var model);
		byte_fallback = model.GetBoolean(nameof(byte_fallback), false);
		ignore_merges = model.GetBoolean(nameof(ignore_merges), false);
		vocab         = model.GetStringArray(nameof(vocab));
		merges        = model.GetStringArray(nameof(merges));
		scores        = model.GetFloatArray(nameof(scores));
		bos_token     = tokenizer.GetString(nameof(bos_token));
		eos_token     = tokenizer.GetString(nameof(eos_token));
		pad_token     = tokenizer.GetString(nameof(pad_token));
		unk_token     = tokenizer.GetString(nameof(unk_token));
	}
#else
	void LoadJson() {
		var tokenizer = JsonUtility.FromJson<Tokenizer>(tokenizerJson.text);
		added_tokens = tokenizer.added_tokens;
		if(tokenizer.pre_tokenizer != null) {
			pre_tokenizer_pattern = tokenizer.pre_tokenizer.pattern;
			pre_tokenizer_behavior = tokenizer.pre_tokenizer.behavior;
		}
		byte_fallback = tokenizer.model.byte_fallback;
		ignore_merges = tokenizer.model.ignore_merges;
		vocab         = tokenizer.model.vocab;
		merges        = tokenizer.model.merges;
		scores        = tokenizer.model.scores;
		bos_token     = tokenizer.bos_token;
		eos_token     = tokenizer.eos_token;
		pad_token     = tokenizer.pad_token;
		unk_token     = tokenizer.unk_token;
	}
	[System.Serializable]
	class Tokenizer {
		public string[] added_tokens;
		public PreTokenizer pre_tokenizer;
		public Model model;
		public string bos_token;
		public string eos_token;
		public string pad_token;
		public string unk_token;
	}
	[System.Serializable]
	class PreTokenizer {
		public string pattern;
		public string behavior;
	}
	[System.Serializable]
	class Model {
		public bool byte_fallback;
		public bool ignore_merges;
		public string[] vocab;
		public string[] merges;
		public float [] scores;
	}
#endif
	[System.NonSerialized] public int bos_token_id;
	[System.NonSerialized] public int eos_token_id;
	[System.NonSerialized] public int pad_token_id;
	[System.NonSerialized] public int unk_token_id;
	[System.NonSerialized] public int vocab_size;
	// WorldLLM patch: readiness flag and per-phase load timings (seconds) for the Udon tokenizer check.
	[System.NonSerialized] public bool isReady;
	[System.NonSerialized] public float loadSecondsJson, loadSecondsRegex, loadSecondsBytes, loadSecondsDict, loadSecondsTotal;
	private bool byte_level;
	private Regex added_token_regex;
	private Regex pre_tokenizer_regex;
	private DataDictionary vocab_ids;
	private DataDictionary merges_ids;
	private DataDictionary added_token_ids;
	void LoadTokenizer() {
		var t0 = Time.realtimeSinceStartup;
		LoadJson();
		var t1 = Time.realtimeSinceStartup;
		loadSecondsJson = t1 - t0;
		bos_token_id = System.Array.IndexOf(vocab, bos_token);
		eos_token_id = System.Array.IndexOf(vocab, eos_token);
		pad_token_id = System.Array.IndexOf(vocab, pad_token);
		unk_token_id = System.Array.IndexOf(vocab, unk_token);
		if(added_tokens != null && added_tokens.Length > 0) {
			var pattern = new System.Text.StringBuilder();
			for(int i=added_tokens.Length-1; i>=0; i--)
				pattern.AppendFormat("|{0}", Regex.Escape(added_tokens[i])); // prioritize longer matches
			added_token_regex = new Regex(pattern.ToString().Substring(1));
		}
		if(!string.IsNullOrEmpty(pre_tokenizer_pattern))
			pre_tokenizer_regex = new Regex(pre_tokenizer_pattern);
		var t2 = Time.realtimeSinceStartup;
		loadSecondsRegex = t2 - t1;
		vocab  = ConvertToByteStringArray(vocab);
		merges = ConvertToByteStringArray(merges);
		var t3 = Time.realtimeSinceStartup;
		loadSecondsBytes = t3 - t2;
		vocab_ids  = BuildIndexDict(vocab);
		merges_ids = BuildIndexDict(merges);
		loadSecondsDict = Time.realtimeSinceStartup - t3;
		vocab_size = vocab.Length;
		byte_level = !byte_fallback && System.Array.IndexOf(vocab, "\0") >= 0;
		if(added_tokens != null && added_tokens.Length > 0) {
			var raw_added_tokens = Take(added_tokens, added_tokens.Length);
			added_tokens = ConvertToByteStringArray(added_tokens);
			added_token_ids = new DataDictionary();
			for(int i=added_tokens.Length-1; i>=0; i--)
				if(vocab_ids.TryGetValue(added_tokens[i], out var id))
					added_token_ids[raw_added_tokens[i]] = id;
		}
		loadSecondsTotal = Time.realtimeSinceStartup - t0;
		isReady = true;
	}

	public void OnEnable() {
		if(vocab != null || chunkedLoading)
			return;
#if UDON
		if(tokenizerChunkJsons != null && tokenizerChunkJsons.Length > 0) {
			BeginChunkedLoad();
			return;
		}
#endif
		LoadTokenizer();
	}

	// ---- WorldLLM patch: chunked loading.
	// tokenizerBodyJson / tokenizerChunkJsons are tokenizer.json cut into a body and chunks beforehand.
	// Loading the whole tokenizer.json in OnEnable stalls one frame for ~0.5 s; here one JSON chunk or one slice
	// of byte-string conversion / dictionary insertion is processed per frame. The resulting tables are identical
	// to LoadTokenizer(), including BuildIndexDict's insertion order.
	[Header("Chunked loading (optional)")]
	public TextAsset tokenizerBodyJson;
	public TextAsset[] tokenizerChunkJsons;
	public int chunkSliceSize = 8192;
	[System.NonSerialized] public float loadStepMaxSeconds;
	private bool chunkedLoading;
	private int loadPhase, loadIndex;
	private float loadStart;
	private string[] chunkPaths;
	private int[] chunkStarts;
#if UDON
	void BeginChunkedLoad() {
		chunkedLoading = true;
		loadStart = Time.realtimeSinceStartup;
		VRCJson.TryDeserializeFromJson(tokenizerBodyJson.text, out var tokenizer);
		added_tokens  = tokenizer.GetStringArray(nameof(added_tokens));
		if(tokenizer.DataDictionary.TryGetValue("pre_tokenizer", TokenType.DataDictionary, out var pre_tokenizer)) {
			pre_tokenizer_pattern  = pre_tokenizer.GetString("pattern");
			pre_tokenizer_behavior = pre_tokenizer.GetString("behavior");
		}
		tokenizer.DataDictionary.TryGetValue("model", TokenType.DataDictionary, out var model);
		byte_fallback = model.GetBoolean(nameof(byte_fallback), false);
		ignore_merges = model.GetBoolean(nameof(ignore_merges), false);
		bos_token     = tokenizer.GetString(nameof(bos_token));
		eos_token     = tokenizer.GetString(nameof(eos_token));
		pad_token     = tokenizer.GetString(nameof(pad_token));
		unk_token     = tokenizer.GetString(nameof(unk_token));
		var vocabCount  = tokenizer.GetInt("vocab_count");
		var mergesCount = tokenizer.GetInt("merges_count");
		var scoresCount = tokenizer.GetInt("scores_count");
		vocab  = new string[vocabCount];
		merges = mergesCount > 0 ? new string[mergesCount] : null;
		scores = scoresCount > 0 ? new float[scoresCount] : null;
		tokenizer.DataDictionary.TryGetValue("chunks", TokenType.DataList, out var chunks);
		var n = chunks.DataList.Count;
		chunkPaths  = new string[n];
		chunkStarts = new int[n];
		for(int i=0; i<n; i++) {
			chunkPaths[i]  = chunks.DataList[i].GetString("path");
			chunkStarts[i] = chunks.DataList[i].GetInt("start");
		}
		loadSecondsJson = Time.realtimeSinceStartup - loadStart;
		loadStepMaxSeconds = loadSecondsJson;
		loadPhase = 1;
		loadIndex = 0;
		SendCustomEventDelayedFrames(nameof(_TokenizerLoadStep), 1);
	}
	public void _TokenizerLoadStep() {
		var t0 = Time.realtimeSinceStartup;
		var mergesLength = merges == null ? 0 : merges.Length;
		if(loadPhase == 1) { // parse one JSON chunk
			VRCJson.TryDeserializeFromJson(tokenizerChunkJsons[loadIndex].text, out var arr);
			var list = arr.DataList;
			var start = chunkStarts[loadIndex];
			var path = chunkPaths[loadIndex];
			if(path == "model.scores") {
				for(int i=0, n=list.Count; i<n; i++)
					scores[start+i] = (float)list[i].Double;
			} else {
				var dst = path == "model.vocab" ? vocab : merges;
				for(int i=0, n=list.Count; i<n; i++)
					dst[start+i] = list[i].String;
			}
			if(++loadIndex >= tokenizerChunkJsons.Length) {
				loadPhase = 2;
				loadIndex = 0;
			}
			loadSecondsJson += Time.realtimeSinceStartup - t0;
		} else if(loadPhase == 2) { // special token ids and regexes, as in LoadTokenizer()
			bos_token_id = System.Array.IndexOf(vocab, bos_token);
			eos_token_id = System.Array.IndexOf(vocab, eos_token);
			pad_token_id = System.Array.IndexOf(vocab, pad_token);
			unk_token_id = System.Array.IndexOf(vocab, unk_token);
			if(added_tokens != null && added_tokens.Length > 0) {
				var pattern = new System.Text.StringBuilder();
				for(int i=added_tokens.Length-1; i>=0; i--)
					pattern.AppendFormat("|{0}", Regex.Escape(added_tokens[i]));
				added_token_regex = new Regex(pattern.ToString().Substring(1));
			}
			if(!string.IsNullOrEmpty(pre_tokenizer_pattern))
				pre_tokenizer_regex = new Regex(pre_tokenizer_pattern);
			loadPhase = 3;
			loadIndex = 0;
			loadSecondsRegex = Time.realtimeSinceStartup - t0;
		} else if(loadPhase == 3) { // byte strings: vocab then merges, one slice per frame
			// Slices never straddle the vocab/merges boundary; advance by the count actually processed.
			if(loadIndex < vocab.Length) {
				var count = Mathf.Min(chunkSliceSize, vocab.Length - loadIndex);
				ConvertRange(vocab, loadIndex, count);
				loadIndex += count;
			} else {
				var count = Mathf.Min(chunkSliceSize, vocab.Length + mergesLength - loadIndex);
				ConvertRange(merges, loadIndex - vocab.Length, count);
				loadIndex += count;
			}
			if(loadIndex >= vocab.Length + mergesLength) {
				vocab_ids  = new DataDictionary();
				merges_ids = new DataDictionary();
				loadPhase = 4;
				loadIndex = 0;
			}
			loadSecondsBytes += Time.realtimeSinceStartup - t0;
		} else if(loadPhase == 4) { // dictionaries in BuildIndexDict order: from "\0" to the end, then the head
			if(loadIndex < vocab.Length) {
				var count = Mathf.Min(chunkSliceSize, vocab.Length - loadIndex);
				InsertRange(vocab_ids, vocab, loadIndex, count);
				loadIndex += count;
			} else {
				var count = Mathf.Min(chunkSliceSize, vocab.Length + mergesLength - loadIndex);
				InsertRange(merges_ids, merges, loadIndex - vocab.Length, count);
				loadIndex += count;
			}
			if(loadIndex >= vocab.Length + mergesLength)
				loadPhase = 5;
			loadSecondsDict += Time.realtimeSinceStartup - t0;
		} else { // finalize, as in LoadTokenizer()
			vocab_size = vocab.Length;
			byte_level = !byte_fallback && System.Array.IndexOf(vocab, "\0") >= 0;
			if(added_tokens != null && added_tokens.Length > 0) {
				var raw_added_tokens = Take(added_tokens, added_tokens.Length);
				added_tokens = ConvertToByteStringArray(added_tokens);
				added_token_ids = new DataDictionary();
				for(int i=added_tokens.Length-1; i>=0; i--)
					if(vocab_ids.TryGetValue(added_tokens[i], out var id))
						added_token_ids[raw_added_tokens[i]] = id;
			}
			loadSecondsTotal = Time.realtimeSinceStartup - loadStart;
			chunkedLoading = false;
			isReady = true;
		}
		loadStepMaxSeconds = Mathf.Max(loadStepMaxSeconds, Time.realtimeSinceStartup - t0);
		if(!isReady)
			SendCustomEventDelayedFrames(nameof(_TokenizerLoadStep), 1);
	}
	static void ConvertRange(string[] strs, int from, int count) {
		if(strs == null || count <= 0)
			return;
		var slice = new string[count];
		System.Array.Copy(strs, from, slice, 0, count);
		slice = ConvertToByteStringArray(slice);
		System.Array.Copy(slice, 0, strs, from, count);
	}
	// Inserts the k-th..(k+count-1)-th keys of BuildIndexDict's order (start..n-1, then 0..start-1).
	static void InsertRange(DataDictionary dict, string[] strs, int from, int count) {
		if(strs == null || count <= 0)
			return;
		var n = strs.Length;
		var start = Mathf.Max(0, System.Array.IndexOf(strs, "\0"));
		for(int k=from; k<from+count; k++) {
			var i = (start + k) % n;
			dict[strs[i]] = i;
		}
	}
#endif

	[System.NonSerialized] public int decodeState;
	public string Decode(int token) {
		if(token < 0 || token >= vocab.Length)
			return string.Format("<error:{0}>", token);
		var bstr = vocab[token];
		var nbstr = bstr.Length;
		var str = "";
		for(int i=0; i<nbstr; i++) {
			var b = char.ConvertToUtf32(bstr, i);
			if(b <= 0b01111111)
				str += char.ConvertFromUtf32(b);
			else if(b > 0b10111111)
				decodeState = b;
			else {
				decodeState = (decodeState << 6) | (0b00111111 & b);
				if((decodeState & -0x800) == 0x3000)
					str += char.ConvertFromUtf32(decodeState - 0x3000);
				else if((decodeState & -0x10000) == 0xe0000)
					str += char.ConvertFromUtf32(decodeState - 0xe0000);
				else if((decodeState & -0x200000) == 0x3c00000)
					str += char.ConvertFromUtf32(decodeState - 0x3c00000);
			}
		}
		return str;
	}

	const int MAX_TOKENS = 16384;
	private int[] tokenArray = new int[MAX_TOKENS];
	private int tokenCount;
	public virtual int[] Encode(string str) {
		tokenCount = 0;
		var i = 0;
		if(added_token_regex != null)
			for(var m=added_token_regex.Match(str); m.Success; m=m.NextMatch()) {
				EncodeSpan(str, i, m.Index);
				tokenArray[tokenCount++] = added_token_ids.TryGetValue(m.Value, out int id) ? id : unk_token_id;
				i = m.Index + m.Length;
			}
		EncodeSpan(str, i, str.Length);
		return Take(tokenArray, tokenCount);
	}
	void EncodeSpan(string str, int start, int stop) {
		if(pre_tokenizer_regex == null) {
			EncodeChunk(str, start, stop);
			return;
		}
		var isolated = pre_tokenizer_behavior == "Isolated";
		var mergedWithNext = pre_tokenizer_behavior == "MergedWithNext";
		var mergedWithPrevious = pre_tokenizer_behavior == "MergedWithPrevious";
		var i = start;
		for(var m=pre_tokenizer_regex.Match(str, start, stop-start); m.Success; ) {
			var j = mergedWithPrevious ? m.Index + m.Length : m.Index;
			EncodeChunk(str, i, j);
			i = j;
			if(isolated) {
				j = i + m.Length;
				EncodeChunk(str, i, j);
				i = j;
			}
			m = m.NextMatch();
			if(mergedWithNext && m.Success) {
				i = m.Index;
				EncodeChunk(str, i, j);
				i = j;
			}
		}
		EncodeChunk(str, i, stop);
	}
	void EncodeChunk(string str, int start, int stop) {
		if(start >= stop)
			return;
		if(scores != null && scores.Length > 0)
			UnigramEncode(System.Text.Encoding.UTF8.GetBytes(str, start, stop-start));
		else
			BytePairEncode(System.Text.Encoding.UTF8.GetBytes(str, start, stop-start));
	}

	private string[] partArray = new string[MAX_TOKENS];
	private void BytePairEncode(byte[] bytes) {
		var nbytes = bytes.Length;
		var chars = new char[nbytes];
		System.Array.Copy(bytes, chars, nbytes);
		if(ignore_merges && vocab_ids.TryGetValue(new string(chars), out int id)) {
			tokenArray[tokenCount++] = id;
			return;
		}
		var parts = partArray;
		var nparts = 0;
		for(int i=0; i<nbytes;) {
			var b = bytes[i];
			if(b >= 0xC0 && !byte_level) {
				var next = Mathf.Min(nbytes, i + Mathf.Max(2, (b>>4)-11));
				var part = new string(chars, i, next-i);
				if(vocab_ids.ContainsKey(part) || !byte_fallback) {
					parts[nparts++] = part;
					i = next;
					continue;
				}
			}
			parts[nparts++] = new string(chars, i++, 1);
		}
		BytePairMerge(parts, nparts);
	}
	private void BytePairMerge(string[] parts, int nparts) {
		while(true) {
			var minRank = 0x7FFFFFFF;
			var minPos = -1;
			for(int i=1; i<nparts; i++) {
				var pair = string.Format("{0} {1}", parts[i-1], parts[i]);
				if(merges_ids.TryGetValue(pair, out int rank) && rank < minRank) {
					minRank = rank;
					minPos = i;
				}
			}
			if(minPos < 0) {
				for(int i=0; i<nparts; i++)
					tokenArray[tokenCount++] = vocab_ids.TryGetValue(parts[i], out int x) ? x : unk_token_id;
				return;
			}
			parts[minPos-1] += parts[minPos];
			System.Array.Copy(parts, minPos+1, parts, minPos, (--nparts)-minPos);
		}
	}
	private void UnigramEncode(byte[] bytes) {
		var nbytes = bytes.Length;
		var chars = new char[nbytes];
		System.Array.Copy(bytes, chars, nbytes);
		var bstr = new string(chars);
		var dp = new Vector2[nbytes];
		for(int i=nbytes-1; i>=0; i--) {
			var bestSum = float.NegativeInfinity;
			var bestIdx = i+1;
			var dpJ = default(Vector2);
			var b = bytes[i];
			var next = b >= 0xC0 && !byte_level ? Mathf.Min(nbytes, i + Mathf.Max(2, (b>>4)-11)) : i+1;
			for(int j=nbytes; j>=next; dpJ=dp[--j]) {
				var chunk = bstr.Substring(i, j-i);
				if(!vocab_ids.TryGetValue(chunk, out int index))
					continue;
				var sum = dpJ.x + scores[index];
				if(sum > bestSum) {
					bestSum = sum;
					bestIdx = j;
				}
			}
			if(float.IsInfinity(bestSum)) {
				var j = byte_fallback ? i+1 : next;
				bestSum = j<nbytes ? dp[j].x : default;
				bestIdx = j;
			}
			dp[i] = new Vector2(bestSum, bestIdx);
		}
		for(int i=0; i<nbytes;) {
			var j = (int)dp[i].y;
			var chunk = bstr.Substring(i, j-i);
			tokenArray[tokenCount++] = vocab_ids.TryGetValue(chunk, out int x) ? x : unk_token_id;
			i = j;
		}
	}

	static DataDictionary BuildIndexDict(string[] strs) {
		var dict = new DataDictionary();
		if(strs == null || strs.Length == 0)
			return dict;
		// start from token <0x00>, to deprioritized byte fallback in the dictionary
		var start = Mathf.Max(0, System.Array.IndexOf(strs, "\0"));
		for(int n=strs.Length, i=start; i<n; i++)
			dict[strs[i]] = i;
		for(int i=0; i<start; i++)
			dict[strs[i]] = i;
		return dict;
	}
	static string[] ConvertToByteStringArray(string[] strs) {
		if(strs == null || strs.Length == 0)
			return strs;
		var n = strs.Length;
		{
			var bytes = System.Text.Encoding.UTF8.GetBytes(string.Join('\0', strs));
			var chars = new char[bytes.Length];
			System.Array.Copy(bytes, chars, bytes.Length);
			strs = new string(chars).Split('\0');
			if(strs.Length != n)
				Debug.LogErrorFormat("ConvertToByteStringArray: expected {0} strings, got {1}", n, strs.Length);
		}
		for(int i=0; i<n; i++) {
			var bstr = strs[i];
			for(int j=bstr.IndexOf('\u00ee'); j>=0; j=bstr.IndexOf('\u00ee', j+1))
				bstr = string.Concat(bstr.Substring(0, j),
					char.ConvertFromUtf32(((bstr[j+1] & 0x3f) << 6) | (bstr[j+2] & 0x3f)), bstr.Substring(j+3));
			strs[i] = bstr;
		}
		return strs;
	}
	static T[] Take<T>(T[] src, int n) {
		var dst = new T[n];
		System.Array.Copy(src, dst, n);
		return dst;
	}
}
}