using UdonSharp;
using UnityEngine;

namespace WorldLLM {
// Best-effort output filter (Task 08). Not a complete moderation system.
// The blocklist is one word per line (# starts a comment). Output and words are folded the same way before a
// substring match: NFKC (full-width ASCII, half-width katakana, combining marks), lower case, and whitespace,
// common separators and invisible format characters (zero-width, bidi controls, soft hyphen) removed. Every word is
// stored in its hiragana and katakana forms as well, so "ばか" also matches "バカ".
// The pipeline feeds each decoded piece through Append. The whole response so far is folded again on every piece
// (native string operations only), so a combining mark that arrives in the next piece still joins its base letter.
[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class LlmOutputFilter : UdonSharpBehaviour {
	public TextAsset blocklist;

	[System.NonSerialized] public string[] words = new string[0];
	[System.NonSerialized] public int hits;
	[System.NonSerialized] public string lastMatch = "";
	private bool loaded;
	private string raw = "";

	void Start() {
		Load();
	}

	public void Load() {
		loaded = true;
		words = new string[0];
		if(blocklist == null)
			return;
		foreach(var line in blocklist.text.Split('\n')) {
			var w = line.Trim();
			if(w.Length > 0 && !w.StartsWith("#"))
				AddFolded(Fold(w));
		}
	}

	// Adds a word at runtime (world owner tools and tests).
	public void AddWord(string word) {
		if(!loaded)
			Load();
		AddFolded(Fold(word));
	}

	void AddFolded(string w) {
		if(w.Length == 0)
			return;
		AddUnique(w);
		AddUnique(ToHiragana(w));
		AddUnique(ToKatakana(w));
	}

	void AddUnique(string w) {
		if(System.Array.IndexOf(words, w) >= 0)
			return;
		var next = new string[words.Length + 1];
		System.Array.Copy(words, next, words.Length);
		next[words.Length] = w;
		words = next;
	}

	public void BeginResponse() {
		if(!loaded)
			Load();
		raw = "";
	}

	// Returns true when the response so far contains a blocked word.
	public bool Append(string piece) {
		if(!loaded)
			Load();
		raw += piece;
		return Find(Fold(raw));
	}

	public bool IsBlocked(string text) {
		if(!loaded)
			Load();
		return Find(Fold(text));
	}

	bool Find(string text) {
		foreach(var w in words)
			if(text.Contains(w)) {
				hits++;
				lastMatch = w;
				return true;
			}
		return false;
	}

	public static string Fold(string text) {
		if(string.IsNullOrEmpty(text))
			return "";
		var t = text.Normalize(System.Text.NormalizationForm.FormKC).ToLowerInvariant();
		// Spaces, separators and invisible format characters (escaped so they stay visible in the source). Supplementary
		// variation selectors (U+E0100-E01EF, surrogate pairs) are removed by a scan that runs only when one is present.
		var removed = new string[] {
			"\u0020", "\u000a", "\u000d", "\u0009", "\u30fb", "\u002e", "\u002c", "\u002d", "\u005f", "\u002a", "\u002f", "\u3001",
			"\u3002", "\u30fc", "\u007e", "\u301c", "\u00ad", "\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\u202a", "\u202b",
			"\u202c", "\u202d", "\u202e", "\u2060", "\u2061", "\u2062", "\u2063", "\u2064", "\u2065", "\u2066", "\u2067", "\u2068", "\u2069",
			"\ufeff", "\u061c", "\u180e", "\u034f", "\u206a", "\u206b", "\u206c", "\u206d", "\u206e", "\u206f", "\ufe00", "\ufe01",
			"\ufe02", "\ufe03", "\ufe04", "\ufe05", "\ufe06", "\ufe07", "\ufe08", "\ufe09", "\ufe0a", "\ufe0b", "\ufe0c", "\ufe0d",
			"\ufe0e", "\ufe0f",
		};
		foreach(var r in removed)
			t = t.Replace(r, "");
		if(t.IndexOf('\udb40') >= 0)
			t = LlmPipeline.RemoveSupplementaryVariationSelectors(t);   // rare: only scanned when present
		return t;
	}

	public static string ToHiragana(string text) {
		var chars = text.ToCharArray();
		for(int i = 0; i < chars.Length; i++)
			if(chars[i] >= 'ァ' && chars[i] <= 'ヶ')
				chars[i] = (char)(chars[i] - 0x60);
		return new string(chars);
	}

	public static string ToKatakana(string text) {
		var chars = text.ToCharArray();
		for(int i = 0; i < chars.Length; i++)
			if(chars[i] >= 'ぁ' && chars[i] <= 'ゖ')
				chars[i] = (char)(chars[i] + 0x60);
		return new string(chars);
	}
}
}
