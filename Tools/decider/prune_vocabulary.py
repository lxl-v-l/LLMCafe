"""Which of the decider's 248 thousand tokens the cafe needs, and what the rest are written as.

A third of decider-0.8b's weights are its token embedding table. The cafe only ever feeds it English instructions and
Japanese (or ASCII) guest utterances, so most rows are never read. A token that is dropped is replaced by the pieces
it was merged from (the tokenizer's own merge tree), down to pieces that are kept; single bytes are always kept, so
every text can still be written. The model then sees a finer split than it was trained on for those words, which is
why each level is graded with the native evaluations before it is used:

    set DECIDER_VOCAB=<level>  and run the native evaluations

Levels (each adds to the one before):
    scripts   drop tokens with letters of other scripts (never produced by Japanese or ASCII text: exact)
    kanji3    drop runs of three or more kanji (almost all Chinese)
    ascii     keep only the ASCII tokens the conversation model's vocabulary also has, plus those the prompts use
    kanji2    drop runs of two kanji as well
    words     the "ascii" level without the long English words: ASCII tokens over three characters stay only if the
              prompts or vocabulary-keep.txt use them (English utterances are then read from short pieces)

    python prune_vocabulary.py            prints the size of every level

Where the files are: the decider's tokenizer.json is looked for next to this file, or at DECIDER_TOKENIZER; the
conversation model's tokenizer.json at CONVERSATION_TOKENIZER (needed from the "ascii" level on). Both can also be set
in paths.local.json next to this file or one folder up: {"DECIDER_TOKENIZER": "...", "CONVERSATION_TOKENIZER": "..."}.
"""
import json
import os
import re
from pathlib import Path

from tokenizers import Tokenizer

BASE = Path(__file__).resolve().parent


def location(name, default):
    """A file that is not in Git (weights, tokenizers): the environment, then paths.local.json, then the default."""
    if os.environ.get(name):
        return Path(os.environ[name])
    for folder in (BASE, BASE.parent):
        if (folder / 'paths.local.json').exists():
            value = json.loads((folder / 'paths.local.json').read_text(encoding='utf-8')).get(name)
            if value:
                return Path(value)
    return Path(default)


def beside(name, elsewhere):
    """A small file that travels with this script: next to it, or where the repository keeps it."""
    return BASE / name if (BASE / name).exists() else BASE.parent / elsewhere


DECIDER = location('DECIDER_TOKENIZER', BASE / 'tokenizer.json')
CONVERSATION = location('CONVERSATION_TOKENIZER', BASE / 'conversation-tokenizer.json')
PROMPT_TOKENS = BASE / 'prompt-ascii-tokens.json'  # written by a run with DECIDER_VOCAB=record
LEVELS = ['scripts', 'kanji3', 'ascii', 'kanji2']

KANA = re.compile(r'[\u3040-\u30ff]')
KANJI_RUN = re.compile(r'[\u4e00-\u9fff ]+')
JAPANESE = re.compile(r'[\x00-\x7f\u3000-\u30ff\u4e00-\u9fff\uff00-\uffef\u2010-\u2027\u2190-\u2193\u25a0-\u25ff\u2605\u2606\u266a\u00d7]*')


def category(text):
    if '\ufffd' in text or len(text.encode('utf-8')) <= 1:
        return 'piece'  # a byte or part of a character: always kept
    if re.fullmatch(r'[\x00-\x7f]*', text):
        return 'ascii'
    if KANA.search(text):
        return 'kana'
    if KANJI_RUN.fullmatch(text):
        return 'kanji%d' % min(len(text.strip()), 3)
    if JAPANESE.fullmatch(text):
        return 'punctuation'
    return 'other'


def plan(level):
    """(keep: set of token ids, split: {dropped id: [kept ids]}) for a level."""
    tokenizer = Tokenizer.from_file(str(DECIDER))
    data = json.loads(DECIDER.read_text(encoding='utf-8'))
    vocab = data['model']['vocab']
    size = tokenizer.get_vocab_size(with_added_tokens=True)
    texts = [tokenizer.decode([i], skip_special_tokens=False) for i in range(size)]
    kinds = [category(t) for t in texts]
    special = {t['id'] for t in data['added_tokens']}
    # The answer labels (A, B, ... of up to 255 options) are read from these rows by the baked graph.
    special |= set(json.loads(beside('decider-label-table.json', 'verification/decider-label-table.json').read_text(encoding='utf-8'))['letter_ids'])

    words = level == 'words'
    if words:
        level = 'ascii'
    dropped = {'other'}
    if level in LEVELS[1:]:
        dropped.add('kanji3')
    if level in LEVELS[3:]:
        dropped.add('kanji2')
    keep = {i for i in range(size) if kinds[i] not in dropped or i in special}
    if level in LEVELS[2:]:
        other = Tokenizer.from_file(str(CONVERSATION))
        shared = {other.decode([i], skip_special_tokens=False) for i in range(other.get_vocab_size())}
        prompts = set(json.loads(PROMPT_TOKENS.read_text(encoding='utf-8')))
        if words:
            listed = (BASE.parent / 'vocabulary-keep.txt').read_text(encoding='utf-8')
            for line in listed.splitlines():
                for word in line.split():
                    prompts.update(tokenizer.encode(word, add_special_tokens=False).ids)
                    prompts.update(tokenizer.encode(' ' + word, add_special_tokens=False).ids)
            shared = {text for text in shared if len(text) <= 3}
        keep -= {i for i in range(size) if kinds[i] == 'ascii' and texts[i] not in shared and i not in prompts and i not in special}

    parents = {}
    for merge in data['model']['merges']:
        a, b = merge if isinstance(merge, list) else merge.split(' ')
        parents.setdefault(vocab[a + b], (vocab[a], vocab[b]))
    split = {}

    def pieces(token):
        if token in keep:
            return [token]
        if token not in split:
            if token not in parents:  # a base token: nothing to split it into
                keep.add(token)
                return [token]
            a, b = parents[token]
            split[token] = pieces(a) + pieces(b)
        return split[token]

    for token in range(size):
        pieces(token)
    return keep, {t: p for t, p in split.items() if t not in keep}


def apply(tokenizer, level):
    """Makes a transformers tokenizer write dropped tokens as their pieces (level 'record' notes the ASCII tokens used)."""
    encode = tokenizer.encode
    if level == 'record':
        seen = set(json.loads(PROMPT_TOKENS.read_text(encoding='utf-8'))) if PROMPT_TOKENS.exists() else set()

        def recording(text, **kwargs):
            ids = encode(text, **kwargs)
            new = {i for i in ids if i not in seen and re.fullmatch(r'[\x00-\x7f]*', tokenizer.decode([i]))}
            if new:
                seen.update(new)
                PROMPT_TOKENS.write_text(json.dumps(sorted(seen)), encoding='utf-8')
            return ids
        tokenizer.encode = recording
        return None
    keep, split = plan(level)

    def pruned(text, **kwargs):
        return [piece for token in encode(text, **kwargs) for piece in split.get(token, (token,))]
    tokenizer.encode = pruned
    return keep


def from_environment(tokenizer):
    level = os.environ.get('DECIDER_VOCAB')
    if not level:
        return ''
    keep = apply(tokenizer, level)
    print(f'vocabulary level {level}: {len(keep) if keep else "recording"} tokens kept')
    return '-vocab-' + level


if __name__ == '__main__':
    for name in LEVELS + ['words']:
        kept, table = plan(name)
        print(name, 'kept', len(kept), 'dropped', len(table), 'longest split', max(map(len, table.values())))
