"""A smaller text vocabulary for the conversation model (LFM2.5-1.2B-JP) and the voice model that shares it (MioTTS-0.4B).

Both models keep one embedding row per token, and 9,723 of the 64,400 text tokens are written in scripts the cafe
never reads or writes (Cyrillic, Hangul, Arabic, ...). This makes a self-consistent smaller model instead of patching
ids in the world:

  - the tokenizer loses the dropped tokens and every merge that produced them, and the rest is renumbered in order;
  - the GGUF loses the same rows of token_embd (the output head is the same matrix), and its tokenizer is rewritten.

The pruned GGUF runs in llama.cpp, so the native checks grade exactly what the world will run, and the same file is
what convert_hf_model.py turns into textures.

Kept: special tokens, single bytes and partial characters, everything with kana or kanji, Japanese punctuation, every
ASCII token, any other token the prompts or the brain's source use, and whatever those were merged from. Tokens above
the text vocabulary (MioTTS's speech tokens) are all kept. Japanese and English chat, and the voice reading it, are
untouched.

With "english" as a last argument the long English words go too (ASCII tokens over three characters that the prompts
do not use, except the words listed in vocabulary-keep.txt). The native order and reply checks came out identical, but
the NPC can then only spell English from short pieces and the voice reads it from those. The world was built this way,
to stay under VRChat's size limit: 17,013 text tokens.

    python prune_text_vocabulary.py <model folder with tokenizer.json> <source gguf> <output folder> [output gguf name]
    python prune_text_vocabulary.py <model folder>                      prints what would be kept

With "world" as a last argument the tokens are not chosen again: the list the world was built with is read from
text-vocabulary.json (17,013 text tokens). Use it to rebuild the same models; choosing again gives a slightly different
list whenever the prompts or the brain's source have changed.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

BASE = Path(__file__).resolve().parent
TEXT_VOCABULARY = 64400  # ids below this are text; MioTTS puts its speech tokens above
SOURCES = ['vocabulary-keep.txt', 'npc-reply-system-prompt.txt', 'lfm-order-extraction-prompt.txt', 'CafeNpcBrain.cs', 'CafeNpcController.cs',
           'integration/LlmPipeline.cs', 'integration/LlmWorkerController.cs', 'evaluate_service_prompts.py',
           'verification/lfm-order-prompt-native-v3.json', 'verification/lfm-reply-prompt-native-v3.json']
DROP_ENGLISH = 'english' in sys.argv[2:]
AS_WORLD = 'world' in sys.argv[2:]
OPTIONS = ('english', 'world')
KANA_OR_KANJI = re.compile(r'[぀-ヿ一-鿿]')
JAPANESE = re.compile(r'[\x00-\x7f　-ヿ一-鿿＀-￯‐-‧←-↓■-◿★☆♪×]*')
ASCII = re.compile(r'[\x00-\x7f]*')


def plan(folder: Path):
    """Token ids to keep (ascending) for the tokenizer in folder."""
    path = folder / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(path))
    data = json.loads(path.read_text(encoding='utf-8'))
    vocab = data['model']['vocab']
    size = TEXT_VOCABULARY
    special = {t['id'] for t in data['added_tokens']}
    used = set()
    for name in SOURCES:
        # In the repository the sources are at these paths; in an exported copy they are somewhere under the parent.
        source = BASE / name
        if not source.exists():
            found = sorted(BASE.parent.rglob(Path(name).name))
            if not found:
                print('not found, skipped:', name)
                continue
            source = found[0]
        text = source.read_text(encoding='utf-8')
        if name.endswith('.json'):
            text = json.dumps(json.loads(text), ensure_ascii=False)  # the outputs as text, not \u escapes
        for line in text.splitlines():
            used.update(tokenizer.encode(line, add_special_tokens=False).ids)
            used.update(tokenizer.encode(' ' + line.strip(), add_special_tokens=False).ids)
    keep, kinds = set(), {}
    for token in range(size):
        text = tokenizer.decode([token], skip_special_tokens=False)
        if token in special:
            kind = 'special'
        elif '�' in text or len(text.encode('utf-8')) <= 1:
            kind = 'piece'
        elif ASCII.fullmatch(text):
            kind = 'ascii kept' if not DROP_ENGLISH or len(text) <= 3 or token in used else 'ascii dropped'
        elif KANA_OR_KANJI.search(text) and JAPANESE.fullmatch(text):
            kind = 'japanese'
        elif JAPANESE.fullmatch(text):
            kind = 'punctuation'
        else:
            kind = 'other dropped' if token not in used else 'other kept'
        kinds[kind] = kinds.get(kind, 0) + 1
        if not kind.endswith('dropped'):
            keep.add(token)
    # A token can only be produced if the two it is merged from can: keep the whole ancestry.
    parents = {}
    for merge in data['model']['merges']:
        a, b = merge if isinstance(merge, list) else merge.split(' ')
        parents.setdefault(vocab[a + b], (vocab[a], vocab[b]))
    stack = list(keep)
    while stack:
        for parent in parents.get(stack.pop(), ()):
            if parent not in keep:
                keep.add(parent)
                kinds['ancestor'] = kinds.get('ancestor', 0) + 1
                stack.append(parent)
    if AS_WORLD:
        keep = set(json.loads((BASE / 'text-vocabulary.json').read_text(encoding='utf-8'))['kept'])
        kinds = {'from text-vocabulary.json': len(keep)}
    return sorted(keep), kinds, data, parents


def write_tokenizer(data, parents, rows, folder: Path, target: Path):
    """tokenizer.json and its companions with only the kept tokens, renumbered."""
    new = {old: i for i, old in enumerate(rows)}
    vocab = data['model']['vocab']
    data['model']['vocab'] = {token: new[i] for token, i in sorted(vocab.items(), key=lambda kv: kv[1]) if i in new}
    merges = []
    for merge in data['model']['merges']:
        a, b = merge if isinstance(merge, list) else merge.split(' ')
        if vocab[a + b] in new and parents[vocab[a + b]] == (vocab[a], vocab[b]):
            merges.append(merge)
    data['model']['merges'] = merges
    for token in data['added_tokens']:
        token['id'] = new[token['id']]
    data['model']['vocab'].update({t['content']: t['id'] for t in data['added_tokens'] if t['content'] in vocab})
    text = json.dumps(data, ensure_ascii=False)
    target.mkdir(parents=True, exist_ok=True)
    (target / 'tokenizer.json').write_text(text, encoding='utf-8')
    for name in ('tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja', 'generation_config.json', 'config.json'):
        if (folder / name).exists():
            shutil.copy(folder / name, target / name)
    config = json.loads((target / 'tokenizer_config.json').read_text(encoding='utf-8'))
    if 'added_tokens_decoder' in config:
        config['added_tokens_decoder'] = {str(new[int(i)]): v for i, v in config['added_tokens_decoder'].items() if int(i) in new}
        (target / 'tokenizer_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding='utf-8')
    model = json.loads((target / 'config.json').read_text(encoding='utf-8'))
    for key in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
        if isinstance(model.get(key), int):
            model[key] = new[model[key]]
    model['vocab_size'] = len(rows)
    (target / 'config.json').write_text(json.dumps(model, indent=1), encoding='utf-8')
    return new, len(merges)


def write_gguf(source: Path, target: Path, rows, new):
    import gguf
    reader = gguf.GGUFReader(source)
    architecture = bytes(reader.fields['general.architecture'].parts[-1]).decode('utf-8')
    tokens = reader.fields['tokenizer.ggml.tokens'].contents()
    kept = set(rows)
    writer = gguf.GGUFWriter(target, arch=architecture)
    for field in reader.fields.values():
        if field.name == 'general.architecture' or field.name.startswith('GGUF.'):
            continue
        value = field.contents()
        kind = field.types[0]
        if field.name in ('tokenizer.ggml.tokens', 'tokenizer.ggml.token_type', 'tokenizer.ggml.scores'):
            value = [value[i] for i in rows]
        elif field.name == 'tokenizer.ggml.merges':
            index = {token: i for i, token in enumerate(tokens)}
            value = [m for m in value if index.get(m.replace(' ', '', 1)) in kept and all(index.get(p) in kept for p in m.split(' ', 1))]
        elif field.name.startswith('tokenizer.ggml.') and field.name.endswith('_token_id'):
            value = new[int(value)]
        elif field.name.endswith('.vocab_size'):
            value = len(rows)
        writer.add_key_value(field.name, value, kind, sub_type=field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None)
    for tensor in reader.tensors:
        data = tensor.data
        if tensor.name == 'token_embd.weight':
            assert data.shape[0] == len(tokens), (data.shape, len(tokens))
            data = np.ascontiguousarray(data[rows])  # one row of quantization blocks per token
        writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    folder = Path(sys.argv[1])
    rows, kinds, data, parents = plan(folder)
    print(len(rows), 'tokens kept:', kinds)
    if len([a for a in sys.argv if a not in OPTIONS]) < 4:
        return
    source, target = Path(sys.argv[2]), Path(sys.argv[3])
    import gguf
    total = len(gguf.GGUFReader(source).fields['tokenizer.ggml.tokens'].data)
    rows += list(range(TEXT_VOCABULARY, total))  # reserved rows and MioTTS's speech tokens stay, after the text
    new, merges = write_tokenizer(data, parents, rows, folder, target)
    name = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] not in OPTIONS else source.stem + '-vocab.gguf'
    write_gguf(source, target / name, rows, new)
    (target / 'vocabulary.json').write_text(json.dumps({'size': len(rows), 'merges': merges, 'kept': rows}), encoding='utf-8')
    print('wrote', target / name, (target / name).stat().st_size, 'bytes;', merges, 'merges')


if __name__ == '__main__':
    main()
