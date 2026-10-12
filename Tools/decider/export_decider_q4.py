"""Export the pinned equal-head Qwen3.5 decider GGUF to ShaderGPT textures.

Usage: export_decider_q4.py [QUANT] [MODEL FOLDER] [VOCABULARY LEVEL]

QUANT defaults to Q4_0 (Q5_K_M keeps the native decisions that Q4_0 flips). MODEL FOLDER is the model as downloaded
from Hugging Face, with a download-manifest.json naming its repository and revision; it defaults to the folder of this
file. Reads ../decider-export/<name>-<QUANT>.gguf and writes ../decider-export/<name>-<quant>-textures.

The third argument is a vocabulary level of prune_vocabulary.py: only the embedding rows of the
tokens that level keeps are written, renumbered in order, with vocabulary.json telling the world how to renumber what
the (unchanged) tokenizer produces: map[old id] is the new id, or -(n+1) for a dropped token whose pieces start at
splits[n] (a count, then that many new ids)."""
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from accelerate import init_empty_weights
from gguf import GGUFReader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_gguf_pytorch_utils import TensorProcessor as HfTensorProcessor
from transformers.modeling_gguf_pytorch_utils import get_gguf_hf_weights_map

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
# ShaderGPT's Python folder (convert_hf_model.py): SHADERGPT_PYTHON, or where this repository keeps it.
sys.path.insert(0, os.environ.get('SHADERGPT_PYTHON') or str(ROOT / 'projects/world-llm/vrc-world-llm/Python'))
import convert_hf_model as converter
from convert_hf_tokenizer import export_tokenizer


class Qwen35GgufProcessor(converter.TensorProcessor):
    original_small_weights = {}
    kept_rows = None  # token ids whose embedding rows are written; None: all

    def process(self, tensor, name: str, **kwargs):
        if name in self.original_small_weights:
            return self.original_small_weights[name]
        if name == 'token_embd.weight' and self.kept_rows is not None:
            tensor = tensor._replace(data=tensor.data[self.kept_rows])  # one row of blocks per token
        quant = super().process(tensor, name, **kwargs)
        if name.endswith('.ssm_a'):
            return torch.log(-quant.dequantize()).flatten()
        if name.endswith('norm.weight') and not name.endswith('.ssm_norm.weight'):
            return quant.dequantize() - 1
        return quant


class TextRotaryProcessor(converter.Qwen3_5TextRotaryEmbedding):
    def process(self, module, state_dict: dict, name: str):
        positions = torch.arange(self.max_pos)[None, None, :].expand(3, 1, -1)
        cosine, sine = module(positions.float(), positions)
        state_dict[f'{name}.weight'] = torch.cat((cosine[..., :cosine.shape[-1]//2],
                                                sine[..., :sine.shape[-1]//2]), dim=-1).squeeze(0)
        for axis in (1, 2):
            positions = torch.zeros(3, 1, 32, dtype=torch.long)
            positions[axis, 0] = torch.arange(32)
            cosine, sine = module(positions.float(), positions)
            state_dict[f'{name}_mrope{axis}.weight'] = torch.cat((cosine[..., :cosine.shape[-1]//2],
                                                                sine[..., :sine.shape[-1]//2]), dim=-1).squeeze(0)


def hf_mapping(model) -> dict[str, str]:
    mapping = get_gguf_hf_weights_map(model, HfTensorProcessor(), model_type='qwen35')
    for layer, kind in enumerate(model.config.layer_types):
        if kind == 'linear_attention':
            mapping[f'blk.{layer}.ssm_dt.bias'] = f'model.layers.{layer}.linear_attn.dt_bias'
    return mapping


def main() -> None:
    torch.set_num_threads(8)
    quant = sys.argv[1] if len(sys.argv) > 1 else 'Q4_0'
    source = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else BASE
    pinned = json.loads((source / 'download-manifest.json').read_text(encoding='utf-8'))
    if source == BASE:
        name = 'decider-2b'
        target = BASE.parent / ('decider-export/q4-textures' if quant == 'Q4_0' else f'decider-export/{quant.lower()}-textures')
    else:
        name = pinned['repository'].split('/')[-1]
        target = BASE.parent / f'decider-export/{name}-{quant.lower()}-textures'
    gguf = BASE.parent / f'decider-export/{name}-{quant}.gguf'
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    vocabulary = None
    if len(sys.argv) > 3:
        sys.path.insert(0, str(BASE.parent / 'decider-0.8b-research'))
        sys.path.insert(0, str(BASE))  # or next to this file
        import prune_vocabulary
        keep, split = prune_vocabulary.plan(sys.argv[3])
        rows = sorted(keep)
        new = {old: i for i, old in enumerate(rows)}
        size = max(max(rows), max(split)) + 1
        table, pieces = [0] * size, []
        for old in range(size):
            if old in new:
                table[old] = new[old]
            elif old in split:
                table[old] = -(len(pieces) + 1)
                pieces += [len(split[old])] + [new[p] for p in split[old]]
        vocabulary = {'level': sys.argv[3], 'size': len(rows), 'map': table, 'splits': pieces}
        Qwen35GgufProcessor.kept_rows = rows
        config.vocab_size = len(rows)
        target = target.with_name(target.name + '-vocab-' + sys.argv[3])
    if config.model_type != 'qwen3_5_text' or config.linear_num_key_heads != config.linear_num_value_heads:
        raise ValueError('This exporter covers the pinned equal-head Qwen3.5 text checkpoint only')
    reader = GGUFReader(gguf)
    parameter_count = sum(int(t.n_elements) for t in reader.tensors)
    if parameter_count > 2_000_000_000:
        raise ValueError('Per-model parameter budget exceeded')
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    model.tie_weights()
    mapping = hf_mapping(model)
    missing = [t.name for t in reader.tensors if t.name not in mapping]
    if missing:
        raise ValueError(f'Unmapped GGUF tensors: {missing}')
    with safe_open(source / 'model.safetensors', framework='pt', device='cpu') as original:
        source_keys = set(original.keys())
        for tensor in reader.tensors:
            if tensor.tensor_type.name in ('F16', 'F32'):
                key = mapping[tensor.name]
                if key not in source_keys and key.startswith('model.'):
                    key = 'model.language_model.' + key[len('model.'):]
                if key not in source_keys:
                    raise ValueError(f'Original small tensor missing: {key}')
                Qwen35GgufProcessor.original_small_weights[tensor.name] = original.get_tensor(key).float()
    converter.get_gguf_hf_weights_map = hf_mapping
    converter.TENSOR_PROCESSORS['qwen35'] = Qwen35GgufProcessor
    converter.Qwen3_5TextRotaryEmbedding = TextRotaryProcessor
    target.mkdir(parents=True, exist_ok=True)
    converter.export_model(model, target, gguf_file=str(gguf))
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    export_tokenizer(tokenizer, target)
    if vocabulary:
        (target / 'vocabulary.json').write_text(json.dumps(vocabulary), encoding='utf-8')
    files = [p for p in target.rglob('*') if p.is_file()]
    manifest = {
        'repository': pinned['repository'], 'revision': pinned['revision'],
        'converter_source_archive': {'llama_tag': 'b11433', 'sha256': '5bdfff68355dd0291ce0cc68f0e6038cb1f3b992aeacfad5d7623dfe448ad18c'},
        'gguf_bytes': gguf.stat().st_size,
        'stored_parameters': parameter_count, 'parameter_limit_per_model': 2_000_000_000,
        'texture_export_files': len(files), 'texture_export_bytes': sum(p.stat().st_size for p in files),
        'output_directory': str(target),
        'source_small_tensors_restored': len(Qwen35GgufProcessor.original_small_weights),
        'scope': 'Offline weight/tokenizer conversion only. Unquantized small tensors are restored directly from original HF weights to avoid GGUF nonlinear/offset rounding loss. Equal key/value head counts require no V-head permutation. Text RoPE uses the required three-axis positions. No Unity inference, decision parity, runtime adoption or final bundle/memory proof.',
    }
    with gguf.open('rb') as stream:
        manifest['gguf_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if source == BASE:
        report = 'decider-q4-export.json' if quant == 'Q4_0' else f'decider-{quant.lower()}-export.json'
    else:
        report = f'{name}-{quant.lower()}{"-vocab-" + vocabulary["level"] if vocabulary else ""}-export.json'
    (BASE.parent / 'verification').mkdir(parents=True, exist_ok=True)
    (BASE.parent / 'verification' / report).write_bytes((json.dumps(manifest, indent=2)+'\n').encode('utf-8'))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
