"""Write the baked MioCodec decoder (bake_miocodec.py) as ShaderGPT textures, with a testcase for Unity.

Run with the converter's env (the one used for convert_hf_model.py):

    python export_miocodec_textures.py <baked dir> <output folder> [megabyte threshold, default 0.05] [bits, 8 or 4]

Matrices and convolution kernels above the threshold are stored at 8 or 4 bits (ShaderGPT's own format with one scale
for every four weights: 10 or 6 bits a weight in the world); the transposed
convolutions stay in floating point because ShaderGPT runs their kernel sizes through a path without quantization.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ShaderGPT's Python folder (convert_hf_model.py): SHADERGPT_PYTHON, or where this repository keeps it.
sys.path.insert(0, os.environ.get('SHADERGPT_PYTHON') or str(Path(__file__).resolve().parents[2] / 'world-llm/vrc-world-llm/Python'))
import convert_hf_model as converter
from quantization import CustomQuantTensor


class Config:
    def __init__(self, values):
        self.__dict__.update(values)

    def to_json_file(self, path):
        Path(path).write_text(json.dumps(self.__dict__, indent=1), encoding='utf-8')


class Node(torch.nn.Module):
    pass


def main():
    baked, folder = Path(sys.argv[1]), Path(sys.argv[2])
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else .05
    bits = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    CustomQuantTensor.__init__.__defaults__ = (bits,)  # the converter builds it without a width
    weights = torch.load(baked / 'baked.pt', map_location='cpu', weights_only=True)
    model = Node()
    model.config = Config(json.loads((baked / 'config.json').read_text(encoding='utf-8')))
    model.dtype = torch.float32
    for name, tensor in weights.items():
        *path, leaf = name.split('.')
        node = model
        for part in path:
            if not hasattr(node, part):
                node.add_module(part, Node())
            node = getattr(node, part)
        node.register_buffer(leaf, tensor)

    def quantize(name, shape):
        # istft.* are exact constants (0/1 fold kernel, FFT basis): kept in floating point.
        return 'upsample_layers' not in name and 'conv_upsample' not in name and not name.startswith('istft.') and np.prod(shape) >= threshold * 1e6

    converter.export_model(model, folder, force_write=True, force_quant=quantize)

    reference = np.load(baked / 'reference.npz')
    testcase = {'input_ids': reference['tokens'].tolist(), 'waveform': reference['waveform'].tolist(),
                'prenet': reference['prenet'].reshape(-1).tolist(), 'prior': reference['prior'].reshape(-1).tolist(),
                'decoder': reference['decoder'].reshape(-1).tolist()}
    (folder / 'testcase.json').write_text(json.dumps(testcase), encoding='utf-8')
    print('testcase:', {k: len(v) for k, v in testcase.items()})


if __name__ == '__main__':
    main()
