"""Voices for MioTTS that do not come from its bundled presets.

MioTTS's LM only sees the text; the voice is the 128-number speaker embedding MioCodec decodes with. The bundled
presets were made from audio that may not be used commercially, so this builds embeddings from other reference
audio and decodes the same speech tokens with each one. References here are Supertonic 3 outputs (its licence
claims no rights in the output), as they are and raised in pitch: synthetic voices, no real speaker.

Reads the speech tokens evaluate_miotts.py saved and the wavs voices_supertonic.py wrote.

    python voices_miotts.py <tts work dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch
from miocodec import MioCodecModel


def main():
    work = Path(sys.argv[1])
    codec = MioCodecModel.from_pretrained(config_path=str(work / 'download/miocodec-v2/config.yaml'),
                                          weights_path=str(work / 'download/miocodec-v2/model.safetensors')).eval()
    rate = int(codec.config.sample_rate)
    rows = json.loads((work / 'out/miotts/0.4B-Q8_0_jp_female.json').read_text(encoding='utf-8'))
    out = work / 'out/miotts-voices'
    out.mkdir(parents=True, exist_ok=True)
    embeddings = {}
    for voice in ['F1', 'F2', 'F3', 'F4', 'F5']:
        for suffix in ['', '_up2', '_up3']:
            # Both lines of the voice, about ten seconds together.
            parts = [soundfile.read(work / 'out/voices' / f'{voice}{suffix}_{i}.wav')[0] for i in (0, 3)]
            wav = torch.tensor(np.concatenate(parts), dtype=torch.float32)
            with torch.no_grad():
                embedding = codec.encode(wav, return_content=False, return_global=True).global_embedding
            embeddings[f'{voice}{suffix}'] = embedding
            torch.save(embedding, out / f'{voice}{suffix}.pt')
            for index in (0, 3, 4):
                tokens = torch.tensor(rows[index]['speech_tokens'], dtype=torch.long)
                with torch.no_grad():
                    speech = codec.decode(global_embedding=embedding, content_token_indices=tokens)
                soundfile.write(out / f'{voice}{suffix}_{index}.wav', speech.squeeze().numpy(), rate)
        print(voice, 'written', flush=True)
    preset = torch.load(work / 'download/MioTTS-Inference/presets/jp_female.pt', map_location='cpu', weights_only=True)
    similarity = {name: round(float(torch.nn.functional.cosine_similarity(e, preset, dim=0)), 3) for name, e in embeddings.items()}
    print('cosine similarity to the bundled jp_female:', json.dumps(similarity))


if __name__ == '__main__':
    main()
