"""MioCodec's decoder (speech tokens -> waveform) as a plain set of weights for one fixed speaker.

The world plays one voice, so everything that depends on the speaker embedding is computed here:
- AdaLN-Zero (speaker -> scale, shift, gate) becomes an ordinary LayerNorm weight and bias, and the gate is
  folded into the projection that follows it (attention wo, feed-forward w2).
- The token decoder (finite scalar quantizer + projection) becomes one embedding table of 12800 rows.
- weight_norm on the transposed convolutions is folded into their weights.
- Rotary embeddings: MioCodec pairs neighbouring features (2i, 2i+1); ShaderGPT pairs the two halves of a head
  (i, i + d/2). The rows of wq and wk are reordered so the same rotation applies; q.k is unchanged by it.
- SnakeBeta's alpha and beta are stored as exp(alpha) and 1/sqrt(exp(beta)).

Writes the baked state dict (for export_miocodec_textures.py, run in the converter's env) and a reference:
the tokens of one line with the tensors after each stage, computed by the baked weights in plain PyTorch, and a
check that this reproduces MioCodec's own decode.

    python bake_miocodec.py <tts work dir> <voice embedding .pt> <output dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from miocodec import MioCodecModel

SPEECH_BASE = 64400
PAD_TOKEN = 4021


def rotary_order(dim, heads):
    head = dim // heads
    within = torch.cat([torch.arange(0, head, 2), torch.arange(1, head, 2)])
    return (torch.arange(heads)[:, None] * head + within[None]).reshape(-1)


def rotary_table(head_dim, positions, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    angles = torch.outer(torch.arange(positions).float(), freqs)
    return torch.cat([angles.cos(), angles.sin()], dim=-1)  # (positions, head_dim): cos half, sin half


def bake(codec, speaker):
    state = {k: v.float() for k, v in codec.state_dict().items()}
    out = {}
    out['embed_tokens.weight'] = codec.decode_token_indices(torch.arange(12800)).float()

    def transformer(name, module, heads, positions):
        dim = state[f'{name}.layers.0.attention.wq.weight'].shape[0]
        order = rotary_order(dim, heads)
        condition = F.silu(speaker)
        for i in range(len(module.layers)):
            p = f'{name}.layers.{i}'
            gates = {}
            for norm in ('attention_norm', 'ffn_norm'):
                if module.use_adaln_zero:
                    shift, scale, gate = F.linear(condition, state[f'{p}.{norm}.condition_proj.1.weight'],
                                                  state[f'{p}.{norm}.condition_proj.1.bias']).chunk(3)
                    out[f'{p}.{norm}.weight'], out[f'{p}.{norm}.bias'], gates[norm] = 1 + scale, shift, gate
                else:
                    out[f'{p}.{norm}.weight'], out[f'{p}.{norm}.bias'] = state[f'{p}.{norm}.weight'], state[f'{p}.{norm}.bias']
            out[f'{p}.attention.wq.weight'] = state[f'{p}.attention.wq.weight'][order]
            out[f'{p}.attention.wk.weight'] = state[f'{p}.attention.wk.weight'][order]
            out[f'{p}.attention.wv.weight'] = state[f'{p}.attention.wv.weight']
            wo, w2 = state[f'{p}.attention.wo.weight'], state[f'{p}.feed_forward.w2.weight']
            out[f'{p}.attention.wo.weight'] = wo * gates['attention_norm'][:, None] if gates else wo
            out[f'{p}.feed_forward.w2.weight'] = w2 * gates['ffn_norm'][:, None] if gates else w2
            out[f'{p}.feed_forward.w1.weight'] = state[f'{p}.feed_forward.w1.weight']
            out[f'{p}.feed_forward.w3.weight'] = state[f'{p}.feed_forward.w3.weight']
        if module.use_adaln_zero:
            shift, scale = F.linear(condition, state[f'{name}.norm.condition_proj.1.weight'], state[f'{name}.norm.condition_proj.1.bias']).chunk(2)
            out[f'{name}.norm.weight'], out[f'{name}.norm.bias'] = 1 + scale, shift
        else:
            out[f'{name}.norm.weight'], out[f'{name}.norm.bias'] = state[f'{name}.norm.weight'], state[f'{name}.norm.bias']
        if f'{name}.output_proj.weight' in state:
            out[f'{name}.output_proj.weight'], out[f'{name}.output_proj.bias'] = state[f'{name}.output_proj.weight'], state[f'{name}.output_proj.bias']
        out[f'{name}.rotary_emb.weight'] = rotary_table(dim // heads, positions, module.rope_theta)

    transformer('wave_prenet', codec.wave_prenet, codec.wave_prenet.n_heads, 512)
    transformer('wave_decoder', codec.wave_decoder, codec.wave_decoder.n_heads, 1024)
    for name in ('wave_conv_upsample.weight', 'wave_conv_upsample.bias', 'wave_upsampler.out_proj.weight', 'wave_upsampler.out_proj.bias'):
        out[name] = state[name]
    out.update(istft(state['istft_head.out.weight'], state['istft_head.out.bias'], int(codec.config.n_fft), int(codec.config.hop_length)))
    for name, value in state.items():
        if name.startswith(('wave_prior_net.', 'wave_post_net.', 'wave_upsampler.resnet_blocks.')):
            out[name] = value
    up = codec.wave_upsampler
    for i, layer in enumerate(up.upsample_layers):
        torch.nn.utils.parametrize.remove_parametrizations(layer, 'weight')
        out[f'wave_upsampler.upsample_layers.{i}.weight'] = layer.weight.detach().float()
        out[f'wave_upsampler.upsample_layers.{i}.bias'] = layer.bias.detach().float()
    snakes = [(f'wave_upsampler.snake_activations.{i}', s) for i, s in enumerate(up.snake_activations)] + [('wave_upsampler.out_snake', up.out_snake)]
    for name, snake in snakes:
        out[f'{name}.weight_alpha'] = snake.alpha.detach().float().exp()
        out[f'{name}.weight_beta'] = (1.0 / (snake.beta.detach().float().exp() + 1e-9)).sqrt()
    return {k: v.detach().contiguous() for k, v in out.items()}


def istft(weight, bias, n_fft, hop):
    """The head and the inverse STFT as three weights.

    out: the head's rows regrouped as [magnitude bins, zeros][phase bins, zeros], each half a multiple of four wide,
         because ShaderGPT stores four values per texel and a split must fall on a texel boundary.
    cos / sin: the inverse real FFT times the Hann window and 1 / (sum of squared windows), per frame.
    fold: overlap-add as a convolution. A frame's n_fft samples are n_fft / hop blocks of `hop`; output block t is
          block i of frame t - i, summed over i. Kernel 7 with padding 3 reaches frames t - 3 .. t.
    """
    bins = n_fft // 2 + 1
    half = -(-bins // 4) * 4
    head_weight, head_bias = torch.zeros(2 * half, weight.shape[1]), torch.zeros(2 * half)
    head_weight[:bins], head_weight[half:half + bins] = weight[:bins], weight[bins:]
    head_bias[:bins], head_bias[half:half + bins] = bias[:bins], bias[bins:]
    n, k = torch.arange(n_fft)[:, None].double(), torch.arange(bins)[None].double()
    scale = torch.full((bins,), 2.0 / n_fft, dtype=torch.float64)
    scale[0] = scale[-1] = 1.0 / n_fft
    window = torch.hann_window(n_fft, dtype=torch.float64)[:, None]
    envelope = n_fft / hop * 3 / 8  # sum of squared Hann windows at hop = n_fft / 4, away from the ends
    cos, sin = torch.zeros(n_fft, half), torch.zeros(n_fft, half)
    cos[:, :bins] = (torch.cos(2 * torch.pi * k * n / n_fft) * scale * window / envelope).float()
    sin[:, :bins] = (-torch.sin(2 * torch.pi * k * n / n_fft) * scale * window / envelope).float()
    blocks = n_fft // hop
    fold = torch.zeros(hop, n_fft, 7)
    for i in range(blocks):
        fold[torch.arange(hop), i * hop + torch.arange(hop), 3 - i] = 1
    return {'istft_head.out.weight': head_weight, 'istft_head.out.bias': head_bias,
            'istft.cos.weight': cos, 'istft.sin.weight': sin, 'istft.fold.weight': fold}


def decode(w, config, tokens, stages=None):
    """The decoder in plain PyTorch from the baked weights. Layout: (time, features) for the transformers, (channels, time) for convolutions."""
    def keep(name, value):
        if stages is not None:
            stages[name] = value.detach().numpy().astype(np.float32)

    def transformer(name, x, layers, heads, window):
        length, dim = x.shape
        head = dim // heads
        table = w[f'{name}.rotary_emb.weight'][:length]
        cos, sin = table[:, None, :head // 2], table[:, None, head // 2:]

        def rotate(v):
            v = v.view(length, heads, head)
            a, b = v[..., :head // 2], v[..., head // 2:]
            return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)

        position = torch.arange(length)
        mask = (position[:, None] - position[None, :]).abs() <= window // 2
        for i in range(layers):
            p = f'{name}.layers.{i}'
            h = F.layer_norm(x, (dim,), w[f'{p}.attention_norm.weight'], w[f'{p}.attention_norm.bias'], 1e-5)
            q, k = rotate(h @ w[f'{p}.attention.wq.weight'].T), rotate(h @ w[f'{p}.attention.wk.weight'].T)
            v = (h @ w[f'{p}.attention.wv.weight'].T).view(length, heads, head)
            scores = torch.einsum('thd,shd->hts', q, k) / head ** 0.5
            attention = scores.masked_fill(~mask[None], float('-inf')).softmax(-1)
            x = x + torch.einsum('hts,shd->thd', attention, v).reshape(length, dim) @ w[f'{p}.attention.wo.weight'].T
            h = F.layer_norm(x, (dim,), w[f'{p}.ffn_norm.weight'], w[f'{p}.ffn_norm.bias'], 1e-5)
            x = x + (F.silu(h @ w[f'{p}.feed_forward.w1.weight'].T) * (h @ w[f'{p}.feed_forward.w3.weight'].T)) @ w[f'{p}.feed_forward.w2.weight'].T
        x = F.layer_norm(x, (dim,), w[f'{name}.norm.weight'], w[f'{name}.norm.bias'], 1e-5)
        if f'{name}.output_proj.weight' in w:
            x = x @ w[f'{name}.output_proj.weight'].T + w[f'{name}.output_proj.bias']
        return x

    def resnet(p, x, groups):
        h = F.conv1d(F.silu(F.group_norm(x[None], groups, w[f'{p}.norm1.weight'], w[f'{p}.norm1.bias'], 1e-6)), w[f'{p}.conv1.weight'], w[f'{p}.conv1.bias'], padding=1)
        h = F.conv1d(F.silu(F.group_norm(h, groups, w[f'{p}.norm2.weight'], w[f'{p}.norm2.bias'], 1e-6)), w[f'{p}.conv2.weight'], w[f'{p}.conv2.bias'], padding=1)
        return x + h[0]

    def snake(p, x):
        return x + (torch.sin(x * w[f'{p}.weight_alpha'][:, None]) * w[f'{p}.weight_beta'][:, None]) ** 2

    x = w['embed_tokens.weight'][tokens]
    x = transformer('wave_prenet', x, config['prenet_layers'], config['prenet_heads'], config['window'])
    keep('prenet', x)
    x = F.conv_transpose1d(x.T[None], w['wave_conv_upsample.weight'], w['wave_conv_upsample.bias'], stride=2)[0]  # (512, 2T)
    for i in range(config['resnet_blocks']):
        x = resnet(f'wave_prior_net.blocks.{i}', x, 32)
    keep('prior', x)
    x = transformer('wave_decoder', x.T, config['decoder_layers'], config['decoder_heads'], config['window']).T
    keep('decoder', x)
    for i in range(config['resnet_blocks']):
        x = resnet(f'wave_post_net.blocks.{i}', x, 32)
    for i, (factor, kernel) in enumerate(zip(config['upsample_factors'], config['upsample_kernels'])):
        p = f'wave_upsampler.upsample_layers.{i}'
        x = F.conv_transpose1d(x[None], w[f'{p}.weight'], w[f'{p}.bias'], stride=factor, padding=(kernel - factor) // 2)[0]
        x = snake(f'wave_upsampler.snake_activations.{i}', x)
        x = resnet(f'wave_upsampler.resnet_blocks.{i}', x, min(32, x.shape[0]))
    x = snake('wave_upsampler.out_snake', (x.T @ w['wave_upsampler.out_proj.weight'].T + w['wave_upsampler.out_proj.bias']).T)
    keep('upsampled', x)
    n_fft, hop = config['n_fft'], config['hop']
    x = x.T @ w['istft_head.out.weight'].T + w['istft_head.out.bias']  # (frames, magnitude half | phase half)
    half = x.shape[1] // 2
    magnitude, phase = x[:, :half].exp().clamp(max=100), x[:, half:]
    segments = (magnitude * phase.cos()) @ w['istft.cos.weight'].T + (magnitude * phase.sin()) @ w['istft.sin.weight'].T  # (frames, n_fft)
    keep('segments', segments)
    blocks = F.conv1d(segments.T[None], w['istft.fold.weight'], padding=3)[0]  # (hop, frames)
    # Sample m of the flattened blocks is sample m - (n_fft - hop) / 2 of MioCodec's output ("same" padding).
    return blocks.T.reshape(-1)


def main():
    work, voice, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    codec = MioCodecModel.from_pretrained(config_path=str(work / 'download/miocodec-v2/config.yaml'),
                                          weights_path=str(work / 'download/miocodec-v2/model.safetensors')).eval()
    speaker = torch.load(voice, map_location='cpu', weights_only=True).float()
    rows = json.loads((work / 'out/unity-sample-1step.json').read_text(encoding='utf-8'))
    tokens = [t - SPEECH_BASE for t in rows[2]['tokens'] if SPEECH_BASE <= t < SPEECH_BASE + 12800]
    # The graph has a fixed length: pad with the token the LM emits during silence (pad_with_silence.py).
    tokens = torch.tensor(tokens + [PAD_TOKEN] * (-len(tokens) % 32))
    with torch.no_grad():
        original = codec.decode(global_embedding=speaker, content_token_indices=tokens).squeeze().float()
        stft_length = codec._calculate_target_stft_length(codec._calculate_original_audio_length(len(tokens)))
        weights = bake(codec, speaker)
        config = {'model_type': 'miocodec_decoder', 'vocab_size': 12800, 'pad_token': PAD_TOKEN, 'sampling_rate': int(codec.config.sample_rate),
                  'n_fft': int(codec.config.n_fft), 'hop': int(codec.config.hop_length), 'window': 65,
                  'prenet_layers': len(codec.wave_prenet.layers), 'prenet_heads': codec.wave_prenet.n_heads,
                  'decoder_layers': len(codec.wave_decoder.layers), 'decoder_heads': codec.wave_decoder.n_heads,
                  'resnet_blocks': len(codec.wave_prior_net.blocks),
                  'upsample_factors': list(codec.wave_upsampler.upsample_factors), 'upsample_kernels': list(codec.wave_upsampler.kernel_sizes)}
        stages = {}
        baked = decode(weights, config, tokens, stages)
    print(f'{len(tokens)} tokens; frames before upsampling: MioCodec {stft_length}, twice the tokens {2 * len(tokens)}')
    print(f'samples: MioCodec {len(original)}, baked {len(baked)}')
    pad = (config['n_fft'] - config['hop']) // 2
    inner = slice(4 * config['hop'], len(baked) - 4 * config['hop'])  # the first and last frames have a partial window sum
    difference = (original[inner.start - pad:inner.stop - pad] - baked[inner]).abs().max()
    print(f'baked vs MioCodec away from the ends: max abs difference {float(difference):.2e} (signal peak {float(original.abs().max()):.3f})')
    torch.save(weights, out / 'baked.pt')
    (out / 'config.json').write_text(json.dumps(config, indent=1), encoding='utf-8')
    np.savez(out / 'reference.npz', tokens=tokens.numpy(), waveform=baked.numpy().astype(np.float32), **stages)
    print({k: v.shape for k, v in stages.items()}, 'weights', sum(v.numel() for v in weights.values()))


if __name__ == '__main__':
    main()
