using UnityEngine;

namespace ShaderGPT.Models {
// MioCodec's decoder (speech tokens -> waveform) for one fixed speaker, from the weights baked by
// bake_miocodec.py. Lives in the ShaderGPT assembly because AutoModel only finds models there.
//
// Layouts: the transformers work on (time, features); the convolutions on (channels, time).
// The sequence has a fixed length: the caller pads the tokens with config.pad_token (a token that decodes to
// silence) up to the length the graph was built for, so no stage needs a length mask.
[System.Serializable]
public class MioCodecDecoderConfig : PretrainedConfig {
	public int sampling_rate;
	public int n_fft;
	public int hop;
	public int window; // local attention: each frame sees window/2 frames on either side
	public int prenet_layers;
	public int prenet_heads;
	public int decoder_layers;
	public int decoder_heads;
	public int resnet_blocks;
	public int pad_token;
	public int[] upsample_factors;
	public int[] upsample_kernels;
}
[RegisterModel(typeof(MioCodecDecoderConfig), "miocodec_decoder")]
public class MioCodecDecoder : PretrainedModel {
	public MioCodecDecoder(Functional fn, MioCodecDecoderConfig config): base(fn, config) {}
	public new MioCodecDecoderConfig config => (MioCodecDecoderConfig)base.config;
	const float NormEps = 1e-5f, GroupNormEps = 1e-6f;

	// Waveform frames per token, and samples per frame (one row of the output).
	public int frames_per_token {get {
		int frames = 2;
		foreach(var factor in config.upsample_factors)
			frames *= factor;
		return frames;
	}}
	// Row t of the output holds samples t*hop .. t*hop+hop-1; the speech starts (n_fft - hop) / 2 samples in.
	public int leading_samples => (config.n_fft - config.hop) / 2;

	// ids: one row per position with (unused or token, position, sequence length, 0).
	Tensor Transformer(string name, Tensor hidden_states, TensorView ids, int layers, int heads) {
		var head_dim = hidden_states.size1*4 / heads;
		var half = config.window / 2;
		using(~ctx) {
			var position_embeddings = ~nn.Embedding($"{name}.rotary_emb", (ids, 1));
			for(int i=0; i<layers; i++) {
				var path = $"{name}.layers.{i}";
				var normed = nn.LayerNorm($"{path}.attention_norm", hidden_states, NormEps);
				var (q,k,v) = nn.Linear(($"{path}.attention.wq", $"{path}.attention.wk", $"{path}.attention.wv"), normed);
				ctx.Release(normed);
				var (query,key) = fn.Rotary(new TensorView(q.tensor, q.size0, q.size1+k.size1), position_embeddings,
					groups:heads+heads).Split((q.size1, k.size1), axis:1);
				var attention = ~ctx|fn.ScaledDotProductAttention(~query, key, ~v,
					heads:heads, scale:1f/System.MathF.Sqrt(head_dim), stable:true,
					window:new(int.MinValue, 0, 1, (ids, 2), -half, half+1, 1, (ids, 1)));
				hidden_states = ~ctx|nn.Linear($"{path}.attention.wo", ~attention, add:~hidden_states);

				normed = nn.LayerNorm($"{path}.ffn_norm", hidden_states, NormEps);
				var (gate,up) = nn.Linear(($"{path}.feed_forward.w1", $"{path}.feed_forward.w3"), normed);
				ctx.Release(normed);
				hidden_states = ~ctx|nn.Linear($"{path}.feed_forward.w2", ~gate, prolog:("silu", up), add:~hidden_states);
			}
		}
		hidden_states = ~ctx|nn.LayerNorm($"{name}.norm", ~hidden_states, NormEps);
		if(state_dict.ContainsKey($"{name}.output_proj.weight"))
			hidden_states = ~ctx|nn.Linear($"{name}.output_proj", ~hidden_states);
		return hidden_states;
	}
	// Releases its input.
	Tensor ResNetBlock(string path, Tensor input, int groups) {
		var hidden_states = fn.GroupNorm(input, groups, state_dict[$"{path}.norm1.weight"], state_dict[$"{path}.norm1.bias"], GroupNormEps);
		hidden_states = ~ctx|fn.Act2Fn("silu", ~hidden_states);
		hidden_states = ~ctx|nn.Conv1d($"{path}.conv1", ~hidden_states, 3);
		hidden_states = ~ctx|fn.GroupNorm(~hidden_states, groups, state_dict[$"{path}.norm2.weight"], state_dict[$"{path}.norm2.bias"], GroupNormEps);
		hidden_states = ~ctx|fn.Act2Fn("silu", ~hidden_states);
		hidden_states = ~ctx|nn.Conv1d($"{path}.conv2", ~hidden_states, 3);
		return ~ctx|fn.Add(~input, ~hidden_states);
	}
	// x + sin(alpha x)^2 / beta, with alpha and 1/sqrt(beta) per channel (weight_alpha, weight_beta). Releases its input.
	// channelsFirst: the channel axis is the rows (convolution layout) instead of the columns.
	Tensor SnakeBeta(string path, Tensor input, bool channelsFirst) {
		var alpha = state_dict[$"{path}.weight_alpha"];
		var beta = state_dict[$"{path}.weight_beta"];
		var scaled = channelsFirst
			? ~ctx|fn.Mul(input, ~fn.Transpose(alpha, size0:input.size0, expand:true))
			: fn.Mul(input, alpha);
		var sine = channelsFirst
			? ~ctx|fn.Act2Fn("sin", ~scaled, mul:~fn.Transpose(beta, size0:input.size0, expand:true))
			: ~ctx|fn.Act2Fn("sin", ~scaled, mul:beta);
		return ~ctx|fn.AddCMul(~input, ~sine, sine);
	}

	// input_ids: (tokens, 1) with (token, position, tokens, 0). frame_ids: (2*tokens, 1) with (0, position, 2*tokens, 0).
	// Returns (frames, ceil(hop/4)) : one row of `hop` samples per frame.
	// probe, if given, sees the tensors after the stages the PyTorch reference records.
	public Tensor Decode(TensorView input_ids, TensorView frame_ids, System.Action<string,Tensor> probe=null) {
		var hidden_states = nn.Embedding("embed_tokens", (input_ids, 0));
		hidden_states = Transformer("wave_prenet", hidden_states, input_ids, config.prenet_layers, config.prenet_heads);
		probe?.Invoke("prenet", hidden_states);

		hidden_states = ~ctx|fn.Transpose(~hidden_states);
		hidden_states = ~ctx|nn.ConvTranspose1d("wave_conv_upsample", ~hidden_states, 2, stride:2);
		for(int i=0; i<config.resnet_blocks; i++)
			hidden_states = ResNetBlock($"wave_prior_net.blocks.{i}", hidden_states, 32);
		probe?.Invoke("prior", hidden_states);

		hidden_states = ~ctx|fn.Transpose(~hidden_states);
		hidden_states = Transformer("wave_decoder", hidden_states, frame_ids, config.decoder_layers, config.decoder_heads);
		hidden_states = ~ctx|fn.Transpose(~hidden_states);
		probe?.Invoke("decoder", hidden_states);

		for(int i=0; i<config.resnet_blocks; i++)
			hidden_states = ResNetBlock($"wave_post_net.blocks.{i}", hidden_states, 32);
		for(int i=0; i<config.upsample_factors.Length; i++) {
			hidden_states = ~ctx|nn.ConvTranspose1d($"wave_upsampler.upsample_layers.{i}", ~hidden_states,
				config.upsample_kernels[i], stride:config.upsample_factors[i]);
			hidden_states = SnakeBeta($"wave_upsampler.snake_activations.{i}", hidden_states, channelsFirst:true);
			hidden_states = ResNetBlock($"wave_upsampler.resnet_blocks.{i}", hidden_states, Mathf.Min(32, hidden_states.size0));
		}
		hidden_states = ~ctx|fn.Transpose(~hidden_states);
		hidden_states = ~ctx|nn.Linear("wave_upsampler.out_proj", ~hidden_states);
		hidden_states = SnakeBeta("wave_upsampler.out_snake", hidden_states, channelsFirst:false);

		// Head: [log magnitude | phase] per frame, then the inverse FFT with the window as two matrices.
		hidden_states = ~ctx|nn.Linear("istft_head.out", ~hidden_states);
		var (log_magnitude, phase) = hidden_states.Chunk2(axis:1);
		var magnitude = fn.Exp(log_magnitude);
		magnitude = ~ctx|fn.Clamp(~magnitude, max:100f);
		var real = fn.Act2Fn("cos", phase, mul:magnitude);
		var imag = ~ctx|fn.Act2Fn("sin", phase, mul:~magnitude);
		ctx.Release(hidden_states);
		var segments = ~ctx|nn.Linear("istft.cos", ~real);
		segments = ~ctx|nn.Linear("istft.sin", ~imag, add:~segments);
		// Overlap-add: each frame's n_fft samples are n_fft/hop blocks; the fold kernel sums block i of frame t-i.
		segments = ~ctx|fn.Transpose(~segments);
		var waveform = ~ctx|nn.Conv1d("istft.fold", ~segments, 7);
		return ~ctx|fn.Transpose(~waveform);
	}
}
}
