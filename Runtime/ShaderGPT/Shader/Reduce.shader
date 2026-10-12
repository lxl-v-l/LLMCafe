Shader "GPT/Reduce" {
Properties {
	DimO("DimO", Vector) = (1, 1, 0, 0)
	DimX("DimX", Vector) = (1, 1, 0, 0)
	DimWin("DimWin", Vector) = (0, 0, 0, 0)
	[HideInInspector]
	TexO("TexO", 2D) = "black" {}
	TexX("TexX", 2D) = "black" {}
	TexWin("TexWin", 2D) = "black" {}

	DimHead("DimHead", Vector) = (0, 0, 0, 0)
	Win    ("Win",     Vector) = (0, 1000000000, 0, 0)
	Win2   ("Win2",    Vector) = (0, 1000000000, 0, 0)
	Swizzle0("Swizzle0", Vector) = (1, 0, 0, 0)
	Swizzle1("Swizzle1", Vector) = (0, 1, 0, 0)
	Swizzle2("Swizzle2", Vector) = (0, 0, 1, 0)
	Swizzle3("Swizzle3", Vector) = (0, 0, 0, 1)
	Bias("Bias", Float) = 0
	Seed("Seed", Float) = 0
	DstBlend("DstBlend", Int) = 0
	InstCount("InstCount", Int) = 1
}
SubShader {
HLSLINCLUDE
#include "UnityCG.cginc"
#include "Common.hlsl"
#include "Function.hlsl"

DECLARE_OUTPUT(O); // output
DECLARE_TENSOR(X); // input
DECLARE_TENSOR(Win);
uniform uint3 DimHead; // xy: group size, z: index mod
uniform float4 Win;
uniform float4 Win2;
uniform float4 Swizzle0;
uniform float4 Swizzle1;
uniform float4 Swizzle2;
uniform float4 Swizzle3;
uniform float Bias;
uniform float Seed;
uniform uint DstBlend;
uniform uint InstCount;

static const float POS_INF = asfloat(0x7F800000);
static const float NEG_INF = asfloat(0xFF800000);
float masked_min4(float4 v, bool4 m) {
	v = m ? v : POS_INF;
	return min(min(v.x, v.y), min(v.z, v.w));
}
float masked_max4(float4 v, bool4 m) {
	v = m ? v : NEG_INF;
	return max(max(v.x, v.y), max(v.z, v.w));
}
void update_minmax(inout float4 O, float4 X) {
	O[2] = X[0] < O[0] ? X[2] : X[0] > O[0] ? O[2] : min(X[2], O[2]);
	O[3] = X[1] > O[1] ? X[3] : X[1] < O[1] ? O[3] : min(X[3], O[3]);
	O[0] = min(X[0], O[0]);
	O[1] = max(X[1], O[1]);
}

float4 main(uint2 pos, uint threadId, uint groupSize, float4 screenPos) {
	// torch.sum, torch.aminmax
	// O[i,j][e] = torch.sum(pow(X[i,j*K:j*K+K][:], e))
	// O[i,j][e] = torch.sum(exp(X[i,j*K:j*K+K][:], SUMEXP2_BIAS[e]))
	// O[i,j].xz = torch.min(X[i,j*K:j*K+K][:])
	// O[i,j].yw = torch.max(X[i,j*K:j*K+K][:])

	uint I = DimHead.x;
	uint J = DimHead.y;
	int2 range = ceil(GetRange(SampleWin(uint2(min(DimWin.x-1, pos.x), 0)), Win, Win2));
	#if defined(REDUCE_SUMPOW) || defined(REDUCE_SUMEXP)
		float4 O = 0;
	#elif defined(REDUCE_MINMAX)
		float4 O = float4(POS_INF, NEG_INF, POS_INF, POS_INF);
	#endif
	// clamp window & out-of-bound read on X
	uint stopK = I*min(J, (uint)max(0, min(int(DimX.y - pos.y*J), (((int)range.y+3)>>2) - int(pos.y*J%DimHead.z))));
	for(uint k=threadId; k<stopK; k+=groupSize) {
		uint2 coord = uint2(pos.x*I + k%I, pos.y*J + k/I);
		float4 X = SampleX(coord);
		#if !defined(REDUCE_CONTINUE)
			int4 index = (pos.y*J%DimHead.z + k/I)*4 + uint4(0,1,2,3);
			bool4 mask = range.x <= index && index < range.y;
			#if defined(REDUCE_SUMPOW)
				X = float4(dot(1, mask), dot(1, mask?X:0), dot(1, mask?X*X:0), dot(1, mask?X*X*X:0));
			#elif defined(REDUCE_SUMEXP)
				// use four overlapping windows of length 256 to cover as much exponent range as possible
				// overlap size should >= 24(f32 mantissa) + 24(max count) = 48 to avoid precision loss
				const float4 SUMEXP2_BIAS = float4(3,1,-1,-3)/2*(256-48);
				X = mask ? X / log(2) : -32768;
				X = float4( // naive exp2(X + SUMEXP2_BIAS[i]) would lose precision when bias is large
					dot(1, min(exp2(X - trunc(X)), 2) * exp2(trunc(X) + SUMEXP2_BIAS[0])),
					dot(1, min(exp2(X - trunc(X)), 2) * exp2(trunc(X) + SUMEXP2_BIAS[1])),
					dot(1, min(exp2(X - trunc(X)), 2) * exp2(trunc(X) + SUMEXP2_BIAS[2])),
					dot(1, min(exp2(X - trunc(X)), 2) * exp2(trunc(X) + SUMEXP2_BIAS[3])));
			#elif defined(REDUCE_MINMAX)
				[branch] if(Bias != 0) // torch.distributions.gumbel
					// Seed (cafe NPC patch): the generator sets it to the token position. Time alone repeats the same
					// noise for every step run within one frame, which biases sampling toward the same tokens.
					X += Bias * rand_gumbel(uint4(coord, asuint(_Time.y) + (uint)Seed, asuint(_SinTime.w)));
				float Xmin = masked_min4(X, mask);
				float Xmax = masked_max4(X, mask);
				X = float4(Xmin, Xmax, masked_min4(index, X == Xmin && mask), masked_min4(index, X == Xmax && mask));
			#endif
		#endif
		#if defined(REDUCE_SUMPOW) || defined(REDUCE_SUMEXP)
			O += X;
		#elif defined(REDUCE_MINMAX)
			update_minmax(O, X);
		#endif
	}
#if defined(REDUCE_MINMAX)
	if(DimO.w != 0) {
		bool2 lane = uint2(screenPos.xy) & 1;
		update_minmax(O, asfloat(QuadReadAcrossX(asuint(O), lane)));
		update_minmax(O, asfloat(QuadReadAcrossY(asuint(O), lane)));
	}
#else
	O = OutputO(screenPos, O);
#endif
	return float4(
		dot(Swizzle0, Swizzle0 == 0 ? 0 : O),
		dot(Swizzle1, Swizzle1 == 0 ? 0 : O),
		dot(Swizzle2, Swizzle2 == 0 ? 0 : O),
		dot(Swizzle3, Swizzle3 == 0 ? 0 : O));
}
float4 vert(float2 uv : TEXCOORD0) : SV_Position {
	return ClipO(uv);
}
[maxvertexcount(3)]
#if defined(REDUCE_SUMPOW) || defined(REDUCE_SUMEXP)
[instance(32)]
#endif
void geom(triangle float4 input[3] : SV_Position, inout TriangleStream<GeomOutput> stream, uint instId : SV_GSInstanceID) {
	GeomInstance(input, stream, instId, InstCount);
}
float4 frag(float4 screenPos : SV_Position) : SV_Target {
	uint4 pos = GetIndexO(screenPos, InstCount);
	float4 o = main(pos.xy, pos.z, pos.w, screenPos);
	if(DstBlend > 1)
		o.w = screenPos.z ? 1 : 0;
	return o; // OutputO is merged into main
}
ENDHLSL
	Tags { "PreviewType"="Plane" } // prevent freezing Unity editor
	Pass {
		Cull Off
		Blend One [DstBlend], One [DstBlend]
HLSLPROGRAM
#pragma target 5.0
#pragma vertex vert
#pragma geometry geom
#pragma fragment frag
#pragma shader_feature_local _ REDUCE_CONTINUE
#pragma shader_feature_local REDUCE_SUMPOW REDUCE_SUMEXP REDUCE_MINMAX
ENDHLSL
	}
}
}