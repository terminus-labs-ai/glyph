/// Global illumination intensity
#define GI_INTENSITY 1.5

/// Frame constants updated per frame
cbuffer FrameConstants : register(b0)
{
    float4x4 ViewProjection;
    float3 CameraPosition;
};

Texture2D<float4> AlbedoTex : register(t0);

/// Vertex shader output
struct VSOutput
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
    float3 Normal   : NORMAL;
};

float4 TransformPosition(float3 pos)
{
    return mul(ViewProjection, float4(pos, 1.0));
}

/// Pixel shader entry point — writes to color target
float4 PSMain(VSOutput input) : SV_Target
{
    return AlbedoTex.Sample(DefaultSampler, input.TexCoord);
}

/// Compute shader for screen-space effects
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    /* compute work */
}
