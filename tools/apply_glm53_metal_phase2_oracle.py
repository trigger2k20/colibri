#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HDR = ROOT / "c" / "backend_metal.h"
SRC = ROOT / "c" / "backend_metal.mm"


def once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {n}")
    return text.replace(old, new, 1)

h = HDR.read_text()
s = SRC.read_text()

# A tiny standalone hook used only by the numerical oracle. It exercises the exact
# clamped pipeline used by the batched GLM-5.3 routed-expert path, without needing a
# 195 GB model or expert slabs.
needle = '''extern "C" int coli_metal_silu_mul(float *g, const float *u, size_t n) {\n'''
insert = '''extern "C" int coli_metal_silu_mul_clamped(float *g, const float *u, size_t n, float limit) {\n  if (!g_dev || n == 0 || !(limit > 0.0f)) return n == 0 ? 1 : 0;\n  @autoreleasepool {\n    id<MTLCommandBuffer> cb = [g_queue commandBuffer];\n    id<MTLBuffer> gb = cpubuf(g_dev, g, n*4);\n    id<MTLBuffer> ub = cpubuf(g_dev, u, n*4);\n    if (!gb || !ub) return 0;\n    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];\n    [e setComputePipelineState:g_moe_silu_clamped];\n    [e setBuffer:gb offset:0 atIndex:0]; [e setBuffer:ub offset:0 atIndex:1];\n    [e setBytes:&limit length:sizeof(float) atIndex:2];\n    [e dispatchThreads:MTLSizeMake((uint32_t)n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];\n    [e endEncoding];\n    { id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder]; [bl synchronizeResource:gb]; [bl endEncoding]; }\n    [cb commit]; [cb waitUntilCompleted];\n    if (cb.status == MTLCommandBufferStatusError) return 0;\n    memcpy(g, gb.contents, n*4);\n    return 1;\n  }\n}\n\n''' + needle
s = once(s, needle, insert, "clamped standalone hook")

hneedle = '''int coli_metal_silu_mul(float *g, const float *u, size_t n);\n'''
hinsert = hneedle + '''/* Numerical-oracle hook for GLM-5.3 clamped SwiGLU. */\nint coli_metal_silu_mul_clamped(float *g, const float *u, size_t n, float limit);\n'''
h = once(h, hneedle, hinsert, "clamped standalone declaration")

HDR.write_text(h)
SRC.write_text(s)
print("Applied GLM53 Metal phase 2B: clamped-SwiGLU numerical oracle hook")
