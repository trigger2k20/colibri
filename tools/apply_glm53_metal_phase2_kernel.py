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

# Separate shader: GLM-5.3 clamps gate only above +limit and clamps up symmetrically.
s = once(
    s,
    'kernel void moe_silu(device float* g [[buffer(0)]], device const float* u [[buffer(1)]],\n'
    '                     uint i [[thread_position_in_grid]]) { float v=g[i]; g[i]=(v/(1.0f+exp(-v)))*u[i]; }\n',
    'kernel void moe_silu(device float* g [[buffer(0)]], device const float* u [[buffer(1)]],\n'
    '                     uint i [[thread_position_in_grid]]) { float v=g[i]; g[i]=(v/(1.0f+exp(-v)))*u[i]; }\n'
    'kernel void moe_silu_clamped(device float* g [[buffer(0)]], device const float* u [[buffer(1)]],\n'
    '                             constant float& limit [[buffer(2)]],\n'
    '                             uint i [[thread_position_in_grid]]) {\n'
    '  float v=min(g[i], limit);\n'
    '  float uv=clamp(u[i], -limit, limit);\n'
    '  g[i]=(v/(1.0f+exp(-v)))*uv;\n'
    '}\n',
    "clamped shader",
)

s = once(
    s,
    'static id<MTLComputePipelineState> g_gemv, g_moe_gemv, g_moe_silu, g_moe_fwht;\n',
    'static id<MTLComputePipelineState> g_gemv, g_moe_gemv, g_moe_silu, g_moe_silu_clamped, g_moe_fwht;\n',
    "clamped pipeline declaration",
)

s = once(
    s,
    '    g_moe_silu = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"moe_silu"] error:&err];\n'
    '    g_moe_fwht = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"moe_fwht"] error:&err];\n'
    '    auto P=[&](const char*n){ return [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(n)] error:&err]; };\n',
    '    g_moe_silu = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"moe_silu"] error:&err];\n'
    '    g_moe_fwht = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"moe_fwht"] error:&err];\n'
    '    auto P=[&](const char*n){ return [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(n)] error:&err]; };\n'
    '    g_moe_silu_clamped=P("moe_silu_clamped");\n',
    "clamped pipeline init",
)

s = once(
    s,
    '    if (!g_gemv || !g_moe_gemv || !g_moe_silu || !g_moe_fwht || !g_a_rms || !g_a_rope || !g_a_copy ||\n',
    '    if (!g_gemv || !g_moe_gemv || !g_moe_silu || !g_moe_silu_clamped || !g_moe_fwht || !g_a_rms || !g_a_rope || !g_a_copy ||\n',
    "clamped pipeline validation",
)

# Generalize the existing batched submitter with an optional clamped activation.
s = once(
    s,
    'static id<MTLCommandBuffer> moe_submit(int nb, int D, int Iinter, int fmt, int qgs,\n',
    'static id<MTLCommandBuffer> moe_submit_impl(int nb, int D, int Iinter, int fmt, int qgs,\n'
    '                         int clamped, float swiglu_limit,\n',
    "moe submit signature",
)

s = once(
    s,
    '  [e setComputePipelineState:g_moe_silu];\n'
    '  [e setBuffer:gg_buf offset:0 atIndex:0];[e setBuffer:uu_buf offset:0 atIndex:1];\n'
    '  [e dispatchThreads:MTLSizeMake((size_t)R*Iinter,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];\n',
    '  [e setComputePipelineState:clamped ? g_moe_silu_clamped : g_moe_silu];\n'
    '  [e setBuffer:gg_buf offset:0 atIndex:0];[e setBuffer:uu_buf offset:0 atIndex:1];\n'
    '  if (clamped) [e setBytes:&swiglu_limit length:sizeof(float) atIndex:2];\n'
    '  [e dispatchThreads:MTLSizeMake((size_t)R*Iinter,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];\n',
    "moe activation dispatch",
)

# Keep the old API semantically identical by passing clamped=0.
s = s.replace(
    'moe_submit(nb,D,Iinter,fmt,qgs,g,u,d,gs,us,ds,xg,xoff,nr,R,',
    'moe_submit_impl(nb,D,Iinter,fmt,qgs,0,0.0f,g,u,d,gs,us,ds,xg,xoff,nr,R,',
)

# Add GLM-5.3 sync entry point next to the existing sync API.  Async follows after correctness.
needle = '''// Async two-phase API: begin submits the block (own scratch, no wait) so the CPU can\n// overlap disk loads with GPU compute; end waits + scatters. Handle owns everything.\n'''
insert = '''extern "C" int coli_metal_moe_block_clamped(int nb, int D, int Iinter, int fmt, int qgs,\n                         const void *const *g, const void *const *u, const void *const *d,\n                         const float *const *gs, const float *const *us, const float *const *ds,\n                         const float *xg, const int *xoff, const int *nr,\n                         const int *rows, const float *rw, float *out, int S,\n                         float swiglu_limit) {\n  (void)S;\n  if (!(swiglu_limit > 0.0f)) return 0;\n  @autoreleasepool {\n    int R = 0; for (int e=0;e<nb;e++) R += nr[e];\n    if (R == 0) return 1;\n    g_xg = ensure(g_xg,&g_xg_cap,(size_t)R*D*4);\n    g_gg = ensure(g_gg,&g_gg_cap,(size_t)R*Iinter*4);\n    g_uu = ensure(g_uu,&g_uu_cap,(size_t)R*Iinter*4);\n    g_hh = ensure(g_hh,&g_hh_cap,(size_t)R*D*4);\n    id<MTLCommandBuffer> cb = moe_submit_impl(nb,D,Iinter,fmt,qgs,1,swiglu_limit,\n        g,u,d,gs,us,ds,xg,xoff,nr,R,g_xg,g_gg,g_uu,g_hh);\n    if (!cb) return 0;\n    return moe_finish(cb,g_hh,nb,R,D,rows,rw,out);\n  }\n}\n\n'''+needle
s = once(s, needle, insert, "clamped sync API")

# Header declaration; deliberately separate from plain SwiGLU API.
hneedle = '''int coli_metal_moe_block(int nb, int D, int Iinter, int fmt, int qgs,\n                         const void *const *g, const void *const *u, const void *const *d,\n                         const float *const *gs, const float *const *us, const float *const *ds,\n                         const float *xg, const int *xoff, const int *nr,\n                         const int *rows, const float *rw,\n                         float *out, int S);\n'''
hinsert = hneedle + '''\n/* GLM-5.3 routed experts: same batched path, but with its clamped SwiGLU semantics.\n * gate is upper-clamped to +limit; up is clamped to [-limit,+limit]. */\nint coli_metal_moe_block_clamped(int nb, int D, int Iinter, int fmt, int qgs,\n                         const void *const *g, const void *const *u, const void *const *d,\n                         const float *const *gs, const float *const *us, const float *const *ds,\n                         const float *xg, const int *xoff, const int *nr,\n                         const int *rows, const float *rw,\n                         float *out, int S, float swiglu_limit);\n'''
h = once(h, hneedle, hinsert, "clamped header API")

HDR.write_text(h)
SRC.write_text(s)
print("Applied GLM53 Metal phase 2A: separate clamped-SwiGLU batched MoE kernel/API")
