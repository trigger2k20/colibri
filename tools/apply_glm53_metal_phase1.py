#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "c" / "glm53.c"
MAKE = ROOT / "c" / "Makefile"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {n}")
    return text.replace(old, new, 1)


src = SRC.read_text()
make = MAKE.read_text()

src = replace_once(
    src,
    '#include "tok.h"\n#ifdef COLI_VULKAN\n#include "backend_vulkan.h"\nstatic int g_vk_ready = 0;\n#endif\n',
    '#include "tok.h"\n#ifdef COLI_METAL\n#include "backend_metal.h"\nstatic int g_metal_ready = 0;\n#endif\n#ifdef COLI_VULKAN\n#include "backend_vulkan.h"\nstatic int g_vk_ready = 0;\n#endif\n',
    "Metal include",
)

src = replace_once(
    src,
    '    int rows, columns, gs;\n    void *vk;                             /* ColiVkTensor*, caricata alla prima uso */\n} Mat;\n',
    '    int rows, columns, gs;\n    int resident;                         /* eligible for persistent accelerator wrapping */\n    void *metal;                          /* ColiMetalTensor*, created lazily */\n    void *vk;                             /* ColiVkTensor*, caricata alla prima uso */\n} Mat;\n',
    "Mat accelerator fields",
)

src = replace_once(
    src,
    '    Mat mat; memset(&mat, 0, sizeof(mat));\n    mat.rows = rows; mat.columns = columns;\n',
    '    Mat mat; memset(&mat, 0, sizeof(mat));\n    mat.rows = rows; mat.columns = columns; mat.resident = 1;\n',
    "quantize_loaded resident flag",
)

src = replace_once(
    src,
    '        mat.fmt = 4; mat.q4 = packed; mat.s = step; mat.gs = 64;\n        return mat;\n',
    '        mat.fmt = 4; mat.q4 = packed; mat.s = step; mat.gs = 64; mat.resident = 1;\n        return mat;\n',
    "load_mat int4 resident flag",
)

old_mv = '''static void mv(float *out, const Mat *w, const float *x) {\n#ifdef COLI_VULKAN\n    if (g_vk_ready && (w->fmt == 1 || w->fmt == 4)) {\n        Mat *mutable_w = (Mat *)w;\n        if (coli_vk_matmul((ColiVkTensor **)&mutable_w->vk, out, x,\n                           w->fmt == 4 ? (const void *)w->q4 : (const void *)w->q8,\n                           w->s, w->fmt, 1, w->columns, w->rows, w->gs))\n            return;\n    }\n#endif\n'''
new_mv = '''static void mv(float *out, const Mat *w, const float *x) {\n#ifdef COLI_METAL\n    if (g_metal_ready && w->resident && (w->fmt == 1 || w->fmt == 4)) {\n        Mat *mutable_w = (Mat *)w;\n        if (coli_metal_matmul((ColiMetalTensor **)&mutable_w->metal, out, x,\n                              w->fmt == 4 ? (const void *)w->q4 : (const void *)w->q8,\n                              w->s, w->fmt, 1, w->columns, w->rows, w->gs))\n            return;\n    }\n#endif\n#ifdef COLI_VULKAN\n    if (g_vk_ready && (w->fmt == 1 || w->fmt == 4)) {\n        Mat *mutable_w = (Mat *)w;\n        if (coli_vk_matmul((ColiVkTensor **)&mutable_w->vk, out, x,\n                           w->fmt == 4 ? (const void *)w->q4 : (const void *)w->q8,\n                           w->s, w->fmt, 1, w->columns, w->rows, w->gs))\n            return;\n    }\n#endif\n'''
src = replace_once(src, old_mv, new_mv, "mv Metal dispatch")

src = replace_once(
    src,
    '    vision_load(m);\n#ifdef COLI_VULKAN\n',
    '    vision_load(m);\n#ifdef COLI_METAL\n    /* Metal is runtime opt-in.  Failure is non-fatal: the existing CPU path\n     * remains authoritative and streamed routed experts are deliberately\n     * excluded from this first integration step. */\n    if (getenv("COLI_METAL") && atoi(getenv("COLI_METAL"))) {\n        g_metal_ready = coli_metal_init() && coli_metal_available();\n        fprintf(stderr, g_metal_ready\n                ? "Metal: attivo sulle matrici residenti\\n"\n                : "Metal: nessun device utilizzabile, resto su CPU\\n");\n    }\n#endif\n#ifdef COLI_VULKAN\n',
    "Metal initialization",
)

src = replace_once(
    src,
    'static void mat_release(Mat *mat) {\n    free((void *)mat->f); free((void *)mat->q8);\n',
    'static void mat_release(Mat *mat) {\n#ifdef COLI_METAL\n    if (mat->metal) coli_metal_tensor_free((ColiMetalTensor *)mat->metal);\n#endif\n    free((void *)mat->f); free((void *)mat->q8);\n',
    "Metal handle release",
)

old_rule = '''# GLM-5.3-Flash: CPU puro, esperti in streaming dal contenitore int4 gs64.\n# Nessun oggetto CUDA/Vulkan/Metal fra le dipendenze perche' il motore non ne\n# ha ancora: quando li avra', questa riga somigliera' a quella di kimi_k3.\nglm53$(EXE): glm53.c cli_args.h st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h quant.h hyper_connections.h delta_attention.h sparse_index.h vision_tower.h backend_vulkan.h edge_adapter_internal.h edge_adapters.h edge_runtime.h segment_adapter_internal.h segment_adapters.h segment_runtime.h $(VK_OBJ) $(VK_SPV)\n\t$(CC) $(CFLAGS) glm53.c $(VK_OBJ) -o glm53$(EXE) $(LDFLAGS)\n'''
new_rule = '''# GLM-5.3-Flash: routed experts still stream from the int4-gs64 container.\n# METAL=1 accelerates resident matrices only in the first integration step;\n# routed MoE stays on the CPU until its clamped-SwiGLU path is implemented.\nglm53$(EXE): glm53.c cli_args.h st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h quant.h hyper_connections.h delta_attention.h sparse_index.h vision_tower.h backend_metal.h backend_vulkan.h edge_adapter_internal.h edge_adapters.h edge_runtime.h segment_adapter_internal.h segment_adapters.h segment_runtime.h $(METAL_OBJ) $(VK_OBJ) $(VK_SPV)\n\t$(CC) $(CFLAGS) glm53.c $(METAL_OBJ) $(VK_OBJ) -o glm53$(EXE) $(LDFLAGS)\n'''
make = replace_once(make, old_rule, new_rule, "glm53 Makefile target")

SRC.write_text(src)
MAKE.write_text(make)
print("Applied GLM53 Metal phase 1: resident matmul dispatch + build wiring")
