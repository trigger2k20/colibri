#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "c" / "glm53.c"


def once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {n}")
    return text.replace(old, new, 1)


s = SRC.read_text()

s = once(
    s,
    '''#ifdef COLI_METAL
#include "backend_metal.h"
static int g_metal_ready = 0;
#endif
''',
    '''#ifdef COLI_METAL
#include "backend_metal.h"
static int g_metal_ready = 0;
/* Runtime proof for the routed-expert path. These counters intentionally count
 * cache-sized MoE blocks, not individual matmuls: one successful block means
 * gate/up/clamped-SwiGLU/down/scatter all completed on Metal. */
static uint64_t g_metal_moe_attempt = 0;
static uint64_t g_metal_moe_ok = 0;
static uint64_t g_metal_moe_fallback = 0;
static uint64_t g_metal_moe_rows = 0;
#endif
''',
    "Metal MoE counters",
)

s = once(
    s,
    '''#ifdef COLI_METAL
        if (g_metal_ready) {
            const int max_rows = tokens * topk;
''',
    '''#ifdef COLI_METAL
        if (g_metal_ready) {
            g_metal_moe_attempt++;
            const int max_rows = tokens * topk;
''',
    "Metal MoE attempt counter",
)

s = once(
    s,
    '''                metal_done = coli_metal_moe_block_clamped(
                    here, c->hidden, c->moe_inter, 4, 64,
                    mg, mu, md, mgs, mus, mds, xg, xoff, nr, rows, rw,
                    out, tokens, c->swiglu_limit);
            }
            free(xg); free(rw); free(rows); free(nr); free(xoff);
''',
    '''                metal_done = coli_metal_moe_block_clamped(
                    here, c->hidden, c->moe_inter, 4, 64,
                    mg, mu, md, mgs, mus, mds, xg, xoff, nr, rows, rw,
                    out, tokens, c->swiglu_limit);
                if (metal_done) {
                    g_metal_moe_ok++;
                    g_metal_moe_rows += (uint64_t)R;
                }
            }
            if (!metal_done) g_metal_moe_fallback++;
            free(xg); free(rw); free(rows); free(nr); free(xoff);
''',
    "Metal MoE result counters",
)

s = once(
    s,
    '''    if (model.streaming)
        printf("experts hits %ld miss %ld bytes %llu\\n",
               model.hits, model.miss, (unsigned long long)model.ebytes);
    free(logits);
''',
    '''    if (model.streaming)
        printf("experts hits %ld miss %ld bytes %llu\\n",
               model.hits, model.miss, (unsigned long long)model.ebytes);
#ifdef COLI_METAL
    if (g_metal_ready && getenv("GLM53_VERBOSE") && atoi(getenv("GLM53_VERBOSE")))
        printf("metal moe attempts %llu ok %llu fallback %llu rows %llu\\n",
               (unsigned long long)g_metal_moe_attempt,
               (unsigned long long)g_metal_moe_ok,
               (unsigned long long)g_metal_moe_fallback,
               (unsigned long long)g_metal_moe_rows);
#endif
    free(logits);
''',
    "Metal MoE summary",
)

SRC.write_text(s)
print("Applied GLM53 Metal phase 2C.1: routed MoE runtime counters")
