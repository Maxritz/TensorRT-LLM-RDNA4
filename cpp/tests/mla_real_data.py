#!/usr/bin/env python3
# Generate real-valued DeepSeek-V2-Lite MLA Q/KV test vectors (TEST 12 "decode" and
# TEST 13 "prefill") from a real DeepSeek GGUF, for the Vulkan MLA FMHA test in
# cpp/tests/vulkanTraceMain.cpp.
#
# Layout produced (matches loadRealMlaData in vulkanTraceMain.cpp):
#   magic            4 bytes  b"MLA1"
#   batchSize        u32
#   maxPages         u32
#   seqQLen          u32
#   numHeads         u32
#   pageSize         u32
#   D                u32            (192 = dLatent(128) + dRope(64))
#   Q  : batchSize*seqQLen*numHeads*D  float32
#   KV : maxPages*pageSize*D            float32
#
# Q/KV are NOT synthetic RNG: they come from a real matmul of a real DeepSeek token
# embedding row through real dequantized attention projection weights, then arranged
# into the kernel's MLA layout:
#   - Q[b,h] slice of (x_b @ attn_q.weight.T), 16 heads x 192 = 3072-dim.
#   - KV token row = concat( kv_a_mqa latent[:128] , kv_a_mqa latent[512:576](rope) )
#     which is the real DeepSeek MLA absorbed KV form (128 latent + 64 rope = 192).
# This stresses the kernel's exp/scale path with genuine weight statistics instead of
# uniform{-1,1}, while keeping kernel-vs-CPU-ref equivalence identical.
#
# Dependencies: numpy (required) + the `gguf` package (pip install gguf), which
# provides the spec-correct, verified dequantizer for Q4_K_M / Q8_0 / Q6_K. The
# generated cpp/tests/mla_real*.bin artifacts are committed; regeneration is only
# needed to refresh the real-weight vectors (e.g. against a different model layer).
import sys, os, struct
import numpy as np
from gguf import GGUFReader, dequantize

def load_tensor(gguf_path, name):
    """Return a dequantized float32 numpy array for tensor `name` (uses gguf.dequantize)."""
    r = GGUFReader(gguf_path, "r")
    for t in r.tensors:
        if t.name == name:
            deq = dequantize(np.asarray(t.data, dtype=np.uint8), t.tensor_type)
            return np.asarray(deq, dtype=np.float32).reshape([int(d) for d in t.shape]).copy()
    raise SystemExit(f"tensor not found: {name}")

def make_kv_row(full576):
    # full576 = x @ kv_a_mqa.weight.T  (576 = 512 latent + 64 rope).
    # DeepSeek MLA "absorbed" KV: 128 nope-latent + 64 rope = 192.
    return np.concatenate([full576[:128], full576[512:576]]).astype(np.float32)

def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else \
        r"E:\OLLAMA-Models\GGUF\DeepSeek-Coder-V2-Lite-Instruct.Q4_K_M.gguf"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else r"cpp\tests"
    os.makedirs(out_dir, exist_ok=True)

    emb = load_tensor(gguf_path, "token_embd.weight")        # [2048, 102400]
    blk = 0
    wq  = load_tensor(gguf_path, f"blk.{blk}.attn_q.weight")        # [2048, 3072]
    wkv = load_tensor(gguf_path, f"blk.{blk}.attn_kv_a_mqa.weight")  # [2048, 576]
    print(f"loaded: emb={emb.shape} wq={wq.shape} wkv={wkv.shape}")

    numHeads, dLatent, dRope = 16, 128, 64
    D = dLatent + dRope             # 192
    pageSize = 64

    # ---- TEST 12 (decode): batchSize=2, seqQLen=1, maxPages=2 ----
    b1, maxPages1, seqQ1 = 2, 2, 1
    hostQ1 = np.empty((b1, seqQ1, numHeads, D), dtype=np.float32)
    tid0, tid1 = 2000 % emb.shape[1], 7127 % emb.shape[1]
    for b, tid in enumerate((tid0, tid1)):
        x = emb[:, tid]            # [2048]  real token embedding row
        q = (x.astype(np.float32) @ wq.astype(np.float32)).reshape(numHeads, D)
        hostQ1[b, 0] = q
    hostKv1 = np.zeros((maxPages1, pageSize, D), dtype=np.float32)
    kv_len_per_page = (50, 60)
    base_tid = 3000
    for pg, kvlen in enumerate(kv_len_per_page):
        for t in range(kvlen):
            tid = (base_tid + pg * pageSize + t) % emb.shape[1]
            x = emb[:, tid]
            full = x.astype(np.float32) @ wkv.astype(np.float32)   # [576]
            hostKv1[pg, t] = make_kv_row(full)
    write_bin(os.path.join(out_dir, "mla_real.bin"),
              b1, maxPages1, seqQ1, numHeads, pageSize, D, hostQ1, hostKv1)

    # ---- TEST 13 (prefill): batchSize=2, seqQLen=16, maxPages=4 ----
    b2, maxPages2, seqQ2 = 2, 4, 16
    hostQ2 = np.empty((b2, seqQ2, numHeads, D), dtype=np.float32)
    for b in range(b2):
        for s in range(seqQ2):
            tid = (4000 + b * 100 + s) % emb.shape[1]
            x = emb[:, tid]
            q = (x.astype(np.float32) @ wq.astype(np.float32)).reshape(numHeads, D)
            hostQ2[b, s] = q
    hostKv2 = np.zeros((maxPages2, pageSize, D), dtype=np.float32)
    # b0 cold (cacheSeqs=0, page0); b1 has 80 prior tokens -> pages 1,2 used.
    base_tid = 5000
    for pg in range(maxPages2):
        for t in range(pageSize):
            tid = (base_tid + pg * pageSize + t) % emb.shape[1]
            x = emb[:, tid]
            hostKv2[pg, t] = make_kv_row(x.astype(np.float32) @ wkv.astype(np.float32))
    write_bin(os.path.join(out_dir, "mla_real_prefill.bin"),
              b2, maxPages2, seqQ2, numHeads, pageSize, D, hostQ2, hostKv2)

    print(f"wrote {os.path.join(out_dir,'mla_real.bin')} (Q={hostQ1.nbytes} KV={hostKv1.nbytes})")
    print(f"wrote {os.path.join(out_dir,'mla_real_prefill.bin')} (Q={hostQ2.nbytes} KV={hostKv2.nbytes})")
    for nm, q, kv in (("decode", hostQ1, hostKv1), ("prefill", hostQ2, hostKv2)):
        print(f"  {nm}: Q[{q.shape}] mean={q.mean():.4f} std={q.std():.4f} "
              f"min={q.min():.4f} max={q.max():.4f}")
        print(f"  {nm}: KV nonzero={np.count_nonzero(kv)} / {kv.size}")

def write_bin(path, batchSize, maxPages, seqQLen, numHeads, pageSize, D, q, kv):
    with open(path, "wb") as f:
        f.write(b"MLA1")
        f.write(struct.pack("<6I", batchSize, maxPages, seqQLen, numHeads, pageSize, D))
        q.astype(np.float32).tofile(f)
        kv.astype(np.float32).tofile(f)

if __name__ == "__main__":
    main()
