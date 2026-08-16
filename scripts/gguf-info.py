#!/usr/bin/env python3
"""Read a GGUF file's metadata header and tensor index without loading the model.

Answers two questions the server does not print at its default verbosity:

  1. What are the architecture parameters that drive the KV cache?
     (head_count_kv, key_length, value_length, block_count)
  2. How many blocks actually carry a KV cache?

Question 2 matters for hybrid models. Qwen3.8-27B is one: only every fourth
block has attn_k/attn_v; the rest are linear-attention/SSM blocks whose state
does not grow with the context. Assuming every block stores KV overestimates
the cache by 4x here. See EXPERIMENTS.md, experiment 6.

Usage:  python3 gguf-info.py /path/to/model.gguf
        ssh <host> 'python3 - /path/to/model.gguf' < scripts/gguf-info.py

Standard library only, by design: this is a shared machine and reading a header
is not worth a pip install.
"""

import struct
import sys

# GGUF value type tags, from ggml/src/gguf.cpp.
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)

_FIXED = {
    U8: ("<B", 1), I8: ("<b", 1),
    U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4),
    BOOL: ("<?", 1),
    U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
}

# Arrays longer than this are summarised rather than materialised: the token
# vocabulary is a single array of ~150k strings and printing it is useless.
ARRAY_PRINT_LIMIT = 16


class Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError(f"wanted {n} bytes, got {len(b)}")
        return b

    def scalar(self, tag):
        fmt, size = _FIXED[tag]
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self):
        return self.raw(self.scalar(U64)).decode("utf-8", "replace")

    def value(self, tag):
        if tag == STRING:
            return self.string()
        if tag == ARRAY:
            elem = self.scalar(U32)
            count = self.scalar(U64)
            if count > ARRAY_PRINT_LIMIT:
                # Skip it. Fixed-width elements can be seeked past; strings have
                # to be walked, since each carries its own length.
                if elem in _FIXED:
                    self.fh.seek(_FIXED[elem][1] * count, 1)
                elif elem == STRING:
                    for _ in range(count):
                        self.fh.seek(self.scalar(U64), 1)
                else:
                    raise ValueError(f"cannot skip array of type {elem}")
                return f"<{count} items, type {elem}, skipped>"
            return [self.value(elem) for _ in range(count)]
        return self.scalar(tag)


def parse(path):
    with open(path, "rb") as fh:
        r = Reader(fh)
        if r.raw(4) != b"GGUF":
            sys.exit(f"{path}: not a GGUF file (bad magic)")
        version = r.scalar(U32)
        n_tensors = r.scalar(U64)
        n_kv = r.scalar(U64)

        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.scalar(U32))

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.scalar(U32)
            dims = [r.scalar(U64) for _ in range(n_dims)]
            ttype = r.scalar(U32)
            offset = r.scalar(U64)
            tensors.append((name, dims, ttype, offset))

    return version, meta, tensors


def block_index(name):
    """'blk.37.attn_k.weight' -> 37, anything else -> None."""
    parts = name.split(".")
    if len(parts) > 2 and parts[0] == "blk" and parts[1].isdigit():
        return int(parts[1])
    return None


def suffix(name):
    """'blk.37.attn_k.weight' -> 'attn_k'."""
    return ".".join(name.split(".")[2:]).removesuffix(".weight")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: gguf-info.py <model.gguf>")
    path = sys.argv[1]
    version, meta, tensors = parse(path)

    arch = meta.get("general.architecture", "?")
    print(f"file            {path}")
    print(f"gguf version    {version}")
    print(f"tensors         {len(tensors)}")
    print(f"metadata keys   {len(meta)}")
    print()

    print("--- metadata ---")
    for key, val in meta.items():
        if key.startswith(("general.", f"{arch}.")) and not key.startswith(
            ("general.quantization", "tokenizer.")
        ):
            print(f"  {key:<44} {val}")
    print()

    # --- which blocks carry a KV cache ---
    blocks = {}
    for name, *_ in tensors:
        idx = block_index(name)
        if idx is not None:
            blocks.setdefault(idx, set()).add(suffix(name))

    with_kv = sorted(i for i, t in blocks.items() if "attn_k" in t and "attn_v" in t)
    mtp = sorted(i for i, t in blocks.items() if any(s.startswith("nextn") for s in t))
    without = sorted(set(blocks) - set(with_kv) - set(mtp))

    print("--- blocks ---")
    print(f"  total                    {len(blocks)}")
    print(f"  with attn_k/attn_v (KV)  {len(with_kv)}  {with_kv}")
    print(f"  without KV (linear/SSM)  {len(without)}")
    print(f"  MTP heads (nextn.*)      {len(mtp)}  {mtp}")
    print()

    for label, idx in (("first without KV", without), ("first with KV", with_kv)):
        if idx:
            i = idx[0]
            print(f"  blk.{i} ({label}): {', '.join(sorted(blocks[i]))}")
    print()

    # --- KV cache arithmetic ---
    def m(key, default=None):
        return meta.get(f"{arch}.{key}", default)

    k_len, v_len = m("attention.key_length"), m("attention.value_length")
    n_kv_heads = m("attention.head_count_kv")
    if None in (k_len, v_len, n_kv_heads) or not with_kv:
        print("--- KV cache ---\n  not enough metadata to compute")
        return

    if isinstance(n_kv_heads, list):          # per-layer arrays exist in some models
        n_kv_heads = max(n_kv_heads)

    print("--- KV cache per token ---")
    print(f"  (k_len + v_len) x head_count_kv = ({k_len} + {v_len}) x {n_kv_heads}"
          f" = {(k_len + v_len) * n_kv_heads} values/token/layer")
    for label, bytes_per in (("f16", 2.0), ("q8_0", 34 / 32), ("q4_0", 18 / 32)):
        per_tok = (k_len + v_len) * n_kv_heads * bytes_per * len(with_kv)
        print(f"  {label:<5} {per_tok / 1024:8.1f} KiB/token", end="")
        for ctx in (32768, 65536, 131072):
            print(f"   {ctx // 1024}k: {per_tok * ctx / 1024**2:8.0f} MiB", end="")
        print()


if __name__ == "__main__":
    main()
