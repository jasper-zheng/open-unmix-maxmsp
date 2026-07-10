"""Export the causal streaming Open-Unmix separator to ``neural.live~`` .pte+.json.

Builds a :class:`openunmix.export_stream.StreamingUMX` from the pretrained
``umx-realtime`` cores and lowers it with the ``neural_tilde`` ``LiveModule``
exporter for one or more ``(buffer_size, delegate)`` combinations.

Usage (from the repo root, in the SA3 export venv)::

    python export/export_umx_live.py \
        --model models/umx-realtime --out-dir export/artifacts \
        --buffer-sizes 1024 2048 4096 --delegates mlx xnnpack

Each combination writes ``umx_live_<delegate>_<B>.pte`` (+ ``.json``).  A fresh
model instance is built per export so the exporter's warm-up primes/zeroes the
lazy streaming caches (STFT pad, LSTM h/c, ISTFT tails) cleanly.
"""
import argparse
import os
import sys

import cached_conv as cc

# openunmix is an editable install; ensure repo root on path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openunmix.export_stream import build_streaming_umx  # noqa: E402


def export_one(model_dir: str, out_dir: str, buffer_size: int, delegate: str) -> str:
    cc.use_cached_conv(True)
    model = build_streaming_umx(model_dir)
    model.register_all()                      # test_method=False (see StreamingUMX)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"umx_live_{delegate}_{buffer_size}")
    # strict=False: LSTM export traces more reliably in non-strict mode.
    path = model.export_to_pte(
        stem, delegate=delegate, buffer_size=buffer_size, batch=1,
        strict=False, warmup=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/umx-realtime")
    ap.add_argument("--out-dir", default="export/artifacts")
    ap.add_argument("--buffer-sizes", type=int, nargs="+", default=[2048])
    ap.add_argument("--delegates", nargs="+", default=["mlx"])
    args = ap.parse_args()

    results = {}
    for B in args.buffer_sizes:
        for delegate in args.delegates:
            key = f"{delegate}_{B}"
            print(f"\n{'='*70}\n=== EXPORT {key}  (buffer_size={B}, delegate={delegate})\n{'='*70}")
            try:
                path = export_one(args.model, args.out_dir, B, delegate)
                sz = os.path.getsize(path) / 1e6
                results[key] = f"OK  {path}  ({sz:.0f} MB)"
                print(f"--- {key}: wrote {path} ({sz:.0f} MB)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                results[key] = f"FAIL  {type(e).__name__}: {e}"
                print(f"--- {key}: FAILED — {type(e).__name__}: {e}")

    print(f"\n{'='*70}\n=== SUMMARY\n{'='*70}")
    for k, v in results.items():
        print(f"  {k:16s} {v}")
    if any(v.startswith("FAIL") for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
