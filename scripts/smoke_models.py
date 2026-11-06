"""Opt-in smoke test for the optional local models (embedding retrieval backend).

    python scripts/smoke_models.py [--allow-download]

Needs `pip install -r requirements-local-models.txt`. If sentence-transformers
is not importable it prints "SKIPPED: ..." and exits 0. By default models load
with local_files_only=True (cache only); --allow-download permits fetching them
from the Hugging Face Hub. Loads one model at a time, runs one encode and one
rerank, and prints device, timings and peak CUDA memory. Not run by pytest.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from support_pipeline.retrieval import CROSS_ENCODER_MODEL, ENCODER_MODEL  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allow-download", action="store_true",
                        help="allow downloading models from the Hugging Face Hub")
    args = parser.parse_args(argv)

    try:
        import torch  # noqa: PLC0415
        from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: PLC0415
    except Exception as exc:  # ImportError, or torch failing to load
        print(f"SKIPPED: sentence-transformers is not usable ({type(exc).__name__}); "
              "pip install -r requirements-local-models.txt")
        return 0

    local_only = not args.allow_download
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"python={sys.version.split()[0]} torch={torch.__version__} device={device} local_files_only={local_only}")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    query = "I deposited by bank card two hours ago but my balance still shows zero."
    docs = ["Card deposits are usually instant, but may be delayed by issuer checks.",
            "If the client does not receive a reset email, check spam folders."]

    try:
        start = time.perf_counter()
        encoder = SentenceTransformer(ENCODER_MODEL, device=device, local_files_only=local_only)
        loaded = time.perf_counter()
        vectors = encoder.encode([query, *docs], normalize_embeddings=True)
        encoded = time.perf_counter()
        print(f"encoder={ENCODER_MODEL} load_s={loaded - start:.2f} encode_s={encoded - loaded:.3f} "
              f"dim={len(vectors[0])}")
        del encoder
        if device == "cuda":
            torch.cuda.empty_cache()

        start = time.perf_counter()
        reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=device, local_files_only=local_only)
        loaded = time.perf_counter()
        scores = reranker.predict([(query, doc) for doc in docs])
        ranked = time.perf_counter()
        print(f"reranker={CROSS_ENCODER_MODEL} load_s={loaded - start:.2f} rerank_s={ranked - loaded:.3f} "
              f"scores={[round(float(s), 3) for s in scores]}")
        del reranker
    except Exception as exc:  # noqa: BLE001
        hint = " (models not cached; rerun with --allow-download)" if local_only else ""
        print(f"FAILED: {type(exc).__name__}: {exc}{hint}")
        return 1

    if device == "cuda":
        print(f"cuda_max_memory_allocated_mb={torch.cuda.max_memory_allocated() / 2**20:.1f}")
        torch.cuda.empty_cache()
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
