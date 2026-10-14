"""Download the pinned Talkie base quantization and WMDP-Bio shard."""

from __future__ import annotations

import hashlib

from talk_wmd.pilot import download_dataset, download_model


MODEL_SHA256 = "9de40b2cf1c3513ade97a7ecc12a1dd2b471cb5b635d7482a5870d8fbd799bcd"


def verify_sha256(path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch for {path}. Do not use this model file.")


def main() -> None:
    model = download_model()
    verify_sha256(model, MODEL_SHA256)
    dataset = download_dataset()
    print(f"Model verified: {model} ({model.stat().st_size:,} bytes)")
    print(f"Dataset ready: {dataset} ({dataset.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
