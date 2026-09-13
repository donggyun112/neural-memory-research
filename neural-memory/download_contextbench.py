from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path


DATA_URL = (
    "https://huggingface.co/datasets/Contextbench/ContextBench/resolve/main/"
    "data/full.parquet?download=true"
)
SOURCE_URL = "https://huggingface.co/datasets/Contextbench/ContextBench"
KEEP_COLUMNS = (
    "instance_id",
    "repo",
    "repo_url",
    "language",
    "base_commit",
    "gold_context",
    "problem_statement",
    "source",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize ContextBench")
    parser.add_argument(
        "--output", type=Path, default=Path("local-data/contextbench.jsonl")
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parquet = args.output.with_suffix(".parquet")
    temporary = parquet.with_suffix(".parquet.part")
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "keymem-neural-memory/0.1"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            target.write(chunk)
    os.replace(temporary, parquet)

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("install the data group with: uv sync --group data") from error
    table = pq.read_table(parquet, columns=list(KEEP_COLUMNS))
    output_tmp = args.output.with_suffix(".jsonl.part")
    with output_tmp.open("w", encoding="utf-8") as target:
        for record in table.to_pylist():
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(output_tmp, args.output)
    manifest = {
        "dataset": "Contextbench/ContextBench",
        "config": "default",
        "rows": table.num_rows,
        "parquet_sha256": digest.hexdigest(),
        "source": SOURCE_URL,
        "code_license": "Apache-2.0",
        "note": "embedded source spans remain subject to their repository licenses",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
