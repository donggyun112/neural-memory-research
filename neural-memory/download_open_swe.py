from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from collections import Counter
from pathlib import Path


DATASET = "nvidia/Open-SWE-Traces"
API = "https://datasets-server.huggingface.co"
DEFAULT_LICENSES = ("Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a reproducible, license-filtered Open-SWE trajectory sample"
    )
    parser.add_argument("--output", type=Path, default=Path("local-data/open-swe-v1.jsonl"))
    parser.add_argument("--rows", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--config", default="v1.0")
    parser.add_argument("--split", default="sweagent")
    parser.add_argument("--licenses", nargs="+", default=list(DEFAULT_LICENSES))
    parser.add_argument(
        "--resolved-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _get_json(path: str, query: dict[str, object], retries: int = 8) -> dict[str, object]:
    url = f"{API}/{path}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"User-Agent": "keymem-neural-memory/0.1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise RuntimeError("dataset server returned a non-object payload")
            return payload
        except Exception as error:
            if attempt + 1 == retries:
                raise
            delay = min(30, 2**attempt)
            if isinstance(error, HTTPError) and error.code == 429:
                retry_after = error.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, min(60, int(retry_after)))
            time.sleep(delay)
    raise AssertionError("unreachable")


def _row_count(config: str, split: str) -> int:
    payload = _get_json("size", {"dataset": DATASET})
    size = payload.get("size")
    if not isinstance(size, dict) or not isinstance(size.get("splits"), list):
        raise RuntimeError("dataset size response has no splits")
    for item in size["splits"]:
        if (
            isinstance(item, dict)
            and item.get("config") == config
            and item.get("split") == split
            and isinstance(item.get("num_rows"), int)
        ):
            return item["num_rows"]
    raise RuntimeError(f"unknown Open-SWE split: {config}/{split}")


def main() -> None:
    args = parse_args()
    if args.rows < 1 or not 1 <= args.batch_size <= 100 or args.workers < 1:
        raise ValueError("rows/workers must be positive and batch-size must be in [1, 100]")
    total = _row_count(args.config, args.split)
    rng = random.Random(args.seed)
    allowed = set(args.licenses)
    offsets = list(range(0, total, args.batch_size))
    rng.shuffle(offsets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    licenses: Counter[str] = Counter()
    accepted = 0
    seen: set[int] = set()
    digest = hashlib.sha256()

    if args.resume and temporary.exists():
        with temporary.open(encoding="utf-8") as existing:
            for line in existing:
                record = json.loads(line)
                provenance = record.get("_hf")
                if (
                    not isinstance(provenance, dict)
                    or provenance.get("config") != args.config
                    or provenance.get("split") != args.split
                    or not isinstance(provenance.get("row_idx"), int)
                ):
                    raise RuntimeError("partial download does not match requested split")
                row_idx = provenance["row_idx"]
                seen.add(row_idx)
                license_name = record.get("license")
                if isinstance(license_name, str):
                    licenses[license_name] += 1
                digest.update(line.encode())
                accepted += 1
        processed_offsets = {index - index % args.batch_size for index in seen}
        offsets = [offset for offset in offsets if offset not in processed_offsets]
        print(f"resuming at {accepted}/{args.rows}", flush=True)

    def fetch(offset: int) -> tuple[int, dict[str, object] | None, Exception | None]:
        try:
            payload = _get_json(
                "rows",
                {
                    "dataset": DATASET,
                    "config": args.config,
                    "split": args.split,
                    "offset": offset,
                    "length": min(args.batch_size, total - offset),
                },
            )
        except Exception as error:
            return offset, None, error
        return offset, payload, None

    chunk_size = args.workers * 2
    mode = "a" if args.resume and temporary.exists() else "w"
    with temporary.open(mode, encoding="utf-8") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for start in range(0, len(offsets), chunk_size):
                payloads = executor.map(fetch, offsets[start : start + chunk_size])
                for offset, payload, error in payloads:
                    if error is not None:
                        print(
                            f"skipping offset {offset}: {type(error).__name__}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    assert payload is not None
                    rows = payload.get("rows")
                    if not isinstance(rows, list):
                        continue
                    for wrapped in rows:
                        if not isinstance(wrapped, dict) or wrapped.get("truncated_cells"):
                            continue
                        row_idx = wrapped.get("row_idx")
                        row = wrapped.get("row")
                        if (
                            not isinstance(row_idx, int)
                            or row_idx in seen
                            or not isinstance(row, dict)
                        ):
                            continue
                        license_name = row.get("license")
                        if not isinstance(license_name, str) or license_name not in allowed:
                            continue
                        if args.resolved_only and row.get("resolved") not in (1, True):
                            continue
                        seen.add(row_idx)
                        licenses[license_name] += 1
                        record = dict(row)
                        record["_hf"] = {
                            "dataset": DATASET,
                            "config": args.config,
                            "split": args.split,
                            "row_idx": row_idx,
                        }
                        encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode()
                        digest.update(encoded)
                        output.write(encoded.decode())
                        accepted += 1
                        if accepted % 100 == 0:
                            print(f"downloaded {accepted}/{args.rows}", flush=True)
                        if accepted >= args.rows:
                            break
                    if accepted >= args.rows:
                        break
                if accepted >= args.rows:
                    break
    if accepted < args.rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"only found {accepted} acceptable rows out of {args.rows}")
    os.replace(temporary, args.output)
    manifest = {
        "dataset": DATASET,
        "config": args.config,
        "split": args.split,
        "sample_rows": accepted,
        "source_rows": total,
        "seed": args.seed,
        "resolved_only": args.resolved_only,
        "licenses": dict(sorted(licenses.items())),
        "sha256": digest.hexdigest(),
        "source": f"https://huggingface.co/datasets/{DATASET}",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
