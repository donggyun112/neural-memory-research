from __future__ import annotations

import argparse

from neural_memory.text_encoders import encode_with_gemma


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the frozen Gemma feature path")
    parser.add_argument("--model", default="google/gemma-3-270m")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--pooling", choices=("last", "mean"), default="last")
    args = parser.parse_args()
    features = encode_with_gemma(
        ("fix parser", "def parse(): pass"),
        model_name=args.model,
        device=args.device,
        batch_size=2,
        max_length=args.max_length,
        pooling=args.pooling,
    )
    print(
        f"shape={tuple(features.shape)},"
        f"norms={[round(value, 6) for value in features.norm(dim=-1).tolist()]}"
    )


if __name__ == "__main__":
    main()
