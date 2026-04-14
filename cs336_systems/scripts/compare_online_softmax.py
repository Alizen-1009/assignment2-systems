from __future__ import annotations

import argparse
import statistics
import time

import torch

from cs336_systems.online_softmax import online_softmax_v1, online_softmax_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare online softmax implementations against torch.")
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["1024", "4096", "32,1024", "8,128,64"],
        help="Comma-separated tensor shapes to benchmark.",
    )
    parser.add_argument("--dim", type=int, default=-1, help="Dimension along which softmax is computed.")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs.")
    parser.add_argument("--repeats", type=int, default=50, help="Number of measured runs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64", "bfloat16"],
        help="Tensor dtype.",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }

    device = torch.device(args.device)
    dtype = dtype_map[args.dtype]

    print("Comparing weighted-softmax reductions")
    print("Reference: sum(softmax(x, dim) * v, dim)")

    for shape_text in args.shapes:
        shape = tuple(int(part) for part in shape_text.split(","))
        torch.manual_seed(args.seed)
        x = torch.randn(*shape, device=device, dtype=dtype)
        v = torch.randn(*shape, device=device, dtype=dtype)

        print(f"\nshape={shape} dim={args.dim} device={device} dtype={dtype}")

        results: list[tuple[str, torch.Tensor, float, float]] = []
        for name, fn in (
            ("torch_reference", lambda x_, v_, dim_: torch.sum(torch.softmax(x_, dim=dim_) * v_, dim=dim_)),
            ("online_softmax_v1", online_softmax_v1),
            ("online_softmax_v2", online_softmax_v2),
        ):
            for _ in range(args.warmup):
                out = fn(x, v, args.dim) if name == "torch_reference" else fn(x, v, dim=args.dim)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            timings_ms: list[float] = []
            out = fn(x, v, args.dim) if name == "torch_reference" else fn(x, v, dim=args.dim)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            for _ in range(args.repeats):
                start = time.perf_counter()
                out = fn(x, v, args.dim) if name == "torch_reference" else fn(x, v, dim=args.dim)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                end = time.perf_counter()
                timings_ms.append((end - start) * 1000.0)

            mean_ms = statistics.mean(timings_ms)
            std_ms = statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0
            results.append((name, out, mean_ms, std_ms))

        ref_out = results[0][1]
        print(f"{'torch_reference':<18} mean={results[0][2]:>9.4f} ms  std={results[0][3]:>8.4f} ms")

        for name, out, mean_ms, std_ms in results[1:]:
            abs_diff = (out - ref_out).abs()
            max_abs = abs_diff.max().item()
            mean_abs = abs_diff.mean().item()
            print(
                f"{name:<18} mean={mean_ms:>9.4f} ms  std={std_ms:>8.4f} ms  "
                f"max_abs={max_abs:>10.4e}  mean_abs={mean_abs:>10.4e}"
            )


if __name__ == "__main__":
    main()
