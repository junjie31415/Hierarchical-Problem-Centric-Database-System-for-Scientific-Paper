import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch weak_methods/run_weak_methods.py from the project root."
    )
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "80")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "weak_methods/outputs/default"),
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to weak_methods/run_weak_methods.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(project_root / "weak_methods" / ".cache" / "matplotlib"),
    )
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "weak_methods/run_weak_methods.py",
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--output-dir",
        args.output_dir,
        *args.extra_args,
    ]
    subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
