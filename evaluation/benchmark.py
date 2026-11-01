from pathlib import Path

from _benchmark import reproduce

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    result = reproduce(ROOT / "benchmark", ROOT / "results/benchmark")
    print(
        f"Evaluated {result['counts']['lanes']} models on "
        f"{result['counts']['dataset_images']} images."
    )
    print(f"Results: {ROOT / 'results/benchmark'}")
