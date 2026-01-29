import json
from pathlib import Path


def load_pids(path: str) -> set[str]:
    data = json.loads(Path(path).read_text())

    if isinstance(data, list):
        return set(map(str, data))

    if isinstance(data, dict):
        for k in ["pids", "patient_ids", "patients"]:
            if k in data:
                return set(map(str, data[k]))

    raise ValueError(f"Unknown split format in {path}")


def main():
    train = load_pids("data/splits/train.json")
    val = load_pids("data/splits/val.json")
    test = load_pids("data/splits/test.json")

    print("Split sizes (patients):")
    print("  train:", len(train))
    print("  val  :", len(val))
    print("  test :", len(test))

    print("\nOverlaps (must be 0):")
    print("  train ∩ val :", len(train & val))
    print("  train ∩ test:", len(train & test))
    print("  val ∩ test  :", len(val & test))


if __name__ == "__main__":
    main()
