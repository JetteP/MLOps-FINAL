import json
import pickle
import random
import pandas as pd
from pathlib import Path

TCGA_DIR = "tcga_data"
OUTPUT_DIR = "data/splits"
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def main():
    random.seed(SEED)

    # Load embeddings (keys = patient IDs)
    with open(f"{TCGA_DIR}/tcga_titan_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    embedding_pids = set(embeddings.keys())

    # Load reports (pid field)
    report_pids = set()  # <-- DIT ONTBRAK
    with open(f"{TCGA_DIR}/tcga_reports.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            report_pids.add(obj["pid"])

    # Load labels (patient_id field)
    labels = pd.read_csv(f"{TCGA_DIR}/tcga_patient_to_cancer_type.csv")
    label_pids = set(labels["patient_id"].unique())

    # Intersection (final cohort)
    usable_pids = sorted(list(embedding_pids & report_pids & label_pids))
    print(f"Usable patients after intersection: {len(usable_pids)}")

    # Patient-aware train/val/test split
    random.shuffle(usable_pids)

    n_total = len(usable_pids)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)

    train_pids = usable_pids[:n_train]
    val_pids = usable_pids[n_train : n_train + n_val]
    test_pids = usable_pids[n_train + n_val :]

    print(f"Train patients: {len(train_pids)}")
    print(f"Val patients:   {len(val_pids)}")
    print(f"Test patients:  {len(test_pids)}")

    # Save splits
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for name, split in [
        ("train", train_pids),
        ("val", val_pids),
        ("test", test_pids),
    ]:
        with open(f"{OUTPUT_DIR}/{name}.json", "w") as f:
            json.dump(split, f, indent=2)


if __name__ == "__main__":
    main()
