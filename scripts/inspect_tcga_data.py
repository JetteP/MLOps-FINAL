import pickle
import json
import pandas as pd
import numpy as np

TCGA_DIR = "tcga_data"


def extract_embeddings_from_value(value):
    """
    Normalize different embedding storage formats to a list of numpy arrays.
    """
    embeddings = []

    if isinstance(value, dict):
        # Common TCGA case: dict of {slide_id: embedding}
        for v in value.values():
            embeddings.append(np.asarray(v))

    elif isinstance(value, list):
        for v in value:
            embeddings.append(np.asarray(v))

    elif isinstance(value, np.ndarray):
        if value.ndim == 1:
            embeddings.append(value)
        else:
            for i in range(value.shape[0]):
                embeddings.append(value[i])

    else:
        raise TypeError(f"Unsupported embedding container type: {type(value)}")

    return embeddings


def main():
    # Load embeddings
    with open(f"{TCGA_DIR}/tcga_titan_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    # Load reports
    reports = []
    with open(f"{TCGA_DIR}/tcga_reports.jsonl") as f:
        for line in f:
            reports.append(json.loads(line))

    # Load labels
    labels = pd.read_csv(f"{TCGA_DIR}/tcga_patient_to_cancer_type.csv")

    print("=== TCGA DATA INSPECTION ===")
    print(f"Patients with embeddings: {len(embeddings)}")
    print(f"Patients with reports: {len(reports)}")
    print(f"Patients with labels: {labels['patient_id'].nunique()}")

    # Inspect embedding structure
    example_pid = next(iter(embeddings))
    example_value = embeddings[example_pid]
    example_embeddings = extract_embeddings_from_value(example_value)

    print(f"Example patient ID: {example_pid}")
    print(f"Number of embeddings for example patient: {len(example_embeddings)}")
    print(f"Embedding dimension: {example_embeddings[0].shape}")

    # Count patients with multiple embeddings
    multi_embedding_patients = 0
    for v in embeddings.values():
        if len(extract_embeddings_from_value(v)) > 1:
            multi_embedding_patients += 1

    print(f"Patients with >1 embedding: {multi_embedding_patients}")

    # Label statistics
    num_classes = labels["cancer_type"].nunique()
    print(f"Number of cancer types: {num_classes}")


if __name__ == "__main__":
    main()
