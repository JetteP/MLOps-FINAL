import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

# 1. Paden instellen [cite: 34]
data_dir = Path("tcga_data")
output_dir = Path("data/splits")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Bestanden laden en inspecteren [cite: 35, 269]
print("--- Data Inspectie ---")
with open(data_dir / "tcga_titan_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Inspecteer de structuur van de eerste patiënt
first_pid = list(embeddings.keys())[0]
first_entry = embeddings[first_pid]

# Bepaal de dimensie (moet 768 zijn) [cite: 48, 224]
if isinstance(first_entry, dict):
    # Als het een dict is, pakken we de eerste embedding om de vorm te checken
    sample_val = list(first_entry.values())[0]
    dim = sample_val.shape[0] if hasattr(sample_val, 'shape') else len(sample_val)
else:
    dim = first_entry.shape[0]

print(f"Patiënt ID voorbeeld: {first_pid}")
print(f"Type data per patiënt: {type(first_entry)}")
print(f"Embedding dimensie: {dim}") [cite: 48]

reports = pd.read_json(data_dir / "tcga_reports.jsonl", lines=True)
labels = pd.read_csv(data_dir / "tcga_patient_to_cancer_type.csv")

# 3. Definieer het Cohort (Intersectie) [cite: 39, 40]
common_pids = set(embeddings.keys()) & set(reports['pid']) & set(labels['pid'])
print(f"Totaal bruikbare patiënten (cohort): {len(common_pids)}") [cite: 48]

# 4. Patient-Aware Split (70/15/15) - NON-NEGOTIABLE [cite: 41, 42, 51, 279, 280]
pids_list = sorted(list(common_pids))
train_pids, temp_pids = train_test_split(pids_list, test_size=0.30, random_state=42)
val_pids, test_pids = train_test_split(temp_pids, test_size=0.50, random_state=42)

# 5. Exporteer JSONL voor VLLM (Stap 1.4) [cite: 45, 46, 85, 301]
def save_split_jsonl(pids, filename):
    split_reports = reports[reports['pid'].isin(pids)]
    # Sla op in de tcga_data map zodat de VLLM scripts ze kunnen vinden 
    split_reports.to_json(data_dir / filename, orient='records', lines=True)
    print(f"Opgeslagen: {filename} met {len(pids)} patiënten")

save_split_jsonl(train_pids, "train.jsonl")
save_split_jsonl(val_pids, "val.jsonl")
save_split_jsonl(test_pids, "test.jsonl")

# Sla metadata op voor de report tabel [cite: 48]
metadata = {
    "total_pids": len(common_pids),
    "dim": dim,
    "train_count": len(train_pids),
    "val_count": len(val_pids),
    "test_count": len(test_pids)
}
with open(output_dir / "splits_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)