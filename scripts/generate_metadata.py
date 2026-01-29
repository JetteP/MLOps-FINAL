import json
from pathlib import Path

import pandas as pd

# 1. Laad de labels
labels_df = pd.read_csv("tcga_data/tcga_patient_to_cancer_type.csv")

# 2. Maak een alfabetische mapping (zodat index 0 altijd hetzelfde is)
unique_types = sorted(labels_df["cancer_type"].unique())
label_map = {name: i for i, name in enumerate(unique_types)}

# 3. Opslaan
output_dir = Path("data/metadata")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "label_map.json", "w") as f:
    json.dump(label_map, f, indent=4)

print(f"Stable label map gegenereerd voor {len(label_map)} klassen.")
