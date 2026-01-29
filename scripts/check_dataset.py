import sys
from pathlib import Path

from torch.utils.data import DataLoader

# soms vindt python je eigen projectmodules niet automatisch
# daarom voegen we hier de huidige werkmap toe aan sys.path
sys.path.append(str(Path.cwd()))

# noqa: E402 zorgt dat ruff niet zeurt dat deze import niet bovenaan staat
from src.data.tcga_dataset import TCGADataset  # noqa: E402


def main():
    # dit script is een snelle sanity check
    print("Loading TCGA Dataset...")
    
    dataset = TCGADataset(
        split_json="data/splits/train.json",
        embeddings_path="tcga_data/tcga_titan_embeddings.pkl",
        labels_path="tcga_data/tcga_patient_to_cancer_type.csv",
        label_map_path="data/label_map.json"
    )

    print("Dataset loaded successfully!")
    print(f"Total Patients (Train Split): {len(dataset)}")
    print(f"Number of Classes: {len(dataset.label_map)}")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # we pakken één batch om te checken of alles klopt
    x_img, x_txt, y, pids = next(iter(dataloader))

    print("\nBatch Information:")
    # image features horen hier [batch, 768] te zijn
    print(f"Feature shape (Image): {x_img.shape} (Expected: [32, 768])")
    print(f"Feature shape (Text):  {x_txt.shape} (Expected: [32, 768])")
    print(f"Labels shape:          {y.shape}     (Expected: [32])")
    
    print("\nClass Mapping (first 5):")
    for i, (k, v) in enumerate(dataset.label_map.items()):
        if i >= 5:
            break  # Ruff wil dit op een nieuwe regel
        print(f"  {k}: {v}")

    print("\n--- 1.3 Verification is check, slay ---")


if __name__ == "__main__":
    main()