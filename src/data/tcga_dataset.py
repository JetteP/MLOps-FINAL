import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TCGADataset(Dataset):
    def __init__(
        self,
        split_json,
        embeddings_path,
        labels_path,
        label_map_path,
        text_embeddings_path=None,
    ):
        """
        Custom Dataset voor TCGA Multi-modal data.
        """
        # 1. Laad de PIDs voor deze split (train/val/test)
        with open(split_json, "r") as f:
            self.pids = json.load(f)

        # 2. Laad visuele embeddings (.pkl)
        with open(embeddings_path, "rb") as f:
            self.img_embeddings = pickle.load(f)

        # 3. Laad labels en label mapping
        self.labels_df = pd.read_csv(labels_path).set_index("patient_id")
        with open(label_map_path, "r") as f:
            self.label_map = json.load(f)

        # 4. Optioneel: Laad tekst embeddings (voor na de vLLM stap)
        self.text_embeddings = None
        if text_embeddings_path and Path(text_embeddings_path).exists():
            with open(text_embeddings_path, "rb") as f:
                self.text_embeddings = pickle.load(f)

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]

        # --- Visuele Data (x_img) ---
        img_data = self.img_embeddings[pid]
        # Aggregatie-strategie: Mean Pooling als er meerdere embeddings zijn
        if isinstance(img_data, dict):
            vectors = [np.asarray(v) for v in img_data.values()]
            x_img = np.mean(vectors, axis=0)
        else:
            x_img = np.asarray(img_data)

        # --- Tekstuele Data (x_txt) ---
        # Placeholder voor Sectie 1.4: als er nog geen embeddings zijn, geven we zeros
        if self.text_embeddings and pid in self.text_embeddings:
            x_txt = np.asarray(self.text_embeddings[pid])
        else:
            x_txt = np.zeros(768)  # vLLM output is meestal 768-dim

        # --- Label (y) ---
        cancer_type = self.labels_df.loc[pid, "cancer_type"]
        y = self.label_map[cancer_type]

        return (
            torch.tensor(x_img, dtype=torch.float32),
            torch.tensor(x_txt, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            pid,
        )
