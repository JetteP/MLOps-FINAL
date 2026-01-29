import json
from pathlib import Path

import pandas as pd


def main():
    # dit is het pad naar het csv-bestand waarin per patiënt staat
    # welk type kanker die persoon heeft
    csv_path = Path("tcga_data/tcga_patient_to_cancer_type.csv")

    # hier willen we metadata-bestanden opslaan
    # Path zorgt ervoor dat we netjes met paden werken
    output_dir = Path("data/metadata")

    # maak de map aan als die nog niet bestaat
    # parents=True betekent: maak ook bovenliggende mappen aan
    # exist_ok=True betekent: geen fout als hij al bestaat
    output_dir.mkdir(parents=True, exist_ok=True)

    # dit is het uiteindelijke bestand waarin we de label mapping opslaan
    # dus: kanker-type -> nummer
    output_file = Path("data/label_map.json")

    print(f"Reading labels from {csv_path}...")

    # lees het csv-bestand in als een pandas DataFrame
    # dit maakt het makkelijk om kolommen te selecteren en te bewerken
    labels_df = pd.read_csv(csv_path)

    # haal alle unieke kanker-types uit de kolom 'cancer_type'
    # sorted zorgt ervoor dat de volgorde altijd hetzelfde is
    # dit is belangrijk voor reproduceerbaarheid
    unique_types = sorted(labels_df["cancer_type"].unique())

    # tel hoeveel verschillende classes we hebben
    num_classes = len(unique_types)

    print(f"Found {num_classes} unique cancer types.")

    # maak een mapping van kanker-type naam naar een integer label
    # enumerate geeft (index, waarde), dus:
    # eerste type -> 0, tweede -> 1, enzovoort
    label_map = {name: i for i, name in enumerate(unique_types)}

    # schrijf deze mapping weg naar een json-bestand
    # indent=4 maakt het bestand leesbaar voor mensen
    # FIX: Hier gebruiken we nu 'output_file' in plaats van de string
    with open(output_file, "w") as f:
        json.dump(label_map, f, indent=4)

    print(
        f"Success! Label map saved to data/label_map.json with {num_classes} classes."
    )

    # extra sanity check:
    # UCEC is een veelvoorkomend TCGA cancer type
    # als die ontbreekt, klopt er waarschijnlijk iets niet in de data
    if "UCEC" in label_map:
        print("Confirmed: 'UCEC' is present in the map.")
    else:
        print("WARNING: 'UCEC' is MISSING!")


if __name__ == "__main__":
    # dit zorgt ervoor dat main() alleen draait
    # als we dit bestand direct uitvoeren
    # en niet als we het importeren in een ander script
    main()