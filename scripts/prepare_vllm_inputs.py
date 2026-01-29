import json
from pathlib import Path

# Configuratie
TCGA_DIR = "tcga_data"
SPLIT_DIR = "data/splits"
OUTPUT_DIR = "data/vllm_inputs"
PROMPT_PREFIX = (
    "You are an experienced clinical pathologist. "
    "Rewrite the following diagnostic report into concise, structured "
    "histopathological findings. Remove irrelevant administrative details. "
    "Do not add new information or speculate.\n\nREPORT:\n"
)


def main():
    # 1. Laad alle rapporten
    reports = {}
    with open(f"{TCGA_DIR}/tcga_reports.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            reports[obj["pid"]] = obj["report"]

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 2. Maak JSONL per split met de "Clinical Filter" prompt
    for split in ["train", "val", "test"]:
        with open(f"{SPLIT_DIR}/{split}.json") as f:
            pids = json.load(f)

        output_path = f"{OUTPUT_DIR}/{split}_decoder_input.jsonl"
        with open(output_path, "w") as out:
            for pid in pids:
                if pid in reports:
                    # Dit is de cruciale stap voor sectie 1.4
                    record = {"pid": pid, "prompt": PROMPT_PREFIX + reports[pid]}
                    out.write(json.dumps(record) + "\n")

        print(f"Gereed voor VLLM Decoder: {output_path}")


if __name__ == "__main__":
    main()
