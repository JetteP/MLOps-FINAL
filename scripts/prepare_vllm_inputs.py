import json
from pathlib import Path

# Configuratie
TCGA_DIR = "tcga_data"
SPLIT_DIR = "data/splits"
OUTPUT_DIR = "data/vllm_inputs"
PROMPT_TEMPLATE = """You are a clinical pathology assistant.

TASK:
Rewrite the given pathology report into a concise, standardized medical description for machine learning.

RULES:
Output ONLY the rewritten medical description.
Do NOT include explanations, headings, bullet points, or markdown.
Do NOT include <think> or any reasoning text.
Return exactly 1–3 sentences.
Preserve only medically relevant findings (tumor type, anatomical site, grade/differentiation, invasion, spread/metastasis if stated).
Remove irrelevant or administrative text such as specimen handling details, formatting artifacts, and disclaimers.
Do NOT add new facts or assumptions. If information is not present, do not invent it.
Keep the output in 1–3 short sentences, using consistent medical wording.
Output plain text only (no bullet points, no JSON).

INPUT REPORT:
{report}
"""



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
                    record = {
                        "pid": pid,
                        "prompt": PROMPT_TEMPLATE.format(report=reports[pid])
                    }


                    out.write(json.dumps(record) + "\n")

        print(f"Gereed voor VLLM Decoder: {output_path}")


if __name__ == "__main__":
    main()
