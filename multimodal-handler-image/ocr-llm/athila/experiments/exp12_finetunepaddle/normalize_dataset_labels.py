from pathlib import Path

DATASET_ROOT = Path(
    r"C:\projekdosen\tutoring\tutoring-flow"
    r"\multimodal-handler-image\ocr-llm\athila"
    r"\experiments\exp12_finetunepaddle\dataset_paddle"
)

def normalize(text: str) -> str:
    replace_map = {
        "\t": " ",
        "–": "-",
        "−": "-",
        "≤": "<=",
        "≥": ">=",
        "×": "*",
        "\"": "",      # buang quote
        "%": "",       # buang persen (opsional)
    }
    for k, v in replace_map.items():
        text = text.replace(k, v)

    # rapihin spasi
    text = " ".join(text.split())
    return text


for split in ["train", "test"]:
    txt_path = DATASET_ROOT / split / f"{split}.txt"
    new_lines = []

    for line in open(txt_path, encoding="utf-8"):
        if "\t" not in line:
            continue
        img, label = line.rstrip("\n").split("\t", 1)
        label = normalize(label)
        new_lines.append(f"{img}\t{label}")

    txt_path.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"[OK] normalized: {txt_path}")
