import os
from pathlib import Path

# ========= CONFIG =========
SRC_DATASET = Path(r"C:\projekdosen\tutoring\Agentic Multimodal Tutor - SLL\playwithOCR\dataset")
OUT_ROOT = Path(r"C:\projekdosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp12_finetunepaddle\dataset_paddle")

SPLITS = ["train", "test"]

for split in SPLITS:
    img_dir = SRC_DATASET / split / "images"
    gt_dir  = SRC_DATASET / split / "gt"

    out_split_dir = OUT_ROOT / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    out_label = out_split_dir / f"{split}.txt"
    lines = []

    for gt_file in gt_dir.glob("*.txt"):
        img_file = img_dir / (gt_file.stem + ".jpg")

        # 🔥 SKIP kalau image ga ada
        if not img_file.exists():
            print(f"[SKIP] image missing: {img_file}")
            continue

        text = gt_file.read_text(encoding="utf-8").strip().replace("\n", " ")

        # 🔥 SKIP kalau label kosong
        if not text:
            print(f"[SKIP] empty label: {gt_file}")
            continue

        # 🔥 PATH RELATIF (AMAN BUAT TRAINING)
        rel_img_path = img_file.as_posix()

        lines.append(f"{rel_img_path}\t{text}")

    out_label.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {split}: {len(lines)} samples → {out_label}")
