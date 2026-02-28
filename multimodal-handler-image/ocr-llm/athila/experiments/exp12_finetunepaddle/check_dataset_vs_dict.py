# check_dataset_vs_dict.py
from pathlib import Path

DICT_PATH = r"C:\projekdosen\tutoring\PaddleOCR\ppocr\utils\custom_dict.txt"
DATASET = r"C:\projekdosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp12_finetunepaddle\dataset_paddle"

dict_chars = set(open(DICT_PATH, encoding="utf-8").read().splitlines())

unknown = set()

for split in ["train", "test"]:
    txt = Path(DATASET) / split / f"{split}.txt"
    for line in open(txt, encoding="utf-8"):
        try:
            _, label = line.strip().split("\t", 1)
        except:
            continue
        for ch in label:
            if ch not in dict_chars:
                unknown.add(ch)

print("❌ UNKNOWN CHARS FOUND:")
for ch in sorted(unknown):
    print(repr(ch))
