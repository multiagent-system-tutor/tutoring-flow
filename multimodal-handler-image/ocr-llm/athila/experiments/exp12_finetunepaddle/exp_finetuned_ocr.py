import os
import cv2
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

# =====================================================
# DATASET
# =====================================================
DATASET_DIR = r"C:\projekdosen\tutoring\Agentic Multimodal Tutor - SLL\dataset\UTS\SOAL2"
IMAGES_DIR = DATASET_DIR
GT_DIR = DATASET_DIR

# =====================================================
# LIMIT
# =====================================================
USE_LIMIT = True
LIMIT_COUNT = 50

# =====================================================
# OUTPUT
# =====================================================
OUT_DIR = r"C:\projekdosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp12_finetunepaddle\results"
BBOX_DIR = os.path.join(OUT_DIR, "bbox")
os.makedirs(BBOX_DIR, exist_ok=True)

# =====================================================
# METRIC
# =====================================================
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_cer(reference, hypothesis):
    if not reference:
        return 0.0
    ref = " ".join(reference.split())
    hyp = " ".join(hypothesis.split())
    return levenshtein_distance(ref, hyp) / len(ref)

def read_ground_truth(base):
    path = os.path.join(GT_DIR, f"{base}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

# =====================================================
# INIT OCR (FINETUNED – PURE OCR MODE)
# =====================================================
print("Initializing PaddleOCR (FINETUNED - PURE OCR MODE)...")

DICT_PATH = r"C:\projekdosen\tutoring\PaddleOCR\ppocr\utils\custom_dict.txt"
REC_MODEL = r"C:\projekdosen\tutoring\PaddleOCR\output\rec_svtr_pseudocode\best_accuracy"

# sanity check (BIAR KALAU SALAH KETAHUAN AWAL)
# PADDED DICT: Force add a dummy character to handle potential index out of range
lines = open(DICT_PATH, encoding="utf-8").read().splitlines()
PADDED_DICT_PATH = os.path.join(os.path.dirname(DICT_PATH), "padded_dict.txt")
with open(PADDED_DICT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
    # Add MULTIPLE padding lines to cover any mismatch
    for i in range(100):   # Increased to 100
        f.write(f"\n<PAD{i}>")

print(f"Created padded dict at {PADDED_DICT_PATH} with {len(lines)+100} lines")
dict_len = len(lines) + 100
print("DICT LENGTH :", dict_len)
print("MODEL PATH  :", REC_MODEL)

ocr = PaddleOCR(
    # ---------- MODEL ----------
    # det_model_dir=None,  # Commented out to use DEFAULT detection model
    rec_model_dir=REC_MODEL,
    rec_char_dict_path=PADDED_DICT_PATH,
    rec_algorithm="SVTR_LCNet",   # Using SVTR_LCNet as it matches SVTR architecture better than default CRNN

    # ---------- FORCE PURE OCR ----------
    # mode="ocr",         # This might not be a valid init arg, but keeping if user added it
    # layout=False,
    # table=False,
    # recovery=False,     # These might also be non-standard init args for PaddleOCR class, check docs!

    # ---------- MUST MATCH TRAIN ----------
    rec_image_shape="3,32,320",
    max_text_length=120,
    use_space_char=True,
    drop_score=0.0,

    # ---------- SAFE ----------
    use_angle_cls=False,
    lang="en",
    enable_mkldnn=False,
    show_log=True,     # Enable logging
    use_gpu=False      # Ensure matches training (and avoids gpu errors)
)

# =====================================================
# LOAD IMAGES
# =====================================================
image_files = (
    glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGES_DIR, "*.png")) +
    glob.glob(os.path.join(IMAGES_DIR, "*.jpeg"))
)

# ... (rest of code)

if USE_LIMIT:
    image_files = image_files[:LIMIT_COUNT]

print(f"Found {len(image_files)} images")

results = []

# =====================================================
# MAIN LOOP
# =====================================================
for idx, img_path in enumerate(image_files):
    fname = os.path.basename(img_path)
    base = os.path.splitext(fname)[0]
    gt_text = read_ground_truth(base)

    print(f"\n[{idx+1}/{len(image_files)}] {fname}")
    start = time.time()

    try:
        ocr_result = ocr.ocr(img_path, cls=False)
    except Exception:
        import traceback
        print(f"  [OCR ERROR] {traceback.format_exc()}")
        continue

    extracted = []
    bboxes = []

    if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0:
        res = ocr_result[0]
        if res is None:
            print("  [DEBUG] ocr_result[0] is None (No text detected)")
            continue
            
        try:
            for line in res:
                if isinstance(line, list) and len(line) >= 2:
                    bboxes.append(line[0])
                    # Debug line structure if needed
                    # print(f"DEBUG LINE: {line}")
                    extracted.append(line[1][0])
        except Exception as e:
            print(f"  [OCR PROCESSING ERROR] {e}")
            print(f"  Result structure: {res}")
            continue

    raw_text = "\n".join(extracted)

    # ---------- BBOX VIS ----------
    if bboxes:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for box in bboxes:
                box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [box], True, (255, 0, 0), 2)

            vis_path = os.path.join(BBOX_DIR, f"vis_{fname}")
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.axis("off")
            plt.savefig(vis_path, bbox_inches="tight")
            plt.close()

    # ---------- SAVE TEXT ----------
    txt_out_dir = os.path.join(OUT_DIR, "txt")
    os.makedirs(txt_out_dir, exist_ok=True)
    txt_path = os.path.join(txt_out_dir, f"{base}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    # ---------- METRIC ----------
    elapsed = time.time() - start
    cer = calculate_cer(gt_text, raw_text)

    print(f"  OCR len: {len(raw_text)} | CER: {cer:.2%} | Time: {elapsed:.2f}s")
    print(f"  Saved text to: {txt_path}")

    results.append({
        "filename": fname,
        "time": elapsed,
        "cer_raw": cer,
        "raw_text": raw_text,
        "ground_truth": gt_text
    })

    pd.DataFrame(results).to_csv(
        os.path.join(OUT_DIR, "finetuned_results.csv"),
        index=False
    )

# =====================================================
# SUMMARY
# =====================================================
print("\nDONE. Total processed:", len(results))

if results:
    df = pd.DataFrame(results)
    print("\n=== SUMMARY ===")
    print("Avg Time :", df["time"].mean())
    print("Avg CER  :", df["cer_raw"].mean())
else:
    print("\nNO VALID OCR RESULTS (CHECK MODEL / DICT)")
