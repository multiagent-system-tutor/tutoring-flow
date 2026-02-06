#!/usr/bin/env python
# coding: utf-8

# # Eksperimen 11: OCR Pipeline (PaddleOCR + Qwen 2.5)
# 
# This notebook implements the full pipeline for Experiment 4 using PaddleOCR for text extraction and Qwen 2.5 (3B Instruct) for text correction/refinement.
# 
# **Key Improvements:**
# - Robust Ollama response parsing.
# - Incremental CSV saving to prevent data loss.
# - Full dataset processing loop.
# 

# In[10]:


# Install required packages
import subprocess
import sys

# Install required packages
print("Installing/Verifying dependencies...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama", "paddleocr", "opencv-python", "numpy", "pandas", "matplotlib", "seaborn", "lmdb", "albumentations==1.3.1"])
except subprocess.CalledProcessError as e:
    print(f"Error during installation: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

import os
import cv2
import matplotlib.pyplot as plt
import time
import glob
import numpy as np
import pandas as pd
import subprocess
from paddleocr import PaddleOCR


# ### DATASET

# In[11]:


TRAIN_DIR = r'f:\projek dosen\tutoring\Agentic Multimodal Tutor - SLL\playwithOCR\dataset\train'
TEST_DIR = r'f:\projek dosen\tutoring\Agentic Multimodal Tutor - SLL\playwithOCR\dataset\test'

# Used for Interface/Testing Loop
IMAGES_DIR = os.path.join(TEST_DIR, 'images')
GT_DIR = os.path.join(TEST_DIR, 'gt')

# Fine-tuning Output Directory
FINETUNE_MODEL_DIR = r'output/rec_finetune/best_model'

# ===================== LIMIT PROCESSING =====================
USE_LIMIT = True  # Set to True to limit the number of processed files
LIMIT_COUNT = 50   # Number of files to process if limit is active


# ### CER

# In[12]:


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


# ### GROUND TRUTH

# In[13]:


def read_ground_truth(filename_base):
    path = os.path.join(GT_DIR, f"{filename_base}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


# ### DATASET PREPARATION FOR FINE-TUNING

# In[ ]:


def prepare_dataset_labels(base_dir, output_file, mode='train'):
    """Generates PaddleOCR label file from gt folder."""
    img_dir = os.path.join(base_dir, 'images')
    gt_dir = os.path.join(base_dir, 'gt')
    
    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        print(f"[{mode.upper()}] Directory missing: {img_dir} or {gt_dir}")
        return
        
    labels = []
    valid_count = 0
    
    # Supported extensions
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(img_dir, ext)))
        
    for img_path in images:
        filename = os.path.basename(img_path)
        # Assumes GT file has same basename + .txt
        basename = os.path.splitext(filename)[0]
        gt_path = os.path.join(gt_dir, basename + '.txt')
        
        if os.path.exists(gt_path):
            with open(gt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            # Flatten text for simple Rec training
            text_flat = text.replace('\n', ' ')
            
            # PaddleOCR expects tab separation
            labels.append(f"{img_path}\t{text_flat}")
            valid_count += 1
            
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))
        
    print(f"[{mode.upper()}] Generated {output_file} with {valid_count} samples.")

# Create dataset directory for list files
os.makedirs('dataset_lists', exist_ok=True)

# Prepare Train and Test Lists
prepare_dataset_labels(TRAIN_DIR, 'dataset_lists/rec_gt_train.txt', mode='train')
prepare_dataset_labels(TEST_DIR, 'dataset_lists/rec_gt_test.txt', mode='test')


# In[ ]:


import subprocess
import sys
import os

# Fungsi untuk menjalankan Fine-Tuning PaddleOCR via Python
def run_finetuning(config_path, pretrain_path, save_dir='output/rec_finetune'):
    # Path ke repo PaddleOCR (Assuming standard location relative to inputs or hardcoded)
    paddleocr_repo = r"f:\projek dosen\tutoring\PaddleOCR" 
    train_script = os.path.join(paddleocr_repo, "tools", "train.py")
    
    # Cek apakah config dan script training ada
    if not os.path.exists(config_path):
        print(f"Error: Config file tidak ditemukan di {config_path}")
        return
    if not os.path.exists(train_script):
        print(f"Error: Training script tidak ditemukan di {train_script}")
        return

    print(f"[INFO] Menggunakan Config: {config_path}")
    print(f"[INFO] Pretrained Model: {pretrain_path}")
    print(f"[INFO] Training Script: {train_script}")

    # Set PYTHONPATH agar bisa import modul dari repo PaddleOCR
    env = os.environ.copy()
    env["PYTHONPATH"] = paddleocr_repo + os.pathsep + env.get("PYTHONPATH", "")

    # Susun command argumen
    cmd = [
        sys.executable, train_script,
        "-c", config_path,
        "-o",
        f"Global.pretrained_model={pretrain_path}",
        f"Global.save_model_dir={save_dir}",
        "Train.dataset.data_dir=./",
        "Train.dataset.label_file_list=['dataset_lists/rec_gt_train.txt']", 
        "Eval.dataset.data_dir=./",
        "Eval.dataset.label_file_list=['dataset_lists/rec_gt_test.txt']"
    ]
    
    print("Memulai Fine-tuning... (Output akan muncul di bawah)")
    print(f"Command Eksekusi: {' '.join(cmd)}\n")
    print("-" * 50)
    
    # Eksekusi process dengan streaming output
    try:
        # bufsize=1 means line buffered
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        
        # Baca output baris per baris
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
        process.stdout.close()
        process.wait()
        
        if process.returncode == 0:
             print("\n" + "="*50)
             print("[SUCCESS] Fine-tuning Berhasil! Model tersimpan di", save_dir)
             print("="*50)
        else:
             print(f"\n[FAILURE] Training gagal dengan exit code {process.returncode}")
    except Exception as e:
        print(f"Gagal menjalankan training: {e}")

# ==========================================
# KONFIGURASI PATH (WAJIB DIISI)
# ==========================================
# Masukkan Absolute Path ke file .yml config kamu
# Contoh: r"F:\projek dosen\tutoring\...\Configs\en_PP-OCRv4_rec.yml"
CONFIG_PATH = r"f:\projek dosen\tutoring\PaddleOCR\configs\rec\PP-OCRv4\en_PP-OCRv4_mobile_rec.yml"

# Masukkan Absolute Path ke folder pretrain model kamu
# Contoh: r"F:\projek dosen\tutoring\...\Pretrain\en_PP-OCRv4_rec_train\best_accuracy"
PRETRAIN_PATH = r"f:\projek dosen\tutoring\PaddleOCR\pretrain_models\en_PP-OCRv4_mobile_rec_pretrained.pdparams"

# Jalankan (Uncomment baris di bawah ini setelah path diisi)
run_finetuning(CONFIG_PATH, PRETRAIN_PATH)


# ### OCR INIT

# In[14]:


print("Initializing PaddleOCR...")

# Check if Fine-tuned model exists
use_model_dir = None
if os.path.exists(FINETUNE_MODEL_DIR):
    print(f"\033[92mFound Fine-tuned Model at {FINETUNE_MODEL_DIR}. Using it!\033[0m")
    use_model_dir = FINETUNE_MODEL_DIR
    # To use SPECIFICALLY the fine-tuned REC model:
    ocr = PaddleOCR(rec_model_dir=use_model_dir, 
                    lang="en", 
                    enable_mkldnn=False, 
                    use_angle_cls=True)
else:
    print("\033[93mFine-tuned model NOT found. Using Default Pre-trained Model.\033[0m")
    ocr = PaddleOCR(lang="en", enable_mkldnn=False, use_angle_cls=True)


# ### LLM CALL (AMAN)

# In[ ]:


# ===================== LLM CALL (ROBUST) =====================
def run_llm(prompt):
    # Run subprocess with robust encoding handling
    try:
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:3b-instruct"],
            input=prompt,
            text=True,
            capture_output=True,
            encoding='utf-8',       # Ensure UTF-8 for I/O
            errors='replace'        # Replace chars that fail to encode/decode (fixes charmap error)
        )
        if result.returncode != 0:
            print(f"  [LLM ERROR] Exit Code: {result.returncode}")
            print(f"  [LLM STDERR] {result.stderr[:200]}...") # Print part of stderr
            return None
            
        return result.stdout.strip()
    except Exception as e:
        print(f"  [LLM EXCEPTION] {e}")
        return None


# ### FILE LIST

# In[16]:


image_files = (
    glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGES_DIR, "*.png")) +
    glob.glob(os.path.join(IMAGES_DIR, "*.jpeg"))
)


results = []

# Apply Limit if Enabled
if USE_LIMIT and LIMIT_COUNT > 0:
    print(f"Limiting processing to first {LIMIT_COUNT} images.")
    image_files = image_files[:LIMIT_COUNT]

print(f"Found {len(image_files)} images.")


# ### MAIN LOOP

# In[17]:


for idx, image_path in enumerate(image_files):
    filename = os.path.basename(image_path)
    filename_base = os.path.splitext(filename)[0]
    gt_text = read_ground_truth(filename_base)

    print(f"\nProcessing [{idx+1}/{len(image_files)}]: {filename}...")
    start_time = time.time()

    # ---------- OCR ----------
    ocr_result = ocr.predict(image_path)
    extracted_lines = []
    bboxes = []

    if ocr_result and len(ocr_result) > 0:
        if isinstance(ocr_result[0], dict) and "rec_texts" in ocr_result[0]:
            extracted_lines = ocr_result[0]["rec_texts"]
            if "dt_polys" in ocr_result[0]:
                bboxes = ocr_result[0]["dt_polys"]
        elif isinstance(ocr_result[0], list):
            for line in ocr_result[0]:
                if isinstance(line, list) and len(line) >= 2:
                    if isinstance(line[1], (tuple, list)):
                        extracted_lines.append(line[1][0])
                    if isinstance(line[0], list):
                        bboxes.append(line[0])

    raw_text = "\n".join(extracted_lines)

    # ---------- VISUALIZATION & BBOX ----------
    if bboxes:
        img_vis = cv2.imread(image_path)
        if img_vis is not None:
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            for box in bboxes:
                box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_vis, [box], True, (255, 0, 0), 2)
            
            # Save Image
            os.makedirs(r'results/bbox', exist_ok=True)
            vis_path = os.path.join(r'results/bbox', f'vis_{filename}')
            plt.figure(figsize=(10, 10))
            plt.imshow(img_vis)
            plt.axis('off')
            plt.savefig(vis_path, bbox_inches='tight')
            plt.close()
            
            # Save Coords TXT
            txt_path = os.path.join(r'results/bbox', f'bbox_{filename_base}.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                for i, box in enumerate(bboxes):
                    text = extracted_lines[i] if i < len(extracted_lines) else ''
                    f.write(f'{box} | {text}\n')

    # ---------- LLM ----------
    final_text = raw_text

    if raw_text.strip():
        if not os.path.exists("prompt_correction.txt"):
            print("  [ERROR] prompt_correction.txt not found!")
            continue
        with open("prompt_correction.txt", "r", encoding="utf-8") as f:
            prompt = f.read().replace("{OCR_TEXT}", raw_text)

        print("  [LLM] running...")
        llm_out = run_llm(prompt)

        if llm_out is None:
            print("  [LLM] timeout -> skip")
            final_text = raw_text
        else:
            final_text = (
                llm_out
                .replace("```plaintext", "")
                .replace("```", "")
                .strip()
            )
            print("  [LLM] done")

    # ---------- METRIC ----------
    elapsed = time.time() - start_time
    cer_raw = calculate_cer(gt_text, raw_text)
    cer_refined = calculate_cer(gt_text, final_text)

    print(
        f"  OCR Length: {len(raw_text)} | "
        f"CER Raw: {cer_raw:.2%} | "
        f"CER Refined: {cer_refined:.2%} | "
        f"Time: {elapsed:.2f}s"
    )

    results.append({
        "filename": filename,
        "time": elapsed,
        "cer_raw": cer_raw,
        "cer_refined": cer_refined,
        "raw_text": raw_text,
        "final_text": final_text,
        "ground_truth": gt_text
    })

    # Save partial results incrementally
    if len(results) > 0:
        pd.DataFrame(results).to_csv('results/exp4_results.csv', index=False)

print("\nDONE. Total processed:", len(results))


# In[ ]:


# ===================== VISUALIZE METRICS =====================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if results:
    df = pd.DataFrame(results)
    print(f"Average Time: {df['time'].mean():.4f}s")
    print(f"Average CER (Raw): {df['cer_raw'].mean():.2%}")
    print(f"Average CER (Refined): {df['cer_refined'].mean():.2%}")
    
    try:
        plt.figure(figsize=(12, 6))
        # Melt for seaborn
        df_melted = df.melt(id_vars=['filename'], value_vars=['cer_raw', 'cer_refined'], var_name='Stage', value_name='CER')
        
        sns.barplot(data=df_melted, x='filename', y='CER', hue='Stage', palette='viridis')
        plt.title('Comparison of CER: Raw OCR vs Qwen 2.5 Refinement')
        plt.xlabel('Filename')
        plt.ylabel('Character Error Rate (0.0 - 1.0)')
        # Too many x-labels might clutter, maybe strip?
        if len(df) > 20:
            plt.xticks([]) # Hide x labels if too many
        else:
            plt.xticks(rotation=45, ha='right')
            
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        save_path = 'results/cer_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {save_path}")
        plt.show()
    except Exception as e:
        print(f"Error plotting: {e}")

