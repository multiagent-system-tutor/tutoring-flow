
import os
import cv2
import matplotlib.pyplot as plt
import time
import glob
import numpy as np
import pandas as pd
import subprocess
from paddleocr import PaddleOCR
import seaborn as sns

# ===================== CONFIGURATION =====================
DATASET_DIR = r'f:/projek dosen/tutoring/Agentic Multimodal Tutor - SLL/dataset/UTS/SOAL2'
IMAGES_DIR = DATASET_DIR
GT_DIR = DATASET_DIR
RESULTS_DIR = r'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

USE_LIMIT = True  
LIMIT_COUNT = 5 # Small count for analysis/testing multiple methods

# ===================== FUNCTIONS =====================
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
    if len(ref) == 0: return 1.0 if len(hyp) > 0 else 0.0
    return levenshtein_distance(ref, hyp) / len(ref)

def read_ground_truth(filename_base):
    path = os.path.join(GT_DIR, f"{filename_base}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def preprocess_image(image_path, method='otsu', temp_filename='temp_preprocessed.jpg'):
    """
    Applies B&W conversion and thresholding, then converts back to RGB.
    """
    img = cv2.imread(image_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == 'otsu':
        # Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive_gaussian':
        # Adaptive Gaussian Thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    elif method == 'adaptive_mean':
        # Adaptive Mean Thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    elif method == 'binary':
        # Simple Binary Thresholding
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        return None

    # Convert back to RGB (3 channels) as requested "rgban itu"
    result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Save to temp file
    cv2.imwrite(temp_filename, result)
    return temp_filename

def run_llm(prompt):
    try:
        # Assuming ollama is in path
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:3b-instruct"],
            input=prompt,
            text=True,
            capture_output=True,
            encoding='utf-8',
            errors='replace'
        )
        if result.returncode != 0:
            print(f"  [LLM ERROR] Exit: {result.returncode}")
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"  [LLM EXCEPTION] {e}")
        return None

# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    print("Initializing PaddleOCR...")
    # NOTE: Using standard PaddleOCR init
    ocr = PaddleOCR(lang="en", enable_mkldnn=False, use_angle_cls=True)

    # Load Files
    image_files = (
        glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) +
        glob.glob(os.path.join(IMAGES_DIR, "*.png")) +
        glob.glob(os.path.join(IMAGES_DIR, "*.jpeg"))
    )

    if USE_LIMIT and LIMIT_COUNT > 0:
        print(f"Limiting to first {LIMIT_COUNT} images for analysis.")
        image_files = image_files[:LIMIT_COUNT]

    print(f"Found {len(image_files)} images.")

    results = []
    # Methods to analyze
    PREPROCESS_METHODS = ['none', 'otsu', 'adaptive_gaussian', 'adaptive_mean']

    # Load prompt template
    prompt_template = ""
    if os.path.exists("prompt_correction.txt"):
        with open("prompt_correction.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
    else:
        print("WARNING: prompt_correction.txt not found. LLM step might be skipped or fail.")

    for idx, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        filename_base = os.path.splitext(filename)[0]
        gt_text = read_ground_truth(filename_base)

        print(f"\nProcessing [{idx+1}/{len(image_files)}]: {filename}...")
        
        for method in PREPROCESS_METHODS:
            print(f"  > Method: {method}")
            start_time = time.time()
            
            # --- Preprocessing ---
            temp_file = f"temp_{method}_{filename}"
            if method == 'none':
                input_path = image_path
            else:
                input_path = preprocess_image(image_path, method, temp_file)
                if input_path is None: input_path = image_path # Fallback

            # --- OCR ---
            try:
                # Using predict() as in the original notebook
                ocr_result = ocr.predict(input_path)
            except Exception as e:
                print(f"    [OCR ERROR] {e}")
                ocr_result = []


            extracted_lines = []
            
            # Parse OCR Result (Standard list of lists format)
            if ocr_result:
                # Handle list of results for each image
                # ocr_result is typically [ [ [box], (text, conf) ], ... ] for one image
                # wrapper might return it inside an extra list
                
                # Check structure
                if isinstance(ocr_result, list):
                    # Check if inner is list
                    for line in ocr_result:
                        if isinstance(line, list):
                            # It's a line result? Or image result?
                            # Standard: result = [line1, line2...]
                            # line1 = [box, (text, conf)]
                            if len(line) == 2 and isinstance(line[1], tuple):
                                # This looks like a text line
                                extracted_lines.append(line[1][0])
                            elif isinstance(line[0], list) and len(line[0]) == 4: # Box is 4 pts
                                # It's a line
                                if isinstance(line[1], tuple):
                                     extracted_lines.append(line[1][0])
                            else:
                                # Start of list of lines?
                                for res in line:
                                    if isinstance(res, list) and len(res) == 2 and isinstance(res[1], tuple):
                                        extracted_lines.append(res[1][0])
                elif isinstance(ocr_result, dict) and "rec_texts" in ocr_result:
                    extracted_lines = ocr_result["rec_texts"]

            raw_text = "\n".join(extracted_lines)

            # --- LLM ---
            final_text = raw_text
            if raw_text.strip() and prompt_template:
                prompt = prompt_template.replace("{OCR_TEXT}", raw_text)
                llm_out = run_llm(prompt)
                if llm_out:
                    final_text = llm_out.replace("```plaintext", "").replace("```", "").strip()

            # --- Metrics ---
            elapsed = time.time() - start_time
            cer_raw = calculate_cer(gt_text, raw_text)
            cer_refined = calculate_cer(gt_text, final_text)

            print(f"    Raw CER: {cer_raw:.2%} | Refined CER: {cer_refined:.2%} | Time: {elapsed:.2f}s")

            results.append({
                "filename": filename,
                "method": method,
                "cer_raw": cer_raw,
                "cer_refined": cer_refined,
                "time": elapsed,
                "raw_text": raw_text,
                "final_text": final_text
            })

            # Cleanup
            if method != 'none' and os.path.exists(temp_file):
                os.remove(temp_file)

    # --- SAVE RESULTS ---
    if results:
        df = pd.DataFrame(results)
        result_path = os.path.join(RESULTS_DIR, 'exp6_analysis_results.csv')
        df.to_csv(result_path, index=False)
        print(f"\nResults saved to {result_path}")
        
        # --- VISUALIZATION ---
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df, x='filename', y='cer_refined', hue='method', palette='viridis')
            plt.title('CER Refined by Preprocessing Method')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'method_comparison.png'))
            print("Comparison graph saved to method_comparison.png")
        except Exception as e:
            print(f"Visualization failed: {e}")
