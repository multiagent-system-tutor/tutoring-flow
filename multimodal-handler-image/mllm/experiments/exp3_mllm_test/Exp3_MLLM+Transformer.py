"""
Handwriting Pseudocode to Plaintext Converter
Using OpenCV for preprocessing & Transformers (LightOnOCR) for extraction.

Usage:
    python handwriting_parser.py
    python handwriting_parser.py --folder "path/to/folder"
"""

from __future__ import annotations

import base64
import json
import time
import sys
import os
import argparse
import io
from typing import Any, Dict, Optional, Tuple, List
from glob import glob
import difflib
import re

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import (
    AutoProcessor, 
    Qwen2VLForConditionalGeneration,
    AutoModelForVision2Seq,
    AutoTokenizer
)
from qwen_vl_utils import process_vision_info

# Check for tkinter availability (optional for CLI usage)
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# ============================================================
# Helpers & Image Conversion
# ============================================================

def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def to_data_url_png(img_bgr: np.ndarray) -> str:
    """Convert visual debug image to base64 string for JSON output."""
    rgb_pil = cv2_to_pil(img_bgr)
    b = pil_to_bytes(rgb_pil)
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def resize_maintain_aspect(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculates Character Error Rate (CER).
    CER = (S + D + I) / N
    S: Substitutions, D: Deletions, I: Insertions, N: Length of Reference
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
        
    # Manual Levenshtein distance implementation
    # Create matrix
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    distance = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        distance[i][0] = i
    for k in range(1, cols):
        distance[0][k] = k

    for col in range(1, cols):
        for row in range(1, rows):
            if reference[row - 1] == hypothesis[col - 1]:
                cost = 0
            else:
                cost = 1
            distance[row][col] = min(
                distance[row - 1][col] + 1,      # deletion
                distance[row][col - 1] + 1,      # insertion
                distance[row - 1][col - 1] + cost # substitution
            )
            
    lev_dist = distance[rows - 1][cols - 1]
    return lev_dist / len(reference)

# ============================================================
# File Selection
# ============================================================


def select_gt_folder() -> Optional[str]:
    """Opens a file dialog to select a Ground Truth folder."""
    if not HAS_TKINTER:
        return None
        
    root = tk.Tk()
    root.withdraw() 
    
    # Prompt user
    response = tk.messagebox.askyesno("Ground Truth", "Do you have a separate folder for Ground Truth (files .txt)?")
    if not response:
        return None

    folder_path = filedialog.askdirectory(
        title="Select Ground Truth Folder (.txt files)"
    )
    return folder_path if folder_path else None

def select_folder() -> str:
    """Opens a file dialog to select a folder."""
    if not HAS_TKINTER:
        raise RuntimeError("Tkinter not available. Please provide folder path via command line arguments.")
        
    root = tk.Tk()
    root.withdraw() # Hide main window
    
    folder_path = filedialog.askdirectory(
        title="Pilih Folder dengan Gambar Pseudocode"
    )
    
    if not folder_path:
        print("Tidak ada folder yang dipilih. Keluar.")
        sys.exit(0)
        
    return folder_path

# ============================================================
# OpenCV Preprocess
# ============================================================

def preprocess_image(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cleans image for better OCR/Vision tasks."""
    # 1. Denoising
    den = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
    
    # 2. Convert to Gray
    gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
    
    # 3. Adaptive threshold
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 15
    )
    
    # 4. Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return den, th

def draw_text_regions(img_bgr: np.ndarray, binary_map: np.ndarray) -> np.ndarray:
    """Draws bounding boxes around suspected text regions."""
    debug_img = img_bgr.copy()
    
    # Dilate to merge words into blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.morphologyEx(binary_map, cv2.MORPH_DILATE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 500: # Filter small noise
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    return debug_img

# ============================================================
# Model Logic (MLLM)
# ============================================================


class MLLMExtractor:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        print(f"Loading local MLLM: {model_name} (This may take a while)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("[WARNING] CUDA not available. Running on CPU. This will be slow!")
        else:
            print(f"[Info] Running on GPU: {torch.cuda.get_device_name(0)}")
        
        try:
            # 1. Load Processor
            print("[Info] Loading AutoProcessor...")
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

            # 2. Load Model
            print("[Info] Loading MLLM Model...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype="auto", 
                device_map=self.device
            ).eval()
            
            print(f"MLLM loaded successfully.")

        except Exception as e:
            raise RuntimeError(f"Failed to load local MLLM: {e}")

    def extract_logic(self, img_pil: Image.Image) -> Tuple[str, str]:
        """
        Returns (raw_text, refined_text)
        """
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_pil},
                    {"type": "text", "text": "Extract text from this image. Maintain the original layout/format, and provide the results in a neat text format. Do not convert it to any other programming language. Do not provide explanations. correct all words such as words that must be there, namely kamus, algoritma, endprogram."}
                ],
            }
        ]
        
        # Inference using standard Qwen2-VL pipeline
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process inputs (Image + Text)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        refined_text = self._refine_pseudocode(output_text)
        return output_text, refined_text
    
    def _refine_pseudocode(self, text: str) -> str:
        # Same refinement logic
        corrections = {
            r'(?i)\bprogran\b': 'program',
            r'(?i)\bporgram\b': 'program',
            r'(?i)\bkamu[s5]\b': 'kamus',
            r'(?i)\bkamus\s+': 'kamus\n',
            r'(?i)\bku\s*mus\b': 'kamus',
            r'(?i)\balgoritma\b': 'algoritma',
            r'(?i)\balgorythm\b': 'algoritma',
            r'(?i)\binterger\b': 'integer',
            r'(?i)\balgoritma\b': 'algoritma',
            r'(?i)\binput\b': 'input', 
            r'(?i)\boutput\b': 'output',
        }
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text)
        return text.strip()

# ============================================================
# Main Pipeline
# ============================================================

def process_student_pseudocode(
    image_path: str,
    extractor: MLLMExtractor,
    max_dimension: int = 2048,
    output_dir: str = "output_results",
    gt_folder: Optional[str] = None
) -> Dict[str, Any]:

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    filename = os.path.basename(image_path)
    base_name = os.path.splitext(filename)[0]
    
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print('='*60)
    
    t_start = time.perf_counter()

    # 1. Load Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image file: {image_path}")

    # Resize if too large
    h, w = img_bgr.shape[:2]
    print(f"[Original Size: {w}x{h}]")
    if max(h, w) > max_dimension:
        img_bgr = resize_maintain_aspect(img_bgr, width=max_dimension) if w > h else resize_maintain_aspect(img_bgr, height=max_dimension)
        h, w = img_bgr.shape[:2]
        print(f"[Resized to: {w}x{h}]")

    t_load_end = time.perf_counter()

    # 2. Preprocess (Mainly for debug visualization)
    denoised_img, binary_map = preprocess_image(img_bgr)
    debug_image_bgr = draw_text_regions(denoised_img, binary_map)
    
    t_preprocess_end = time.perf_counter()

    # 3. MLLM Extraction
    print(f"Processing with MLLMExtractor...")
    
    # ERROR HANDLING NOTE: We pass the original (resized) image, NOT the denoised one.
    raw_text, refined_text = extractor.extract_logic(cv2_to_pil(img_bgr))

    t_inference_end = time.perf_counter()
    
    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(refined_text)
        
    # Save DEBUG Image (Bounding Boxes)
    debug_img_path = os.path.join(output_dir, f"{base_name}_debug.png")
    cv2.imwrite(debug_img_path, debug_image_bgr)
        
    # 4. Accuracy Check (Ground Truth)
    # Determine GT path
    if gt_folder:
         gt_path = os.path.join(gt_folder, f"{base_name}.txt")
    else:
         gt_path = os.path.join(os.path.dirname(image_path), f"{base_name}.txt")
    
    cer_raw = None
    cer_refined = None
    
    # Logic to prioritize "Clean" GT (without NG_ prefix) to avoid auto-generated artifacts
    gt_found = False
    final_gt_path = gt_path
    
    if base_name.startswith("NG_"):
        clean_name = base_name[3:] # Remove "NG_"
        clean_gt_path = os.path.join(os.path.dirname(gt_path), f"{clean_name}.txt")
        if os.path.exists(clean_gt_path):
            final_gt_path = clean_gt_path
            gt_found = True
            
    # If not found via clean name, check the exact name
    if not gt_found and os.path.exists(gt_path):
        final_gt_path = gt_path
        gt_found = True
            
    if gt_found:
        print(f"[Ground Truth Found]: {os.path.basename(final_gt_path)}")
        try:
            with open(final_gt_path, "r", encoding="utf-8") as f:
                gt_text = f.read().strip()
                
            # Calculate CER
            cer_raw = calculate_cer(gt_text, raw_text)
            cer_refined = calculate_cer(gt_text, refined_text)
            
            print(f"   -> CER Raw    : {round(cer_raw * 100, 2)}%")
            print(f"   -> CER Refined: {round(cer_refined * 100, 2)}%")
            
        except Exception as e:
            print(f"[Warning] Could not read GT file: {e}")
            gt_text = None
    else:
        # If no GT, display informative deviation
        cer_diff = calculate_cer(refined_text, raw_text)
        print(f"[Info] No Ground Truth file found. (Checked: {os.path.basename(gt_path)})")
        print(f"[Info] Comparing Raw vs Refined (Deviation):")
        print(f"   -> Deviation  : {round(cer_diff * 100, 2)}%")

    # Timing calculations
    time_load = t_load_end - t_start
    time_preprocess = t_preprocess_end - t_load_end
    time_inference = t_inference_end - t_preprocess_end
    time_total = t_inference_end - t_start
    
    print(f"\n[Saved to {output_path}]")
    print(f"[Debug Image saved to {debug_img_path}]")
    print(f"[Total time: {round(time_total, 2)}s]")
    
    return {
        "time_total": round(time_total, 4),
        "time_load": round(time_load, 4),
        "time_preprocess": round(time_preprocess, 4),
        "time_inference": round(time_inference, 4),
        "cer_raw": cer_raw,
        "cer_refined": cer_refined,
        "text": refined_text,
        "image_debug_path": debug_img_path,
        "output_file": output_path
    }

# ============================================================
# Execution
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Convert handwritten pseudocode to plaintext using MLLM.")
    parser.add_argument("--folder", type=str, help="Path to folder containing images.")
    parser.add_argument("--gt_folder", type=str, help="Path to seperate Ground Truth folder (optional).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace model name")
    parser.add_argument("--output", type=str, default="output_results", help="Output directory")
    
    args = parser.parse_args()

    # Determine folder path
    folder_path = args.folder
    gt_folder_path = args.gt_folder
    
    if not folder_path:
        print("Please select a folder with images...")
        try:
            folder_path = select_folder()
            # Ask for GT folder if not provided via CLI
            if not gt_folder_path:
                 gt_folder_path = select_gt_folder()
        except Exception as e:
            print(f"Error selecting folder: {e}")
            sys.exit(1)
            
    print(f"Target Images Folder: {folder_path}")
    if gt_folder_path:
        print(f"Target GT Folder    : {gt_folder_path}")
    else:
        print(f"Target GT Folder    : Same as Images")
    
    # Find all image files in folder (Case-insensitive check)
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    
    # Check all files in the directory
    try:
        for fname in os.listdir(folder_path):
            full_path = os.path.join(folder_path, fname)
            if os.path.isfile(full_path):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_extensions:
                    image_files.append(full_path)
    except Exception as e:
        print(f"[Error] scanning directory: {e}")
        sys.exit(1)

    image_files.sort()
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        sys.exit(1)
    
    print(f"\nFound {len(image_files)} image file(s)")
    print("="*60)
    
    # INIT EXTRACTOR
    try:
        extractor = MLLMExtractor(model_name=args.model)
    except Exception as e:
        print(f"[FATAL] Could not initialize model: {e}")
        sys.exit(1)

    try:
        results = []
        successful = 0
        failed = 0
        
        for idx, img_path in enumerate(image_files, 1):
            print(f"\n[File {idx}/{len(image_files)}]")
            try:
                result = process_student_pseudocode(
                    img_path, 
                    extractor=extractor,
                    output_dir=args.output,
                    gt_folder=gt_folder_path
                )
                results.append(result)
                successful += 1
            except Exception as e:
                import traceback
                print(f"[ERROR] Failed to process {os.path.basename(img_path)}: {e}")
                print(traceback.format_exc())
                failed += 1
        
        # Final Summary
        print("\n" + "="*60)
        print(" BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"Total files: {len(image_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if results:
            total_time = sum(r['time_total'] for r in results)
            avg_time = total_time / len(results)
            
            # CER
            cers_raw = [r['cer_raw'] for r in results if r['cer_raw'] is not None]
            cers_ref = [r['cer_refined'] for r in results if r['cer_refined'] is not None]
            
            avg_cer_raw = (sum(cers_raw)/len(cers_raw)) if cers_raw else 0.0
            avg_cer_ref = (sum(cers_ref)/len(cers_ref)) if cers_ref else 0.0
            
            print(f"Total processing time: {round(total_time, 2)}s")
            print(f"Avg Time per file: {round(avg_time, 2)}s")
            
            if cers_raw:
                print(f"Avg CER (Raw): {round(avg_cer_raw * 100, 2)}%")
                print(f"Avg CER (Refined): {round(avg_cer_ref * 100, 2)}%")
            else:
                print("Avg CER: N/A (No Ground Truth found)")

        print(f"\nResults saved to: {args.output}/")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()