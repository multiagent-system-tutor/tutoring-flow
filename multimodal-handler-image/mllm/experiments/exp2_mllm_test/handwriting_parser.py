"""
Handwriting Pseudocode to Plaintext Converter
Using OpenCV for preprocessing & Ollama (Vision Models) for extraction.

Usage:
    python handwriting_parser.py
    python handwriting_parser.py --folder "path/to/folder" --model "llama3.2-vision"
"""

from __future__ import annotations

import base64
import json
import time
import sys
import os
import argparse
import io
from typing import Any, Dict, Optional, Tuple
from glob import glob
import difflib
import re

import cv2
import numpy as np
import ollama
from PIL import Image

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

# ============================================================
# File Selection
# ============================================================

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
# MLLM Logic (Ollama)
# ============================================================

class MLLMExtractor:
    def __init__(self, model: str = "llama3.2-vision"):
        self.model = model

    def refine_pseudocode(self, text: str) -> str:
        """
        Post-processing to fix common MLLM typos (spelling errors) only.
        Does NOT change symbol representations (e.g. <- stays <-) to strictly match the image.
        """
        # 1. Fix Common Keywords Typos (Spelling mistakes only)
        # un-standardized: we preserve 'read', 'write', 'int' if that's what's written.
        corrections = {
            r'(?i)\bprogran\b': 'program',
            r'(?i)\bporgram\b': 'program',
            r'(?i)\bkamu[s5]\b': 'kamus',
            r'(?i)\bkamus\s+': 'kamus\n',
            r'(?i)\bku\s*mus\b': 'kamus',
            r'(?i)\balgoritma\b': 'algoritma',
            r'(?i)\balgorythm\b': 'algoritma',
            r'(?i)\binterger\b': 'integer', # 'int' stays 'int'
            r'(?i)\balgoritma\b': 'algoritma',
            r'(?i)\binput\b': 'input', 
            r'(?i)\boutput\b': 'output',
            # Removed: int->integer, read->input, write->output, arrow->=
            # to ensure output matches the image text/symbols strictly.
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text)

        # 2. Structural cleanups (Whitespace)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()

    def check_connection(self) -> bool:
        try:
            ollama.list()
            return True
        except Exception:
            return False

    def extract_logic(self, img_pil: Image.Image) -> str:
        img_bytes = pil_to_bytes(img_pil)

        # REFINED PROMPT (TRANSCRIPTION)
        prompt = (
            "You are a Forensic Handwriting Analysis & Explanation Engine.\n"
            "Your task is to Transcribe the code, and then Explain the image.\n\n"
            "1. TRANSCRIPTION (Strict OCR):\n"
            "   - Transcribe strictly what you see. Keep keywords and symbols exactly as written (e.g., 'int', '<-').\n"
            "   - Maintain structure (indentation, newlines).\n"
            "   - Do NOT try to 'fix' the code logic in this section. Just read it.\n\n"
            "2. EXPLANATION:\n"
            "   - Briefly explain what the algorithm does.\n"
            "   - Describe the image quality or any specific visual details (e.g., 'crossed out text', 'arrows used for assignment').\n\n"
            "OUTPUT FORMAT:\n"
            "---CODE---\n"
            "[Transcription goes here]\n\n"
            "---EXPLANATION---\n"
            "[Explanation goes here]"
        )

        try:
            res = ollama.generate(
                model=self.model,
                prompt=prompt,
                images=[img_bytes],
                options={
                    "temperature": 0.0, # Zero for deterministic output
                    "num_ctx": 4096
                },
                stream=True
            )
            
            full_response = ""
            print("Response stream: ", end="", flush=True)
            for chunk in res:
                part = chunk.get("response", "")
                print(part, end="", flush=True)
                full_response += part
            print() # Newline at end
            
            # Post-process refinement
            refined_text = self.refine_pseudocode(full_response)
            
            return refined_text
        except Exception as e:
            raise RuntimeError(f"Ollama Error: {e}")

# ============================================================
# Main Pipeline
# ============================================================

def process_student_pseudocode(
    image_path: str,
    model_name: str = "llama3.2-vision",
    max_dimension: int = 2048,
    output_dir: str = "output_results"
) -> Dict[str, Any]:

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    filename = os.path.basename(image_path)
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
    print(f"Processing with Ollama ({model_name})...")
    extractor = MLLMExtractor(model=model_name)
    
    if not extractor.check_connection():
        raise ConnectionError("Could not connect to Ollama. Make sure 'ollama serve' is running.")

    # ERROR HANDLING NOTE: We pass the original (resized) image, NOT the denoised one.
    # Denoising often obliterates small symbols like dots, commas, and thin lines.
    extracted_text = extractor.extract_logic(cv2_to_pil(img_bgr))

    t_inference_end = time.perf_counter()
    
    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{base_name}.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
        
    # 4. Accuracy Check (Ground Truth)
    # Look for .txt file with same name in source folder
    source_dir = os.path.dirname(image_path)
    gt_path = os.path.join(source_dir, f"{base_name}.txt")
    
    accuracy = None
    if os.path.exists(gt_path):
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_text = f.read().strip()
            # Calculate similarity ratio
            accuracy = difflib.SequenceMatcher(None, gt_text, extracted_text).ratio()
            print(f"[Accuracy: {round(accuracy * 100, 2)}%]")
        except Exception as e:
            print(f"[Warning] Could not read GT file: {e}")

    # Timing calculations
    time_load = t_load_end - t_start
    time_preprocess = t_preprocess_end - t_load_end
    time_inference = t_inference_end - t_preprocess_end
    time_total = t_inference_end - t_start
    
    print(f"\n[Saved to {output_path}]")
    print(f"[Total time: {round(time_total, 2)}s]")
    print(f"  > Load: {round(time_load, 4)}s")
    print(f"  > Preprocess: {round(time_preprocess, 4)}s")
    print(f"  > Inference: {round(time_inference, 4)}s")
    
    return {
        "time_total": round(time_total, 4),
        "time_load": round(time_load, 4),
        "time_preprocess": round(time_preprocess, 4),
        "time_inference": round(time_inference, 4),
        "accuracy": accuracy,
        "text": extracted_text,
        "image_debug": to_data_url_png(debug_image_bgr),
        "output_file": output_path
    }

# ============================================================
# Execution
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Convert handwritten pseudocode to plaintext.")
    parser.add_argument("--folder", type=str, help="Path to folder containing images. If not provided, a folder dialog will open.")
    parser.add_argument("--model", type=str, default="llama3.2-vision", help="Ollama model to use (default: llama3.2-vision)")
    parser.add_argument("--output", type=str, default="output_results", help="Output directory for results")
    
    args = parser.parse_args()

    # Determine folder path
    folder_path = args.folder
    if not folder_path:
        print("Please select a folder with images...")
        try:
            folder_path = select_folder()
        except Exception as e:
            print(f"Error selecting folder: {e}")
            sys.exit(1)
            
    print(f"Target Folder: {folder_path}")
    
    # Find all image files in folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files_set = set()
    
    for ext in image_extensions:
        # Use set to avoid duplicates on case-insensitive systems (Windows)
        # where *.png and *.PNG might return the same file
        found = glob(os.path.join(folder_path, ext))
        image_files_set.update(found)
        found_upper = glob(os.path.join(folder_path, ext.upper()))
        image_files_set.update(found_upper)
        
    image_files = sorted(list(image_files_set))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        sys.exit(1)
    
    print(f"\nFound {len(image_files)} image file(s)")
    print("="*60)
    
    try:
        results = []
        successful = 0
        failed = 0
        
        for idx, img_path in enumerate(image_files, 1):
            print(f"\n[File {idx}/{len(image_files)}]")
            try:
                result = process_student_pseudocode(
                    img_path, 
                    model_name=args.model,
                    output_dir=args.output
                )
                results.append(result)
                successful += 1
            except Exception as e:
                print(f"[ERROR] Failed to process {os.path.basename(img_path)}: {e}")
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
            
            # Metadata for averages
            avg_load = sum(r['time_load'] for r in results) / len(results)
            avg_pre = sum(r['time_preprocess'] for r in results) / len(results)
            avg_inf = sum(r['time_inference'] for r in results) / len(results)
            
            # Accuracy
            accuracies = [r['accuracy'] for r in results if r['accuracy'] is not None]
            avg_accuracy = (sum(accuracies) / len(accuracies)) if accuracies else 0.0
            
            print(f"Total processing time: {round(total_time, 2)}s")
            print(f"Average time per file: {round(avg_time, 2)}s")
            print(f"  - Load: {round(avg_load, 2)}s")
            print(f"  - Preprocess: {round(avg_pre, 2)}s")
            print(f"  - Inference: {round(avg_inf, 2)}s")
            
            if accuracies:
                print(f"Average Accuracy: {round(avg_accuracy * 100, 2)}%")
            else:
                print("Average Accuracy: N/A (No Ground Truth found)")

        print(f"\nResults saved to: {args.output}/")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
