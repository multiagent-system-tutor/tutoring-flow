
import pandas as pd
import numpy as np
import random

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

def calculate_cer_normalized(reference, hypothesis):
    """Normalizes all whitespace (space, tab, newline) to single space."""
    if pd.isna(reference) or reference is None: reference = ""
    if pd.isna(hypothesis) or hypothesis is None: hypothesis = ""
    
    reference = str(reference)
    hypothesis = str(hypothesis)

    if not reference.strip():
        return 0.0
    
    ref = " ".join(reference.split())
    hyp = " ".join(hypothesis.split())
    
    if len(ref) == 0:
        return 0.0
        
    return levenshtein_distance(ref, hyp) / len(ref)

def add_spacing_noise(text):
    if pd.isna(text): return ""
    lines = str(text).split('\n')
    modified_lines = []
    for line in lines:
        # ALWAYS add spacing to EVERY line to be sure
        # Add explicit large amount of spaces
        prefix = " " * random.randint(15, 40)
        # Occasionally mix in tabs
        if random.random() > 0.7:
             prefix += "\t" * random.randint(2, 4)
             
        modified_lines.append(prefix + line)
    return "\n".join(modified_lines)

def format_for_md(text):
    if pd.isna(text): return ""
    # Replace newlines with <br>
    # Replace spaces with &nbsp; or a visible symbol like • for visualization if requested, 
    # but here let's strictly replace the added spacing noise to be visible.
def format_for_md(text):
    if pd.isna(text): return ""
    val = str(text).replace('|', '\|')
    val = val.replace('\n', '<br>')
    val = val.replace('\t', '→')
    # Use middle dot to visualize spaces clearly as requested
    val = val.replace(' ', '·') 
    return val

import os

def main():
    input_csv_path = r'c:\projekdosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp9cekspasi\results\exp4_results.csv'
    intermediate_csv_path = r'c:\projekdosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp9cekspasi\spacing_experiment_data.csv'
    
    # --- Step 1: Generate Data and Save to CSV ---
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"File not found at {input_csv_path}")
        return

    # Filter for non-empty Ground Truth
    df = df[df['ground_truth'].notna() & (df['ground_truth'].str.strip() != "")]
    
    # Take top 5 for demonstration
    sample_df = df.head(5).copy()

    data_to_save = []
    for idx, row in sample_df.iterrows():
        orig = row['final_text']
        gt = row['ground_truth']
        
        # Create modified version
        mod = add_spacing_noise(orig)
        
        data_to_save.append({
            'filename': row['filename'],
            'original_text': orig,
            'modified_text': mod,
            'ground_truth': gt
        })
    
    # Save to intermediate CSV to simulate data persistence
    df_saved = pd.DataFrame(data_to_save)
    df_saved.to_csv(intermediate_csv_path, index=False)
    print(f"Intermediate data saved to {intermediate_csv_path}")

    # --- Step 2: Read from CSV and Verify ---
    print("Reading back from CSV to calculate CER...")
    df_loaded = pd.read_csv(intermediate_csv_path)

    md_output = []
    md_output.append("| No | Original OCR | Modified OCR (Added Spaces) | Ground Truth | CER Comparison (Normalized) |")
    md_output.append("|---|---|---|---|---|")

    for idx, row in df_loaded.iterrows():
        original_text = row['original_text']
        modified_text = row['modified_text']
        gt_text = row['ground_truth']
        
        # Calculate CERs using the reloaded data (Normalized logic)
        cer_orig = calculate_cer_normalized(gt_text, original_text)
        cer_mod = calculate_cer_normalized(gt_text, modified_text)
        
        comparison_str = f"Org: {cer_orig:.2%}<br>Mod: {cer_mod:.2%}<br>Diff: {cer_mod - cer_orig:.4f}"
        
        md_output.append(f"| {idx+1} | {format_for_md(original_text)} | {format_for_md(modified_text)} | {format_for_md(gt_text)} | {comparison_str} |")

    output_str = "\n".join(md_output)
    
    with open("spacing_experiment.md", "w", encoding="utf-8") as f:
        f.write("# Experiment: Effect of Added Spacing on Normalized CER (Verified via CSV)\n\n")
        f.write(f"Data was generated, saved to `{os.path.basename(intermediate_csv_path)}`, and then reloaded for this calculation.\n")
        f.write("This confirms that even after passing through CSV storage (which preserves whitespace), the Normalized CER remains robust.\n\n")
        f.write(output_str)

    print("Experiment complete. Results saved to spacing_experiment.md")

if __name__ == "__main__":
    main()
