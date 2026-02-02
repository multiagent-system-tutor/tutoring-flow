
import pandas as pd
import numpy as np

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
    """Current logic: Normalizes all whitespace (space, tab, newline) to single space."""
    if pd.isna(reference) or reference is None: reference = ""
    if pd.isna(hypothesis) or hypothesis is None: hypothesis = ""
    
    reference = str(reference)
    hypothesis = str(hypothesis)

    if not reference.strip():
        return 0.0
    
    ref = " ".join(reference.split())
    hyp = " ".join(hypothesis.split())
    
    return levenshtein_distance(ref, hyp) / len(ref)

def calculate_cer_strict(reference, hypothesis):
    """Strict logic: Keeps all whitespace characters exactly as they are."""
    if pd.isna(reference) or reference is None: reference = ""
    if pd.isna(hypothesis) or hypothesis is None: hypothesis = ""
    
    reference = str(reference)
    hypothesis = str(hypothesis)

    if not reference:
        return 0.0

    # We do NOT normalize whitespace here
    ref = reference
    hyp = hypothesis
    
    return levenshtein_distance(ref, hyp) / len(ref)

def calculate_cer_no_whitespace(reference, hypothesis):
    """No whitespace logic: Removes all spaces, tabs, and newlines."""
    if pd.isna(reference) or reference is None: reference = ""
    if pd.isna(hypothesis) or hypothesis is None: hypothesis = ""
    
    reference = str(reference)
    hypothesis = str(hypothesis)

    # Remove all whitespace
    ref = "".join(reference.split())
    hyp = "".join(hypothesis.split())
    
    if len(ref) == 0:
        return 0.0
        
    return levenshtein_distance(ref, hyp) / len(ref)

def main():
    csv_path = r'c:\projekdosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp9cekspasi\results\exp4_results.csv'
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found at {csv_path}")
        return

    with open("cer_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Loaded {len(df)} rows from {csv_path}\n")

        # Calculate CERs
        df['cer_strict'] = df.apply(lambda row: calculate_cer_strict(row['ground_truth'], row['final_text']), axis=1)
        df['cer_normalized'] = df.apply(lambda row: calculate_cer_normalized(row['ground_truth'], row['final_text']), axis=1)
        df['cer_no_whitespace'] = df.apply(lambda row: calculate_cer_no_whitespace(row['ground_truth'], row['final_text']), axis=1)

        # Calculate Averages
        avg_strict = df['cer_strict'].mean()
        avg_normalized = df['cer_normalized'].mean()
        avg_no_whitespace = df['cer_no_whitespace'].mean()

        f.write("\n=== CER Analysis Results ===\n")
        f.write(f"1. Strict (Exact Whitespace):         {avg_strict:.2%} (Includes tabs, newlines, spaces as distinct)\n")
        f.write(f"2. Normalized (Join with single space): {avg_normalized:.2%} (Current Logic)\n")
        f.write(f"3. No Whitespace (Remove all):        {avg_no_whitespace:.2%} (Focus purely on characters)\n")

        f.write("\n--- Details ---\n")
        f.write(f"Difference (Normalized - Strict):       {avg_normalized - avg_strict:.4f}\n")
        f.write(f"Difference (Normalized - No Whitespace):{avg_normalized - avg_no_whitespace:.4f}\n")
        
        # Check individual examples where difference is high
        df['diff_norm_strict'] = abs(df['cer_normalized'] - df['cer_strict'])
        df['diff_norm_nowhite'] = abs(df['cer_normalized'] - df['cer_no_whitespace'])
        
        f.write("\nTop 3 examples where Strict vs Normalized differs most:\n")
        top_diff_strict = df.nlargest(3, 'diff_norm_strict')
        for idx, row in top_diff_strict.iterrows():
            f.write(f"  File: {row['filename']}\n")
            f.write(f"    Strict CER: {row['cer_strict']:.2%}\n")
            f.write(f"    Norm   CER: {row['cer_normalized']:.2%}\n")
            f.write("-" * 30 + "\n")

        f.write("\nTop 3 examples where Normalized vs No Whitespace differs most:\n")
        top_diff_nowhite = df.nlargest(3, 'diff_norm_nowhite')
        for idx, row in top_diff_nowhite.iterrows():
            f.write(f"  File: {row['filename']}\n")
            f.write(f"    Norm CER:      {row['cer_normalized']:.2%}\n")
            f.write(f"    No White CER:  {row['cer_no_whitespace']:.2%}\n")
            f.write("-" * 30 + "\n")
    
    print("\nAnalysis complete. Results written to cer_analysis_report.txt")

if __name__ == "__main__":
    main()
