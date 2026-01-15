import time
import json
import re
import datetime
from vllm import LLM, SamplingParams

# Konfigurasi Model
# Ganti dengan path model lokal atau nama model di HuggingFace
# Contoh: "facebook/opt-125m" (kecil untuk tes), "meta-llama/Llama-3.1-8B-Instruct" (besar)
MODEL_NAME = "facebook/opt-125m" 

def extract_last_json(text: str):
    matches = re.findall(r"\{[\s\S]*\}", text)
    if not matches:
        return None

    for cand in reversed(matches):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None

def run_experiment_vllm(plan, date_str):
    prompt = f"""
You are an AI teacher and problem generator.

Plan: {plan}
Date: {date_str}

Task:
Generate ONE concrete math problem based on the plan.
If an image helps understanding, describe the image clearly in the problem text.

Rules:
- You MUST fill all fields with real content.
- Do NOT repeat placeholders or templates.
- Output ONLY valid JSON.
- No explanation outside JSON.

Required JSON format:
{{
  "soal": "...",
  "image_description": "...",
  "solution": "..."
}}
"""
    
    # Inisialisasi Sampling Parameters
    sampling_params = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=1024)

    # Inisialisasi LLM (Load model ke GPU/VRAM)
    # Warning: Ini akan memakan VRAM. Pastikan GPU cukup.
    print(f"Loading model {MODEL_NAME} using vLLM...")
    llm = LLM(model=MODEL_NAME, trust_remote_code=True)

    print("Starting generation...")
    start_time = time.time()

    # Generate
    outputs = llm.generate([prompt], sampling_params)

    end_time = time.time()
    inference_time = end_time - start_time

    # Ambil hasil generate
    result_text = outputs[0].outputs[0].text

    print(f"Inference finished in {inference_time:.4f} seconds.")
    return result_text, inference_time

if __name__ == "__main__":
    plan_input = "Geometri: Menghitung luas bangun datar gabungan (Persegi dan Segitiga)"
    date_input = datetime.date.today().strftime("%Y-%m-%d")

    # Jalankan eksperimen
    # NOTE: vLLM biasanya butuh GPU NVIDIA (CUDA).
    try:
        result_text, duration = run_experiment_vllm(plan_input, date_input)

        if result_text:
            print("\n--- Raw Model Output ---")
            print(result_text)

            print("\n--- Parsed JSON ---")
            parsed = extract_last_json(result_text)
            if parsed:
                print(json.dumps(parsed, indent=4, ensure_ascii=False))
            else:
                print("JSON parsing failed, raw text returned.")
    except Exception as e:
        print(f"\nError running vLLM: {e}")
        print("Tip: Pastikan library vllm terinstall ('pip install vllm') dan CUDA tersedia.")
