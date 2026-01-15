import json
import os

nb_path = r'c:\projekdosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp2_ocr_ollama\explore.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# New content for the last cell
new_summary_code = [
    "%pip install matplotlib seaborn\n",
    "print(\"\\n=== SUMMARY ===\")\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "df = pd.DataFrame(results)\n",
    "if not df.empty:\n",
    "    print(f\"Average Time: {df['time'].mean():.4f}s\")\n",
    "    print(f\"Average CER (Raw): {df['cer_raw'].mean():.2%}\")\n",
    "    print(f\"Average CER (Refined): {df['cer_refined'].mean():.2%}\")\n",
    "    print(\"\\nDetailed Results exported to 'exp2_results.csv'\")\n",
    "    df.to_csv('exp2_results.csv', index=False)\n",
    "    \n",
    "    # --- Visualization ---\n",
    "    try:\n",
    "        plt.figure(figsize=(12, 6))\n",
    "        df_melted = df.melt(id_vars=['filename'], value_vars=['cer_raw', 'cer_refined'], var_name='Stage', value_name='CER')\n",
    "        \n",
    "        sns.barplot(data=df_melted, x='filename', y='CER', hue='Stage', palette='viridis')\n",
    "        plt.title('Comparison of CER: Raw OCR vs LLM Refinement')\n",
    "        plt.xlabel('Filename')\n",
    "        plt.ylabel('Character Error Rate (0.0 - 1.0)')\n",
    "        plt.xticks(rotation=45, ha='right')\n",
    "        plt.grid(axis='y', linestyle='--', alpha=0.5)\n",
    "        plt.tight_layout()\n",
    "        \n",
    "        # Save the plot\n",
    "        save_path = 'cer_comparison.png'\n",
    "        plt.savefig(save_path, dpi=300, bbox_inches='tight')\n",
    "        print(f\"Graph saved to {save_path}\")\n",
    "        \n",
    "        plt.show()\n",
    "    except Exception as e:\n",
    "        print(f\"Error plotting graph: {e}\")\n",
    "else:\n",
    "    print(\"No results to show.\")"
]

found_count = 0
for cell in cells:
    source = cell.get('source', [])
    source_text = "".join(source)
    
    if "print(\"\\n=== SUMMARY ===\")" in source_text or "df.to_csv('exp2_results.csv'" in source_text:
        cell['source'] = new_summary_code
        found_count += 1
        print("Updated Summary Cell to include savefig")
        break

print(f"Total updated cells: {found_count}")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
