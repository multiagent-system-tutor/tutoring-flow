import json

nb_path = r'f:\projek dosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp11_finetuningcopy\explore.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Print cells 9, 10, 11
    for i in range(8, 12):
        if i < len(data['cells']):
            cell = data['cells'][i]
            print(f"Cell {i} Type: {cell['cell_type']}")
            print(f"Source: {cell['source']}")
            print("-" * 20)
except Exception as e:
    print(e)
