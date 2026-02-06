import json

nb_path = r'f:\projek dosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp11_finetuningcopy\explore.ipynb'

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total cells: {len(data['cells'])}")
    for i, cell in enumerate(data['cells']):
        if cell['cell_type'] == 'markdown':
            content = "".join(cell['source'])
            print(f"Cell {i} [MD]: {content[:50]}...")
except Exception as e:
    print(e)
