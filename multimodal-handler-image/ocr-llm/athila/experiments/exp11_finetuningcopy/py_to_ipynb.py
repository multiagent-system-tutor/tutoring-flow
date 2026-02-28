import nbformat
import os

py_file = r'f:\projek dosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp11_finetuningcopy\explore.py'
nb_file = r'f:\projek dosen\tutoring\tutoring-flow\multimodal-handler-image\ocr-llm\athila\experiments\exp11_finetuningcopy\explore_recreated.ipynb'

def py_to_ipynb(py_path, ipynb_path):
    if not os.path.exists(py_path):
        print(f"Error: {py_path} not found.")
        return

    with open(py_path, 'r', encoding='utf-8') as f:
        code_lines = f.readlines()

    nb = nbformat.v4.new_notebook()
    
    # Simple logic: Split by likely cell delimiters or just put everything in one cell
    # Since nbconvert --to python keeps cells separated by special comments, we can try to use them.
    # Usually it uses "# In[ ]:" or similar. 
    
    current_cell_lines = []
    
    # Attempt to detect standard nbconvert separators (e.g. # In[...])
    # However, standard export often keeps them. Let's inspect the content structure flexibly.
    # We will treat '# In[' as a new cell marker.
    
    for line in code_lines:
        if line.strip().startswith("# In[") or line.strip().startswith("# %%"):
            if current_cell_lines:
                # Add previous cell
                source = "".join(current_cell_lines).strip()
                if source:
                    nb['cells'].append(nbformat.v4.new_code_cell(source))
                current_cell_lines = []
        else:
            current_cell_lines.append(line)
            
    # Add last cell
    if current_cell_lines:
        source = "".join(current_cell_lines).strip()
        if source:
             nb['cells'].append(nbformat.v4.new_code_cell(source))

    # Save notebook
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Successfully converted {py_path}\nTo: {ipynb_path}")

if __name__ == "__main__":
    py_to_ipynb(py_file, nb_file)
