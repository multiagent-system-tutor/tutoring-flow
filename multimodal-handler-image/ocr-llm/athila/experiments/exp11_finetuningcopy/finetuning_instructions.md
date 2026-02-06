
# Workflow for Running PaddleOCR Fine-tuning

## 1. Prerequisites
- **Config File**: You need the `.yml` config file (e.g., `en_PP-OCRv4_rec.yml`).
- **Pretrained Model**: Download the model (e.g., `en_PP-OCRv4_rec_train`) and unzip it.
- **Dataset**: Run the dataset preparation cell in the notebook to generate `rec_gt_train.txt`.

## 2. Path Setup
The command paths depend on where your files are located.

### Scenario A: Files in `exp11_finetuningcopy` (Simplest)
Copy your `configs` folder and `pretrain_models` folder into `exp11_finetuningcopy`.
Then run:
```bash
python -m paddleocr.tools.train -c configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml \
-o Global.pretrained_model=./pretrain_models/en_PP-OCRv4_rec_train/best_accuracy \
Train.dataset.data_dir=./ \
Train.dataset.label_file_list=['dataset_lists/rec_gt_train.txt'] \
Eval.dataset.data_dir=./ \
Eval.dataset.label_file_list=['dataset_lists/rec_gt_test.txt'] \
Global.save_model_dir=./output/rec_finetune
```

### Scenario B: Files Elsewhere (Use Absolute Paths)
If your config and model are in different folders, use **Absolute Paths**.
Example (Replace `F:/path/to/...` with your actual paths):

```bash
python -m paddleocr.tools.train -c "F:/path/to/PaddleOCR/configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml" \
-o Global.pretrained_model="F:/path/to/models/en_PP-OCRv4_rec_train/best_accuracy" \
Train.dataset.data_dir=./ \
Train.dataset.label_file_list=['dataset_lists/rec_gt_train.txt'] \
Eval.dataset.data_dir=./ \
Eval.dataset.label_file_list=['dataset_lists/rec_gt_test.txt'] \
Global.save_model_dir=./output/rec_finetune
```

## 3. Notes
- **`Train.dataset.data_dir=./`**: This works because your `rec_gt_train.txt` contains **absolute paths** to the images.
- **`label_file_list`**: This is relative because the notebook generates `dataset_lists/` inside the current folder.
