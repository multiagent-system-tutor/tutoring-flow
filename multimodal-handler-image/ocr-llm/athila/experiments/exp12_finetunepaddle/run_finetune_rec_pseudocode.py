import subprocess
from pathlib import Path

# =====================================================
# PATH CONFIG
# =====================================================
EXPERIMENT_ROOT = Path(
    r"C:\projekdosen\tutoring\tutoring-flow"
    r"\multimodal-handler-image\ocr-llm\athila"
    r"\experiments\exp12_finetunepaddle"
)

PADDLEOCR_ROOT = Path(r"C:\projekdosen\tutoring\PaddleOCR")

DATASET_ROOT = EXPERIMENT_ROOT / "dataset_paddle"
TRAIN_LABEL = DATASET_ROOT / "train" / "train.txt"
TEST_LABEL  = DATASET_ROOT / "test" / "test.txt"

CONFIG_PATH = PADDLEOCR_ROOT / "configs" / "rec" / "rec_svtrnet_pseudocode.yml"
DICT_PATH   = PADDLEOCR_ROOT / "ppocr" / "utils" / "custom_dict.txt"

PRETRAINED_MODEL = PADDLEOCR_ROOT / "en_PP-OCRv4_rec_train" / "best_accuracy"
OUTPUT_DIR = PADDLEOCR_ROOT / "output" / "rec_svtr_pseudocode"

# =====================================================
# STEP 1 — CHARACTER DICT (WITH SPACE)
# =====================================================
DICT_CONTENT = """a
b
c
d
e
f
g
h
i
j
k
l
m
n
o
p
q
r
s
t
u
v
w
x
y
z
A
B
C
D
E
F
G
H
I
J
K
L
M
N
O
P
Q
R
S
T
U
V
W
X
Y
Z
0
1
2
3
4
5
6
7
8
9
+
-
*
/
=
<
>
(
)
[
]
:
;
,
.
_
←
≤
≥
×
 """  # ← BARIS TERAKHIR ADALAH SPASI

DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
DICT_PATH.write_text(DICT_CONTENT, encoding="utf-8")
print(f"[OK] custom_dict.txt ditulis ke {DICT_PATH}")

# =====================================================
# STEP 2 — YAML CONFIG (EVAL PRAKTIS MATI)
# =====================================================
CONFIG_YAML = f"""
Global:
  use_gpu: False
  epoch_num: 50
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: {OUTPUT_DIR.as_posix()}
  save_epoch_step: 5
  eval_batch_step: 999999999   # 🔥 EVAL PRAKTIS MATI, TAPI TIPE INT
  character_dict_path: {DICT_PATH.as_posix()}
  max_text_length: 120
  infer_mode: False
  use_space_char: True

Architecture:
  model_type: rec
  algorithm: SVTR
  Backbone:
    name: SVTRNet
    img_size: [32, 320]
    out_char_num: 84
  Neck:
    name: SequenceEncoder
    encoder_type: reshape
  Head:
    name: CTCHead

Loss:
  name: CTCLoss

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    learning_rate: 0.0005

PostProcess:
  name: CTCLabelDecode
  character_dict_path: {DICT_PATH.as_posix()}
  use_space_char: True

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: .
    label_file_list:
      - {TRAIN_LABEL.as_posix()}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - RecResizeImg:
          image_shape: [3, 32, 320]
      - CTCLabelEncode:
          character_dict_path: {DICT_PATH.as_posix()}
          max_text_length: 120
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: True
    batch_size_per_card: 8
    drop_last: True
    num_workers: 2

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: .
    label_file_list:
      - {TEST_LABEL.as_posix()}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - RecResizeImg:
          image_shape: [3, 32, 320]
      - CTCLabelEncode:
          character_dict_path: {DICT_PATH.as_posix()}
          max_text_length: 120
      - KeepKeys:
          keep_keys: ['image', 'label', 'length']
  loader:
    shuffle: False
    batch_size_per_card: 8
    drop_last: False
    num_workers: 2
"""

CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATH.write_text(CONFIG_YAML.strip(), encoding="utf-8")
print(f"[OK] Config ditulis ke {CONFIG_PATH}")

# =====================================================
# STEP 3 — CHECK PRETRAINED
# =====================================================
if not PRETRAINED_MODEL.with_suffix(".pdparams").exists():
    raise FileNotFoundError(f"Pretrained model tidak ditemukan: {PRETRAINED_MODEL}")

print("[OK] Pretrained model ditemukan")

# =====================================================
# STEP 4 — RUN TRAINING
# =====================================================
print("\n🔥 MULAI FINE-TUNING PADDLEOCR (RECOGNITION)\n")

cmd = [
    "python",
    str(PADDLEOCR_ROOT / "tools" / "train.py"),
    "-c",
    str(CONFIG_PATH),
    "-o",
    f"Global.pretrained_model={PRETRAINED_MODEL.as_posix()}",
]

subprocess.run(cmd, cwd=PADDLEOCR_ROOT)
