from PIL import Image 
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel 
import os
import requests
from my_timer import my_timer


@my_timer
def run_trOCR(model_name="microsoft/trocr-base-printed", images=""):
    """
    There are 3 main models to choose from, small, base and large. 
    Some other fine-tuned models: IAM Handwritten, SROIE Receipts
    """
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {device}")
    model.to(device)  # Move model to GPU
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_new_tokens=1000)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)


if __name__ == "__main__":
    model_id = "microsoft/trocr-base-handwritten"

    image = Image.open("../dataset/HW_bagus.jpeg").convert("RGB")
    run_trOCR(model_id, image)

    image = Image.open("../dataset/HW_jelek.jpg").convert("RGB")
    run_trOCR(model_id, image)

    # image = Image.open("../dataset/trocr_image.jpg").convert("RGB")
    # run_trOCR(model_id, image)

    image = Image.open("../dataset/tes_full.png").convert("RGB")
    run_trOCR(model_id, image)

    image = Image.open("../dataset/tes_sebaris.png").convert("RGB")
    run_trOCR(model_id, image)

    image = Image.open("../dataset/HW_sebaris.png").convert("RGB")
    run_trOCR(model_id, image)