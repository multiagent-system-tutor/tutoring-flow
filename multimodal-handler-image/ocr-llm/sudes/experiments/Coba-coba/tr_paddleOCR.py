# from paddleocr import PaddleOCR
# import cv2
# from my_timer import my_timer
# import os

# os.environ["FLAGS_use_mkldnn"] = "0"
# os.environ["FLAGS_enable_pir_api"] = "0"



# @my_timer
# def run_PaddleOR(img):
#     ocr = PaddleOCR(
#             use_textline_orientation=False,  # pengganti use_angle_cls
#             lang="en",
#             ocr_version="PP-OCRv4")
#     result = ocr.ocr(img)
#     for line in result[0]:
#         print(line[1][0])
    
#     for line in result[0]:
#         box = line[0]
#         text = line[1][0]

#     # ubah ke format numpy
#     pts = [(int(x), int(y)) for x, y in box]
#     pts = pts + [pts[0]]  # tutup kotak

#     for i in range(len(pts)-1):
#         cv2.line(img, pts[i], pts[i+1], (0,255,0), 2)

#     # tulis teks
#     cv2.putText(
#         img,
#         text,
#         (pts[0][0], pts[0][1]-5),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.6,
#         (0,0,255),
#         2
#     )

#     cv2.imshow("OCR Result", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     image = cv2.imread("../dataset/HW_bagus.jpeg")
#     run_PaddleOR(image)


from paddleocr import PaddleOCR
import cv2
from my_timer import my_timer
import numpy as np
import os
os.environ["FLAGS_use_onednn"] = "0"


@my_timer
def run_paddle(image):
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='en'
    )

    result= ocr.predict(image)
    print(result[1][1][0])


if __name__ == "__main__":
    # image1 = cv2.imread("../dataset/HW_bagus.jpeg")
    # run_PaddleOR(image1)

    image2 = cv2.imread("../dataset/HW_jelek.jpg")
    run_paddle(image2)

