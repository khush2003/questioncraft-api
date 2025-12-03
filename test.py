
# from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
# from pypdfium2 import PdfDocument
import logging

# ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory

# Configure logging
logging.basicConfig(level=logging.INFO)

# pdf = PdfDocument('test.pdf')

# for i in range(len(pdf)):
#     page = pdf.get_page(i)
#     image = page.render(
#         scale=3
#     )
#     pil_image = image.to_pil()
#     pil_image.save(f'img_{i}.png')
    


img_path = 'img_test.png'

# def perform_ocr(img):
#     try:
#         img = np.array(img)
#         result = ocr.ocr(img)
#         text = ""
#         for idx in range(len(result)):
#             res = result[idx]
#             for line in res:
#                 text += line[1][0] + " "
#                 print(line[1][0])

#         print("\n\n\n ---------- \n")
#         print(text)
#         return text
#     except Exception as e:
#         logging.error(f"Error processing image: {e}")
#         return ""
    
    
# if __name__ == "__main__":
    # img_paths = ['img_02.jpg', 'img_test.png', 'img_test_1.png', 'img_test_2.png']  # Add your image paths here
    # perform_ocr_batch(img_paths)
    # asyncio.run(perform_ocr_batch_async(img_paths))