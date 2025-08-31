import cv2
import numpy as np
import pathlib
from pathlib import Path
import os

class Preprocessing:
    def __init__(self):
        pass

    def load_process_data(self, image_path):
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError('Could not read image')

        cleaned_bgr = self.clean_artifacts(image_bgr)
        image_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)

        cv2.imshow('Original Image', image_bgr)
        cv2.imshow('Cleaned Image', cleaned_bgr)

        processed_data = {
            'original_path': image_path,
            'for_rgb': image_rgb.copy(),
            'for_hsv': cleaned_bgr.copy(),
            'for_lab': cleaned_bgr.copy(),
            'for_deconvolution': image_rgb.copy(),
            'cleaned_bgr': cleaned_bgr
        }

        return processed_data

    @staticmethod
    def clean_artifacts(image_bgr, kernel_size=5, background_threshold=220):
        cleaned_image = cv2.medianBlur(image_bgr, kernel_size)

        gray = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2GRAY)
        _, background_mask = cv2.threshold(gray, background_threshold, 255, cv2.THRESH_BINARY)

        if np.any(background_mask):
            mean_tissue_color = cv2.mean(cleaned_image, mask=cv2.bitwise_not(background_mask))[:3]
            cleaned_image[background_mask > 0] = mean_tissue_color

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channels = cv2.split(cleaned_image)
        channels_eq = [clahe.apply(ch) for ch in channels]
        cleaned_image = cv2.merge(channels_eq)

        return cleaned_image

    def process_folder(self, input_folder, output_folder=None):
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.svs']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(Path(input_folder).glob(f'*{ext}'))
            image_paths.extend(Path(input_folder).glob(f'*{ext.upper()}'))

        all_processed_data=[]
        for image_path in image_paths:
            try:
                processed_data = self.load_process_data(str(image_path))
                all_processed_data.append(processed_data)
                if output_folder:
                    self.load_process_data(processed_data, output_folder)
            except Exception as e:
              print(e)
        return all_processed_data

    def _save_processed_image(self, processed_data, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        original_path = Path(processed_data['original_path'])
        base_name = original_path.stem
        output_path = Path(output_folder) / f"{base_name}_cleaned.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(processed_data['for_rgb'], cv2.COLOR_RGB2BGR))
