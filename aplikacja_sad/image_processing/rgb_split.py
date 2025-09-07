from PIL import Image
import os


def rgb_split(image_path):
    try:
        img = Image.open(image_path)
        r, g, b = img.split()

        return {
            'Red Channel': r,
            'Green Channel': g,
            'Blue Channel': b
        }
    except Exception as e:
        raise Exception(f"RGB split failed: {str(e)}")