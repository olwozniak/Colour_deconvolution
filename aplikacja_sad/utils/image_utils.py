from PIL import Image

def load_image(filepath):
    try:
        return Image.open(filepath)
    except Exception as e:
        raise Exception(f"Failed to load image: {str(e)}")

def resize_image(image, scale):
    width = int(image.width * scale)
    height = int(image.height * scale)
    return image.resize((width, height), Image.Resampling.LANCZOS)