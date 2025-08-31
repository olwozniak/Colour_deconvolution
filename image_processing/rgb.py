import cv2
import numpy as np


class RGB:
    def __init__(self):
        pass

    def analyze_rgb_channels(processed_data):
        image_rgb = processed_data['for_rgb']

        r_channel = image_rgb[:, :, 0]  # Czerwony
        g_channel = image_rgb[:, :, 1]  # Zielony
        b_channel = image_rgb[:, :, 2]  # Niebieski

        return {
            'original_rgb': image_rgb,
            'r_channel': r_channel,
            'g_channel': g_channel,
            'b_channel': b_channel
        }
    def segment_he_rgb(rgb_data, hematoxylin_threshold=100, eosin_threshold=150):
        r_channel = rgb_data['r_channel']
        g_channel = rgb_data['g_channel']
        b_channel = rgb_data['b_channel']

        _,hematoxylin_mask = cv2.threshold(r_channel, hematoxylin_threshold, 255, cv2.THRESH_BINARY_INV)
        _, eosin_mask = cv2.threshold(g_channel, eosin_threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        hematoxylin_mask = cv2.morphologyEx(hematoxylin_mask, cv2.MORPH_OPEN, kernel)
        eosin_mask = cv2.morphologyEx(eosin_mask, cv2.MORPH_OPEN, kernel)

        return {
            'hematoxylin_mask': hematoxylin_mask,
            'eosin_mask': eosin_mask,
            'combined_mask': cv2.bitwise_or(hematoxylin_mask, eosin_mask)
        }