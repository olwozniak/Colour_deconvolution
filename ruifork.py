import cv2
from sklearn.decomposition import PCA
import numpy as np
from decomposition import Decomposition

class Ruifork(Decomposition):

    @staticmethod
    def estimate_stain_matrix(od_flat, threshold=0.05, n_components=2):

        foreground_mask = np.linalg.norm(od_flat, axis=1) > threshold
        od_foreground = od_flat[foreground_mask]
        pca = PCA(n_components=n_components)
        pca.fit(od_foreground)
        M = pca.components_.T

        return M

    @staticmethod
    def process_image(image, visualise=True, threshold=0.15):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        og_shape = image_rgb.shape
        image_od = Ruifork.convert_OD(image_rgb)
        od_flat = image_od.reshape(-1, 3)
        stain_matrix = Ruifork.estimate_stain_matrix(od_flat, threshold=threshold)
        concentrations = Ruifork.deconv_stains(od_flat, stain_matrix)
        reconstructed_image = Ruifork.reconstruct(concentrations, stain_matrix, og_shape)
        stain_images = Ruifork.extract_individual_stains(concentrations, stain_matrix, og_shape)
        if visualise:
            Ruifork.visualise_image(image_rgb, reconstructed_image, stain_images, concentrations)
        return stain_images, stain_matrix, concentrations

