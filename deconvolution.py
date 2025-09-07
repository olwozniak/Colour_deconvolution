import numpy as np
import matplotlib.pyplot as plt
from wavelet import Wavelet
import cv2
from ruifork import Ruifork
from sklearn.decomposition import FastICA

class Deconvolution:
    @staticmethod
    def convert_OD(image, I_0=255):
        image = np.clip(image.astype(np.float64), 1, None)
        od_image = -np.log10(image / I_0)
        return od_image

    @staticmethod
    def deconv_stains(od_flat, stain_matrix):
        if stain_matrix.shape[0] != 3:
            stain_matrix = stain_matrix.T
        stain_matrix_inv = np.linalg.pinv(stain_matrix)
        concentration = np.dot(od_flat, stain_matrix_inv.T)
        return concentration

    @staticmethod
    def reconstruct(concentration, stain_matrix, og_shape):
        od_recon = np.dot(concentration, stain_matrix.T)
        rgb_recon = 255 * np.exp(-od_recon)
        rgb_recon = np.clip(rgb_recon, 0, 255).astype(np.uint8)
        rgb_recon = rgb_recon.reshape(og_shape)

        return rgb_recon

    @staticmethod
    def extract_individual_stains(concentration, stain_matrix, og_shape):
        stain_images = []
        for i in range(stain_matrix.shape[1]):
            od_single = np.outer(concentration[:, i], stain_matrix[:, i])
            od_single = np.clip(od_single, 0, None)
            rgb_single = 255 * np.exp(-od_single)
            rgb_single = np.clip(rgb_single, 0, 255).astype(np.uint8)
            rgb_single = rgb_single.reshape(og_shape)
            stain_images.append(rgb_single)

        return stain_images

    @staticmethod
    def normalize_stain_matrix(stain_matrix):
        stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0)

        for i in range(stain_matrix.shape[1]):
            if stain_matrix[:, i].sum() < 0:
                stain_matrix[:, i] *= -1
        order = []
        for i in range(stain_matrix.shape[1]):
            r, g, b = stain_matrix[:, i]
            if b > r:
                order.insert(0, i)
            else:
                order.append(i)

        stain_matrix = stain_matrix[:, order]

        return stain_matrix

    @staticmethod
    def image_diff(img1, img2):
        diff = np.abs(img1.astype(float) - img2.astype(float))
        diff = diff / np.max(diff)
        return diff

    @staticmethod
    def visualise_image(og_image, image_recon, stain_image, concentration):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(og_image)
        axes[0, 0].set_title('Oryginalny obraz')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(image_recon)
        axes[0, 1].set_title('Obraz rekonstruowany')
        axes[0, 1].axis('off')

        diff = Deconvolution.image_diff(og_image, image_recon)
        axes[0, 2].imshow(diff, cmap='hot')
        axes[0, 2].set_title('Różnica (oryginał - rekonstrukcja)')
        axes[0, 2].axis('off')

        for i in range(min(2, len(stain_image))):
            axes[1, i].imshow(stain_image[i])
            axes[1, i].set_title(f'Barwnik {i + 1}')
            axes[1, i].axis('off')

        if concentration.shape[1] > 0:
            axes[1, 2].hist(concentration[:, 0], bins=50, alpha=0.7, label='Barwnik 1')
        if concentration.shape[1] > 1:
            axes[1, 2].hist(concentration[:, 1], bins=50, alpha=0.7, label='Barwnik 2')
        axes[1, 2].set_title('Histogram stężeń barwników')
        axes[1, 2].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def process_image(image, visualise=True, method='wavelet', threshold=0.15):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        og_shape = image_rgb.shape
        image_od = Deconvolution.convert_OD(image_rgb)
        od_flat = image_od.reshape(-1, 3)

        if method == 'wavelet':
            wavelet_bands = Wavelet.wavelet_decomposition(image_od, levels=3)
            three_channel_bands = Wavelet.prep_3_channel(wavelet_bands, og_shape[:2])
            selected_bands = Wavelet.select_best_bands(three_channel_bands, num_bands_to_select=20,
                                                       variance_threshold=1e-10, plot_selection=False)

            if selected_bands is not None:
                ica_matrix, _ = Wavelet.prepare_ica(selected_bands, image_od)
                stain_matrix, _ = Wavelet.estimate_stain_matrix(ica_matrix, od_flat, n_comp=2)
            else:
                ica = FastICA(n_components=2, random_state=42, max_iter=1000)
                ica_components = ica.fit_transform(od_flat)
                stain_matrix = ica.mixing_

        else:
            stain_matrix = Ruifork.estimate_stain_matrix(od_flat, threshold=threshold)

        stain_matrix = Deconvolution.normalize_stain_matrix(stain_matrix)
        stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0)
        concentrations = Deconvolution.deconv_stains(od_flat, stain_matrix)
        reconstructed_image = Deconvolution.reconstruct(concentrations, stain_matrix, og_shape)
        stain_images = Deconvolution.extract_individual_stains(concentrations, stain_matrix, og_shape)

        if visualise:
            Deconvolution.visualise_image(image_rgb, reconstructed_image, stain_images, concentrations)

        return stain_images, stain_matrix, concentrations