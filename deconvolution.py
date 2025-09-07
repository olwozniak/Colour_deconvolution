import numpy as np
import matplotlib.pyplot as plt

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
    def extract_individual_stains(concentration, stain_matrix, og_shape, ):
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
