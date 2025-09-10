import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from deconvolution import Deconvolution
import matplotlib.pyplot as plt
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

class Normalization:
    def __init__(self, target_image):
        self.target_image = target_image
        self.target_od = Deconvolution.convert_OD(target_image)
        self.target_od_flat = self.target_od.reshape(-1, 3)
        self.target_stain_matrix = None
        self.target_concentrations = None
        self.target_stats = {}

    def extract_target_stains_stats(self, method='reference'):
        self.target_stain_matrix = self.estimate_stain_matrix(self.target_od, method)
        self.target_concentrations = Deconvolution.deconv_stains(self.target_od_flat, self.target_stain_matrix)
        n_stains = self.target_stain_matrix.shape[1]

        for i in range(n_stains):
            stain_conc = self.target_concentrations[:, i]
            stain_conc = stain_conc[np.isfinite(stain_conc)]

            if len(stain_conc) > 1:
                fg_mask, fg_mean, fg_std = self.cluster_stain_channel(stain_conc, method='kmeans')
                self.target_stats[i] = (fg_mean, fg_std)
            else:
                self.target_stats[i] = (np.mean(stain_conc) if len(stain_conc) > 0 else 0,
                                        np.std(stain_conc) if len(stain_conc) > 1 else 1.0)

    def estimate_stain_matrix(self, image_od, method='reference'):
        od_flat = image_od.reshape(-1, 3)
        if method == 'reference':
            stain_matrix = np.array([
                [0.65, 0.07, 0.27],
                [0.70, 0.99, 0.11],
                [0.29, 0.11, 0.96]
            ]).T
        else:
            from sklearn.decomposition import FastICA
            ica = FastICA(n_components=3, random_state=42, max_iter=1000)
            ica_components = ica.fit_transform(od_flat)
            stain_matrix = ica.mixing_

        return Deconvolution.normalize_stain_matrix(stain_matrix)

    def cluster_stain_channel(self, stain_channel, method='kmeans'):
        flat_channel = stain_channel.flatten()
        flat_channel = flat_channel[np.isfinite(flat_channel)].reshape(-1, 1)
        if len(flat_channel) < 2:
            return np.ones(len(stain_channel), dtype=bool), np.mean(stain_channel), np.std(stain_channel) if len(
                stain_channel) > 0 else 1.0

        try:
            if method == 'kmeans':
                kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(flat_channel)
                centers = kmeans.cluster_centers_.flatten()
                labels = kmeans.labels_
            elif method == 'em':
                gmm = GaussianMixture(n_components=2, random_state=0).fit(flat_channel)
                labels = gmm.predict(flat_channel)
                centers = gmm.means_.flatten()
            elif method == 'vb':
                bgm = BayesianGaussianMixture(n_components=2, random_state=0).fit(flat_channel)
                labels = bgm.predict(flat_channel)
                centers = bgm.means_.flatten()
            else:
                raise ValueError("Method must be 'kmeans', 'em', or 'vb'")

            fg_label = np.argmax(centers)
            fg_mask = (labels == fg_label)
            fg_values = flat_channel[fg_mask]

            if len(fg_values) > 0:
                fg_mean = np.mean(fg_values)
                fg_std = np.std(fg_values)
            else:
                fg_mean = 0
                fg_std = 1

            return fg_mask, fg_mean, fg_std

        except Exception as e:
            print(f"Error in clustering: {e}")
            return np.ones(len(flat_channel), dtype=bool), np.mean(flat_channel), np.std(flat_channel) if len(
                flat_channel) > 0 else 1.0

    def all_pixel_quantile_norm(self, image):
        source_flat = image.reshape(-1,3)
        target_flat = self.target_image.reshape(-1,3)
        normalized_flat = np.zeros_like(source_flat)
        for c in range(3):
            src_sorted = np.sort(source_flat[:,c])
            target_sorted = np.sort(target_flat[:,c])
            min_length = min(len(src_sorted), len(target_sorted))
            src_sorted = src_sorted[:min_length]
            target_sorted = target_sorted[:min_length]

            unique_src, src_indices = np.unique(source_flat[:, c], return_inverse=True)
            normalized_flat[:, c] = np.interp(source_flat[:, c], src_sorted, target_sorted)

        return normalized_flat.reshape(image.shape).astype(np.uint8)

    def colour_map_quantile_norm(self, image):
        source_colour = np.unique(image.reshape(-1, 3), axis=0)
        target_colour = np.unique(self.target_image.reshape(-1, 3), axis=0)

        # Sprawdź, czy mamy wystarczająco dużo punktów do interpolacji
        if len(source_colour) < 2 or len(target_colour) < 2:
            print("Warning: Not enough unique colors for quantile normalization. Returning original image.")
            return image

        normalized_colours = np.zeros_like(source_colour)
        for c in range(3):
            src_channel = source_colour[:, c]
            target_channel = target_colour[:, c]

            # Sortuj kanały
            src_sorted = np.sort(src_channel)
            target_sorted = np.sort(target_channel)

            # Upewnij się, że tablice mają tę samą długość
            min_length = min(len(src_sorted), len(target_sorted))
            if min_length < 2:
                # Jeśli nie ma wystarczająco punktów, użyj prostej normalizacji liniowej
                src_min, src_max = np.min(src_channel), np.max(src_channel)
                target_min, target_max = np.min(target_channel), np.max(target_channel)

                if src_max - src_min > 0:
                    normalized = (src_channel - src_min) * (target_max - target_min) / (src_max - src_min) + target_min
                else:
                    normalized = np.full_like(src_channel, target_min)
            else:
                # Przycinaj do tej samej długości
                src_sorted = src_sorted[:min_length]
                target_sorted = target_sorted[:min_length]

                # Interpolacja kwantylowa
                normalized = np.interp(src_channel, src_sorted, target_sorted)

            normalized_colours[:, c] = normalized

        # Tworzenie mapy kolorów
        color_map = {}
        for src, norm in zip(source_colour, normalized_colours):
            color_map[tuple(src)] = norm

        # Zastosowanie mapowania do całego obrazu
        normalized_image = np.zeros_like(image)
        flat_image = image.reshape(-1, 3)

        for i, pixel in enumerate(flat_image):
            pixel_tuple = tuple(pixel)
            if pixel_tuple in color_map:
                normalized_image.reshape(-1, 3)[i] = color_map[pixel_tuple]
            else:
                # Znajdź najbliższy kolor w source_colour
                distances = np.linalg.norm(source_colour - pixel, axis=1)
                closest_idx = np.argmin(distances)
                normalized_image.reshape(-1, 3)[i] = normalized_colours[closest_idx]

        return normalized_image.astype(np.uint8)

    def reinhard_normalization(self, image):
        source_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        src_mean, src_std = np.mean(source_lab, axis=(0,1)), np. std(source_lab, axis=(0,1))
        trg_mean, trg_std = np.mean(target_lab, axis=(0,1)), np.std(target_lab, axis=(0,1))

        normalized_lab = (source_lab - src_mean) * (trg_std / src_std) + trg_mean
        normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)

        result = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

        return result

    def stain_specific_normalization(self, image, cluster_method='kmeans'):
        image_od = Deconvolution.convert_OD(image)
        od_flat = image_od.reshape(-1, 3)

        if self.target_stain_matrix is None:
            self.extract_target_stains_stats()

        stain_matrix = self.target_stain_matrix
        concentrations = Deconvolution.deconv_stains(od_flat, stain_matrix)
        n_stains = stain_matrix.shape[1]
        normalized_concentrations = np.zeros_like(concentrations)

        for i in range(n_stains):
            conc = concentrations[:, i]
            valid_conc = conc[np.isfinite(conc)]
            if len(valid_conc) == 0:
                normalized_concentrations[:, i] = conc
                continue

            fg_mask, fg_mean, fg_std = self.cluster_stain_channel(conc, cluster_method)
            if i not in self.target_stats:
                self.extract_target_stains_stats()

            trg_mean, trg_std = self.target_stats[i]
            if fg_std == 0:
                fg_std = 1e-6

            normalized_foreground = (conc[fg_mask] - fg_mean) * (trg_std / fg_std) + trg_mean
            normalized_concentrations[fg_mask, i] = normalized_foreground
            normalized_concentrations[~fg_mask, i] = conc[~fg_mask]

        normalized_od_flat = np.dot(normalized_concentrations, stain_matrix.T)
        normalized_rgb = 255 * np.exp(-normalized_od_flat)
        normalized_rgb = np.clip(normalized_rgb, 0, 255).astype(np.uint8)
        normalized_rgb = normalized_rgb.reshape(image.shape)

        return normalized_rgb

    def normalize(self, image, method='stain_specific', **kwargs):
        if method == 'all_pixel_quantile':
            return self.all_pixel_quantile_norm(image)
        elif method == 'colour_map_quantile':
            return self.colour_map_quantile_norm(image)
        elif method == 'reinhard':
            return self.reinhard_normalization(image)
        elif method == 'stain_specific':
            valid_kwargs = {}
            if 'cluster_method' in kwargs:
                valid_kwargs['cluster_method'] = kwargs['cluster_method']
            return self.stain_specific_normalization(image, **valid_kwargs)
        else:
            raise ValueError("Unsupported normalization method")

    def visualize_normalization(self, original, normalized, title="Normalization Result"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(normalized)
        axes[1].set_title(title)
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()
