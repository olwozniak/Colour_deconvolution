import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from wavelet import Wavelet


class Normalization:
    def __init__(self, target_image):
        self.target_image = target_image
        self.wavelet_processor = Wavelet()
        self.target_dtsin_dtstd = self._extract_target_stains_statistics(target_image)

    def all_pixel_quantile_norm(self, image):
        source_flat = image.reshape(-1,3)
        target_flat = self.target_image.reshape(-1,3)
        normalized_flat = np.zeros_like(source_flat)
        for c in range(3):
            src_sorted = np.sort(source_flat[:,c])
            target_sorted = np.sort(target_flat[:,c])
            normalized_flat[:,c] = np.interp(source_flat[:, c], src_sorted,target_sorted)

        normalized_image = normalized_flat.reshape(image.shape)
        return normalized_image.astype(np.uint8)

    def colour_dmap_quantile_norm(self, image):
        source_colour = np.unique(image.reshape(-1, 3), axis=0)
        target_colour = np.unique(self.target_image.reshape(-1, 3), axis=0)

        normalized_colours = np.zeros_like(source_colour)
        for c in range(3):
            src_sorted = np.sort(source_colour[:,c])
            target_sorted = np.sort(target_colour[:,c])
            normalized_colours[:,c] = np.interp(source_colour[:,c], src_sorted,target_sorted)

        normalized_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel = image[i, j]
                dist = np.linalg.norm(normalized_colours - pixel, axis=1)
                idx = np.argmin(dist)
                normalized_image[i, j] = normalized_colours[idx]

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

    def vb_reinhard_weighted(self, image, n_components=3):
        source_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2LAB).astype(np.float32)

        src_flat = source_lab.reshape(-1, 3)
        trg_flat = target_lab.reshape(-1, 3)

        gmm_src = BayesianGaussianMixture(n_components=n_components, random_state=0).fit(src_flat)
        src_probs = gmm_src.predict_proba(src_flat)

        gmm_trg = BayesianGaussianMixture(n_components=n_components, random_state=0).fit(trg_flat)

        src_means = gmm_src.means_
        src_stds = np.sqrt(gmm_src.covariances_.diagonal(axis1=1, axis2=2))
        trg_means = gmm_trg.means_
        trg_stds = np.sqrt(gmm_trg.covariances_.diagonal(axis1=1, axis2=2))

        normalized_flat = np.zeros_like(src_flat)
        for i in range(len(src_flat)):
            pixel = src_flat[i]
            probs = src_probs[i]
            transformed_pixel = np.zeros(3)
            for j in range(n_components):
                term = (pixel - src_means[j]) * (trg_stds[j] / src_stds[j]) + trg_means[j]
                transformed_pixel += probs[j] * term
            normalized_flat[i] = transformed_pixel

        normalized_lab = normalized_flat.reshape(source_lab.shape)
        normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

        return result

    def stain_clustering(self, stain_channel, method='kmeans'):
        flat_channel = stain_channel.flatten()
        flat_channel = flat_channel[np.isfinite(flat_channel)].reshape(-1,1)

        if method == 'kmeans':
            kmeans = KMeans(n_clusters=2, random_state=0).fit(flat_channel)
            labels = kmeans.labels_
            centers = kmeans.clusters_centers_
            fg_label = np.argmax(centers)
            bg_label = 1 - fg_label
            fg_mask = (labels == fg_label).reshape(stain_channel.shape)
        elif method == 'em':
            gmm = GaussianMixture(n_components=2, random_state=0).fit(flat_channel)
            labels = gmm.predict(flat_channel)
            centers = gmm.means_
            fg_label = np.argmax(centers)
            bg_label = 1 - fg_label
            fg_mask = (labels == fg_label).reshape(stain_channel.shape)
        elif method == 'vb':
            bgm = BayesianGaussianMixture(n_components=2, random_state=0).fit(flat_channel)
            labels = bgm.predict(flat_channel)
            centers = bgm.means_
            fg_label = np.argmax(centers)
            bg_label = 1 - fg_label
            fg_mask = (labels == fg_label).reshape(stain_channel.shape)
        else:
            raise ValueError('method must be either kmeans, em or vb')

        return fg_mask, centers[fg_label], centers[bg_label]

    def cvd_mm_normalization(self, image, cluster_method='kmeans', deconv_method='wavelet'):
        source_stains = self.color_deconvolution(image)
        target_stains = self.color_deconvolution(self.target_image)

        normalized_stains = np.zeros_like(source_stains)
        n_stains = source_stains.shape[2]



































