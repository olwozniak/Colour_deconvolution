import cv2
import numpy as np
from sklearn.mixture import BayesianGaussianMixture


class Reinhard:
    def __init__(self, target_image):
        self.target_image = target_image

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