from sklearn.decomposition import PCA
import numpy as np

class Ruifork():

    @staticmethod
    def estimate_stain_matrix(od_flat, threshold=0.05, n_components=2):

        foreground_mask = np.linalg.norm(od_flat, axis=1) > threshold
        od_foreground = od_flat[foreground_mask]
        pca = PCA(n_components=n_components)
        pca.fit(od_foreground)
        M = pca.components_.T

        return M

