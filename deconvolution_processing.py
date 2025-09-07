from deconvolution import Deconvolution
from wavelet import Wavelet
from ruifork import Ruifork
import cv2
from sklearn.decomposition import FastICA
import numpy as np

class Process:
    @staticmethod
    def process_image(image, visualise=True, method='wavelet', threshold=0.15):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        og_shape = image_rgb.shape
        image_od = Deconvolution.convert_OD(image_rgb)
        od_flat = image_od.reshape(-1, 3)
        method = method.lower()
        valid_methods = ['wavelet', 'ruifork', 'reference']
        if method not in valid_methods:
            raise ValueError(f"Unknown method: {method}. Choose from {valid_methods}")

        if method == 'wavelet':
            wavelet_bands = Wavelet.wavelet_decomposition(image_od, levels=3)
            three_channel_bands = Wavelet.prep_3_channel(wavelet_bands, og_shape[:2])
            selected_bands = Wavelet.select_best_bands(
                three_channel_bands,
                num_bands_to_select=20,
                variance_threshold=1e-10,
                plot_selection=False
            )

            if selected_bands is not None:
                ica_matrix, _ = Wavelet.prepare_ica(selected_bands, image_od)
                stain_matrix, _ = Wavelet.estimate_stain_matrix(ica_matrix, od_flat, n_comp=2)
            else:
                ica = FastICA(n_components=2, random_state=42, max_iter=1000)
                ica_components = ica.fit_transform(od_flat)
                stain_matrix = ica.mixing_
        elif method == 'reference':
            stain_matrix = np.array([
                [0.65, 0.07],
                [0.70, 0.99],
                [0.29, 0.11]
            ])
        elif method == 'ruifork':
            stain_matrix = Ruifork.estimate_stain_matrix(od_flat, threshold=threshold)
        else:
            raise ValueError(f"Nieznana metoda: {method}")
        stain_matrix = Deconvolution.normalize_stain_matrix(stain_matrix)
        concentrations = Deconvolution.deconv_stains(od_flat, stain_matrix)
        reconstructed_image = Deconvolution.reconstruct(concentrations, stain_matrix, og_shape)
        stain_images = Deconvolution.extract_individual_stains(concentrations, stain_matrix, og_shape)

        if visualise:
            Deconvolution.visualise_image(image_rgb, reconstructed_image, stain_images, concentrations)

        return stain_images, stain_matrix, concentrations
