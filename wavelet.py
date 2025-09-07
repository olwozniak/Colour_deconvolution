import cv2
import pywt
import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from deconvolution import Deconvolution


class Wavelet():
    def __init__(self):
        pass

    @staticmethod
    def wavelet_decomposition(od_image, wavelet='haar', levels=5):
        all_bands = []
        for i in range(3):
            channel = od_image[:, :, i]
            channel_bands = []

            coeffs = pywt.wavedec2(channel, wavelet, level=levels)
            approximation = coeffs[0]
            channel_bands.append(('approx', approximation))

            for level_idx in range(1, len(coeffs)):
                h, v, d = coeffs[level_idx]
                channel_bands.append((f'h{level_idx}', h))
                channel_bands.append((f'v{level_idx}', v))
                channel_bands.append((f'd{level_idx}', d))

            all_bands.append(channel_bands)
        return all_bands

    @staticmethod
    def prep_3_channel(wavelet_bands, original_shape):
        band_types = {}
        for channel_idx, channel_bands in enumerate(wavelet_bands):
            for band_name, band_data in channel_bands:
                if band_name not in band_types:
                    band_types[band_name] = [None, None, None]
                if band_data.shape != original_shape:
                    resized_data = cv2.resize(band_data, (original_shape[1], original_shape[0]),
                                              interpolation=cv2.INTER_CUBIC)
                    band_types[band_name][channel_idx] = resized_data
                else:
                    band_types[band_name][channel_idx] = band_data

        three_channel_bands = []

        for band_name, channel_data in band_types.items():
            if all(data is not None for data in channel_data):
                three_channel_band = np.stack(channel_data, axis=-1)
                three_channel_bands.append((band_name, three_channel_band))

        return three_channel_bands

    @staticmethod
    def select_best_bands(three_channel_bands, num_bands_to_select=20, variance_threshold=1e-4, plot_selection=False):
        band_data = []
        kurtosis_values = []
        variance_values = []
        band_names = []

        for band_name, band in three_channel_bands:
            flat_band = band.ravel()
            band_variance = np.var(flat_band)
            if band_variance < variance_threshold:
                continue

            normalized_band = (flat_band - np.mean(flat_band)) / (np.std(flat_band) + 1e-10)
            kurt = stats.kurtosis(normalized_band)
            band_data.append((band_name, band))
            kurtosis_values.append(abs(kurt))
            variance_values.append(band_variance)
            band_names.append(band_name)

        if not band_data:
            return None

        num_available = len(band_data)
        if num_bands_to_select > num_available:
            num_bands_to_select = num_available

        selected_indices = np.argsort(kurtosis_values)[-num_bands_to_select:]
        selected_bands = [band_data[i] for i in selected_indices]

        return selected_bands

    @staticmethod
    def prepare_ica(selected_bands, image_od):
        ica_data = []
        target_size = image_od.shape[0] * image_od.shape[1]
        for band_name, band_data in selected_bands:
            flat_band = band_data.reshape(-1)

            if len(flat_band) != target_size:
                original_shape = band_data.shape[:2]

                scale_factor = np.sqrt(target_size / len(flat_band))
                new_shape = (int(original_shape[0] * scale_factor),
                             int(original_shape[1] * scale_factor))

                resized_band = cv2.resize(band_data, (new_shape[1], new_shape[0]),
                                          interpolation=cv2.INTER_CUBIC)
                flat_band = resized_band.reshape(-1)

                if len(flat_band) > target_size:
                    flat_band = flat_band[:target_size]
                elif len(flat_band) < target_size:
                    flat_band = np.pad(flat_band, (0, target_size - len(flat_band)),
                                       mode='constant')

            ica_data.append(flat_band)

        ica_matrix = np.column_stack(ica_data)
        od_flat = image_od.reshape(-1, 3)

        if ica_matrix.shape[0] != od_flat.shape[0]:
            raise ValueError(
                f"Inconsistent sizes after processing: ICA matrix has {ica_matrix.shape[0]} samples, OD has {od_flat.shape[0]} samples")

        return ica_matrix, od_flat

    @staticmethod
    def estimate_stain_matrix(ica_matrix, od_flat, n_comp=2):
        ica = FastICA(n_components=n_comp, random_state=42, max_iter=1000)
        components = ica.fit_transform(ica_matrix)

        stain_matrix = np.zeros((3, n_comp))

        regressor = LinearRegression(fit_intercept=False)

        for channel_idx in range(3):
            regressor.fit(components, od_flat[:, channel_idx])
            stain_matrix[channel_idx, :] = regressor.coef_

        return stain_matrix, components

    @staticmethod
    def recon_and_vis(concentrations, stain_matrix, og_shape, og_image, visualise=True):
        reconstructed_image = Deconvolution.reconstruct(concentrations, stain_matrix, og_shape)

        stain_images = Deconvolution.extract_individual_stains(concentrations, stain_matrix, og_shape)

        if visualise:
            Deconvolution.visualise_image(og_image, reconstructed_image, stain_images, concentrations)

        return stain_images
