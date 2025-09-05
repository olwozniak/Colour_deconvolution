import numpy as np
from skimage import metrics
from scipy import stats
import cv2

class Compare:

    def compare_normalization_effectiveness(original, normalized, target):
        """
        Porównuje skuteczność normalizacji między obrazami
        """
        results = {}

        # 1. Podobieństwo strukturalne (SSIM)
        results['ssim_original'] = metrics.structural_similarity(
            cv2.cvtColor(original, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(target, cv2.COLOR_BGR2GRAY),
            win_size=7
        )

        results['ssim_normalized'] = metrics.structural_similarity(
            cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(target, cv2.COLOR_BGR2GRAY),
            win_size=7
        )

        # 2. RMSE (Root Mean Square Error)
        results['rmse_original'] = np.sqrt(metrics.mean_squared_error(
            original.flatten(), target.flatten()
        ))

        results['rmse_normalized'] = np.sqrt(metrics.mean_squared_error(
            normalized.flatten(), target.flatten()
        ))

        # 3. Korelacja między histogramami
        def histogram_correlation(img1, img2):
            hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        results['hist_corr_original'] = histogram_correlation(original, target)
        results['hist_corr_normalized'] = histogram_correlation(normalized, target)

        return results


def compare_stain_separation(original, normalized):
    """
    Porównuje jakość separacji barwników przed i po normalizacji
    """

    # Przetwarzanie obrazu oryginalnego
    stain_images_orig = process_image(original, visualise=False)

    # Przetwarzanie obrazu znormalizowanego
    stain_images_norm = process_image(normalized, visualise=False)

    # Ocena jakości separacji
    comparison_results = {}

    for i, (stain_orig, stain_norm) in enumerate(zip(stain_images_orig, stain_images_norm)):
        # Kontrast barwników
        contrast_orig = np.std(stain_orig)
        contrast_norm = np.std(stain_norm)
        comparison_results[f'stain_{i}_contrast_improvement'] = contrast_norm / contrast_orig

        # Entropia (miara informacji)
        entropy_orig = stats.entropy(stain_orig.flatten())
        entropy_norm = stats.entropy(stain_norm.flatten())
        comparison_results[f'stain_{i}_entropy_ratio'] = entropy_norm / entropy_orig

    return comparison_results, stain_images_orig, stain_images_norm


# Porównanie separacji barwników
stain_comparison, stains_orig, stains_norm = compare_stain_separation(original, normalized)
