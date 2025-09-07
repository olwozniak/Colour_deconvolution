import numpy as np
from skimage import metrics
from scipy import stats
import cv2


class Compare:

    @staticmethod
    def resize_to_match(img, target_shape):
        """Resize image to match target shape"""
        if img.shape[:2] != target_shape[:2]:
            return cv2.resize(img, (target_shape[1], target_shape[0]))
        return img

    @staticmethod
    def compare_normalization_effectiveness(original, normalized, target):
        """
        Porównuje skuteczność normalizacji między obrazami
        """
        results = {}

        # Sprawdzenie i dostosowanie rozmiarów
        target_shape = target.shape
        original = Compare.resize_to_match(original, target_shape)
        normalized = Compare.resize_to_match(normalized, target_shape)

        # Konwersja do skali szarości dla SSIM
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        normalized_gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        # DODATKOWO: Upewnij się, że obrazy w skali szarości mają ten sam rozmiar
        print(f"Rozmiary po konwersji:")
        print(f"Original gray: {original_gray.shape}")
        print(f"Normalized gray: {normalized_gray.shape}")
        print(f"Target gray: {target_gray.shape}")

        # 1. Podobieństwo strukturalne (SSIM) - z dodatkowym sprawdzeniem
        try:
            results['ssim_original'] = metrics.structural_similarity(
                original_gray, target_gray, win_size=7
            )
        except ValueError as e:
            print(f"Błąd SSIM oryginalny: {e}")
            results['ssim_original'] = 0

        try:
            results['ssim_normalized'] = metrics.structural_similarity(
                normalized_gray, target_gray, win_size=7
            )
        except ValueError as e:
            print(f"Błąd SSIM znormalizowany: {e}")
            results['ssim_normalized'] = 0

        # 2. RMSE (Root Mean Square Error)
        results['rmse_original'] = np.sqrt(metrics.mean_squared_error(
            original.flatten(), target.flatten()
        ))

        results['rmse_normalized'] = np.sqrt(metrics.mean_squared_error(
            normalized.flatten(), target.flatten()
        ))

        # 3. Korelacja między histogramami
        def histogram_correlation(img1, img2):
            # Upewnij się, że obrazy mają ten sam rozmiar
            img2_resized = Compare.resize_to_match(img2, img1.shape)
            hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([img2_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        results['hist_corr_original'] = histogram_correlation(original, target)
        results['hist_corr_normalized'] = histogram_correlation(normalized, target)

        # 4. PSNR - z obsługą błędów
        try:
            results['psnr_original'] = metrics.peak_signal_noise_ratio(target, original)
        except:
            results['psnr_original'] = 0

        try:
            results['psnr_normalized'] = metrics.peak_signal_noise_ratio(target, normalized)
        except:
            results['psnr_normalized'] = 0

        return results

    # ... reszta metody compare_stain_separation pozostaje bez zmian ...

    @staticmethod
    def compare_stain_separation(original, normalized, visualise=False):
        """
        Porównuje jakość separacji barwników przed i po normalizacji
        """
        from wavelet import Wavelet

        # Przetwarzanie obrazu oryginalnego
        stain_images_orig = Wavelet.process_image(original, visualise=visualise)

        # Przetwarzanie obrazu znormalizowanego
        stain_images_norm = Wavelet.process_image(normalized, visualise=visualise)

        # Ocena jakości separacji
        comparison_results = {}

        for i, (stain_orig, stain_norm) in enumerate(zip(stain_images_orig, stain_images_norm)):
            # Kontrast barwników
            contrast_orig = np.std(stain_orig.astype(np.float32))
            contrast_norm = np.std(stain_norm.astype(np.float32))
            comparison_results[f'stain_{i}_contrast_improvement'] = contrast_norm / max(contrast_orig, 1e-10)

            # Entropia (miara informacji)
            entropy_orig = stats.entropy(stain_orig.flatten() + 1)
            entropy_norm = stats.entropy(stain_norm.flatten() + 1)
            comparison_results[f'stain_{i}_entropy_ratio'] = entropy_norm / max(entropy_orig, 1e-10)

            # Różnica w jasności
            brightness_orig = np.mean(stain_orig.astype(np.float32))
            brightness_norm = np.mean(stain_norm.astype(np.float32))
            comparison_results[f'stain_{i}_brightness_ratio'] = brightness_norm / max(brightness_orig, 1e-10)

            # Jaskrawość (variance)
            variance_orig = np.var(stain_orig.astype(np.float32))
            variance_norm = np.var(stain_norm.astype(np.float32))
            comparison_results[f'stain_{i}_variance_ratio'] = variance_norm / max(variance_orig, 1e-10)

        return comparison_results, stain_images_orig, stain_images_norm

    # ... reszta metod pozostaje bez zmian ...

    @staticmethod
    def print_comparison_results(results):
        """
        Wyświetla wyniki porównania w czytelnej formie
        """
        print("=== WYNIKI PORÓWNANIA NORMALIZACJI ===")
        print(f"SSIM - Oryginalny: {results['ssim_original']:.4f}")
        print(f"SSIM - Znormalizowany: {results['ssim_normalized']:.4f}")
        ssim_improvement = ((results['ssim_normalized'] - results['ssim_original']) / max(results['ssim_original'],
                                                                                          1e-10) * 100)
        print(f"Poprawa SSIM: {ssim_improvement:+.2f}%")
        print()

        print(f"RMSE - Oryginalny: {results['rmse_original']:.2f}")
        print(f"RMSE - Znormalizowany: {results['rmse_normalized']:.2f}")
        rmse_improvement = ((results['rmse_original'] - results['rmse_normalized']) / max(results['rmse_original'],
                                                                                          1e-10) * 100)
        print(f"Poprawa RMSE: {rmse_improvement:+.2f}%")
        print()

        print(f"PSNR - Oryginalny: {results['psnr_original']:.2f} dB")
        print(f"PSNR - Znormalizowany: {results['psnr_normalized']:.2f} dB")
        psnr_improvement = results['psnr_normalized'] - results['psnr_original']
        print(f"Poprawa PSNR: {psnr_improvement:+.2f} dB")
        print()

        print(f"Korelacja histogramu - Oryginalny: {results['hist_corr_original']:.4f}")
        print(f"Korelacja histogramu - Znormalizowany: {results['hist_corr_normalized']:.4f}")
        hist_improvement = ((results['hist_corr_normalized'] - results['hist_corr_original']) / max(
            results['hist_corr_original'], 1e-10) * 100)
        print(f"Poprawa korelacji: {hist_improvement:+.2f}%")

    @staticmethod
    def print_stain_comparison(stain_results):
        """
        Wyświetla wyniki porównania separacji barwników
        """
        print("\n=== WYNIKI SEPARACJI BARWNIKÓW ===")
        num_stains = len([k for k in stain_results.keys() if k.startswith('stain_0_')])

        for i in range(num_stains):
            print(f"\nBarwnik {i + 1}:")
            print(f"  Poprawa kontrastu: {stain_results[f'stain_{i}_contrast_improvement']:.4f}")
            print(f"  Stosunek entropii: {stain_results[f'stain_{i}_entropy_ratio']:.4f}")
            print(f"  Stosunek jasności: {stain_results[f'stain_{i}_brightness_ratio']:.4f}")
            print(f"  Stosunek wariancji: {stain_results[f'stain_{i}_variance_ratio']:.4f}")

            # Interpretacja
            contrast = stain_results[f'stain_{i}_contrast_improvement']
            if contrast > 1.2:
                print("    ✓ Znaczna poprawa kontrastu")
            elif contrast > 1.0:
                print("    ↺ Niewielka poprawa kontrastu")
            else:
                print("    ✗ Pogorszenie kontrastu")