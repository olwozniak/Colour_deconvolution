import cv2
import matplotlib.pyplot as plt
import numpy as np
from deconvolution import Deconvolution
from deconvolution_processing import Process
from normalization import Normalization


class Compare:

    @staticmethod
    def compare_methods(image, target_image=None):
        """
        Porównuje wszystkie kombinacje metod dekonwolucji i normalizacji
        """
        # Lista dostępnych metod
        deconv_methods = ['wavelet', 'ruifork', 'reference']
        norm_methods = ['stain_specific', 'reinhard', 'all_pixel_quantile', 'colour_map_quantile', None]

        # Przygotowanie obrazu źródłowego
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if target_image is not None:
            target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        # Tworzenie siatki wykresów
        n_rows = len(deconv_methods)
        n_cols = len(norm_methods)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Przetwarzanie każdej kombinacji
        for i, deconv_method in enumerate(deconv_methods):
            for j, norm_method in enumerate(norm_methods):
                try:
                    # Normalizacja jeśli wybrana
                    if norm_method is not None and target_image is not None:
                        normalizer = Normalization(target_rgb)
                        normalizer.extract_target_stains_stats()
                        normalized_image = normalizer.normalize(image_rgb, method=norm_method)
                    else:
                        normalized_image = image_rgb

                    # Dekonwolucja
                    stain_images, stain_matrix, concentrations = Process.process_image(
                        cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR),
                        visualise=False,
                        method=deconv_method,
                        normalization_method=None
                    )

                    # Wyświetlanie wyniku
                    if stain_images is not None and len(stain_images) > 0:
                        combined_stains = np.hstack(stain_images[:2]) if len(stain_images) >= 2 else stain_images[0]
                        axes[i, j].imshow(combined_stains)
                    axes[i, j].set_title(f'Deconv: {deconv_method}\nNorm: {norm_method if norm_method else "None"}')
                    axes[i, j].axis('off')

                except Exception as e:
                    axes[i, j].text(0.5, 0.5, f'Error:\n{str(e)}',
                                    ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].set_title(f'Deconv: {deconv_method}\nNorm: {norm_method if norm_method else "None"}')
                    axes[i, j].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_clustering_methods(image, target_image, deconv_method='wavelet', norm_method='stain_specific'):
        """
        Porównuje różne metody klasteryzacji w normalizacji specyficznej dla plam
        """
        clustering_methods = ['kmeans', 'em', 'vb']

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        # Tworzenie siatki wykresów
        n_cols = len(clustering_methods) + 1  # +1 dla oryginału
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 8))

        # Wyświetl oryginalny obraz
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        axes[1, 0].imshow(target_rgb)
        axes[1, 0].set_title('Target Image')
        axes[1, 0].axis('off')

        results = {}

        for j, cluster_method in enumerate(clustering_methods, 1):
            try:
                # Normalizacja z różnymi metodami klasteryzacji
                normalizer = Normalization(target_rgb)
                normalizer.extract_target_stains_stats()

                normalized_image = normalizer.normalize(
                    image_rgb,
                    method=norm_method,
                    cluster_method=cluster_method
                )

                # Dekonwolucja
                stain_images, stain_matrix, concentrations = Process.process_image(
                    cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR),
                    visualise=False,
                    method=deconv_method,
                    normalization_method=None
                )

                # Wyświetl wyniki
                if stain_images is not None and len(stain_images) > 0:
                    axes[0, j].imshow(normalized_image)
                    axes[0, j].set_title(f'Normalized\n({cluster_method})')
                    axes[0, j].axis('off')

                    axes[1, j].imshow(stain_images[0])  # pierwsza plama
                    axes[1, j].set_title(f'Stain 0\n({cluster_method})')
                    axes[1, j].axis('off')

                # Oblicz metryki
                metrics = Compare.calculate_stain_metrics(stain_images, concentrations, stain_matrix, image_rgb)
                results[cluster_method] = metrics

            except Exception as e:
                print(f"Error with clustering method {cluster_method}: {e}")
                axes[0, j].text(0.5, 0.5, f'Error:\n{str(e)}',
                                ha='center', va='center', transform=axes[0, j].transAxes)
                axes[0, j].set_title(f'Error: {cluster_method}')
                axes[0, j].axis('off')

                axes[1, j].axis('off')
                results[cluster_method] = {'error': str(e)}

        plt.tight_layout()
        plt.show()

        # Generuj analizę metod klasteryzacji
        Compare.analyze_clustering_results(results)

        return results

    @staticmethod
    def calculate_stain_metrics(stain_images, concentrations, stain_matrix, original_image):
        """Oblicza metryki jakości dla wyników dekonwolucji"""
        metrics = {}

        if stain_images is None or len(stain_images) == 0:
            return metrics

        # 1. Kontrast plam
        for idx, stain_img in enumerate(stain_images):
            gray_stain = cv2.cvtColor(stain_img, cv2.COLOR_RGB2GRAY)
            metrics[f'stain_{idx}_contrast'] = np.std(gray_stain)
            metrics[f'stain_{idx}_mean'] = np.mean(gray_stain)

        # 2. Separowalność plam
        if len(stain_images) >= 2:
            stain1_gray = cv2.cvtColor(stain_images[0], cv2.COLOR_RGB2GRAY)
            stain2_gray = cv2.cvtColor(stain_images[1], cv2.COLOR_RGB2GRAY)

            # Międzyklasowa wariancja
            between_class_var = np.var(np.concatenate([stain1_gray.flatten(), stain2_gray.flatten()]))
            metrics['between_class_variance'] = between_class_var

            # Współczynnik korelacji
            correlation = np.corrcoef(stain1_gray.flatten(), stain2_gray.flatten())[0, 1]
            metrics['stain_correlation'] = abs(correlation)

        # 3. Jakość rekonstrukcji
        try:
            reconstructed = Deconvolution.reconstruct(concentrations, stain_matrix, original_image.shape)
            mse = np.mean((original_image.astype(float) - reconstructed.astype(float)) ** 2)
            metrics['reconstruction_mse'] = mse

            if mse > 0:
                metrics['psnr'] = 20 * np.log10(255.0 / np.sqrt(mse))
        except:
            metrics['reconstruction_mse'] = float('inf')
            metrics['psnr'] = 0

        # 4. Entropia
        for idx, stain_img in enumerate(stain_images):
            gray_stain = cv2.cvtColor(stain_img, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray_stain], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            metrics[f'stain_{idx}_entropy'] = entropy

        return metrics

    @staticmethod
    def analyze_clustering_results(results):
        """
        Analizuje i wyświetla wnioski z porównania metod klasteryzacji
        """
        print("=" * 80)
        print("ANALIZA METOD KLASTERYZACJI")
        print("=" * 80)

        valid_results = {k: v for k, v in results.items() if isinstance(v, dict) and 'error' not in v}

        if not valid_results:
            print("Brak wyników do analizy")
            return

        # Znajdź najlepszą metodę dla każdej metryki
        best_methods = {}
        metrics_to_analyze = ['between_class_variance', 'stain_correlation',
                              'reconstruction_mse', 'stain_0_contrast', 'stain_1_contrast']

        for metric in metrics_to_analyze:
            best_value = None
            best_method = None

            for method, metrics in valid_results.items():
                if metric in metrics:
                    value = metrics[metric]

                    # Określ czy wyższa czy niższa wartość jest lepsza
                    if metric in ['reconstruction_mse', 'stain_correlation']:
                        if best_value is None or value < best_value:
                            best_value = value
                            best_method = method
                    else:
                        if best_value is None or value > best_value:
                            best_value = value
                            best_method = method

            if best_method is not None:
                best_methods[metric] = (best_method, best_value)

        # Wyświetl wyniki
        print("\nNAJLEPSZE METODY KLASTERYZACJI:")
        print("-" * 50)
        for metric, (method, value) in best_methods.items():
            metric_name = {
                'between_class_variance': 'Separacja plam',
                'stain_correlation': 'Korelacja plam',
                'reconstruction_mse': 'Błąd rekonstrukcji',
                'stain_0_contrast': 'Kontrast plamy 0',
                'stain_1_contrast': 'Kontrast plamy 1'
            }.get(metric, metric)

            print(f"{metric_name:20s}: {method:8s} (wartość: {value:.4f})")

        # Ranking ogólny
        method_scores = {method: 0 for method in valid_results.keys()}
        for metric, (method, value) in best_methods.items():
            method_scores[method] += 1

        print("\nRANKING OGÓLNY METOD KLASTERYZACJI:")
        print("-" * 50)
        for method, score in sorted(method_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{method:8s}: {score} najlepszych wyników")

        # Szczegółowa analiza każdej metody
        print("\nSZczEGÓŁOWA ANALIZA:")
        print("-" * 50)
        for method, metrics in valid_results.items():
            print(f"\nMetoda: {method}")
            if 'between_class_variance' in metrics:
                var = metrics['between_class_variance']
                if var > 1000:
                    sep_quality = "DOSKONAŁA"
                elif var > 500:
                    sep_quality = "DOBRA"
                else:
                    sep_quality = "SŁABA"
                print(f"  → Separacja plam: {var:.1f} ({sep_quality})")

            if 'stain_correlation' in metrics:
                corr = metrics['stain_correlation']
                if corr < 0.1:
                    corr_quality = "BARDZO DOBRA"
                elif corr < 0.3:
                    corr_quality = "DOBRA"
                else:
                    corr_quality = "SŁABA"
                print(f"  → Korelacja plam: {corr:.3f} ({corr_quality})")

            if 'reconstruction_mse' in metrics:
                mse = metrics['reconstruction_mse']
                if mse < 100:
                    mse_quality = "DOSKONAŁA"
                elif mse < 500:
                    mse_quality = "DOBRA"
                else:
                    mse_quality = "SŁABA"
                print(f"  → Jakość rekonstrukcji: {mse:.1f} ({mse_quality})")

        # Rekomendacja końcowa
        best_overall = max(method_scores.items(), key=lambda x: x[1])[0]
        print(f"\nREKOMENDACJA: Najlepsza metoda klasteryzacji → {best_overall.upper()}")
        print("=" * 80)

    @staticmethod
    def quantitative_comparison(image, target_image=None):
        """
        Ilościowe porównanie metod za pomocą różnych metryk
        """
        deconv_methods = ['wavelet', 'ruifork', 'reference']
        norm_methods = ['stain_specific', 'reinhard', 'all_pixel_quantile', 'colour_map_quantile', None]

        results = {}
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if target_image is not None:
            target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        for deconv_method in deconv_methods:
            for norm_method in norm_methods:
                try:
                    key = f'{deconv_method}_{norm_method if norm_method else "none"}'
                    print(f"Processing: {key}")

                    # Normalizacja
                    if norm_method is not None and target_image is not None:
                        normalizer = Normalization(target_rgb)
                        normalizer.extract_target_stains_stats()
                        normalized_image = normalizer.normalize(image_rgb, method=norm_method)
                    else:
                        normalized_image = image_rgb

                    # Dekonwolucja
                    stain_images, stain_matrix, concentrations = Process.process_image(
                        cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR),
                        visualise=False,
                        method=deconv_method,
                        normalization_method=None
                    )

                    # Obliczanie metryk
                    metrics = Compare.calculate_stain_metrics(stain_images, concentrations, stain_matrix, image_rgb)
                    results[key] = metrics

                except Exception as e:
                    print(f"Error with {deconv_method}_{norm_method}: {e}")
                    results[f'{deconv_method}_{norm_method if norm_method else "none"}'] = {'error': str(e)}

        return results

    @staticmethod
    def plot_quantitative_results(quantitative_results, save_path=None):
        """
        Wizualizacja wyników ilościowych
        """
        methods = list(quantitative_results.keys())
        if not methods:
            return

        # Pobierz wszystkie dostępne metryki
        all_metrics = set()
        for method_data in quantitative_results.values():
            if isinstance(method_data, dict):
                all_metrics.update(method_data.keys())

        metrics = [m for m in all_metrics if not m.startswith('stain_') or m.endswith('_contrast')]

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 6, 8))

        if n_metrics == 1:
            axes = [axes]

        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

        for i, metric in enumerate(metrics):
            values = []
            valid_methods = []
            method_colors = []

            for idx, method in enumerate(methods):
                if (metric in quantitative_results[method] and
                        not isinstance(quantitative_results[method][metric], str)):
                    values.append(quantitative_results[method][metric])
                    valid_methods.append(method)
                    method_colors.append(colors[idx])

            if values:
                bars = axes[i].bar(valid_methods, values, color=method_colors)
                axes[i].set_title(metric, fontsize=12, fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)

                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                 f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def generate_comprehensive_analysis(results):
        """
        Generuje szczegółową analizę i wnioski z wyników
        """
        print("=" * 80)
        print("KOMPREHENSYWNA ANALIZA WYNIKÓW")
        print("=" * 80)

        # Znajdź najlepsze metody dla każdej metryki
        best_methods = {}
        metrics_to_analyze = ['reconstruction_mse', 'between_class_variance',
                              'stain_correlation', 'psnr', 'stain_0_contrast', 'stain_1_contrast']

        for metric in metrics_to_analyze:
            best_value = None
            best_method = None

            for method, metrics_dict in results.items():
                if (isinstance(metrics_dict, dict) and metric in metrics_dict and
                        not isinstance(metrics_dict[metric], str)):

                    value = metrics_dict[metric]

                    # Określ czy wyższa czy niższa wartość jest lepsza
                    if metric in ['reconstruction_mse', 'stain_correlation']:
                        if best_value is None or value < best_value:
                            best_value = value
                            best_method = method
                    else:
                        if best_value is None or value > best_value:
                            best_value = value
                            best_method = method

            if best_method is not None:
                best_methods[metric] = (best_method, best_value)

        # Analiza dekonwolucji
        deconv_performance = {'wavelet': 0, 'ruifork': 0, 'reference': 0}
        for metric, (method, value) in best_methods.items():
            for deconv in deconv_performance.keys():
                if deconv in method:
                    deconv_performance[deconv] += 1

        # Analiza normalizacji
        norm_performance = {
            'stain_specific': 0, 'reinhard': 0,
            'all_pixel_quantile': 0, 'colour_map_quantile': 0, 'none': 0
        }
        for metric, (method, value) in best_methods.items():
            for norm in norm_performance.keys():
                if norm in method or (norm == 'none' and 'none' in method):
                    norm_performance[norm] += 1

        # Wyświetl wyniki
        print("\nNAJLEPSZE KOMBINACJE DLA POSZCZEGÓLNYCH METRYK:")
        print("-" * 60)
        for metric, (method, value) in best_methods.items():
            print(f"{metric:25s}: {method:30s} (wartość: {value:.4f})")

        print("\nWNIOSKI - WYDajNOŚĆ METOD DEKONWOLUCJI:")
        print("-" * 60)
        best_deconv = max(deconv_performance.items(), key=lambda x: x[1])
        for deconv, score in sorted(deconv_performance.items(), key=lambda x: x[1], reverse=True):
            print(f"{deconv:15s}: {score:2d} najlepszych wyników")
        print(f"→ Najlepsza metoda dekonwolucji: {best_deconv[0]}")

        print("\nWNIOSKI - WYDajNOŚĆ METOD NORMALIZACJI:")
        print("-" * 60)
        best_norm = max(norm_performance.items(), key=lambda x: x[1])
        for norm, score in sorted(norm_performance.items(), key=lambda x: x[1], reverse=True):
            print(f"{norm:20s}: {score:2d} najlepszych wyników")
        print(f"→ Najlepsza metoda normalizacji: {best_norm[0]}")

        # Szczegółowa analiza jakości
        print("\nANALIZA JAKOŚCI:")
        print("-" * 60)

        # Analiza rekonstrukcji
        mse_values = []
        for method, metrics in results.items():
            if 'reconstruction_mse' in metrics and not isinstance(metrics['reconstruction_mse'], str):
                mse_values.append(metrics['reconstruction_mse'])

        if mse_values:
            avg_mse = np.mean(mse_values)
            print(f"Średni błąd rekonstrukcji (MSE): {avg_mse:.4f}")
            if avg_mse < 100:
                print("→ Doskonała jakość rekonstrukcji")
            elif avg_mse < 500:
                print("→ Dobra jakość rekonstrukcji")
            else:
                print("→ Umiarkowana jakość rekonstrukcji")

        # Analiza separacji plam
        correlation_values = []
        for method, metrics in results.items():
            if 'stain_correlation' in metrics and not isinstance(metrics['stain_correlation'], str):
                correlation_values.append(metrics['stain_correlation'])

        if correlation_values:
            avg_corr = np.mean(correlation_values)
            print(f"Średnia korelacja między plamami: {avg_corr:.4f}")
            if avg_corr < 0.1:
                print("→ Doskonała separacja plam")
            elif avg_corr < 0.3:
                print("→ Dobra separacja plam")
            else:
                print("→ Słaba separacja plam")

        # Rekomendacja końcowa
        print("\nREKOMENDACJA KOŃCOWA:")
        print("-" * 60)
        best_overall = best_methods.get('between_class_variance', ('unknown', 0))[0]
        print(f"Rekomendowana kombinacja: {best_overall}")

        best_deconv_part = best_overall.split('_')[0]
        best_norm_part = best_overall.split('_')[1] if '_' in best_overall else 'none'

        print(f"→ Metoda dekonwolucji: {best_deconv_part}")
        print(f"→ Metoda normalizacji: {best_norm_part}")

        print("=" * 80)

