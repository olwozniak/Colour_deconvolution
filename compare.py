import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from deconvolution import Deconvolution
from deconvolution_processing import Process
from normalization import Normalization


class Compare:

    @staticmethod
    def compare_methods(image, target_image, save_dir=None):
        deconv_methods = ['wavelet', 'ruifork', 'reference']
        norm_methods = ['stain_specific', 'reinhard', 'all_pixel_quantile', 'colour_map_quantile', None]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_rgb = None
        if target_image is not None:
            target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        results = {}
        timing_results = {}

        n_rows = len(deconv_methods)
        n_cols = len(norm_methods)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, deconv_method in enumerate(deconv_methods):
            for j, norm_method in enumerate(norm_methods):
                method_name = f"{deconv_method}_{norm_method if norm_method else 'none'}"
                start_time = time.time()

                try:
                    if norm_method is not None and target_rgb is not None:
                        normalizer = Normalization(target_rgb)
                        normalizer.extract_target_stains_stats()
                        normalized_image = normalizer.normalize(image_rgb, method=norm_method)
                    else:
                        normalized_image = image_rgb

                    stain_images, stain_matrix, concentrations = Process.process_image(
                        cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR),
                        visualise=False,
                        method=deconv_method,
                        normalization_method=None
                    )

                    if stain_images is not None and len(stain_images) > 0:
                        combined_stains = np.hstack(stain_images[:2]) if len(stain_images) >= 2 else stain_images[0]
                        axes[i, j].imshow(combined_stains)

                    axes[i, j].set_title(f'Deconv: {deconv_method}\nNorm: {norm_method if norm_method else "None"}')
                    axes[i, j].axis('off')

                    results[method_name] = {
                        'stain_images': stain_images,
                        'stain_matrix': stain_matrix,
                        'concentrations': concentrations,
                        'normalized_image': normalized_image
                    }

                except Exception as e:
                    axes[i, j].text(0.5, 0.5, f'Error:\n{str(e)}',
                                    ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].set_title(f'Deconv: {deconv_method}\nNorm: {norm_method if norm_method else "None"}')
                    axes[i, j].axis('off')

                    results[method_name] = {'error': str(e)}
                end_time = time.time()
                timing_results[method_name] = end_time - start_time

        plt.tight_layout()
        plot_path = None
        if save_dir:
            plot_path = os.path.join(save_dir, "methods_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Wykres porównania metod zapisano jako: {plot_path}")

        plt.show()

        return results, plot_path, timing_results

    @staticmethod
    def compare_clustering_methods(image, target_image, deconv_method='wavelet', norm_method='stain_specific',
                                   save_dir=None):
        clustering_methods = ['kmeans', 'em', 'vb']

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        n_cols = len(clustering_methods) + 1  # +1 dla oryginału
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 8))

        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        axes[1, 0].imshow(target_rgb)
        axes[1, 0].set_title('Target Image')
        axes[1, 0].axis('off')

        results = {}
        timing_results = {}

        for j, cluster_method in enumerate(clustering_methods, 1):
            method_name = f"clustering_{cluster_method}"
            start_time = time.time()

            try:
                normalizer = Normalization(target_rgb)
                normalizer.extract_target_stains_stats()
                cluster_params = {}
                if cluster_method in ['em', 'vb']:
                    cluster_params = {
                        'max_iter': 1000,
                        'tol': 1e-6,
                        'n_init': 5,
                        'random_state': 42
                    }

                normalized_image = normalizer.normalize(
                    image_rgb,
                    method=norm_method,
                    cluster_method=cluster_method,
                    **cluster_params
                )
                stain_images, stain_matrix, concentrations = Process.process_image(
                    cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR),
                    visualise=False,
                    method=deconv_method,
                    normalization_method=None
                )
                if stain_images is not None and len(stain_images) > 0:
                    axes[0, j].imshow(normalized_image)
                    axes[0, j].set_title(f'Normalized\n({cluster_method})')
                    axes[0, j].axis('off')

                    axes[1, j].imshow(stain_images[0])
                    axes[1, j].set_title(f'Stain 0\n({cluster_method})')
                    axes[1, j].axis('off')

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

            end_time = time.time()
            timing_results[cluster_method] = end_time - start_time

        plt.tight_layout()
        plot_path = None
        if save_dir:
            plot_path = os.path.join(save_dir, f"clustering_comparison_{deconv_method}_{norm_method}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Wykres porównania klasteryzacji zapisano jako: {plot_path}")
        plt.show()
        Compare.analyze_clustering_results(results)

        return results, plot_path, timing_results

    @staticmethod
    def calculate_stain_metrics(stain_images, concentrations, stain_matrix, original_image):
        metrics = {}

        if stain_images is None or len(stain_images) == 0:
            return metrics
        for idx, stain_img in enumerate(stain_images):
            gray_stain = cv2.cvtColor(stain_img, cv2.COLOR_RGB2GRAY)
            metrics[f'stain_{idx}_contrast'] = np.std(gray_stain)
            metrics[f'stain_{idx}_mean'] = np.mean(gray_stain)

        if len(stain_images) >= 2:
            stain1_gray = cv2.cvtColor(stain_images[0], cv2.COLOR_RGB2GRAY)
            stain2_gray = cv2.cvtColor(stain_images[1], cv2.COLOR_RGB2GRAY)

            between_class_var = np.var(np.concatenate([stain1_gray.flatten(), stain2_gray.flatten()]))
            metrics['between_class_variance'] = between_class_var

            correlation = np.corrcoef(stain1_gray.flatten(), stain2_gray.flatten())[0, 1]
            metrics['stain_correlation'] = abs(correlation)
        try:
            reconstructed = Deconvolution.reconstruct(concentrations, stain_matrix, original_image.shape)
            mse = np.mean((original_image.astype(float) - reconstructed.astype(float)) ** 2)
            metrics['reconstruction_mse'] = mse

            if mse > 0:
                metrics['psnr'] = 20 * np.log10(255.0 / np.sqrt(mse))
        except:
            metrics['reconstruction_mse'] = float('inf')
            metrics['psnr'] = 0

        for idx, stain_img in enumerate(stain_images):
            gray_stain = cv2.cvtColor(stain_img, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray_stain], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            metrics[f'stain_{idx}_entropy'] = entropy

        return metrics

    @staticmethod
    def analyze_clustering_results(results):
        print("ANALIZA METOD KLASTERYZACJI")

        valid_results = {k: v for k, v in results.items() if isinstance(v, dict) and 'error' not in v}

        if not valid_results:
            print("Brak wyników do analizy")
            return

        best_methods = {}
        metrics_to_analyze = ['between_class_variance', 'stain_correlation',
                              'reconstruction_mse', 'stain_0_contrast', 'stain_1_contrast']

        for metric in metrics_to_analyze:
            best_value = None
            best_method = None

            for method, metrics in valid_results.items():
                if metric in metrics:
                    value = metrics[metric]

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

        print("\nNAJLEPSZE METODY KLASTERYZACJI:")
        for metric, (method, value) in best_methods.items():
            metric_name = {
                'between_class_variance': 'Separacja plam',
                'stain_correlation': 'Korelacja plam',
                'reconstruction_mse': 'Błąd rekonstrukcji',
                'stain_0_contrast': 'Kontrast plamy 0',
                'stain_1_contrast': 'Kontrast plamy 1'
            }.get(metric, metric)

            print(f"{metric_name:20s}: {method:8s} (wartość: {value:.4f})")

        method_scores = {method: 0 for method in valid_results.keys()}
        for metric, (method, value) in best_methods.items():
            method_scores[method] += 1

        print("\nPODSUMOWANIE:")
        for method, score in sorted(method_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{method:8s}: {score} najlepszych wyników")

        print("\nANALIZA WYNIKÓW:")
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

        best_overall = max(method_scores.items(), key=lambda x: x[1])[0]
        print(f"\nREKOMENDACJA: Najlepsza metoda klasteryzacji → {best_overall.upper()}")

    @staticmethod
    def quantitative_comparison(image, target_image=None, save_dir=None):
        deconv_methods = ['wavelet', 'ruifork', 'reference']
        norm_methods = ['stain_specific', 'reinhard', 'all_pixel_quantile', 'colour_map_quantile', None]

        results = {}
        timing_results = {}
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_rgb = None
        if target_image is not None:
            target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

        for deconv_method in deconv_methods:
            for norm_method in norm_methods:
                method_name = f'{deconv_method}_{norm_method if norm_method else "none"}'
                start_time = time.time()

                try:
                    print(f"Processing: {method_name}")
                    if norm_method is not None and target_rgb is not None:
                        normalizer = Normalization(target_rgb)
                        normalizer.extract_target_stains_stats()
                        normalized_image = normalizer.normalize(image_rgb, method=norm_method)
                    else:
                        normalized_image = image_rgb

                    stain_images, stain_matrix, concentrations = Process.process_image(
                        cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR),
                        visualise=False,
                        method=deconv_method,
                        normalization_method=None
                    )

                    metrics = Compare.calculate_stain_metrics(stain_images, concentrations, stain_matrix, image_rgb)
                    results[method_name] = metrics

                except Exception as e:
                    print(f"Error with {deconv_method}_{norm_method}: {e}")
                    results[method_name] = {'error': str(e)}

                end_time = time.time()
                timing_results[method_name] = end_time - start_time

        return results, timing_results

    @staticmethod
    def plot_quantitative_results(results, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        methods = list(results.keys())

        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink',
                  'lightblue', 'wheat', 'palegreen', 'lavender', 'bisque']

        metrics_data = {
            'MSE Rekonstrukcji': [results[m].get('reconstruction_mse', 0) for m in methods],
            'Wariancja międzyklasowa': [results[m].get('between_class_variance', 0) for m in methods],
            'Korelacja plam': [results[m].get('stain_correlation', 0) for m in methods],
            'SSIM': [results[m].get('ssim', 0) for m in methods]
        }

        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[idx // 2, idx % 2]
            bar_colors = [colors[i % len(colors)] for i in range(len(methods))]

            bars = ax.bar(range(len(methods)), values, color=bar_colors, alpha=0.7)
            ax.set_title(metric_name)
            ax.set_ylabel('Wartość')
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right')

            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Wykres wyników ilościowych zapisano jako: {save_path}")

        plt.show()

    @staticmethod
    def plot_timing_results(timing_results, save_path=None):
        if not timing_results:
            print("Brak danych czasowych do wyświetlenia")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        methods = list(timing_results.keys())
        times = [timing_results[m] for m in methods]

        sorted_indices = np.argsort(times)[::-1]
        sorted_methods = [methods[i] for i in sorted_indices]
        sorted_times = [times[i] for i in sorted_indices]

        bars = ax.bar(range(len(sorted_methods)), sorted_times,
                      color=plt.cm.viridis(np.linspace(0, 1, len(sorted_methods))))

        ax.set_title('Czas wykonania metod', fontsize=16, fontweight='bold')
        ax.set_ylabel('Czas [s]', fontsize=12)
        ax.set_xlabel('Metoda', fontsize=12)
        ax.set_xticks(range(len(sorted_methods)))
        ax.set_xticklabels(sorted_methods, rotation=45, ha='right')

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Wykres czasów wykonania zapisano jako: {save_path}")

        plt.show()

        print("\n📊 STATYSTYKI CZASOWE:")
        print("-" * 30)
        print(f"Najszybsza metoda: {min(timing_results, key=timing_results.get)} ({min(timing_results.values()):.2f}s)")
        print(
            f"Najwolniejsza metoda: {max(timing_results, key=timing_results.get)} ({max(timing_results.values()):.2f}s)")
        print(f"Średni czas: {np.mean(list(timing_results.values())):.2f}s")
        print(f"Łączny czas wszystkich metod: {sum(timing_results.values()):.2f}s")

    @staticmethod
    def analyze_timing_performance(timing_results):
        if not timing_results:
            print("Brak danych czasowych do analizy")
            return
        print("ANALIZA WYDAJNOŚCI CZASOWEJ")

        deconv_times = {}
        norm_times = {}

        for method, time_val in timing_results.items():
            if '_' in method:
                deconv, norm = method.split('_', 1)
                if deconv not in deconv_times:
                    deconv_times[deconv] = []
                deconv_times[deconv].append(time_val)

                if norm not in norm_times:
                    norm_times[norm] = []
                norm_times[norm].append(time_val)

        print("\nŚREDNIE CZASY WYKONANIA DEKONWOLUCJI:")
        for deconv, times in deconv_times.items():
            print(f"{deconv:15s}: {np.mean(times):.2f}s (min: {min(times):.2f}s, max: {max(times):.2f}s)")
        print("\nŚREDNIE CZASY WYKONANIA NORMALIZACJI:")
        for norm, times in norm_times.items():
            print(f"{norm:20s}: {np.mean(times):.2f}s (min: {min(times):.2f}s, max: {max(times):.2f}s)")
        print("\nNAJLEPSZE METODY POD WZGLĘDEM CZASU:")
        fastest_methods = sorted(timing_results.items(), key=lambda x: x[1])[:5]
        for method, time_val in fastest_methods:
            print(f"{method:30s}: {time_val:.2f}s")
        return {
            'deconv_times': {k: np.mean(v) for k, v in deconv_times.items()},
            'norm_times': {k: np.mean(v) for k, v in norm_times.items()},
            'fastest_methods': fastest_methods
        }

    @staticmethod
    def generate_comprehensive_analysis(results):
        print("ANALIZA WYNIKÓW")

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

        deconv_performance = {'wavelet': 0, 'ruifork': 0, 'reference': 0}
        for metric, (method, value) in best_methods.items():
            for deconv in deconv_performance.keys():
                if deconv in method:
                    deconv_performance[deconv] += 1

        norm_performance = {
            'stain_specific': 0, 'reinhard': 0,
            'all_pixel_quantile': 0, 'colour_map_quantile': 0, 'none': 0
        }
        for metric, (method, value) in best_methods.items():
            for norm in norm_performance.keys():
                if norm in method or (norm == 'none' and 'none' in method):
                    norm_performance[norm] += 1

        print("\nNAJLEPSZE KOMBINACJE DLA POSZCZEGÓLNYCH METRYK:")
        for metric, (method, value) in best_methods.items():
            print(f"{metric:25s}: {method:30s} (wartość: {value:.4f})")
        print("\nWNIOSKI - WYDajNOŚĆ METOD DEKONWOLUCJI:")
        best_deconv = max(deconv_performance.items(), key=lambda x: x[1])
        for deconv, score in sorted(deconv_performance.items(), key=lambda x: x[1], reverse=True):
            print(f"{deconv:15s}: {score:2d} najlepszych wyników")
        print(f"→ Najlepsza metoda dekonwolucji: {best_deconv[0]}")
        print("\nWNIOSKI - WYDajNOŚĆ METOD NORMALIZACJI:")
        best_norm = max(norm_performance.items(), key=lambda x: x[1])
        for norm, score in sorted(norm_performance.items(), key=lambda x: x[1], reverse=True):
            print(f"{norm:20s}: {score:2d} najlepszych wyników")
        print(f"→ Najlepsza metoda normalizacji: {best_norm[0]}")
        print("\nANALIZA JAKOŚCI:")

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

        print("\nREKOMENDACJA KOŃCOWA:")
        best_overall = best_methods.get('between_class_variance', ('unknown', 0))[0]
        print(f"Rekomendowana kombinacja: {best_overall}")

        best_deconv_part = best_overall.split('_')[0]
        best_norm_part = best_overall.split('_')[1] if '_' in best_overall else 'none'

        print(f"→ Metoda dekonwolucji: {best_deconv_part}")
        print(f"→ Metoda normalizacji: {best_norm_part}")
