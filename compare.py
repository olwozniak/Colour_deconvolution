import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from deconvolution import Deconvolution
from deconvolution_processing import Process
from normalization import Normalization


class Compare:

    @staticmethod
    def compare_methods(image, target_image=None, save_dir=None):
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

        # Zapisz wykres jeśli podano ścieżkę
        if save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"compare_methods_{timestamp}.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Wykres zapisano jako: {save_path}")

        plt.show()

        if save_dir:
            return save_path
        return None

    @staticmethod
    def compare_clustering_methods(image, target_image, deconv_method='wavelet', norm_method='stain_specific',
                                   save_dir=None):
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

        # Zapisz wykres jeśli podano ścieżkę
        if save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"clustering_comparison_{timestamp}.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Wykres zapisano jako: {save_path}")

        plt.show()

        # Generuj analizę metod klasteryzacji
        Compare.analyze_clustering_results(results)

        if save_dir:
            # Zapisz również wyniki analizy do pliku tekstowego
            analysis_filename = f"clustering_analysis_{timestamp}.txt"
            analysis_path = os.path.join(save_dir, analysis_filename)
            Compare.save_analysis_to_file(results, analysis_path)

            return save_path, analysis_path

        return results

    @staticmethod
    def save_analysis_to_file(results, filepath):
        """Zapisuje analizę do pliku tekstowego"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("ANALIZA METOD KLASTERYZACJI\n")
            f.write("=" * 50 + "\n\n")

            for method, metrics in results.items():
                f.write(f"Metoda: {method}\n")
                if isinstance(metrics, dict) and 'error' not in metrics:
                    for key, value in metrics.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  Error: {metrics}\n")
                f.write("\n")

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
            print(f"Wykres ilościowy zapisano jako: {save_path}")
        plt.show()

    @staticmethod
    def quantitative_comparison(image, target_image=None, save_dir=None):
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

        # Zapisz wyniki do pliku jeśli podano ścieżkę
        if save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"quantitative_results_{timestamp}.json"
            results_path = os.path.join(save_dir, results_filename)

            import json
            # Konwersja do formatu JSON-serializable
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                                 for k, v in value.items()}
                else:
                    serializable_results[key] = value

            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)

            print(f"Wyniki ilościowe zapisano jako: {results_path}")

            # Stwórz i zapisz wykres
            plot_filename = f"quantitative_plot_{timestamp}.png"
            plot_path = os.path.join(save_dir, plot_filename)
            Compare.plot_quantitative_results(results, plot_path)

            return results, results_path, plot_path

        return results


# Przykład użycia:
if __name__ == "__main__":
    # Załaduj obrazy
    image = cv2.imread('sciezka/do/obrazu.jpg')
    target_image = cv2.imread('sciezka/do/obrazu_referencyjnego.jpg')

    # Utwórz katalog do zapisu jeśli nie istnieje
    save_directory = "results"
    os.makedirs(save_directory, exist_ok=True)

    # Użyj z zapisem do pliku
    saved_plot_path = Compare.compare_methods(image, target_image, save_dir=save_directory)

    # Wyniki ilościowe z zapisem
    results, results_path, plot_path = Compare.quantitative_comparison(
        image, target_image, save_dir=save_directory
    )