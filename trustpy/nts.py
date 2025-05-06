import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


class NTS:
    def __init__(self, oracle: np.ndarray, predictions: np.ndarray,
                 alpha: float = 1.0, beta: float = 1.0,
                 trust_spectrum: bool = False,
                 show_summary: bool = True,
                 export_summary: bool = True) -> None:
        """
        It computes trust scores for each class, estimates trust density using KDE, and calculates per-class and overall NetTrustScore (NTS). Optionally plots trust spectrum.

        Args:
            oracle (np.ndarray): True labels.
            predictions (np.ndarray): SoftMax probabilities predicted by a model (e.g., DNNs).
            alpha (float): Reward factor for correct predictions. Defaults to 1.0.
            beta (float): Penalty factor for incorrect predictions. Defaults to 1.0.
            trust_spectrum (bool): If True plots the trust spectrum. Defaults to False.
            show_summary (bool): If True, prints a summary table of NTS values. Defaults to True.
            export_summary (bool): If True, saves a summary table of NTS values to a CSV file. Defaults to True.
        """

        assert isinstance(oracle, np.ndarray), 'Oracle/Actual Classes must be a NumPy array'
        assert isinstance(predictions, np.ndarray), 'Predictions/Predicted Classes must be a NumPy array'
        assert isinstance(alpha, (int, float)), 'alpha must be a number'
        assert isinstance(beta, (int, float)), 'beta must be a number'
        assert isinstance(trust_spectrum, bool), 'trust_spectrum must be True/False'
        assert isinstance(show_summary, bool), 'show_summary must be True/False'
        assert isinstance(export_summary, bool), 'export_summary must be True/False'

        assert oracle.ndim == 1, 'Oracle/Actual Classes must be a 1D array'
        assert predictions.ndim == 2, 'Predictions/Predicted Classes must be a 2D array'
        assert oracle.shape[0] == predictions.shape[0], (f'Number of samples mismatch: oracle ({oracle.shape[0]}) vs predictions ({predictions.shape[0]})')

        alpha = float(alpha)
        beta = float(beta)
        assert alpha > 0, 'alpha must be positive'
        assert beta > 0, 'beta must be positive'
        assert np.all((predictions >= 0) & (predictions <= 1)), 'Predictions must be between 0 and 1'
        assert np.allclose(predictions.sum(axis=1), 1, atol=1e-5), 'Each row of SoftMax predictions must sum to 1'

        self.oracle = oracle
        self.predictions = predictions
        self.alpha = alpha
        self.beta = beta
        self.trust_spectrum = trust_spectrum
        self.show_summary = show_summary
        self.export_summary = export_summary

    def compute(self) -> dict:
        """
        Compute the NTS for each class and overall, with optional trust spectrum plot.

        Returns:
            dict: A dictionary containing NTS for each class and the overall NTS
                  Keys: 'class_0', ..., 'class_n', 'overall'
        """
        n_classes = self.predictions.shape[1]
        qa_trust = self._compute_question_answer_trust(n_classes)

        class_nts, density_curves, x_range = self._compute_trust_density(qa_trust)

        if self.trust_spectrum:
            self._plot_trust_spectrum(class_nts, density_curves, x_range, n_classes)

        overall_nts = self._compute_overall_NTS(class_nts, qa_trust)

        nts_dict = {f'class_{i}': f'{nts:.3f}' for i, nts in enumerate(class_nts)}
        nts_dict['overall'] = f'{overall_nts:.3f}'

        if self.show_summary:
            self.print_summary(nts_dict)

        if self.export_summary:
            self.export_summary_to_file(nts_dict)

        return nts_dict

    def _compute_question_answer_trust(self, n_classes: int) -> list:
        """
        Compute the question-answer trust scores for each class.

        Args:
            n_classes (int): Number of classes.

        Returns:
            list: List of lists, each containing trust scores for a class.
        """
        predicted_class = np.argmax(self.predictions, axis=1)
        
        qa_trust = [[] for _ in range(n_classes)]
        for i in range(self.oracle.shape[0]):
            true_label = self.oracle[i]
            pred_label = predicted_class[i]
            max_prob = self.predictions[i, pred_label]
            if pred_label == true_label:
                qa_trust[true_label].append(max_prob**self.alpha)
            else:
                qa_trust[true_label].append((1 - max_prob)**self.beta)
        return qa_trust

    def _compute_trust_density(self, qa_trust: list) -> tuple:
        """
        Compute the NTS and trust density curves for each class.

        Args:
            qa_trust (list): List of trust scores for each class.

        Returns:
            tuple: (class_nts, density_curves, x_range)
                - class_nts (list): NTS for each class.
                - density_curves (list): Density curves for each class.
                - x_range (np.ndarray): X-axis values for density curves.
        """
        class_nts, density_curves = [], []
        x_range = np.linspace(0, 1, 100)
        for target in qa_trust:
            target = np.asarray(target)
            tm = np.mean(target) if len(target) > 0 else 0.0
            class_nts.append(tm)
            kde = KernelDensity(bandwidth=0.5 / np.sqrt(max(len(target), 1)), kernel='gaussian')
            kde.fit(target[:, None] if len(target) > 0 else np.array([[0.5]]))
            logprob = kde.score_samples(x_range[:, None])
            density_curves.append(np.exp(logprob))
        return class_nts, density_curves, x_range

    def _plot_trust_spectrum(self, class_nts: list, density_curves: list, x_range: np.ndarray, n_classes: int,
                             filename: str = "trust_spectrum.png") -> None:
        """
        Plot the trust density curves for each class.

        Args:
            class_nts (list): NTS for each class.
            density_curves (list): Density curves for each class.
            x_range (np.ndarray): X-axis values for density curves.
            n_classes (int): Number of classes.
            filename (str): Name of the saved trust spectrum image.
        """
        assert isinstance(filename, str), 'filename must be a string'
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
            filename += '.png'
                                 
        class_labels = [f'Class {i}' for i in range(n_classes)]
        colors = plt.cm.tab10(np.arange(n_classes))
        fig, ax = plt.subplots(figsize=(6 * n_classes, 6), ncols=n_classes, sharey=True)
        if n_classes == 1:
            ax = [ax]
        for c in range(n_classes):
            ax[c].plot(x_range, density_curves[c], linestyle='dashed', color=colors[c])
            ax[c].fill_between(x_range, density_curves[c], alpha=0.5, color=colors[c])
            ax[c].set_xlabel('Question-Answer Trust', fontsize=24, fontweight='bold')
            if c == 0:
                ax[c].set_ylabel('Trust Density', fontsize=24, fontweight='bold')
            ax[c].tick_params(labelsize=24)
            ax[c].set_title(f'{class_labels[c]}\nNTS = {class_nts[c]:.3f}', fontsize=24)
        plt.tight_layout()
        output_dir = os.path.join(os.getcwd(), "trustpy", "nts")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()

    def _compute_overall_NTS(self, class_nts: list, qa_trust: list) -> float:
        """
        Compute the overall NTS across all classes.

        Args:
            class_nts (list): NTS for each class.
            qa_trust (list): List of trust scores for each class.

        Returns:
            float: Overall NTS.
        """
        overall_nts = sum(tm * len(ts) for tm, ts in zip(class_nts, qa_trust))
        total_samples = sum(len(ts) for ts in qa_trust)
        return overall_nts / total_samples if total_samples > 0 else 0.0

    def print_summary(self, nts_dict: dict) -> None:
        """
        Pretty prints a summary table of NTS, conditional NTS values.

        Args:
            nts_dict (dict): Dictionary of trust scores computed by compute().
        """
        classes = sorted(k.split('_')[1] for k in nts_dict if k.startswith('class_'))
        print(f"{'Class':<10} {'NTS':<10}")
        print("-" * 20)
        for c in classes:
            print(f"{c:<10} {nts_dict[f'class_{c}']:<10}")
        print("-" * 20)
        print(f"{'Overall':<10} {nts_dict.get('overall', '-'):<10}")

    def export_summary_to_file(self, nts_dict: dict, filename: str = "nts_summary.csv") -> None:
        """
        Saves a summary table of NTS, conditional NTS values to a CSV file.

        Args:
            nts_dict (dict): Dictionary of trust scores computed by compute().
            filename (str): Filename to save the summary. Defaults to 'trust_summary.csv'.
        """
        output_dir = os.path.join(os.getcwd(), "trustpy", "nts")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        fields = ['Class', 'NTS']
        classes = sorted(k.split('_')[1] for k in nts_dict if k.startswith('class_'))

        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for c in classes:
                writer.writerow([c, nts_dict[f'class_{c}']])
            writer.writerow(['Overall', nts_dict.get('overall', '-')])

