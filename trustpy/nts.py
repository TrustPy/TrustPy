import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

class NTS:
    def __init__(self, oracle: np.ndarray, predictions: np.ndarray,
                 alpha: float = 1.0, beta: float = 1.0,
                 trust_spectrum: bool = False) -> None:
        """
        It computes trust scores for each class, estimates trust density using KDE, and calculates per-class and overall NetTrustScore (NTS). Optionally plots trust spectrum.
        
        Args:
            oracle (np.ndarray): True labels.
            predictions (np.ndarray): SoftMax probabilities predicted by a model (e.g., DNNs).
            alpha (float): Reward factor for correct predictions. Defaults to 1.0.
            beta (float): Penalty factor for incorrect predictions. Defaults to 1.0.
            trust_spectrum (bool): If True plots the trust spectrum. Defaults to False.
        Raises:
            TypeError: If inputs are of incorrect type.
            ValueError: If array shapes or values are invalid.
        """
        
        assert isinstance(oracle, np.ndarray), 'Oracle/Actual Classes must be a NumPy array'
        assert isinstance(predictions, np.ndarray), 'Predictions/Predicted Classes must be a NumPy array'
        assert isinstance(alpha, (int, float)), 'alpha must be a number'
        assert isinstance(beta, (int, float)), 'beta must be a number'
        assert isinstance(trust_spectrum, bool), 'trust_spectrum must be True/False'
        
        assert oracle.ndim == 1, 'Oracle/Actual Classes must be a 1D array'
        assert predictions.ndim == 2, 'Predictions/Predicted Classes must be a 1D array'
        assert oracle.shape[0] == predictions.shape[0], (f'Number of samples mismatch: oracle ({oracle.shape[0]}) vs predictions ({predictions.shape[0]})')
        
        alpha = float(alpha)
        beta = float(beta)
        assert alpha > 0, 'alpha must be positive'
        assert beta > 0, 'beta must be positive'
        assert np.all((predictions >= 0) & (predictions <= 1)), 'Predictions must be between 0 and 1'
        assert np.allclose(predictions.sum(axis = 1), 1, atol = 1e-5), 'Each row of SoftMax predictions must sum to 1'
        
        self.oracle = oracle
        self.predictions = predictions
        self.alpha = alpha
        self.beta = beta
        self.trust_spectrum = trust_spectrum
        
    def compute(self) -> dict:
        """
        Compute the NTS for each class and overall, with optional trust spectrum plot.
        
        Returns:
            dict: A dictionary containing NTS for each class and the overall NTS
                - Keys are 'class_0', 'class_1', ..., 'class_n', 'overall'
        """
        n_classes = self.predictions.shape[1]        
        qa_trust = self._compute_question_answer_trust(n_classes)

        class_nts, density_curves, x_range = self._compute_trust_density(qa_trust)
        
        if self.trust_spectrum:
            self._plot_trust_spectrum(class_nts, density_curves, x_range, n_classes)
            
        # Compute overall NTS
        overall_nts = self._compute_overall_NTS(class_nts, qa_trust)
        
        nts_dict = {f'class_{i}': f'{nts:.3f}' for i, nts in enumerate(class_nts)}
        nts_dict['overall'] = f'{overall_nts:.3f}'
        return nts_dict

    def _compute_question_answer_trust(self, n_classes: int) -> list:
        """
        Compute the question-answer scores for each class.

        Args:
            n_classes (int): Number of classes.

        Returns:
            list: List of lists. Each sublist includes trust scores for a class.
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

        This method computes the per-class NTS and the density curves, aligning with 'trust density' (density estimate) and contributing to 'trust spectrum' (distribution of trust scores).

        Args:
            qa_trust (list): List of trust scores for each class.

        Returns:
            tuple: (class_nts, density_curves, x_range)
                - class_nts (list): NTS for each class.
                - density_curves (list): Density curves for each class, representing trust density.
                - x_range (np.ndarray): X-axis values for density curves, part of the trust spectrum.
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

    def _plot_trust_spectrum(self, class_nts: list, density_curves: list, x_range: np.ndarray, n_classes: int) -> None:
        """
        Plot the trust density curves for each class, visualizing the trust spectrum.

        This corresponds to the 'trust spectrum' in the paper, showing the distribution of trust scores via density curves.

        Args:
            class_nts (list): NTS for each class.
            density_curves (list): Density curves for each class.
            x_range (np.ndarray): X-axis values for density curves.
            n_classes (int): Number of classes.
        """
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
        plt.savefig(os.path.join('./trust_spectrum.png'))
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

    