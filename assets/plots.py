import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def plot_trust_spectrum(class_nts: list, density_curves: list, x_range: np.ndarray, n_classes: int) -> None:
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

def plot_conditional_trust_densities(correct_trust: list, incorrect_trust: list, n_classes: int) -> None:
        """
        Plot the conditional trust density curves for correct and incorrect predictions per class.

        Args:
            correct_trust (list): Trust scores for correct predictions per class.
            incorrect_trust (list): Trust scores for incorrect predictions per class.
            n_classes (int): Number of classes.
        """
        x_range = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(6 * n_classes, 6), ncols=n_classes, sharey=True)
        if n_classes == 1:
            ax = [ax]
        for c in range(n_classes):∂∂∂
            # Correct predictions
            kde_corr = KernelDensity(bandwidth=0.5 / np.sqrt(max(len(correct_trust[c]), 1)), kernel='gaussian')
            kde_corr.fit(np.array(correct_trust[c] or [0.5])[:, None])
            logprob_corr = kde_corr.score_samples(x_range[:, None])
            ax[c].plot(x_range, np.exp(logprob_corr), label='Correct', color='blue')

            # Incorrect predictions
            kde_incorr = KernelDensity(bandwidth=0.5 / np.sqrt(max(len(incorrect_trust[c]), 1)), kernel='gaussian')
            kde_incorr.fit(np.array(incorrect_trust[c] or [0.5])[:, None])
            logprob_incorr = kde_incorr.score_samples(x_range[:, None])
            ax[c].plot(x_range, np.exp(logprob_incorr), label='Incorrect', color='red')

            ax[c].set_title(f'Class {c}', fontsize=24)
            ax[c].legend()
            ax[c].set_xlabel('Question-Answer Trust', fontsize=24, fontweight='bold')
            if c == 0:
                ax[c].set_ylabel('Trust Density', fontsize=24, fontweight='bold')
            ax[c].tick_params(labelsize=24)
        plt.tight_layout()
        plt.savefig(os.path.join('./conditional_trust_densities.png'))
        plt.close()