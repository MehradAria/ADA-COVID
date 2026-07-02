"""Training progress visualization."""

from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.dpi"] = 100


def plot_training_history(history: Dict[str, List[float]], output_path: str = "ada_covid_training.png") -> None:
    if not history["iter"]:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["iter"], history["src_acc"], label="Source", linewidth=2)
    axes[0].plot(
        history["iter"], history["tgt_acc"], label="Target", linewidth=2, linestyle="--"
    )
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Classification Accuracy During Training")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["iter"], history["dis_acc"], color="red", linewidth=2)
    axes[1].axhline(y=50, color="gray", linestyle=":", label="Chance (50%)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Domain Accuracy (%)")
    axes[1].set_title(
        "Domain Discriminator Accuracy\n(should converge toward 50% - domain confusion)"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("ADA-COVID Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training plot saved to {output_path}")
