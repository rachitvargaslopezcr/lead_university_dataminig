import matplotlib.pyplot as plt
import numpy as np


def plot_varianza_explicada(explained: np.ndarray, cumulative: np.ndarray) -> None:
    """Grafica barras de varianza explicada y curva acumulada."""
    components = [f"PC{i+1}" for i in range(len(explained))]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(components, explained, color="slateblue", alpha=0.85)
    ax1.set_ylabel("% Varianza explicada")
    ax1.set_xlabel("Componentes principales")

    ax2 = ax1.twinx()
    ax2.plot(components, cumulative, color="darkorange", marker="o", linewidth=2)
    ax2.set_ylabel("% Varianza acumulada")
    ax2.set_ylim(0, 105)

    for i, value in enumerate(explained):
        ax1.text(i, value + 1, f"{value:.1f}%", ha="center", fontsize=9)

    plt.title("Varianza explicada por componente")
    fig.tight_layout()
    plt.show()

