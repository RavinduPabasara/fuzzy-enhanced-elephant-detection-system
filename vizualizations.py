import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz


def plot_membership_functions():
    """
    Visualize membership functions for the Enhanced Fuzzy Confidence Controller
    """
    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Fuzzy Logic Membership Functions', fontsize=16)

    # Universe of discourse
    universe = np.linspace(0, 1, 100)

    # 1. Object Complexity Membership Functions
    complexity_labels = ['very_simple', 'simple', 'moderate', 'complex']
    complexity_mfs = [
        fuzz.trimf(universe, [0, 0, 0.2]),  # very_simple
        fuzz.trimf(universe, [0.1, 0.3, 0.5]),  # simple
        fuzz.trimf(universe, [0.4, 0.6, 0.8]),  # moderate
        fuzz.trimf(universe, [0.7, 0.9, 1])  # complex
    ]

    axs[0].set_title('Object Complexity Membership Functions')
    axs[0].set_xlabel('Complexity Level')
    axs[0].set_ylabel('Membership Degree')

    for label, mf in zip(complexity_labels, complexity_mfs):
        axs[0].plot(universe, mf, label=label)
    axs[0].legend()
    axs[0].grid(True)

    # 2. Object Size Membership Functions
    size_labels = ['very_small', 'small', 'medium', 'large']
    size_mfs = [
        fuzz.trimf(universe, [0, 0, 0.2]),  # very_small
        fuzz.trimf(universe, [0.1, 0.3, 0.5]),  # small
        fuzz.trimf(universe, [0.4, 0.6, 0.8]),  # medium
        fuzz.trimf(universe, [0.7, 0.9, 1])  # large
    ]

    axs[1].set_title('Object Size Membership Functions')
    axs[1].set_xlabel('Relative Size')
    axs[1].set_ylabel('Membership Degree')

    for label, mf in zip(size_labels, size_mfs):
        axs[1].plot(universe, mf, label=label)
    axs[1].legend()
    axs[1].grid(True)

    # 3. Detection History Membership Functions
    history_labels = ['unstable', 'moderate', 'consistent']
    history_mfs = [
        fuzz.trimf(universe, [0, 0, 0.3]),  # unstable
        fuzz.trimf(universe, [0.2, 0.5, 0.8]),  # moderate
        fuzz.trimf(universe, [0.7, 1, 1])  # consistent
    ]

    axs[2].set_title('Detection History Membership Functions')
    axs[2].set_xlabel('Detection Stability')
    axs[2].set_ylabel('Membership Degree')

    for label, mf in zip(history_labels, history_mfs):
        axs[2].plot(universe, mf, label=label)
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


# Run the visualization
plot_membership_functions()