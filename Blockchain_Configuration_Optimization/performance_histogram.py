# ==========================================================
# IMPORTS MODULE HERE
# ==========================================================
import matplotlib.pyplot as plt
import numpy as np

def plot_grouped_gap_histogram(*settings):
    """
    Plots grouped bar chart for 7 algorithms across 6 different settings.
    Settings are automatically sorted by their average utility (ascending):
    - Lower utility → left (first)
    - Higher utility → right (last)
    Expects exactly 6 setting arrays (each with 7 numeric values, one per algorithm).
    """
    if len(settings) != 6:
        raise ValueError("This function expects exactly 6 setting arrays.")

    algos = ["BCGA", "RCGA", "PSO", "TLBO", "DE", "ABC", "HYBRID GA"]

    # Distinct hatch patterns for 7 algorithms
    hatches = ['', '//', '\\\\', '||', '--', 'xx', '..']

    # Shades of gray for 7 algorithms
    gray_shades = [0.85, 0.70, 0.55, 0.40, 0.25, 0.10, 0.95]
    colors = [plt.cm.Greys(s) for s in gray_shades]

    def pad(values):
        """Ensure each setting has exactly 7 values (pad with NaN if needed)."""
        return list(values) + [np.nan] * (7 - len(values))

    # Pad and create data array (shape: 6 settings × 7 algorithms)
    padded_settings = [pad(s) for s in settings]
    data = np.array(padded_settings)   # shape (6, 7)

    # --- SORT SETTINGS BY THEIR MEAN (ascending: lower first, higher last) ---
    # Compute mean for each setting, ignoring NaN values
    means = np.nanmean(data, axis=1)
    # Get order indices that sort means from smallest to largest
    sort_idx = np.argsort(means)
    # Reorder data and keep original labels for reference
    data_sorted = data[sort_idx]
    # Create new setting labels based on the sorted order
    # setting_labels = [f"Setting {i+1}" for i, idx in enumerate(sort_idx)]
    # Alternatively, you can simply use "Setting 1,2,3..." in sorted order:
    setting_labels = [f"Setting {i+1}" for i in range(len(settings))]

    n_settings, n_algos = data_sorted.shape

    bar_width = 0.20
    group_spacing = 0.5

    # X positions for each setting group
    indices = np.arange(n_settings) * (n_algos * bar_width + group_spacing)

    plt.figure(figsize=(13, 6))

    # Plot bars for each algorithm
    for i in range(n_algos):
        x_pos = indices + i * bar_width
        bars = plt.bar(
            x_pos,
            data_sorted[:, i],
            width=bar_width,
            edgecolor='black',
            color=colors[i],
            hatch=hatches[i],
            label=algos[i]
        )

        # Annotate values
        for j, val in enumerate(data_sorted[:, i]):
            if not np.isnan(val):
                plt.text(x_pos[j], val + 0.008, f"{val:.2f}",
                         ha='center', fontsize=7, rotation=45)

    # X‑axis labels (sorted order)
    group_centers = indices + (n_algos * bar_width) / 2
    plt.xticks(group_centers, setting_labels, rotation=0)

    plt.xlabel("BLOCKCHAIN CONFIGURATION SETTINGS")
    plt.ylabel("AVERAGE BEST UTILITY")
    plt.title("ALGORITHM PERFORMANCE COMPARISON ON BLOCKCHAIN CONFIGURATION OPTIMIZATION (MINIMIZATION PROBLEM)")

    plt.legend(loc='best', framealpha=0.9)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================================
# EXECUTION STARTS HERE
# ==========================================================
if __name__ == "__main__":
    # Example data for 6 settings, each with 7 algorithm values.
    # The order must match algos: BCGA, RCGA, PSO, TLBO, DE, ABC, HYBRID GA
    setting1 = [0.40, 0.41, 0.42, 0.40, 0.40, 0.41, 0.40]
    setting2 = [0.41, 0.42, 0.42, 0.41, 0.41, 0.42, 0.41]
    setting3 = [0.43, 0.44, 0.44, 0.43, 0.43, 0.44, 0.43]
    setting4 = [0.44, 0.45, 0.45, 0.44, 0.44, 0.45, 0.44]
    setting5 = [0.44, 0.45, 0.45, 0.45, 0.44, 0.46, 0.44]
    setting6 = [0.44, 0.45, 0.46, 0.45, 0.46, 0.46, 0.45]

    plot_grouped_gap_histogram(setting1, setting2, setting3, setting4, setting5, setting6)