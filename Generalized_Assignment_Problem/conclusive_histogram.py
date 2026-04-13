# ==========================================================
# IMPORTS MODULE HERE
# ==========================================================
import matplotlib.pyplot as plt
import numpy as np

def plot_grouped_gap_histogram(gap8, gap10, gap12):
    algos = ["BCGA", "RCGA", "PSO", "DE", "TLBO", "ABC"]

    # Ensure all have 6 values (pad if needed)
    def pad(values):
        return values + [np.nan] * (6 - len(values))

    gap8 = pad(gap8)
    gap10 = pad(gap10)
    gap12 = pad(gap12)

    data = np.array([gap8, gap10, gap12])  # shape (3,6)

    n_groups = data.shape[0]   # 3 GAPs
    n_algos = data.shape[1]    # 6 algorithms

    bar_width = 0.24
    group_spacing = 0.5

    # X positions for groups
    indices = np.arange(n_groups) * (n_algos * bar_width + group_spacing)

    plt.figure()

    # Plot bars
    for i in range(n_algos):
        x_pos = indices + i * bar_width
        bars = plt.bar(x_pos, data[:, i], width=bar_width, label=algos[i])

        # Annotate values
        for j, val in enumerate(data[:, i]):
            if not np.isnan(val):
                plt.text(x_pos[j], val + 0.5, f"{val:.1f}", ha='center', fontsize=8)

    # X-axis labels centered per group
    group_centers = indices + (n_algos * bar_width) / 2
    plt.xticks(group_centers, ["GAP 8", "GAP 10", "GAP 12"])

    plt.xlabel("Dataset (First instance of GAP-8, GAP-10, GAP-12 File)")
    plt.ylabel("Average Best Cost")
    plt.title("Algorithm Performance Comparison on Generalised Assignment Problem (Maximization Problem)")

    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================================
# EXECUTION STARTS HERE
# ==========================================================
if __name__ == "__main__": 
    gap12 = [1425.28, 1422.44, 1435.33, 1427.00, 1410.33, 1412.20]
    gap10 = [948.07, 943.00, 954.40, 936.00, 948.38, 926.15]
    gap8  = [959.00, 1070.00, 1085.00, 1006, 1050]

    plot_grouped_gap_histogram(gap8, gap10, gap12)
