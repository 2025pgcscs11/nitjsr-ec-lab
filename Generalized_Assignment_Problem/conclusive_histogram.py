# ==========================================================
# IMPORTS MODULE HERE
# ==========================================================
import matplotlib.pyplot as plt
import numpy as np

def plot_grouped_gap_histogram(x, y, z):
    algos = ["BCGA", "RCGA", "PSO", "DE", "TLBO", "ABC"]

    # Distinct hatch patterns (more variety)
    hatches = ['', '//', '\\\\', '||', '--', 'xx']

    # Shades of gray (0=black, 1=white) – dark enough to be visible
    gray_shades = [0.85, 0.70, 0.55, 0.40, 0.25, 0.10]
    colors = [plt.cm.Greys(s) for s in gray_shades]

    def pad(values):
        return values + [np.nan] * (6 - len(values))

    x = pad(x)
    y = pad(y)
    z = pad(z)

    data = np.array([x, y, z])

    n_groups = data.shape[0]
    n_algos = data.shape[1]

    bar_width = 0.24
    group_spacing = 0.5

    indices = np.arange(n_groups) * (n_algos * bar_width + group_spacing)

    plt.figure(figsize=(10, 6))

    for i in range(n_algos):
        x_pos = indices + i * bar_width

        bars = plt.bar(
            x_pos,
            data[:, i],
            width=bar_width,
            edgecolor='black',
            color=colors[i],          # different gray fill
            hatch=hatches[i],         # additional pattern
            label=algos[i]
        )

        # Annotate values (adjust vertical offset for readability)
        for j, val in enumerate(data[:, i]):
            if not np.isnan(val):
                plt.text(x_pos[j], val + 0.8, f"{val:.1f}", ha='center', fontsize=8)

    group_centers = indices + (n_algos * bar_width) / 2
    plt.xticks(group_centers, ["GAP 12", "GAP 8", "GAP 10"])

    plt.xlabel("DATASET (First instance of GAP-8, GAP-10, GAP-12 File)")
    plt.ylabel("AVERAGE BEST COST")
    plt.title("ALGORITHM PERFORMANCE COMPARISION ON GAP")

    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================================
# EXECUTION STARTS HERE
# ==========================================================
if __name__ == "__main__": 
    # ["BCGA", "RCGA", "PSO", "DE", "TLBO", "ABC"]
    gap12 = [1428.56, 1422.57, 1440.25, 1423.00, 1436.43, 1423.55]
    gap10 = [945.85, 942.36, 953.30, 954.00, 940.00, 933.95]
    gap8  = [959.00, 1070.00, 1085.00, 1006, 1050, 1065.65]
    
    plot_grouped_gap_histogram(gap12, gap8, gap10)