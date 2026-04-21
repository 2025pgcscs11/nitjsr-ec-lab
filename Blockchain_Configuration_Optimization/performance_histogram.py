# ==========================================================
# IMPORTS MODULE HERE
# ==========================================================
import matplotlib.pyplot as plt
import numpy as np

def plot_grouped_gap_histogram(setting1, setting2):
    algos = ["BCGA", "RCGA", "PSO", "DE", "TLBO", "ABC", "Hybrid GA"]
    
    # Ensure all have 6 values (pad if needed)
    def pad(values):
        return values + [np.nan] * (6 - len(values))
    
    setting1 = pad(setting1)
    setting2 = pad(setting2)
    
    data = np.array([setting1, setting2])  # shape (2,7)
    
    n_groups = data.shape[0]   # 2 BCOs
    n_algos = data.shape[1]    # 7 algorithms
    
    bar_width = 0.12  # Reduced for 7 algorithms
    group_spacing = 0.3
    
    # X positions for groups
    indices = np.arange(n_groups) * (n_algos * bar_width + group_spacing)
    
    plt.figure(figsize=(12, 6))
    
    # Plot bars
    for i in range(n_algos):
        x_pos = indices + i * bar_width
        bars = plt.bar(x_pos, data[:, i], width=bar_width, label=algos[i])
        
        # Annotate values
        for j, val in enumerate(data[:, i]):
            if not np.isnan(val):
                plt.text(x_pos[j], val + 0.002, f"{val:.2f}", 
                        ha='center', fontsize=7, rotation=45)
    
    # X-axis labels centered per group
    group_centers = indices + (n_algos * bar_width) / 2
    plt.xticks(group_centers, ["Setting 1", "Setting 2"])
    
    plt.xlabel("Blockchain Configuration Settings")
    plt.ylabel("Average Best Utility")
    plt.title("Algorithm Performance Comparison on Blockchain Configuration Optimization (Minimization Problem)")
    
    plt.legend(loc='upper right', ncol=1)  # 2 columns for legend
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================================
# EXECUTION STARTS HERE
# ==========================================================
if __name__ == "__main__": 
    setting1 = [0.41, 0.41, 0.42, 0.41, 0.41, 0.41, 0.41]
    setting2 = [0.43, 0.43, 0.44, 0.43, 0.43, 0.43, 0.43]
    
    plot_grouped_gap_histogram(setting1, setting2)