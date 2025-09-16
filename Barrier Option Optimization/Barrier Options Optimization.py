import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def barrier_option_mc_lagrangian(S0, K, B, r, sigma, T, N, M, lambda_penalty, option_type='knock_out', plot_paths=False):
    dt = T / N
    discount_factor = np.exp(-r * T)

    # Generate random normal shocks
    np.random.seed(42)  # For reproducible results
    Z = np.random.normal(size=(M, N))
    
    # Initialize asset paths
    S = np.zeros((M, N + 1))
    S[:, 0] = S0

    # Time array for plotting
    time = np.linspace(0, T, N + 1)

    # Simulate asset paths
    for t in range(1, N + 1):
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

    # Calculate min and max prices
    min_price = np.min(S, axis=1)
    max_price = np.max(S, axis=1)

    # Calculate payoff with penalty
    if option_type == 'knock_out':
        violation = np.maximum(0, B - min_price)
        payoff = np.maximum(S[:, -1] - K, 0) + lambda_penalty * violation
        knock_out_mask = min_price <= B  # Paths that hit the barrier
    else:  # knock_in
        violation = np.maximum(0, min_price - B)
        payoff = np.maximum(S[:, -1] - K, 0) + lambda_penalty * violation
        knock_out_mask = min_price > B  # Paths that never hit the barrier

    option_price = discount_factor * np.mean(payoff)

    # Plotting functions
    if plot_paths:
        plot_combined_results(S, time, K, B, option_type, knock_out_mask, min_price, payoff, option_price)

    return option_price, S, payoff, knock_out_mask

def plot_combined_results(S, time, K, B, option_type, knock_out_mask, min_price, payoff, option_price):
    """Plot all results in a 2x2 grid"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{option_type.replace("_", " ").title()} Barrier Option Analysis\nEstimated Price: ${option_price:.2f}', fontsize=16, fontweight='bold')
    
    # Plot 1: Price paths (top-left)
    plot_price_paths(axes[0, 0], S, time, K, B, option_type, knock_out_mask)
    
    # Plot 2: Final price distribution (top-right)
    plot_final_price_distribution(axes[0, 1], S[:, -1], K, option_type)
    
    # Plot 3: Payoff distribution (bottom-left)
    plot_payoff_distribution(axes[1, 0], payoff, option_type)
    
    # Plot 4: Barrier hit analysis (bottom-right)
    plot_barrier_analysis(axes[1, 1], min_price, B, option_type)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

def plot_price_paths(ax, S, time, K, B, option_type, knock_out_mask):
    """Plot asset price paths with barrier and strike lines"""
    num_paths_to_plot = min(100, S.shape[0])
    
    if option_type == 'knock_out':
        active_paths = ~knock_out_mask
        inactive_paths = knock_out_mask
        active_color = 'green'
        inactive_color = 'red'
    else:  # knock_in
        active_paths = knock_out_mask
        inactive_paths = ~knock_out_mask
        active_color = 'blue'
        inactive_color = 'gray'
    
    # Plot inactive paths
    for i in range(min(num_paths_to_plot, len(inactive_paths))):
        if inactive_paths[i]:
            ax.plot(time, S[i], color=inactive_color, alpha=0.1, linewidth=0.5)
    
    # Plot active paths
    active_count = 0
    for i in range(min(num_paths_to_plot, len(active_paths))):
        if active_paths[i] and active_count < 20:
            ax.plot(time, S[i], color=active_color, alpha=0.7, linewidth=1.5)
            active_count += 1
    
    # Add barrier and strike lines
    ax.axhline(y=B, color='red', linestyle='--', linewidth=2, label=f'Barrier (B={B})')
    ax.axhline(y=K, color='orange', linestyle='--', linewidth=2, label=f'Strike (K={K})')
    ax.axhline(y=S[0, 0], color='black', linestyle='-', linewidth=1, label=f'Initial Price (S0={S[0, 0]})')
    
    ax.set_xlabel('Time (Years)')
    ax.set_ylabel('Asset Price')
    ax.set_title('Price Paths Simulation')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_final_price_distribution(ax, final_prices, K, option_type):
    """Plot distribution of final asset prices"""
    ax.hist(final_prices, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(x=K, color='red', linestyle='--', linewidth=2, label=f'Strike (K={K})')
    ax.axvline(x=np.mean(final_prices), color='green', linestyle='--', linewidth=2, 
               label=f'Mean: ${np.mean(final_prices):.2f}')
    ax.set_xlabel('Final Asset Price')
    ax.set_ylabel('Frequency')
    ax.set_title('Final Price Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_payoff_distribution(ax, payoff, option_type):
    """Plot distribution of option payoffs"""
    ax.hist(payoff, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.axvline(x=np.mean(payoff), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: ${np.mean(payoff):.2f}')
    ax.set_xlabel('Payoff')
    ax.set_ylabel('Frequency')
    ax.set_title('Payoff Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_barrier_analysis(ax, min_prices, B, option_type):
    """Plot analysis of barrier hits"""
    barrier_hits = min_prices <= B
    hit_percentage = np.mean(barrier_hits) * 100
    
    # Bar chart instead of pie chart for better readability
    status = ['Hit Barrier', 'Missed Barrier']
    counts = [np.sum(barrier_hits), np.sum(~barrier_hits)]
    colors = ['lightcoral', 'lightgreen']
    
    bars = ax.bar(status, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Paths')
    ax.set_title(f'Barrier Hit Analysis\nTotal Hit Rate: {hit_percentage:.1f}%')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)

# Run the simulation with plotting
if __name__ == "__main__":
    # Parameters
    S0 = 100
    K = 105
    B = 95
    r = 0.05
    sigma = 0.2
    T = 1.0
    N = 252
    M = 1000  # Reduced for faster plotting
    lambda_penalty = 1000
    
    print("Running Knock-Out Barrier Option Simulation...")
    price_ko, paths_ko, payoffs_ko, mask_ko = barrier_option_mc_lagrangian(
        S0, K, B, r, sigma, T, N, M, lambda_penalty, 'knock_out', True
    )
    
    print("\nRunning Knock-In Barrier Option Simulation...")
    price_ki, paths_ki, payoffs_ki, mask_ki = barrier_option_mc_lagrangian(
        S0, K, B, r, sigma, T, N, M, lambda_penalty, 'knock_in', True
    )
    
    print(f"\nFinal Results:")
    print(f"Knock-Out Barrier Option Price: ${price_ko:.4f}")
    print(f"Knock-In Barrier Option Price: ${price_ki:.4f}")