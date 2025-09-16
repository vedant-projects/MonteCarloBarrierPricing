import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import time
from scipy import stats
import matplotlib.pyplot as plt
from enum import Enum

class LookbackOptionType(Enum):
    FIXED_STRIKE_CALL = "fixed_strike_call"
    FIXED_STRIKE_PUT = "fixed_strike_put"
    FLOATING_STRIKE_CALL = "floating_strike_call"
    FLOATING_STRIKE_PUT = "floating_strike_put"

@dataclass
class LookbackOptionParams:
    """Parameters for lookback option pricing"""
    S0: float  # Initial asset price
    r: float   # Risk-free rate
    sigma: float  # Volatility
    T: float   # Time to maturity (years)
    K: Optional[float] = None  # Strike price (for fixed strike options)
    option_type: LookbackOptionType = LookbackOptionType.FLOATING_STRIKE_CALL

@dataclass
class SimulationParams:
    """Parameters for Monte Carlo simulation"""
    n_paths: int  # Number of Monte Carlo paths
    n_steps: int  # Number of time steps
    lambda_penalty: float  # Lagrangian penalty parameter
    seed: int = 42  # Random seed for reproducibility
    antithetic: bool = True  # Use antithetic variates for variance reduction
    control_variate: bool = False  # Use control variates

@dataclass
class SimulationResults:
    """Container for simulation results"""
    price_estimate: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    computation_time: float
    paths: np.ndarray
    payoffs: np.ndarray
    min_prices: np.ndarray
    max_prices: np.ndarray
    hit_rates: Dict[str, float]

class LookbackOptionPricer:
    """Professional lookback option pricer using Monte Carlo simulation with variance reduction techniques"""
    
    def __init__(self, option_params: LookbackOptionParams, sim_params: SimulationParams):
        self.option_params = option_params
        self.sim_params = sim_params
        self.results: Optional[SimulationResults] = None
        
    def _validate_parameters(self) -> None:
        """Validate input parameters"""
        if self.option_params.S0 <= 0:
            raise ValueError("Initial asset price must be positive")
        if self.option_params.T <= 0:
            raise ValueError("Time to maturity must be positive")
        if self.option_params.sigma <= 0:
            raise ValueError("Volatility must be positive")
        if self.sim_params.n_paths <= 0:
            raise ValueError("Number of paths must be positive")
        if self.sim_params.n_steps <= 0:
            raise ValueError("Number of steps must be positive")
        
        # Validate strike price for fixed strike options
        if self.option_params.option_type in [LookbackOptionType.FIXED_STRIKE_CALL, LookbackOptionType.FIXED_STRIKE_PUT]:
            if self.option_params.K is None or self.option_params.K <= 0:
                raise ValueError("Strike price must be provided and positive for fixed strike options")
    
    def _simulate_paths(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate geometric Brownian motion paths with antithetic variates"""
        np.random.seed(self.sim_params.seed)
        
        dt = self.option_params.T / self.sim_params.n_steps
        n_paths_effective = self.sim_params.n_paths // 2 if self.sim_params.antithetic else self.sim_params.n_paths
        
        # Generate random shocks
        Z = np.random.normal(size=(n_paths_effective, self.sim_params.n_steps))
        
        if self.sim_params.antithetic:
            Z = np.vstack([Z, -Z])  # Antithetic variates
        
        # Initialize paths
        S = np.zeros((self.sim_params.n_paths, self.sim_params.n_steps + 1))
        S[:, 0] = self.option_params.S0
        
        # Simulate paths
        drift = (self.option_params.r - 0.5 * self.option_params.sigma ** 2) * dt
        diffusion = self.option_params.sigma * np.sqrt(dt)
        
        for t in range(1, self.sim_params.n_steps + 1):
            S[:, t] = S[:, t - 1] * np.exp(drift + diffusion * Z[:, t - 1])
        
        # Calculate path extremes
        min_prices = np.min(S, axis=1)
        max_prices = np.max(S, axis=1)
        
        return S, min_prices, max_prices
    
    def _calculate_payoffs(self, S: np.ndarray, min_prices: np.ndarray, max_prices: np.ndarray) -> np.ndarray:
        """Calculate lookback option payoffs with Lagrangian penalty"""
        final_prices = S[:, -1]
        
        if self.option_params.option_type == LookbackOptionType.FLOATING_STRIKE_CALL:
            payoff = final_prices - min_prices
            # Penalty for negative minimum prices (shouldn't occur with GBM)
            penalty = self.sim_params.lambda_penalty * np.maximum(0, -min_prices)
            
        elif self.option_params.option_type == LookbackOptionType.FLOATING_STRIKE_PUT:
            payoff = max_prices - final_prices
            penalty = self.sim_params.lambda_penalty * np.maximum(0, -max_prices)
            
        elif self.option_params.option_type == LookbackOptionType.FIXED_STRIKE_CALL:
            payoff = np.maximum(final_prices - self.option_params.K, 0)
            # Additional value from lookback feature (max(0, max_price - K) - payoff)
            lookback_bonus = np.maximum(max_prices - self.option_params.K, 0) - payoff
            payoff += lookback_bonus
            penalty = self.sim_params.lambda_penalty * np.maximum(0, -min_prices)
            
        elif self.option_params.option_type == LookbackOptionType.FIXED_STRIKE_PUT:
            payoff = np.maximum(self.option_params.K - final_prices, 0)
            lookback_bonus = np.maximum(self.option_params.K - min_prices, 0) - payoff
            payoff += lookback_bonus
            penalty = self.sim_params.lambda_penalty * np.maximum(0, -max_prices)
            
        else:
            raise ValueError(f"Unsupported option type: {self.option_params.option_type}")
        
        return payoff + penalty
    
    def _calculate_analytical_benchmark(self) -> Optional[float]:
        """Calculate analytical price for floating strike lookback options (if available)"""
        # Implementation of analytical formulas would go here
        # For simplicity, returning None - in practice, you'd implement the closed-form solutions
        return None
    
    def price_option(self) -> SimulationResults:
        """Main method to price the lookback option"""
        start_time = time.time()
        
        # Validate parameters
        self._validate_parameters()
        
        # Simulate paths
        S, min_prices, max_prices = self._simulate_paths()
        
        # Calculate payoffs
        payoffs = self._calculate_payoffs(S, min_prices, max_prices)
        
        # Discount and compute statistics
        discount_factor = np.exp(-self.option_params.r * self.option_params.T)
        price_estimate = discount_factor * np.mean(payoffs)
        standard_error = discount_factor * np.std(payoffs) / np.sqrt(self.sim_params.n_paths)
        
        # Confidence interval
        z_score = stats.norm.ppf(0.975)  # 95% CI
        ci_lower = price_estimate - z_score * standard_error
        ci_upper = price_estimate + z_score * standard_error
        
        # Calculate hit rates and other statistics
        hit_rates = {
            'min_below_S0': np.mean(min_prices < self.option_params.S0),
            'max_above_S0': np.mean(max_prices > self.option_params.S0),
            'positive_payoff': np.mean(payoffs > 0)
        }
        
        computation_time = time.time() - start_time
        
        self.results = SimulationResults(
            price_estimate=price_estimate,
            standard_error=standard_error,
            confidence_interval=(ci_lower, ci_upper),
            computation_time=computation_time,
            paths=S,
            payoffs=payoffs,
            min_prices=min_prices,
            max_prices=max_prices,
            hit_rates=hit_rates
        )
        
        return self.results
    
    def generate_report(self) -> None:
        """Generate comprehensive pricing report"""
        if self.results is None:
            raise ValueError("Run price_option() first")
        
        print("=" * 60)
        print("LOOKBACK OPTION PRICING REPORT")
        print("=" * 60)
        print(f"Option Type: {self.option_params.option_type.value}")
        print(f"Initial Price (S0): {self.option_params.S0:.2f}")
        print(f"Risk-Free Rate (r): {self.option_params.r:.3f}")
        print(f"Volatility (σ): {self.option_params.sigma:.3f}")
        print(f"Time to Maturity (T): {self.option_params.T:.2f} years")
        if self.option_params.K is not None:
            print(f"Strike Price (K): {self.option_params.K:.2f}")
        print(f"Number of Paths: {self.sim_params.n_paths:,}")
        print(f"Number of Steps: {self.sim_params.n_steps}")
        print(f"Antithetic Variates: {self.sim_params.antithetic}")
        print("-" * 60)
        print(f"Price Estimate: ${self.results.price_estimate:.4f}")
        print(f"Standard Error: ±{self.results.standard_error:.6f}")
        print(f"95% CI: [${self.results.confidence_interval[0]:.4f}, ${self.results.confidence_interval[1]:.4f}]")
        print(f"Computation Time: {self.results.computation_time:.3f} seconds")
        print(f"Positive Payoff Rate: {self.results.hit_rates['positive_payoff']:.2%}")
        print(f"Minimum Price < S0: {self.results.hit_rates['min_below_S0']:.2%}")
        print(f"Maximum Price > S0: {self.results.hit_rates['max_above_S0']:.2%}")
        print("=" * 60)
    
    def plot_results(self) -> None:
        """Generate diagnostic plots"""
        if self.results is None:
            raise ValueError("Run price_option() first")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Lookback Option Analysis - {self.option_params.option_type.value}', fontsize=16)
        
        # Plot 1: Sample paths
        n_sample_paths = min(50, self.sim_params.n_paths)
        time_axis = np.linspace(0, self.option_params.T, self.sim_params.n_steps + 1)
        for i in range(n_sample_paths):
            axes[0, 0].plot(time_axis, self.results.paths[i], alpha=0.7, linewidth=0.5)
        axes[0, 0].set_xlabel('Time (Years)')
        axes[0, 0].set_ylabel('Asset Price')
        axes[0, 0].set_title('Sample Simulated Paths')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Payoff distribution
        axes[0, 1].hist(self.results.payoffs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(np.mean(self.results.payoffs), color='red', linestyle='--', label=f'Mean: {np.mean(self.results.payoffs):.2f}')
        axes[0, 1].set_xlabel('Payoff')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Payoff Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Minimum prices distribution
        axes[1, 0].hist(self.results.min_prices, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].axvline(np.mean(self.results.min_prices), color='red', linestyle='--', label=f'Mean: {np.mean(self.results.min_prices):.2f}')
        axes[1, 0].set_xlabel('Minimum Price')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Minimum Prices')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Maximum prices distribution
        axes[1, 1].hist(self.results.max_prices, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].axvline(np.mean(self.results.max_prices), color='red', linestyle='--', label=f'Mean: {np.mean(self.results.max_prices):.2f}')
        axes[1, 1].set_xlabel('Maximum Price')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Maximum Prices')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Define parameters
    option_params = LookbackOptionParams(
        S0=100.0,
        r=0.05,
        sigma=0.2,
        T=1.0,
        option_type=LookbackOptionType.FLOATING_STRIKE_CALL
    )
    
    sim_params = SimulationParams(
        n_paths=100000,
        n_steps=252,
        lambda_penalty=1000.0,
        seed=42,
        antithetic=True
    )
    
    # Price the option
    pricer = LookbackOptionPricer(option_params, sim_params)
    results = pricer.price_option()
    
    # Generate report and plots
    pricer.generate_report()
    pricer.plot_results()