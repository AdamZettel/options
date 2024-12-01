import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Black-Scholes price function
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Bisection Solver
def bisection_solver(f, low, high, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        mid = (low + high) / 2
        value = f(mid)
        if abs(value) < tol:
            return mid
        elif value > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2

# General implied volatility function
def implied_volatility(S, K, T, r, market_price, option_type, solver, **kwargs):
    def f(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - market_price
    return solver(f, 1e-9, 5.0, **kwargs)

# Parameters
S = 100  # Underlying price
T = 0.5  # Time to maturity (fixed for smile)
r = 0.05 # Risk-free rate
option_type = 'call'

# Strike prices
strike_prices = np.linspace(80, 120, 15)

# Simulate market prices with a smile-shaped volatility
def generate_market_price(S, K, T, r, base_volatility, smile_factor=0.2):
    # Smile factor exaggerates the volatility for strikes far from ATM
    distance = abs(K - S) / S  # Relative distance from ATM
    implied_vol = base_volatility * (1 + smile_factor * distance**2)
    return black_scholes(S, K, T, r, implied_vol, option_type)

base_volatility = 0.2  # ATM volatility
market_prices = [generate_market_price(S, K, T, r, base_volatility) for K in strike_prices]

# Compute implied volatilities for the smile
implied_vols = [
    implied_volatility(S, K, T, r, market_price, option_type, bisection_solver)
    for K, market_price in zip(strike_prices, market_prices)
]

# Plot the volatility smile
plt.figure(figsize=(8, 6))
plt.plot(strike_prices, implied_vols, marker='o', label="Volatility Smile")
plt.axvline(S, color='gray', linestyle='--', label="ATM Strike (S = 100)")
plt.title("Volatility Smile Example")
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.grid(True)
plt.legend()
plt.show()
