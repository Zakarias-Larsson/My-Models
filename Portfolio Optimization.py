import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

# Define your portfolio
Portfolio = ['SGOL', 'IWD', 'VUG', 'INDA', 'VDC']

# Get historical data from yfinance
tickerData = yf.Tickers(Portfolio)
tickerDf = tickerData.history(interval='1d', start='2010-01-01', end='2023-12-30')
close_prices = tickerDf['Close']

# Calculate the log of returns
log_return = np.log(1 + close_prices.pct_change()).dropna()
number_of_symbols = len(Portfolio)

# Calculate the covariance matrix and correlation matrix
cov_matrix = log_return.cov()
corr_matrix = log_return.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Asset Returns')
plt.show()

# Plot the covariance matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Covariance Matrix of Asset Returns')
plt.show()

# Monte Carlo Simulation parameters
num_of_portfolios = 10000
risk_free_rate = 0.01

# Arrays to store simulation results
all_weights = np.zeros((num_of_portfolios, number_of_symbols))
ret_arr = np.zeros(num_of_portfolios)
vol_arr = np.zeros(num_of_portfolios)
sharpe_arr = np.zeros(num_of_portfolios)

# Simulation: Generate random portfolios
for ind in range(num_of_portfolios):
    weights = np.random.random(number_of_symbols)
    weights /= np.sum(weights)  # Normalize weights to sum to 1

    # Expected portfolio return and volatility
    port_returns = np.dot(weights, log_return.mean()) * 52
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(log_return.cov() * 52, weights)))

    # Sharpe ratio calculation
    sharpe_ratio = (port_returns - risk_free_rate) / port_volatility

    # Store the values
    all_weights[ind, :] = weights
    ret_arr[ind] = port_returns
    vol_arr[ind] = port_volatility
    sharpe_arr[ind] = sharpe_ratio

# Create a DataFrame to store all simulation results
simulations_df = pd.DataFrame({
    'Returns': ret_arr,
    'Volatility': vol_arr,
    'Sharpe Ratio': sharpe_arr
})

# Add the portfolio weights as a column
for i, symbol in enumerate(Portfolio):
    simulations_df[symbol + ' Weight'] = all_weights[:, i]

# Print out a sample of the simulation results
print('')
print('=' * 80)
print('SIMULATIONS RESULT:')
print('-' * 80)
print(simulations_df.head())
print('-' * 80)

# Plot the results
plt.scatter(simulations_df['Volatility'], simulations_df['Returns'], c=simulations_df['Sharpe Ratio'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Monte Carlo Simulation: Portfolio Optimization')
plt.show()

# Find the portfolio with the maximum Sharpe Ratio
max_sharpe_idx = sharpe_arr.argmax()
optimal_weights = all_weights[max_sharpe_idx]

# Print the portfolio with the highest Sharpe Ratio
print('')
print('=' * 80)
print('OPTIMAL PORTFOLIO (MAX SHARPE RATIO):')
print('-' * 80)
print(f"Max Sharpe Ratio: {sharpe_arr[max_sharpe_idx]}")
print(f"Expected Return: {ret_arr[max_sharpe_idx]}")
print(f"Volatility: {vol_arr[max_sharpe_idx]}")
print('Optimal Weights:', dict(zip(Portfolio, optimal_weights)))
print('-' * 80)
