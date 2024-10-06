import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from scipy.stats import pearsonr

# Define the ticker symbol
tickerSymbol = 'SPY'
tickerData = yf.Ticker(tickerSymbol)

# Fetch historical data
tickerDf = tickerData.history(interval='1d', start='2010-1-1', end='2023-12-30')
close_prices = tickerDf['Close']
returns = 100 * close_prices.pct_change().dropna()

Out_Sample_Data = tickerData.history(interval='1d', start='2024-1-1', end='2024-10-04')
close_prices_out = Out_Sample_Data['Close']  # Use Out_Sample_Data instead of tickerDf
returns_out = 100 * close_prices_out.pct_change().dropna()

# Plot the returns
plt.figure(figsize=(10, 4))
plt.plot(returns)
plt.ylabel('Percentage Return', fontsize=16)
plt.title(f'{tickerSymbol} Returns', fontsize=20)
plt.show()

# PACF of squared returns
plot_pacf(returns**2)
plt.show()


# Fit the GARCH(1,1) model
garch_model = arch_model(returns, vol='Garch', p=2, q=1, mean='zero',dist='skewt')
garch_fitted = garch_model.fit(disp="off")
print(garch_fitted.summary())

# Perform rolling window forecast for model evaluation
rolling_predictions = []
test_size = 250 * 10  # Last 2 years for testing
train_size = len(returns) - test_size

for i in range(test_size):
    train = returns[:train_size+i]
    model = arch_model(train, vol='Garch', p=2, q=1, mean='Zero', dist='skewt')
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

# Convert rolling predictions to Pandas Series
rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-test_size:])

# Calculate actual volatility (using absolute returns as a proxy)
actual_volatility = returns[-test_size:].abs()

# Calculate various accuracy metrics
rmse = sqrt(mean_squared_error(actual_volatility, rolling_predictions))
mae = mean_absolute_error(actual_volatility, rolling_predictions)
correlation, _ = pearsonr(actual_volatility, rolling_predictions)

print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Correlation between actual and predicted: {correlation:.4f}")

# Plot in-sample rolling forecast vs actual
plt.figure(figsize=(12, 6))
plt.plot(actual_volatility, label='Actual Volatility', color='blue', alpha=0.5)
plt.plot(rolling_predictions, label='Predicted Volatility', color='red', alpha=0.5)
plt.title('In-sample Volatility Prediction - Rolling Forecast', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend(fontsize=12)
plt.show()

# Scatter plot of predicted vs actual volatility
plt.figure(figsize=(10, 10))
plt.scatter(actual_volatility, rolling_predictions, alpha=0.5)
plt.plot([actual_volatility.min(), actual_volatility.max()], [actual_volatility.min(), actual_volatility.max()], 'r--', lw=2)
plt.xlabel('Actual Volatility')
plt.ylabel('Predicted Volatility')
plt.title('Predicted vs Actual Volatility')
plt.show()

# Forecast for a future period
forecast_horizon = 150
forecast = garch_fitted.forecast(horizon=forecast_horizon)

# Extract the forecasted volatility
forecasted_volatility = np.sqrt(forecast.variance.values[-1, :])

# Create a DataFrame with the forecasted volatility
forecast_dates = pd.date_range(start=returns.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
forecast_df = pd.DataFrame(forecasted_volatility, index=forecast_dates, columns=['Forecasted_Volatility'])

# Plot the forecasted volatility
plt.figure(figsize=(12, 6))
plt.plot(forecast_df['Forecasted_Volatility'])
plt.title(f'Forecasted Volatility (Next {forecast_horizon} days)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.show()

# Calculate and print average forecasted volatility
avg_forecasted_volatility = forecast_df['Forecasted_Volatility'].mean()
print(f"Average forecasted volatility: {avg_forecasted_volatility:.4f}")

# Print confidence intervals for the forecast
lower_bound = np.percentile(forecasted_volatility, 2.5)
upper_bound = np.percentile(forecasted_volatility, 97.5)
print(f"95% Confidence Interval for forecasted volatility: [{lower_bound:.4f}, {upper_bound:.4f}]")

# Residual analysis
residuals = actual_volatility - rolling_predictions
plt.figure(figsize=(12, 6))
plt.plot(residuals.index, residuals)
plt.title('Residuals of Volatility Forecast')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.show()

# QQ plot of residuals
from scipy import stats
plt.figure(figsize=(10, 10))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q plot of residuals")
plt.show()