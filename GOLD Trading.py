import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta

# Define stock ticker and fetch data
tickerSymbol = 'GLD'
tickerData = yf.Ticker(tickerSymbol)

# Download historical stock data
tickerDf = tickerData.history(interval='1d', start='2015-01-01', end='2023-12-30')
priceData = tickerDf['Open']

# Display first few rows of data
print(priceData.head())

# Helper function to calculate cumulative percent change
def check_cumulative_percent_change(price_data, buy_date, potential_sell_date):
    """
    This helper function checks the cumulative percent change
    between a buying and potential selling day to see if it yields overall growth.
    """
    pct_change = price_data.pct_change()[1:]
    sub_series = 1 + pct_change[buy_date: potential_sell_date]
    return sub_series.product() > 1

# Function to identify buying and selling days
def get_buying_selling_days(price_data, b, s):
    pct_change = price_data.pct_change()[1:]

    # Create conditions for buying and selling
    buying_days = pct_change.rolling(b).apply(lambda x: (x > 0).all(), raw=True)
    selling_days = pct_change.rolling(s).apply(lambda x: (x < 0).all(), raw=True)

    return {'buying_days': buying_days, 'potential_selling_days': selling_days}

# Get buying and selling days
info_dict = get_buying_selling_days(priceData, 4, 1)
buying_days = info_dict['buying_days']
potential_selling_days = info_dict['potential_selling_days']

# Create dataframe to store information
df_stocks = pd.DataFrame(index=buying_days.index)
df_stocks['buying_day'] = (buying_days == 1)
df_stocks['potential_selling_day'] = (potential_selling_days == 1)
df_stocks['price'] = priceData

# Filter dataframe to keep only relevant days
df_stocks = df_stocks[(df_stocks.buying_day | df_stocks.potential_selling_day)]

# Simulate investment strategy
def get_investing_result(df_stocks, starting_funds, verbose=False):
    price_data = df_stocks.price
    holding = False
    current_funds = starting_funds
    current_shares = 0
    last_buy_date = None
    events_list = []

    for date, data in df_stocks.iterrows():
        if not holding and data.buying_day:
            num_shares_to_buy = int(current_funds / data.price)
            current_shares += num_shares_to_buy
            current_funds -= num_shares_to_buy * data.price
            last_buy_date = date
            events_list.append(('b', date))
            holding = True
            if verbose:
                print(f'Bought {num_shares_to_buy} shares at ${data.price:.2f} on {date.date()}')

        elif holding and data.potential_selling_day:
            if check_cumulative_percent_change(price_data, last_buy_date, date):
                current_funds += current_shares * data.price
                events_list.append(('s', date))
                holding = False
                current_shares = 0
                if verbose:
                    print(f'Sold at ${data.price:.2f} on {date.date()}')

    final_stock_price = price_data[-1]
    final_value = current_funds + final_stock_price * current_shares
    return round((final_value - starting_funds) / starting_funds, 2), events_list

# Simulate the investment result
percent_change, events_list = get_investing_result(df_stocks, 10000, verbose=True)
print(f"Total percent change: {percent_change * 100:.2f}%")

# Plot the data with buy and sell events
plt.figure(figsize=(10, 4))
plt.plot(priceData, label='Open Price')

y_lims = (int(priceData.min() * 0.95), int(priceData.max() * 1.05))
shaded_y_lims = (int(priceData.min() * 0.5), int(priceData.max() * 1.5))

for idx, event in enumerate(events_list):
    color = 'red' if event[0] == 'b' else 'blue'
    plt.axvline(event[1], color=color, linestyle='--', alpha=0.4)
    if event[0] == 's':
        plt.fill_betweenx(range(*shaded_y_lims), event[1], events_list[idx-1][1], color='k', alpha=0.1)

plt.title(f"{tickerSymbol} Price Data with Buy/Sell Events", fontsize=20)
plt.ylim(*y_lims)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.legend()
plt.show()
