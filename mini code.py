# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Fetch Historical Stock Data
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close']  # Return only closing prices

# Step 2: Prepare the Data for Training
def prepare_data(data, prediction_days):
    """
    Prepare data for supervised learning by shifting 'Close' prices.
    :param data: Pandas Series of stock prices.
    :param prediction_days: Number of days to predict ahead.
    :return: Features (X) and Labels (y).
    """
    df = pd.DataFrame(data)
    df['Prediction'] = df[df.columns[0]].shift(-prediction_days)  # Target variable

    X = np.array(df.drop(['Prediction'], axis=1))[:-prediction_days]
    y = np.array(df['Prediction'])[:-prediction_days]
    return X, y

# Step 3: Train Linear Regression Model
def train_model(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Step 4: Predict Future Prices
def predict_future(model, data, prediction_days):
    """Predict future prices based on the trained model."""
    future_data = np.array(data[-prediction_days:]).reshape(-1, 1)
    predictions = model.predict(future_data)
    return predictions

# Step 5: Main Function
def main():
    # User inputs
    ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    prediction_days = 30  # Predict the next 30 days

    # Fetch stock data
    print("\nFetching data...")
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    print("Data fetched successfully!")

    # Prepare data
    X, y = prepare_data(stock_data, prediction_days)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    print("\nTraining the model...")
    model = train_model(X_train, y_train)

    # Test the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error on Test Data: {mse:.2f}")

    # Predict future prices
    print("\nPredicting future prices...")
    future_prices = predict_future(model, stock_data, prediction_days)
    print(f"Predicted prices for the next {prediction_days} days:")
    print(future_prices)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data, label="Historical Prices")
    future_dates = pd.date_range(stock_data.index[-1], periods=prediction_days+1)[1:]
    plt.plot(future_dates, future_prices, label="Predicted Prices", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"{ticker} Stock Price Prediction")
    plt.legend()
    plt.show()

# Run the tool
if __name__ == "__main__":
    main()