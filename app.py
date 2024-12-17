from flask import Flask, request, render_template
from stock_predictor import fetch_stock_data, prepare_data, train_model, predict_future
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form["ticker"].upper()
        start_date = "2020-01-01"
        end_date = "2023-01-01"
        prediction_days = 30

        # Fetch and process stock data
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        X, y = prepare_data(stock_data, prediction_days)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model and calculate MSE
        model = train_model(X_train, y_train)
        mse = mean_squared_error(y_test, model.predict(X_test))

        # Predict future prices
        future_prices = predict_future(model, stock_data, prediction_days)

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index, stock_data, label="Historical Prices")
        future_dates = pd.date_range(stock_data.index[-1], periods=prediction_days+1)[1:]
        plt.plot(future_dates, future_prices, label="Predicted Prices", linestyle="--")
        plt.legend()
        image_path = os.path.join("static", "plot.png")
        plt.savefig(image_path)
        plt.close()

        return render_template("index.html", mse=round(mse, 2), image_url=image_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
