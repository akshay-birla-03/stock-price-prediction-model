
# Stock Price Prediction with LSTM and ARIMA

## Overview

This project combines the strengths of **Long Short-Term Memory (LSTM)** neural networks and **ARIMA** statistical models to predict stock prices. By leveraging historical stock data, the models aim to provide accurate and insightful predictions for informed decision-making.

---

## Features

- **Data Source**: Automatically fetches historical stock data from Yahoo Finance using the `yfinance` library.
- **LSTM**: Uses deep learning for time series forecasting, capturing long-term dependencies.
- **ARIMA**: Employs statistical methods to model and predict stock prices based on historical patterns.
- **Hybrid Predictions**: Combines predictions from both LSTM and ARIMA for robust results.
- **Performance Metrics**: Evaluates models using Mean Squared Error (MSE) and R-squared values.
- **Visualization**: Plots actual prices alongside model predictions for clear performance insights.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Open the script `stock_price_prediction_lstm_arima.py`.
2. Modify the `ticker` variable to the stock symbol of your choice. For example:
   ```python
   ticker = 'TATAELXSI.NS'
   ```
3. Run the script:
   ```bash
   python stock_price_prediction_lstm_arima.py
   ```
4. View the output metrics in the console and the prediction graphs.

---

## Example Output

The graph below demonstrates the model's ability to predict stock prices accurately:

1. **Blue Line**: Actual Stock Prices
2. **Orange Line**: LSTM Predictions
3. **Green Line**: ARIMA Predictions
4. **Red Line**: Combined Predictions

![Graph Placeholder](placeholder_for_actual_graph.png)

---

## File Structure

```
.
├── stock_price_prediction_lstm_arima.py  # Main script for the project
├── stock_price_prediction_documentation.md  # Detailed project documentation
├── requirements.txt  # List of required Python libraries
└── README.md  # This file
```

---

## Dependencies

- `yfinance`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `keras`
- `statsmodels`

---

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you encounter any problems.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contact

For any inquiries or suggestions, please reach out to [Your Name](mailto:your_email@example.com).
