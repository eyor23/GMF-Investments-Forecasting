# GMF Investment Forecasting Project as Financial Analyst
## Overview

This project focuses on forecasting Tesla (TSLA) stock prices using historical financial data. We employ time series analysis and machine learning techniques to develop predictive models. The project aims to provide insights into TSLA's stock behavior and demonstrate the effectiveness of different forecasting methods.

## Project Structure 

├── notebooks/
│   └── portfolio_analysis.ipynb # Jupyter Notebook containing the main Exploratory Data Analysis (EDA)
|   └── modeling.ipynb # Jupyter Notebook containing the modeling
├── scripts/
│   ├── prepro_eda.py         # Module for loading and cleaning data from YFinance and performing Exploratory Data Analysis (EDA)
├── .gitignore                 # Specifies intentionally untracked files that Git should ignore
├── README.md                  # Project documentation
└── requirements.txt           # List of Python packages required to run the project

## Data Source

* **YFinance:** Historical stock data for TSLA, Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY) is sourced from the YFinance Python library.
* **Time Range:** The data covers the period from January 1, 2015, to January 31, 2025.

## Project Workflow

### 1. Data Acquisition and Preprocessing

* Downloaded historical stock data for TSLA, BND, and SPY using `yfinance`.
* Cleaned and preprocessed the data, handling missing values and ensuring data integrity.
* Performed Exploratory Data Analysis (EDA) to understand the data's characteristics:
    * Visualized closing prices over time.
    * Analyzed daily percentage changes and volatility.
    * Calculated rolling means and standard deviations.
    * Performed outlier detection.
    * Decomposed time series into trend, seasonal, and residual components.
    * Calculated Value at Risk (VaR) and Sharpe Ratio.

### 2. Time Series Forecasting

* Developed time series forecasting models for TSLA stock prices.
* Implemented ARIMA and LSTM models.
* **ARIMA Model:**
    * Utilized `statsmodels.tsa.arima.model.ARIMA` for manual ARIMA parameter selection.
    * Analyzed ACF and PACF plots to determine ARIMA parameters.
    * Fitted the ARIMA model and generated forecasts.
* **LSTM Model:**
    * Scaled the data using `MinMaxScaler`.
    * Created time series sequences for LSTM input.
    * Built and trained an LSTM model using TensorFlow/Keras.
    * Generated forecasts and inverse scaled them.
* **Model Evaluation:**
    * Calculated Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) to evaluate model performance.
    * Visualized actual vs. predicted stock prices.

### 3. Model Evaluation and Results

* **ARIMA Model Performance:**
    * MAE: 89.91
    * RMSE: 111.08
    * MAPE: 34.61%
* **LSTM Model Performance:**
    * MAE: 7.37
    * RMSE: 9.99
    * MAPE: 3.16%
* **Key Findings:**
    * The LSTM model significantly outperformed the ARIMA model, demonstrating its ability to capture the complex and volatile patterns of TSLA's stock.
    * The ARIMA model's linear approach was inadequate for modeling the non-linear behavior of TSLA's stock.
    * The LSTM model showed a high level of accuracy, with an average prediction error of 3.16%.


## Libraries Used

* `yfinance`
* `pandas`
* `numpy`
* `scikit-learn`
* `statsmodels`
* `tensorflow` and `keras`
* `matplotlib`
* `seaborn`

## Instructions for Running the Code

1.  Clone the repository to your local machine.
2.  Create a virtual environment:
    * `python -m venv venv` (or `conda create -n venv python=3.x`)
    * Activate the virtual environment: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows).
3.  Install the required libraries: `pip install -r requirements.txt`
4.  Navigate to the `notebooks/` directory and run the Jupyter Notebook `portfolio_analysis.ipynb` and `modeling.ipynb`to execute the analysis and modeling.

## Future Improvements

* Explore more advanced time series models, such as SARIMA or Prophet.
* Incorporate external factors and features into the models.
* Optimize LSTM hyperparameters for better performance.
* Implement walk-forward validation for more robust model evaluation.
* Develop a user-friendly interface for generating forecasts.
* Deploy the model as a web service.

## Author

Eyor Getachew Mamo (eyor.gech@gmail.com)

## License

MIT License

## Acknowledgments

- `pandas` for data manipulation
- `matplotlib` and `seaborn` for visualization

