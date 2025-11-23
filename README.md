Empirical Analysis of AI-Driven Intraday Stock Forecasting
📌 Project Overview
This repository contains the source code and documentation for a research project titled "An Empirical Study on an AI-Driven System for Intraday Stock Market Prediction." The primary objective of this system is to mitigate the risks associated with manual day trading—such as emotional interference and execution latency—by automating the analysis process. The project empirically contrasts two distinct forecasting methodologies: ARIMA, a classical statistical time-series model, and Artificial Neural Networks (ANN), a non-linear machine learning approach. By leveraging historical OHLCV (Open, High, Low, Close, Volume) data alongside technical indicators, this system aims to identify high-probability entry and exit points for financial assets.

🚀 Key Features
Automated Data Pipeline: Real-time fetching of market data using the yfinance API.

Hybrid Analysis: Implementation of both statistical (ARIMA) and Deep Learning (ANN/MLP) models to handle linear and non-linear market behaviors.

Feature Engineering: Automatic calculation of key technical indicators, specifically Moving Averages (MA) and the Relative Strength Index (RSI).

Comparative Visualization: Side-by-side graphical plotting of model predictions against actual market prices to empirically validate performance.

Error Metrics: Calculation of Root Mean Square Error (RMSE) to quantify prediction accuracy.

🛠️ Tech Stack
Language: Python 3.x

Machine Learning: TensorFlow (Keras), Scikit-learn

Statistics: Statsmodels (ARIMA)

Data Manipulation: Pandas, NumPy

Data Source: Yahoo Finance API

⚙️ Installation & Usage
Clone the Repository

Bash
git clone https://github.com/[YourUsername]/AI-Intraday-Stock-Predictor.git
cd AI-Intraday-Stock-Predictor
Install Dependencies

Bash
pip install -r requirements.txt
Run the Predictor

Bash
python main.py
📊 Methodology
The system architecture follows a strict data mining workflow:

Data Acquisition: Raw stock data is retrieved for a specified ticker (e.g., AAPL, ADANIPORTS.NS).

Pre-processing: Data is cleaned, and features (RSI, MA_10, MA_50) are generated. The dataset is normalized to a 0-1 range to ensure convergence during neural network training.

Model Execution:

ARIMA (5,1,0): processes the data as a linear time series, predicting the next step based on residual errors.

ANN (Feed-Forward): A multi-layer perceptron with Dropout regularization learns complex patterns from historical windows.

Evaluation: Both models are back-tested on unseen data (the most recent 20%), and their RMSE scores are compared.

📚 References & Academic Basis
This project is grounded in the methodologies proposed by the following key research:

Base Methodology: Venkatarathnam, N., et al. (2024). "An Empirical Study on Implementation of AI & ML in Stock Market Prediction." 

Comparative Analysis: Sizan, M. M. H., et al. (2023). "AI-Enhanced Stock Market Prediction: Evaluating Machine Learning Models..." 

Trend Analysis: Sarisa, M., et al. (2024). "Stock Market Prediction Through AI: Analyzing Market Trends With Big Data Integration." 

Systematic Review: Chopra, R. & Sharma, G. D. (2021). "Application of Artificial Intelligence in Stock Market Forecasting..." 

📜 License
Distributed under the MIT License. See LICENSE for more information.
