import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import datetime

# ==========================================
# 1. DATA ACQUISITION & PRE-PROCESSING
# ==========================================

def get_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance.
    """
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Ensure we only keep the 'Close' column for simplicity in this example
    # The Base Paper suggests using Open, High, Low, Vol, but for the specific 
    # ARIMA vs ANN comparison, Close price is the standard target.
    df = data[['Close']].copy()
    
    # Filling missing values if any (Data Cleaning)
    df = df.fillna(method='ffill')
    
    return df

def add_technical_indicators(df):
    """
    Adds Moving Averages (MA) and RSI as features.
    (Based on the methodology from Venkatarathnam et al. [cite: 1343])
    """
    # 1. Moving Averages (Technical Indicator)
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # 2. Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values created by rolling windows
    df.dropna(inplace=True)
    return df

# ==========================================
# 2. MODEL 1: ARIMA (Statistical)
# ==========================================

def train_predict_arima(train_data, test_data):
    """
    Trains an ARIMA model and forecasts future prices.
    ARIMA is linear and good for short-term trends.
    """
    print("\nRunning ARIMA Model...")
    history = [x for x in train_data]
    predictions = []
    
    # Rolling forecast: Train on history, predict next step, add real value to history
    for t in range(len(test_data)):
        # Order (p,d,q) = (5,1,0) is a standard starting point for stocks
        model = ARIMA(history, order=(5,1,0)) 
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_data[t]
        history.append(obs) # Add the actual observation to history for next loop
        
    return predictions

# ==========================================
# 3. MODEL 2: ANN (Artificial Neural Network)
# ==========================================

def create_ann_model(input_dim):
    """
    Builds a standard Feed-Forward Neural Network (MLP).
    ANNs identify complex, non-linear patterns.
    """
    model = Sequential()
    # Input Layer + 1st Hidden Layer (50 Neurons, ReLU activation)
    model.add(Dense(50, input_dim=input_dim, activation='relu'))
    # Dropout layer to prevent overfitting (Coding Skill requirement)
    model.add(Dropout(0.2))
    # 2nd Hidden Layer
    model.add(Dense(25, activation='relu'))
    # Output Layer (1 Neuron for predicted price)
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_predict_ann(df):
    print("\nRunning ANN Model...")
    
    # We use 'Close' price as the target. 
    # Ideally, we use technical indicators as inputs (X), but for direct comparison
    # with ARIMA (which is univariate), we will use Look-Back window approach here.
    
    data = df['Close'].values.reshape(-1, 1)
    
    # Scaling Data (0,1) - Critical for Neural Networks 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences (e.g., use past 60 days to predict day 61)
    prediction_days = 60
    x_train, y_train = [], []
    
    # Splitting 80% for training
    train_len = int(len(scaled_data) * 0.8)
    train_set = scaled_data[0:train_len]
    
    for i in range(prediction_days, len(train_set)):
        x_train.append(train_set[i-prediction_days:i, 0])
        y_train.append(train_set[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # Build and Train Model
    model = create_ann_model(input_dim=x_train.shape[1])
    model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=1)
    
    # Preparing Test Data
    test_set = scaled_data[train_len - prediction_days:]
    x_test = []
    
    for i in range(prediction_days, len(test_set)):
        x_test.append(test_set[i-prediction_days:i, 0])
        
    x_test = np.array(x_test)
    
    # Predicting
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions) # Scale back to actual price
    
    return predictions, train_len

# ==========================================
# 4. MAIN EXECUTION & VISUALIZATION
# ==========================================

if __name__ == "__main__":
    # 1. Set Parameters
    STOCK_TICKER = 'AAPL' # Example: Apple Inc. (Can use 'ADANIPORTS.NS' for Indian market)
    START_DATE = '2020-01-01'
    END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 2. Get Data
    df = get_stock_data(STOCK_TICKER, START_DATE, END_DATE)
    df = add_technical_indicators(df) # Adds RSI, MA [cite: 1284]
    
    # 3. Prepare Data Splits for ARIMA
    train_size = int(len(df) * 0.8)
    train_data, test_data = df['Close'][0:train_size].values, df['Close'][train_size:].values
    
    # 4. Run Models
    # ARIMA Predictions
    arima_preds = train_predict_arima(train_data, test_data)
    
    # ANN Predictions
    ann_preds, train_len = train_predict_ann(df)
    
    # 5. Evaluation (RMSE)
    # We trim ANN predictions to match test_data length if necessary
    rmse_arima = np.sqrt(mean_squared_error(test_data, arima_preds))
    rmse_ann = np.sqrt(mean_squared_error(test_data, ann_preds))
    
    print(f"\n--- Results ---")
    print(f"ARIMA RMSE: {rmse_arima}")
    print(f"ANN RMSE: {rmse_ann}")
    
    # 6. Plotting Results 

[Image of Data Flow Diagram]
 (Conceptual connection)
    plt.figure(figsize=(14, 7))
    
    # Plot Actual Prices
    # We align the x-axis to the dates of the test data
    test_dates = df.index[train_size:]
    
    plt.plot(test_dates, test_data, label='Actual Price', color='black')
    plt.plot(test_dates, arima_preds, label='ARIMA Prediction (Statistical)', color='blue', linestyle='--')
    plt.plot(test_dates, ann_preds, label='ANN Prediction (AI/ML)', color='red')
    
    plt.title(f'Stock Price Prediction: ARIMA vs ANN ({STOCK_TICKER})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
