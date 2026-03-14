# -------------------------------
# *.Library Imports 
# -------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# -------------------------------
# 1.Data Loading and preprocessing
# -------------------------------
def preprocess_stock_data(stock_filename, factors_df):
    stock_df = pd.read_csv(stock_filename)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%Y-%m')
    merged_df = pd.merge(stock_df, factors_df, on='Date', how='inner')
    merged_df['Excess Return'] = merged_df['Monthly Return'] - merged_df['RF']/100
    merged_df = merged_df.rename(columns={
        'Mkt-RF': 'x1',
        'SMB': 'x2',
        'HML': 'x3'
    })
    return merged_df[['Date', 'Excess Return', 'x1', 'x2', 'x3']]

factors_df = pd.read_csv("~/Desktop/Fama_French_3_Factors_Monthly.csv")
factors_df['Date'] = pd.to_datetime(factors_df['Date'], format='%Y-%m')
amzn_df = preprocess_stock_data("~/Desktop/AMZN.csv", factors_df)
aapl_df = preprocess_stock_data("~/Desktop/AAPL.csv", factors_df)
m_df    = preprocess_stock_data("~/Desktop/M.csv", factors_df)


# -------------------------------
# 2.Dataset splitting
# -------------------------------
def train_test_split_time(df, test_ratio=0.2, date_col='Date'):
    df = df.sort_values(date_col).reset_index(drop=True)
    split_index = int(len(df) * (1 - test_ratio))
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = np.asarray(train_df[['x1', 'x2', 'x3']])
    y_train = np.asarray(train_df['Excess Return'])
    X_test = np.asarray(test_df[['x1', 'x2', 'x3']])
    y_test = np.asarray(test_df['Excess Return'])

    return X_train, y_train, X_test, y_test

amzn_X_train, amzn_y_train, amzn_X_test, amzn_y_test = train_test_split_time(amzn_df)
aapl_X_train, aapl_y_train, aapl_X_test, aapl_y_test = train_test_split_time(aapl_df)
m_X_train, m_y_train, m_X_test, m_y_test = train_test_split_time(m_df)


# -------------------------------
# 3.Data standardization
# -------------------------------
def standardize_train_test(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s

amzn_X_train_s, amzn_X_test_s = standardize_train_test(amzn_X_train, amzn_X_test)
aapl_X_train_s, aapl_X_test_s = standardize_train_test(aapl_X_train, aapl_X_test)
m_X_train_s, m_X_test_s = standardize_train_test(m_X_train, m_X_test)