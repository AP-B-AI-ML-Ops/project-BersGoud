import pandas as pd
import csv
import numpy as np

from prefect import flow, task
from sklearn.preprocessing import StandardScaler

@task
def read_financial_data(filename: str):
    # Read financial data from CSV file
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

@task
def preprocess_financial_data(data):
    # Convert data types and preprocess features
    processed_data = []
    for row in data:
        processed_row = {}
        processed_row['Volume'] = int(float(row.pop('v')))
        processed_row['VWAP'] = float(row.pop('vw'))
        processed_row['Open'] = float(row.pop('o'))
        processed_row['Close'] = float(row.pop('c'))
        processed_row['High'] = float(row.pop('h'))
        processed_row['Low'] = float(row.pop('l'))
        processed_row['Timestamp'] = int(row.pop('t'))
        processed_row['Number_of_Trades'] = int(row.pop('n'))
        processed_row['Date'] = pd.to_datetime(row.pop('date')).strftime('%Y-%m-%d')
        processed_data.append(processed_row)
    return processed_data

@task
def save_preprocessed_data(data, filename):
    # Save preprocessed data to a CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

@flow
def prep_data(data_path: str, dest_path: str):
    # Read financial data
    financial_data = read_financial_data(data_path)
    # Preprocess financial data
    preprocessed_data = preprocess_financial_data(financial_data)
    # Save preprocessed data
    save_preprocessed_data(preprocessed_data, dest_path)

if __name__ == "__main__":
    prep_data(data_path="data/apple_stock_data.csv", dest_path="data/apple_stock_data.csv")
