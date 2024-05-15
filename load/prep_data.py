import csv
import os
import pickle

import pandas as pd
from prefect import flow, task
from sklearn.preprocessing import StandardScaler


@task
def read_financial_data(filename: str):
    # Read financial data from CSV file
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data


@task
def preprocess_labels_financial_data(data):
    # Convert data types and preprocess features
    processed_data = []
    for row in data:
        processed_row = {}
        processed_row["Volume"] = int(float(row.pop("v")))
        processed_row["VWAP"] = float(row.pop("vw"))
        processed_row["Open"] = float(row.pop("o"))
        processed_row["Close"] = float(row.pop("c"))
        processed_row["High"] = float(row.pop("h"))
        processed_row["Low"] = float(row.pop("l"))
        processed_row["Timestamp"] = int(row.pop("t"))
        processed_row["Number_of_Trades"] = int(row.pop("n"))
        processed_row["Date"] = pd.to_datetime(row.pop("date")).strftime("%Y-%m-%d")
        processed_data.append(processed_row)
    return processed_data


@task
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


@task
def read_csv_stock_data(filename: str):
    df = pd.read_csv(filename)
    return df


@task
def preprocess_stock_data(df: pd.DataFrame):
    # Assuming 'Close' is the target label
    target_label = "Close"

    # Convert 'Date' column to datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Remove or handle datetime column before standardizing
    X = df.drop(columns=["Date", target_label])
    y = df[target_label]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


@task
def save_preprocessed_data(data, filename):
    # Save preprocessed data to a CSV file
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


@flow
def prep_data_flow(data_path: str, dest_path: str, model_path: str):
    # Read financial data
    financial_data = read_financial_data(data_path)
    # Preprocess financial data
    preprocessed_data = preprocess_labels_financial_data(financial_data)
    # Save preprocessed data
    save_preprocessed_data(preprocessed_data, dest_path)
    df = read_csv_stock_data(data_path)
    # Preprocess stock data
    X, y = preprocess_stock_data(df)
    # Create model_path folder unless it already exists
    os.makedirs(model_path, exist_ok=True)

    # Save preprocessed data
    dump_pickle(X, os.path.join(model_path, "X.pkl"))
    dump_pickle(y, os.path.join(model_path, "y.pkl"))
