import csv
import json
import os
from polygon import RESTClient
from prefect import task, flow
import datetime

API_KEY = '1y_afkcFULM8pHgpKKcybbzAhcHRxkA6'
DATA_PATH = "data"

@task
def fetch_polygon_data(start_date, end_date, frequency):
    client = RESTClient(API_KEY)
    if frequency == 'month':
        frequency_str = 'month'
    elif frequency =='minute':
        frequency_str = 'minute'
    else:
        frequency_str = 'day'
    aggs = client.get_aggs('AAPL', 1, frequency_str, start_date, end_date, raw=True)
    data = json.loads(aggs.data)
    return data

@task
def save_data(data, filename):
    if "results" in data:  # Check if the data has 'results' key
        data = data["results"]  # Extract the results
        fields = list(data[0].keys()) + ['date']  # Add 'date' field
        with open(filename, 'w', newline='') as file:  # Explicitly set the mode to 'w'
            csv_writer = csv.DictWriter(file, fields)
            csv_writer.writeheader()
            for row in data:
                # Convert timestamp to date
                row['date'] = datetime.datetime.fromtimestamp(row['t'] / 1000, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                csv_writer.writerow(row)
    else:
        print("No results found in data.")

@flow
## API allows data to be used from your date to 2 years ago, so assume minimum 2022-07-01 to 2024-05-05 max.
def collect_flow(start_date="2024-05-11", end_date="2024-05-12", frequency="minute"):
    os.makedirs(DATA_PATH, exist_ok=True)
    polygon_data = fetch_polygon_data(start_date, end_date, frequency)
    save_data_task = save_data(polygon_data, os.path.join(DATA_PATH, "apple_stock_data.csv"))
    return save_data_task

if __name__ == "__main__":
    collect_flow()
