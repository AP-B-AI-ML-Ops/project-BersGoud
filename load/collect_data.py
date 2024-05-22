import csv
import datetime
import json
import os

from polygon import RESTClient
from prefect import flow, task

API_KEY = "1y_afkcFULM8pHgpKKcybbzAhcHRxkA6"
DATA_PATH = "data"


@task(name="fetch-polygon-data-collect_data")
def fetch_polygon_data(start_date, end_date):
    client = RESTClient(API_KEY)
    aggs = client.get_aggs("AAPL", 1, "day", start_date, end_date, raw=True)
    data = json.loads(aggs.data)
    return data


@task(name="save-data-collect_data")
def save_data(data, filename):
    if "results" in data:
        data = data["results"]
        fields = list(data[0].keys()) + ["date"]
        with open(filename, "w", newline="") as file:
            csv_writer = csv.DictWriter(file, fields)
            csv_writer.writeheader()
            for row in data:
                # Convert timestamp to date
                row["date"] = datetime.datetime.fromtimestamp(
                    row["t"] / 1000, datetime.timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.writerow(row)
    else:
        print("No results found in data.")


@task(name="generate-query-params-collect_data")
def generate_query_params():
    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=20 * 30)
    return start_date, end_date


@flow(name="collect-flow-collect_data")
def collect_flow():
    os.makedirs(DATA_PATH, exist_ok=True)
    start_date, end_date = generate_query_params()
    polygon_data = fetch_polygon_data(start_date, end_date)
    save_data_task = save_data(
        polygon_data, os.path.join(DATA_PATH, "apple_stock_data.csv")
    )
    return save_data_task
