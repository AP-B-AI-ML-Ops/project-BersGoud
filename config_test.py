# testen van de Polygon.io API-sleutel voor het verkrijgen van gegevens over de Apple beurs (AAPL)

from polygon import RESTClient
import json
from typing import cast
from urllib3 import HTTPResponse

API_KEY = '1y_afkcFULM8pHgpKKcybbzAhcHRxkA6'
client = RESTClient(API_KEY)

aggs = cast(
    HTTPResponse,
    client.get_aggs(
        'AAPL',
        1,
        'day',
        '2022-05-05',
        '2024-05-05',
        raw = True
    ),
)

data = json.loads(aggs.data)
print(data)