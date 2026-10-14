"""Small standard-library client for the DemandPilot serving API."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_URL = os.getenv("DEMANDPILOT_API_URL", "http://localhost:8000").rstrip("/")


class APIError(RuntimeError):
    """Dashboard-safe API failure."""


def request_json(method, path, payload=None, params=None, timeout=5):
    url = f"{API_URL}{path}"
    if params:
        url = f"{url}?{urlencode(params)}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method=method)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise APIError(f"API returned {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise APIError(f"API unavailable: {exc}") from exc


def health():
    return request_json("GET", "/health")


def get_forecast(sku, forecast_date):
    return request_json("GET", f"/forecasts/{sku}", params={"forecast_date": forecast_date})


def inventory_recommendation(p10, p50, p90, on_hand, minimum_order_quantity=0):
    return request_json(
        "POST",
        "/inventory/recommendation",
        payload={
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "on_hand": on_hand,
            "minimum_order_quantity": minimum_order_quantity,
        },
    )
