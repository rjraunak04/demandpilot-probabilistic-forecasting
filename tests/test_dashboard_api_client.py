import json
from io import BytesIO
from unittest.mock import patch

import pytest

from dashboard.api_client import APIError, health, inventory_recommendation, request_json


class Response:
    def __init__(self, payload):
        self.body = BytesIO(json.dumps(payload).encode())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.body.read()


def test_health_client():
    with patch("dashboard.api_client.urlopen", return_value=Response({"status": "ok"})):
        assert health() == {"status": "ok"}


def test_inventory_client_posts_contract():
    with patch("dashboard.api_client.urlopen", return_value=Response({"reorder_quantity": 8})) as call:
        result = inventory_recommendation(10, 15, 20, 12)
        assert result["reorder_quantity"] == 8
        request = call.call_args.args[0]
        assert request.method == "POST"
        assert json.loads(request.data)["p90"] == 20


def test_api_connection_failure_is_dashboard_safe():
    with patch("dashboard.api_client.urlopen", side_effect=TimeoutError("timeout")):
        with pytest.raises(APIError, match="API unavailable"):
            request_json("GET", "/health")
