"""Timestral API client."""
import requests


def forecast(base_url, values, prediction_length):
    """Call the Timestral /forecast endpoint and return the parsed JSON payload."""
    response = requests.post(
        f"{base_url}/forecast",
        json={
            "values": values,
            "prediction_length": prediction_length,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()