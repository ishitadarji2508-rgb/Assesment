import requests
from langchain_core.tools import tool


@tool
def get_delivery_estimate(
    distance_km: float,
    num_items: int,
    rain_flag: int
) -> str:
    """Returns the estimated delivery time from the Flask prediction API."""
    api_url = "http://127.0.0.1:5000/predict"

    request_data = {
        "distance_km": distance_km,
        "num_items": num_items,
        "rain_flag": rain_flag
    }

    try:

        response = requests.post(
            api_url,
            json=request_data,
            timeout=5
        )

        if response.status_code == 200:

            result = response.json()

            estimated_time = result.get(
                "predicted_delivery_time_min"
            )

            return f"Estimated delivery time: {estimated_time} minutes."

        return f"API Error: {response.text}"

    except Exception as error:

        return f"Unexpected Error: {error}"