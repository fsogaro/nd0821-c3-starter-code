import requests
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heroku app test")
    parser.add_argument(
        "--test_env", type=str, default="heroku"
    )
    args = parser.parse_args()

    data = {
        "age": 50,
    }
    if args.test_env == "local":
        post_url = "http://0.0.0.0:5000/predict"
    else:
        post_url = 'https://fast-basin-20342.herokuapp.com/predict'

    print(post_url)
    print(json.dumps(data))
    response = requests.post(post_url, json=data)
    print(response)
    print(response.json())

    logging.info(response)
