
# Write tests using the same syntax as with the requests module.
import json

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Hi human! Welcome to my prediction app!"


def test_get_malformed():
    r = client.get("/items/other")
    assert r.status_code != 200
    assert r.json() == {'detail': 'Not Found'}


def test_post():
    r = client.post("/predict", json={})
    assert r.status_code == 200
    assert r.json()["prediction"] == 0


def test_post_params_case_0():
    r = client.post("/predict", json={"age": 60})
    assert r.status_code == 200
    assert r.json()["data"]["age"] == 60
    assert r.json()["data"]["native-country"] == "United-States"
    assert r.json()["prediction"] == 0

def test_post_params_case_1():
    data = {
        'age': 31,
        'workclass': ' Private',
        'fnlgt': 45781,
        'education': ' Masters',
        'education-num': 14,
        'marital-status': ' Never-married',
        'occupation': ' Prof-specialty',
        'relationship': ' Not-in-family',
        'race': ' White',
        'sex': ' Female',
        'capital-gain': 14084,
        'capital-loss': 0,
        'hours-per-week': 50,
        'native-country': ' United-States'
    }
    r = client.post("/predict", json=data)
    assert r.status_code == 200
    assert r.json()["prediction"] == 1
