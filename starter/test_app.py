

# Write tests using the same syntax as with the requests module.
import json

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.json() == "Hi human! Welcome to my prediction app!"


def test_get_malformed():
    r = client.get("/items/other")
    assert r.status_code != 200


def test_post():
    r = client.post("/predict", json={})
    assert r.status_code == 200


def test_post_params():
    r = client.post("/predict", json={"age": 60})
    assert r.status_code == 200
    assert r.json()["data"]["age"] == 60
    assert r.json()["data"]["native_country"] == "United-States"
    assert r.json()["prediction"] == 1

