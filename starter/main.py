# Put the code for your API here.

# Import Union since our Item object will have tags that can be strings or a list.
import os
from typing import Union, Optional, AnyStr

import pandas as pd
from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel
import requests
import json

from starter.ml.data import process_data
from starter.ml.utils import load_pickle

app = FastAPI()


@app.get("/")
async def welcome_msg():
    return {"message": f"Hi human! Welcome to my prediction app!"}


# Declare the data object with its components and their type.
class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    item_id: int


class CensusData(BaseModel):
    age: Optional[int] = 39
    workclass: Optional[str] = 'State-gov'
    fnlgt: Optional[int] = 77516
    education: Optional[str] = 'Bachelors'
    education_num: Optional[int] = 13
    marital_status: Optional[str] = "Never-married"
    occupation: Optional[str] = "Adm-clerical"
    relationship: Optional[str] = "Not-in-family"
    race: Optional[str] = "White"
    sex: Optional[str] = "Male"
    capital_gain: Optional[int] = 2174
    capital_loss: Optional[int] = 0
    hours_per_week: Optional[int] = 40
    native_country: Optional[str] = "United-States"


# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/predict")
async def predict(data: CensusData):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    local=True
    if local==True:
        mdl = load_pickle(os.path.join("model", "trained_model.pickle"))
        encoder = load_pickle(os.path.join("model", "hot_encoder.pickle"))
        lb = load_pickle(os.path.join("model", "label_encoder.pickle"))
    else:
        raise NotImplementedError("need cloud setup")

    print(data.dict())
    df = pd.DataFrame(data.dict(), index=[0])
    y = inference(df, mdl, encoder, lb, cat_features=cat_features)

    return {"data": data, "prediction": y}


def inference(data, mdl, encoder, lb, cat_features):
    x, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    return mdl.predict(x)
