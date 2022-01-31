# Put the code for your API here.

# Import Union since our Item object will have tags that can be strings or a list.
import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from starter.ml.model import inference_new_data
from starter.ml.utils import load_pickle

app = FastAPI()


@app.get("/")
async def welcome_msg():
    return "Hi human! Welcome to my prediction app!"


class CensusData(BaseModel):
    age: Optional[int] = 39
    workclass: Optional[str] = 'State-gov'
    fnlgt: Optional[int] = 77516
    education: Optional[str] = 'Bachelors'
    education_num: Optional[int] = Field(13, alias="education-num")
    marital_status: Optional[str] = Field("Never-married",
                                          alias="marital-status")
    occupation: Optional[str] = "Adm-clerical"
    relationship: Optional[str] = "Not-in-family"
    race: Optional[str] = "White"
    sex: Optional[str] = "Male"
    capital_gain: Optional[int] = Field(2174, alias="capital-gain")
    capital_loss: Optional[int] = Field(0, alias="capital-loss")
    hours_per_week: Optional[int] = Field(40, alias="hours-per-week")
    native_country: Optional[str] = Field("United-States",
                                          alias="native-country")

    class Config:
        arbitrary_types_allowed = True


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

    # Heroku access to DVC data
    if "DYNO" in os.environ and os.path.isdir(".dvc"):
        os.system("dvc config core.no_scm true")
        if os.system("dvc pull") != 0:
            exit("dvc pull failed")
        os.system("rm -r .dvc .apt/usr/lib/dvc")

    cwd = os.getcwd()
    mdl = load_pickle(os.path.join(cwd, "model", "trained_model.pickle"))
    encoder = load_pickle(os.path.join(cwd, "model", "hot_encoder.pickle"))
    lb = load_pickle(os.path.join(cwd, "model", "label_encoder.pickle"))

    df = pd.DataFrame(jsonable_encoder(data.dict(by_alias=True)), index=[0])
    y = inference_new_data(df, mdl, encoder, lb, cat_features=cat_features)

    return {"data": data, "prediction": y.tolist()[0]}

