import random
from fastapi import APIRouter
import joblib
import pandas as pd
from pydantic import BaseModel, confloat
import logging
import random
import pandas as pd
import numpy as np
from fastapi import APIRouter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

# load and read the file
df = pd.read_csv("https://raw.githubusercontent.com/bw-airbnb-2/DS/master/airbnb.csv", index_col=0)
dataset = df.values

log = logging.getLogger(__name__)
router = APIRouter()

#classifier = joblib.load('app/api/classifier.joblib')
#print('Pickled model loaded!')


class AirBnB(BaseModel):
    """Data model to parse & validate airbnb measurements"""
    {
    "userId":1,
    "name":"Chris",
    "room_type":"large",
    "location":"Japan",
    "price":255.99,
    "accommodates":3,
    "bathrooms":2,
    "bedrooms":2,
    "beds":3,
    "guests_included":2,
    "minimum_nights":3,
    "maximum_nights":6
    }

    def to_df(self):
        return pd.DataFrame([dict(self)])

@router.post('/predict')
def predict_species(airbnb: AirBnB):
    """Predict airbnb species from price and bedrooms"""
    species = classifier.predict(airbnb.to_df())
    return species[0]


@router.get('/random')
def random_airbnb():
    """Return a random airbnb species"""
    return random.choice([{"userId":1, "name":"Chris", "room_type":"large", 
    "location":"Japan",
    "price":255.99,
    "accommodates":3,
    "bathrooms":2,
    "bedrooms":2,
    "beds":3,
    "guests_included":2,
    "minimum_nights":3,
    "maximum_nights":6}])