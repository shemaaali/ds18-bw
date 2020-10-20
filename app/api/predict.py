import logging
import random
from fastapi import APIRouter
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from keras.models import save_model

#import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from category_encoders import OneHotEncoder, OrdinalEncoder


log = logging.getLogger(__name__)
router = APIRouter()


strains = pd.read_csv("https://raw.githubusercontent.com/build-week-med-cabinet2/DS/main/dataset/cannabis.csv")
dataset = strains.values


# one hot encoder on all the non-numerical columns
encoder = OneHotEncoder(use_cat_names=True)
imputer = SimpleImputer()
scaler = StandardScaler()
Logistic_model = LogisticRegression(max_iter=1000)

#X_train_encoded.head()


X = dataset[:,0:5]
y = dataset[:,5]
y=np.reshape(y, (-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

#scaler_y.fit(y)
#yscale=scaler_y.transform(y)
#X = StandardScaler().fit_transform(X)
#y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train_encoded = encoder.fit_transform(X_train)
scaler_x.fit(X_train_encoded)
xscale=scaler_x.transform(X_train_encoded)

'''model = Sequential()
model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history = model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)'''


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""
    Strain:str = Field(..., example="13-Dawgs")
    Type: str = Field(..., example="sativa")
    Rating:float = Field(..., example=40)
    Effects:str = Field(..., example="Euphoric")
    Flavor:str = Field(..., example="Spicy/Herbal")

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])




@router.post('/predict')
async def predict(item: Item):
    """Make random baseline predictions for classification problem."""
    #X_new = pd.read_json(item)
    X_new = item.to_df()
    log.info(X_new)
    model = load_model("keras_model/keras_nn_model.h5", compile=False)
    Dict = {'pharmaceuticals' : 1, 'right strains' : 0, 'dosing' : 0, 'intake method' : 1, 'intake schedule' : 2, 'yes' : 1, 'no' : 0}
    strain_type = Dict.get(X_new['Strain'].iloc[0])
    med_type = Dict.get(X_new['Type'].iloc[0])
    Rating_cab = Dict.get(X_new['Rating'].iloc[0])
    Effects_cab = Dict.get(X_new['Effects'].iloc[0])
    Flavor_cab = Dict.get(X_new['Flavor'].iloc[0])
    print(Rating_cab)
    Xnew = np.array([[X_new['Strain'].iloc[0],
                      X_new['Type'].iloc[0], X_new['Rating'].iloc[0],
                      X_new['Effects'].iloc[0], X_new['Flavor'].iloc[0], str(strain_type), 
                      str(med_type),float(Rating_cab), str(Effects_cab), str(Flavor_cab)]])
    Xnew= scaler_x.transform(Xnew)
    y_pred = model.predict(Xnew)
    y_pred = scaler_y.inverse_transform(y_pred)
    y_pred = float(y_pred[0][0])
    #y_pred = float(random.randint(100, 500))
    return {
        'prediction': y_pred
    }