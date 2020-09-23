from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .api import predict, viz
#from .api import database

description = """
Deploys a logistic regression model fit on the [Palmer AirBnB](https://raw.githubusercontent.com/bw-airbnb-2/DS/master/airbnb.csv) dataset.

<img src="https://github.com/samuelklam/airbnb-pricing-prediction/blob/master/public/img/post-sample-image.jpg?raw=true" width="40%" /> <img src="https://github.com/bw-airbnb-2/marketing/blob/master/assets/whiteFontUpPrice.png" width="30%" />
"""


app = FastAPI(
    title='AirBnB Optimizing Price Predictor API',
    description=description,
    version='0.1',
    docs_url='/',
)

app.include_router(predict.router)
app.include_router(viz.router)
#app.include_router(database.router)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex='https?://.*',
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if __name__ == '__main__':
    uvicorn.run(app)
