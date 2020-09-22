from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api import predict, viz

description = """
Deploys a logistic regression model fit on the [Palmer AirBnB](https://raw.githubusercontent.com/bw-airbnb-2/DS/master/airbnb.csv) dataset.

<img src="https://github.com/samuelklam/airbnb-pricing-prediction/blob/master/public/img/post-sample-image.jpg?raw=true" width="40%" /> <img src="https://tse3.mm.bing.net/th?id=OIP.0HCy6Zoz51bbmucz38NlCQHaEK&pid=Api&P=0&w=343&h=194" width="30%" />
"""


app = FastAPI(
    title='AirBnB Optimizing Price Predictor API',
    description=description,
    version='0.1',
    docs_url='/',
)

app.include_router(predict.router)

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
