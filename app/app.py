from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .api import predict
#from .api import database

description = """
Deploys a logistic regression model fit on the [Palmer MedCabinet](https://raw.githubusercontent.com/build-week-med-cabinet2/DS/main/dataset/cannabis.csv)

<img src="https://d2ebzu6go672f3.cloudfront.net/media/content/images/p6_MedicalMarijuana_ML1710_ts483300738.jpg" width="40%" /> 
"""


app = FastAPI(
    title='Med Cannabis Predictor API',
    description=description,
    version='0.1',
    docs_url='/',
)

app.include_router(predict.router)
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
