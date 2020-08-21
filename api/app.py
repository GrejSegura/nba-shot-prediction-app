from fastapi import FastAPI
import uvicorn
import joblib
from models import *


app = FastAPI()
model = Classifier()

model.load_state_dict(torch.load(r'./models/model_pytorch.pt'))
model.eval()

# Routes
@app.get('/predict/{parameters}')
async def predict(parameters):
    '''
    pre-process pipeline
    '''
    data = parameters ### this should be the data pipeline
    proba = model.predict_pytorch(data)
    if proba > 0.5:
        prediction = 1
    else:
        prediction = 0

    return {'prediction':prediction}

if __name__=='__main__':
    uvicorn.run(app)
