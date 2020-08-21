from fastapi import FastAPI
import uvicorn
import joblib

app = FastAPI()

model_pytorch = open('../models/model_pytorch.pt', 'rb')
model = joblib.load(model_pytorch)

# Routes
@app.get('/predict/{parameters}')
async def predict(parameters):
    '''
    pre-process pipeline
    '''
    data = parameters ### this should be the data pipeline
    proba = model.predict(data)
    if proba > 0.5:
        prediction = 1
    else:
        prediction = 0

    return {'prediction':prediction}

if __name__=='__main__':
    uvicorn.run(app)
