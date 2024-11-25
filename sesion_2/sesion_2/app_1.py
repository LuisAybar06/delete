import uvicorn
from fastapi import FastAPI
from joblib import load
from scipy.sparse import load_npz
import joblib


# 1. Create the app object
app = FastAPI()

# 2. Load the model
classifier = joblib.load('xgb_model.pkl')
x_test_merged_loaded = load_npz('x_test_merged.npz')


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}
 

# 4. Expose the prediction functionality, make a prediction using x_test data
@app.post('/predict')
def predict():
    predictions = classifier.predict(x_test_merged_loaded)
    return {
        'predictions': predictions.tolist()
    }

