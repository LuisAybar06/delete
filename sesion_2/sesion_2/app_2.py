import uvicorn
from fastapi import FastAPI, File, UploadFile
from joblib import load
from scipy.sparse import load_npz
import io

app = FastAPI()

classifier = load('xgb_model.pkl') 

@app.get("/")
def read_root():
    return {"message": "Hello, World"}

# Exponer la funcionalidad de predicción usando el archivo .npz cargado
@app.post("/predict")
async def predict(file: UploadFile = File(...)):  

    contents = await file.read()
    npz_file = io.BytesIO(contents)
    

    x_test_merged_loaded = load_npz(npz_file)
    predictions = classifier.predict(x_test_merged_loaded)
    
    # Retornar las predicciones
    return {"predictions": predictions.tolist()}
