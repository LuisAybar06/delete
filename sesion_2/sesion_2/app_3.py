import uvicorn
from fastapi import FastAPI, File, UploadFile
from joblib import load
from scipy.sparse import load_npz
import io
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, DateTime, Date
from sqlalchemy.orm import sessionmaker
from datetime import datetime


SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
metadata = MetaData()

# Cargar la tabla existente
predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, index=True),
    Column("prediction", Float),
    Column("created_at", DateTime), 
    Column("process_date", Date),
)
metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


app = FastAPI()

classifier = load('xgb_model.pkl') 

@app.get("/")
def read_root():
    return {"message": "Hello, World"}



@app.post("/predict")
async def predict(file: UploadFile = File(...)):  
    contents = await file.read()
    npz_file = io.BytesIO(contents)
    
    x_test_merged_loaded = load_npz(npz_file)
    
    predictions_result = classifier.predict(x_test_merged_loaded)
    
    db = SessionLocal()

    try:
        # Guardar cada predicción en la base de datos
        for prediction in predictions_result:
            # Insertar cada predicción en la tabla
            db.execute(
                predictions.insert().values(
                    prediction=prediction,
                    created_at = datetime.now(),
                    process_date = datetime.now().date()

                )
            )
        # Confirmar los cambios
        db.commit()
        
    except Exception as e:
        db.rollback()
        return {"error": str(e)}
    
    finally:
        db.close()

    return {"predictions": predictions_result.tolist()}
