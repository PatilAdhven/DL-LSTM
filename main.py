from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pickle
import re
import os

app = FastAPI(title="LSTM Text Prediction API")

MODEL_PATH = 'lstm_text_predictor.h5'
TOKENZIER_PATH = 'tokenizer.pkl'
MAX_SEQUENCE_LEN = 20

model = None
tokenizer = None

@app.on_event("startup")
def load_assets():
    global model, tokenizer
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENZIER_PATH):
        print("Loading model and tokenizer...")
        model = load_model(MODEL_PATH)
        with open(TOKENZIER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        print("Assets loaded successfully.")
    else:
        print("Warning: Model or tokenizer not found. Please run `python train_model.py` first.")

class TextInput(BaseModel):
    seed_text: str
    top_n: int = 3

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.get("/")
def root():
    return {"message": "LSTM Text Prediction API is running. Head over to /docs to experiment!"}

@app.post("/predict")
def predict_next(input_data: TextInput):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Train the model first.")

    seed_text = clean_text(input_data.seed_text)
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    
    if len(token_list) >= MAX_SEQUENCE_LEN:
        token_list = token_list[-(MAX_SEQUENCE_LEN-1):]
        
    token_list = pad_sequences([token_list], maxlen=MAX_SEQUENCE_LEN-1, padding='pre')
    
    predicted_probs = model.predict(token_list, verbose=0)[0]
    
    # Get top N predictions
    top_indices = np.argsort(predicted_probs)[-input_data.top_n:][::-1]
    
    predictions = []
    index_word = tokenizer.index_word
    for idx in top_indices:
        if idx in index_word:
            predictions.append(index_word[idx])
            
    return {
        "seed_text": input_data.seed_text,
        "clean_seed": seed_text,
        "predictions": predictions,
        "top_n": input_data.top_n
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
