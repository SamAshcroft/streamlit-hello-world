from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io

app = FastAPI()

# Load models once on startup
text_classifier = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")
image_classifier = pipeline(model="microsoft/resnet-50")

class TextInput(BaseModel):
    text: str

@app.post("/predict/sentiment")
def predict_sentiment(input: TextInput):
    result = text_classifier(input.text)
    return {"result": result}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = image_classifier(image)
    return {"result": result}
