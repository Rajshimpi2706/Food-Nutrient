import os
import tempfile
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from imageai.Classification import ImageClassification
import uvicorn
import torch

# ------------------ APP SETUP ------------------
app = FastAPI(title="Fruit Nutrition Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ USDA CONFIG ------------------
USDA_API_KEY = os.getenv("USDA_API_KEY", "PjcAGiGcTDQiMsiTSFbfR5rIvo8cd0SFUGQIkRP1")
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# ------------------ IMAGEAI MODEL SETUP ------------------
execution_path = os.getcwd()

classifier = ImageClassification()
classifier.setModelTypeAsResNet50()

possible_paths = [
    os.path.join(execution_path, "resnet50-19c8e357.pth"),
    os.path.join(execution_path, "..", "resnet50-19c8e357.pth"),
    os.path.join(execution_path, "..", "FruitNutritionDetector", "resnet50-19c8e357.pth"),
]

model_path = next((os.path.abspath(p) for p in possible_paths if os.path.isfile(p)), None)

if not model_path:
    raise RuntimeError("Model file resnet50-19c8e357.pth not found")

classifier.setModelPath(model_path)

# Patch torch.load for ImageAI compatibility
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)

torch.load = patched_load
classifier.loadModel()

# ------------------ USDA FUNCTIONS ------------------
def fetch_nutritional_data(food_name: str):
    response = requests.get(
        USDA_BASE_URL,
        params={
            "api_key": USDA_API_KEY,
            "query": food_name,
            "pageSize": 1
        },
        timeout=10
    )

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="USDA API failed")

    return response.json()

def extract_nutrients(data):
    foods = data.get("foods")
    if not foods:
        return {"error": "No nutrition data found"}

    food = foods[0]
    nutrients = {}

    for n in food.get("foodNutrients", []):
        if n.get("nutrientName") in {
            "Energy",
            "Protein",
            "Total lipid (fat)",
            "Carbohydrate, by difference",
            "Fiber, total dietary",
        }:
            nutrients[n["nutrientName"]] = n.get("value")

    return {
        "food_name": food.get("description"),
        "nutrients": nutrients,
    }

# ------------------ ROUTES ------------------
@app.get("/")
def root():
    return {"message": "Fruit Nutrition Detection API is running. Use /docs"}

@app.post("/fruit-detection")
async def fruit_detection(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        predictions, probabilities = classifier.classifyImage(temp_path, result_count=1)

        detected_fruit = predictions[0]
        confidence = round(float(probabilities[0]), 3)

        usda_data = fetch_nutritional_data(detected_fruit)
        nutrition_info = extract_nutrients(usda_data)

        return {
            "detected_fruit": detected_fruit,
            "confidence": confidence,
            "nutrition_info": nutrition_info,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "processing_failed", "detail": str(e)},
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ------------------ RUN SERVER ------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5002)
