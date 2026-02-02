import os
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from imageai.Classification import ImageClassification
import uvicorn
import torch

# ------------------ APP SETUP ------------------
app = FastAPI(title="Fruit Nutrition Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ USDA CONFIG ------------------
# Use provided API key by default; can be overridden via environment variable
USDA_API_KEY = os.getenv("USDA_API_KEY", "PjcAGiGcTDQiMsiTSFbfR5rIvo8cd0SFUGQIkRP1")
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Note: a default key is set above per user request. To use an environment variable instead,
# remove the second argument to os.getenv or set USDA_API_KEY in your environment.

# ------------------ IMAGEAI MODEL SETUP ------------------
execution_path = os.getcwd()

classifier = ImageClassification()
classifier.setModelTypeAsResNet50()
# Try to find model in current dir, else try parent FruitNutritionDetector folder
possible_paths = [os.path.join(execution_path, "resnet50-19c8e357.pth"),
                  os.path.join(execution_path, "..", "resnet50-19c8e357.pth"),
                  os.path.join(execution_path, "..", "FruitNutritionDetector", "resnet50-19c8e357.pth")]
model_path = None
for p in possible_paths:
    if os.path.isfile(p):
        model_path = os.path.abspath(p)
        break

if not model_path:
    raise RuntimeError("resnet50-19c8e357.pth model not found. Place it in the raj folder or ../FruitNutritionDetector/")

classifier.setModelPath(model_path)

# Patch torch.load for legacy ImageAI models
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = patched_load

classifier.loadModel()

# ------------------ USDA FUNCTIONS ------------------
def fetch_nutritional_data(food_name: str):
    params = {
        "api_key": USDA_API_KEY,
        "query": food_name,
        "pageSize": 1
    }
    response = requests.get(USDA_BASE_URL, params=params, timeout=10)

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="USDA API failed")

    return response.json()

def extract_nutrients(data):
    if "foods" not in data or not data["foods"]:
        return {"error": "No nutrition data found"}

    food = data["foods"][0]
    nutrients = {}

    for n in food["foodNutrients"]:
        name = n.get("nutrientName")
        value = n.get("value")

        if name in [
            "Energy",
            "Protein",
            "Total lipid (fat)",
            "Carbohydrate, by difference",
            "Fiber, total dietary"
        ]:
            nutrients[name] = value

    return {
        "food_name": food.get("description"),
        "nutrients": nutrients
    }

# ------------------ ROUTES ------------------
@app.get("/")
def root():
    return {
        "message": "Fruit Nutrition Detection API is running. Use /docs for API documentation."
    }

@app.post("/fruit-detection")
async def fruit_detection(file: UploadFile = File(...)):
    # Log request metadata for debugging
    try:
        client_host = getattr(file, 'filename', 'unknown')
        print(f"[fruit-detection] received file: {file.filename}, content_type: {file.content_type}")
    except Exception:
        pass

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files allowed")

    file_path = os.path.join(execution_path, file.filename)

    # Save uploaded image
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        predictions, probabilities = classifier.classifyImage(
            file_path, result_count=1
        )
        detected_fruit = predictions[0]
        confidence = float(probabilities[0])

        usda_data = fetch_nutritional_data(detected_fruit)
        nutrition_info = extract_nutrients(usda_data)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[fruit-detection] error during processing: {e}\n{tb}")
        return JSONResponse(status_code=500, content={"error": "processing_failed", "detail": str(e), "trace": tb.splitlines()[-10:]})

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return JSONResponse(
        content={
            "detected_fruit": detected_fruit,
            "confidence": round(confidence, 3),
            "nutrition_info": nutrition_info
        }
    )

# ------------------ RUN SERVER ------------------
if __name__ == "__main__":
    # Run without reload to avoid unstable reloader behavior in this environment
    uvicorn.run("app:app", host="0.0.0.0", port=5002, reload=False)
