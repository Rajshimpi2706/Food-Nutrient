Fruit Nutrition Detector (raj folder)

Quick start

1. Install dependencies (preferably in a virtualenv):

   pip install -r requirements.txt

2. Ensure `resnet50-19c8e357.pth` is available in this folder or in `../FruitNutritionDetector/`.

3. Set USDA API key (example, Windows PowerShell):

   setx USDA_API_KEY "your_actual_api_key_here"
   Restart your terminal to make it effective.

4. Run the app:

   python app.py

5. Open docs:

   http://127.0.0.1:5000/docs

Notes
- This uses ImageAI ResNet50 (ImageNet) for prototyping. For better fruit detection, fine-tune a model on a fruit dataset.
