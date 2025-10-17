from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import json
import os
from datetime import datetime

app = FastAPI(title="YOLO API Service")

# CORS для работы с Beget и мобильным приложением
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка моделей
models = {
    "default": YOLO('yolov8n.pt'),
    # Добавьте здесь ваши кастомные модели когда будут готовы
    # "model1": YOLO('path/to/model1.pt'),
    # "model2": YOLO('path/to/model2.pt'),
}

@app.on_event("startup")
async def startup_event():
    print("YOLO API Server started")
    for name, model in models.items():
        model.to('cpu')
        print(f"Model {name} loaded successfully")

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.5,
    model_name: str = "default"
):
    """Основной endpoint для предсказания"""
    try:
        # Проверка модели
        if model_name not in models:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not found")
        
        model = models[model_name]
        
        # Проверка типа файла
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Чтение изображения
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
        
        # Конвертация цвета если нужно
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Предсказание
        results = model(image, conf=confidence)
        
        # Обработка результатов
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    'label': model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        # Создаем изображение с bounding boxes
        annotated_image = results[0].plot()
        
        # Конвертируем в base64 для передачи
        _, buffer = cv2.imencode('.jpg', annotated_image)
        import base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "detections": detections,
            "annotated_image": image_base64,
            "model_used": model_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервера"""
    return {
        "status": "healthy", 
        "models_loaded": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models")
async def list_models():
    """Список доступных моделей"""
    return {"available_models": list(models.keys())}