from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import mysql.connector
import cv2
import shutil
import os
from ultralytics import YOLO
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://food-backend-1oar.onrender.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/opt/render/project/uploads"  # Render 臨時儲存路徑os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

model = YOLO("food_yolov8_multi_GoogleColab.pt")

db_config = {
    "host" = "gateway01.us-east-1.prod.aws.tidbcloud.com",
    "port" = 4000,
    "user" = "4LbqsBXa8zkncXb.root",
    "password" = "TdVb2lyWfM1TUuw8",
    "database" = "food_recognition",
    "ssl_ca" = "/etc/ssl/cert.pem",
    "ssl_verify_cert" = True,
  "ssl_verify_identity" = True
}

def connect_db():
    return mysql.connector.connect(**db_config)

@app.options("/predict")
async def options_predict():
    return JSONResponse(status_code=200, content={"message": "CORS preflight successful"})

@app.post("/predict")
async def predict_food(image: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        original_filename = f"{timestamp}_{image.filename}"
        img_path = os.path.join(UPLOAD_DIR, original_filename)

        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        img = cv2.imread(img_path)
        results = model(img, conf=0.5, iou=0.4)

        detections = []
        detected_names = set()

        if results and len(results[0].boxes.cls) > 0:
            boxes = results[0].boxes.xyxy.tolist()
            names = results[0].names
            confidences = results[0].boxes.conf.tolist()
            classes = results[0].boxes.cls.tolist()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(classes[i])
                food_name = names[cls_id]
                conf = confidences[i]

                if food_name in detected_names:
                    continue
                detected_names.add(food_name)

                detection_boxes = [{"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, f"{food_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                conn = connect_db()
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT food_name, serving_size, calories, protein, total_carbohydrate, 
                           total_fat, dietary_fiber, price 
                    FROM food_info 
                    WHERE label=%s
                """, (food_name,))
                food_data = cursor.fetchone()
                logger.info(f"查詢結果 for {food_name}: {food_data}")  # 確認查詢結果
                cursor.close()
                conn.close()

                detection = {
                    "food_name": food_data["food_name"] if food_data else food_name,
                    "label": food_name,
                    "confidence": conf,
                    "boxes": detection_boxes
                }
                if food_data:
                    detection["serving_size"] = food_data["serving_size"]
                    detection["calories"] = food_data["calories"]
                    detection["protein"] = food_data["protein"]
                    detection["total_carbohydrate"] = food_data["total_carbohydrate"]
                    detection["total_fat"] = food_data["total_fat"]
                    detection["dietary_fiber"] = food_data["dietary_fiber"]
                    detection["price"] = food_data["price"]

                detections.append(detection)

            detected_filename = f"detected_{original_filename}"
            detected_img_path = os.path.join(UPLOAD_DIR, detected_filename)
            cv2.imwrite(detected_img_path, img)

            conn = connect_db()
            cursor = conn.cursor()
            for detection in detections:
                calories = detection.get("calories", None)
                price = detection.get("price", None)
                cursor.execute(
                    "INSERT INTO food_history (food_name, confidence, calories, price, image_path) VALUES (%s, %s, %s, %s, %s)",
                    (detection["food_name"], detection["confidence"], calories, price, detected_img_path),
                )
            conn.commit()
            cursor.close()
            conn.close()

            response_data = {
                "detections": detections,
                "image_url": f"http://localhost:8000/uploads/{detected_filename}"
            }
            logger.info(f"📦 偵測結果: {response_data}")
            return JSONResponse(content=response_data)
        else:
            raise HTTPException(status_code=400, detail="未偵測到食物！")

    except Exception as e:
        logger.error(f"❌ 錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
