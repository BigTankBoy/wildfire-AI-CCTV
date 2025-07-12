import cv2
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import glob

# 현재 파일 경로 기준으로 설정
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
print(f"🔍 현재 파일: {current_file}")
print(f"🔍 현재 디렉토리: {current_dir}")

nested_dir = os.path.join(current_dir, "integration")
if os.path.exists(nested_dir):
    pt_files_nested = glob.glob(os.path.join(nested_dir, "*.pt"))
    if pt_files_nested:
        fire_model_path = pt_files_nested[0]
        thermal_model_path = pt_files_nested[1] if len(pt_files_nested) > 1 else None
    else:
        fire_model_path = r"C:\Users\namef\OneDrive\바탕 화면\integration\integration\fire_new.pt"
        thermal_model_path = r"C:\Users\namef\OneDrive\바탕 화면\integration\integration\fire_thermal.pt"
else:
    fire_model_path = r"C:\Users\namef\OneDrive\바탕 화면\integration\integration\fire_new.pt"
    thermal_model_path = r"C:\Users\namef\OneDrive\바탕 화면\integration\integration\fire_thermal.pt"

if not os.path.exists(fire_model_path):
    print("❌ 화재 모델 파일이 없습니다.")
    exit()

fire_model = YOLO(fire_model_path)
thermal_model = YOLO(thermal_model_path) if thermal_model_path and os.path.exists(thermal_model_path) else None

os.makedirs("captures", exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

captured = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    fire_detected, thermal_detected = False, False

    fire_results = fire_model(frame)
    for result in fire_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names.get(cls_id, f"class_{cls_id}")
            if "화재" in label or "연기" in label:
                fire_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if "화재" in label else (0, 165, 255)
                display_label = "Fire" if "화재" in label else "Smoke"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{display_label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if thermal_model:
        thermal_results = thermal_model(frame)
        for result in thermal_results:
            for box in result.boxes:
                thermal_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, "Thermal", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if not captured and (fire_detected or thermal_detected):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captures/detection_{now}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[✅ 캡처됨] {filename}")
        captured = True
        break

    cv2.putText(frame, "Fire & Thermal Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Fire & Thermal Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 프로그램 종료")
        break

cap.release()
cv2.destroyAllWindows()
