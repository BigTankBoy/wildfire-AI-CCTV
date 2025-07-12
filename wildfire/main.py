import cv2
import os
import glob
from datetime import datetime
import numpy as np
from ultralytics import YOLO

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
print(f"🔍 현재 파일: {current_file}")
print(f"🔍 현재 디렉토리: {current_dir}")

def find_model_paths(base_dir):
    nested = os.path.join(base_dir, "integration")
    if os.path.exists(nested):
        pt_list = glob.glob(os.path.join(nested, "*.pt"))
        if pt_list:
            return pt_list[0], pt_list[1] if len(pt_list) > 1 else None
    return None, None

fire_model_path, thermal_model_path = find_model_paths(current_dir)

if not fire_model_path:
    fire_model_path = r"integration/fire_new.pt"
if not thermal_model_path:
    thermal_model_path = r"integration/fire_thermal.pt"

print(f"🔍 Fire 모델 경로: {fire_model_path}")
print(f"🔍 Thermal 모델 경로: {thermal_model_path}")

try:
    fire_model = YOLO(fire_model_path)
    thermal_model = YOLO(thermal_model_path) if thermal_model_path else None
    print("✅ 모델 로딩 완료!")
    print("📋 Fire 모델 클래스:", fire_model.names)
    if thermal_model:
        print("📋 Thermal 모델 클래스:", thermal_model.names)
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    exit(1)

fire_class_id = None
for idx, name in fire_model.names.items():
    if name.lower() == "fire":
        fire_class_id = idx
        break

thermal_class_id = None
if thermal_model:
    for idx, name in thermal_model.names.items():
        if name.lower() in ["thermal", "fire_thermal"]:
            thermal_class_id = idx
            break

result_file = open("cap.txt", "a", encoding="utf-8")
os.makedirs("captures", exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    result_file.close()
    exit(1)
print("✅ 카메라 준비 완료! 🚀 감지 시작 (종료: 'q')")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    fire_results = fire_model(frame)
    for result in fire_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            conf_percent = int(conf * 100)

            if cls_id == fire_class_id:
                display_label = "Fire"
                color = (0, 0, 255)
            else:
                display_label = "Smoke"
                color = (0, 165, 255)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{display_label} {conf_percent}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            remark = ""
            if cls_id == fire_class_id and conf_percent >= 80:
                remark = "★CRITICAL!!!!"
                fname = f"captures/Fire_{ts}_{conf_percent}.jpg"
                cv2.imwrite(fname, frame)
                print(f"[CRITICAL 캡처] {fname}")

            result_file.write(f"{ts} {display_label} {conf_percent} {remark}\n")

    if thermal_model:
        thermal_results = thermal_model(frame)
        for result in thermal_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                conf_percent = int(conf * 100)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Thermal {conf_percent}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                remark = ""
                if cls_id == thermal_class_id and conf_percent >= 80:
                    remark = "★CRITICAL!!!!"
                    fname = f"captures/Thermal_{ts}_{conf_percent}.jpg"
                    cv2.imwrite(fname, frame)
                    print(f"[CRITICAL 캡처] {fname}")
                result_file.write(f"{ts} Thermal {conf_percent} {remark}\n")

    cv2.putText(frame, "Fire & Thermal Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Fire & Thermal Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 종료 중...")
        break

cap.release()
cv2.destroyAllWindows()
result_file.close()
