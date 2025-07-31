import cv2
from ultralytics import YOLO

# Carrega modelo
model = YOLO("weights/helmet.pt")  # Modelo ja criado e treinado, peguei no github do Ultralytics

# Mapear ID da classe para texto legível
class_map = {
    0: "Com Touca",
    1: "Sem Touca"
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = frame.copy()

    if results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{class_map.get(cls_id, 'Desconhecido')} {conf:.2f}"
            color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)  # verde se com capacete, vermelho se sem

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        # Nenhuma detecção - opcional exibir texto
        cv2.putText(annotated_frame, "Nenhuma pessoa detectada", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Detecção de Touca", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
