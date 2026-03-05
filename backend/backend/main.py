"""
=============================================================================
 Prevo Audit AI Agent — Real-Time Object Verification System (FastAPI Backend)
=============================================================================
"""

import asyncio
import base64
import time
import threading

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from pathlib import Path
from ultralytics import YOLO

RECORDINGS_DIR = Path("recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)

OBJECTS_TO_VERIFY = ["cell phone", "laptop", "bottle"]
DISPLAY_NAMES = ["Mobile Phone", "Laptop", "Bottle"]

CONFIDENCE_THRESHOLD = 0.70
INSTANT_VERIFY_THRESHOLD = 0.90
VERIFIED_HOLD_SECONDS = 2.5
MODEL_PATH = "yolov8n.pt"

print("[Prevo Audit AI Agent] Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
print("[Prevo Audit AI Agent] Model ready.")


class LazyCamera:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self._cap = None
        self._thread = None
        self._running = False
        self._clients = 0
        self._clients_lock = threading.Lock()

    def add_client(self):
        with self._clients_lock:
            self._clients += 1
            if self._clients == 1:
                self._start()

    def remove_client(self):
        with self._clients_lock:
            self._clients = max(0, self._clients - 1)
            if self._clients == 0:
                self._stop()

    @property
    def is_active(self):
        return self._running

    def _start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[Prevo Audit AI Agent] Camera OPENED (client connected)")

    def _stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        with self.lock:
            self.frame = None
        print("[Prevo Audit AI Agent] Camera CLOSED (no clients)")

    def _capture_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Prevo Audit AI Agent] WARNING: No camera device found. Using browser camera mode.")
            self._running = False
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        while self._running:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.08)
        cap.release()

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

camera = LazyCamera()


class Session:
    WAITING  = "waiting"
    VERIFIED = "verified"
    WRONG    = "wrong"
    DONE     = "done"

    def __init__(self):
        self.reset()

    def reset(self):
        self.current_index = 0
        self.step_status   = self.WAITING
        self.verify_time   = None
        self.results = [
            {"object": OBJECTS_TO_VERIFY[i], "display": DISPLAY_NAMES[i], "status": "pending"}
            for i in range(len(OBJECTS_TO_VERIFY))
        ]
        self.complete = False
        self.is_recording = False
        self.video_writer = None
        self.recording_filename = ""

    def start_recording(self):
        if self.is_recording:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.recording_filename = str(RECORDINGS_DIR / f"session_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.recording_filename, fourcc, 5.0, (1280, 720))
        self.is_recording = True
        print(f"[Prevo Audit AI Agent] Recording started: {self.recording_filename}")

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        print(f"[Prevo Audit AI Agent] Recording saved: {self.recording_filename}")

    @property
    def current_object(self):
        return OBJECTS_TO_VERIFY[self.current_index]

    @property
    def current_display(self):
        return DISPLAY_NAMES[self.current_index]

    def advance(self):
        self.current_index += 1
        if self.current_index >= len(OBJECTS_TO_VERIFY):
            self.complete    = True
            self.step_status = self.DONE
        else:
            self.step_status = self.WAITING
            self.verify_time = None

    def mark_verified(self, conf):
        if self.step_status == self.WAITING:
            self.step_status = self.VERIFIED
            self.verify_time = time.time()
            self.results[self.current_index]["status"]     = "verified"
            self.results[self.current_index]["confidence"] = f"{conf:.0%}"

    def mark_wrong(self, detected):
        self.step_status = self.WRONG
        self.results[self.current_index]["detected"] = detected

session = Session()

def detect_and_annotate(frame, expected):
    results  = model(frame, verbose=False)[0]
    status   = "waiting"
    top_label = None
    top_conf  = 0.0
    detections = []
    frame_h, frame_w = frame.shape[:2]
    frame_area = frame_w * frame_h

    for box in results.boxes:
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = model.names[cls]
        if label == "person":
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        detections.append({"label": label, "conf": conf, "box": (x1, y1, x2, y2)})

    if detections:
        expected_dets = [d for d in detections if d["label"] == expected]
        other_dets    = [d for d in detections if d["label"] != expected]
        if expected_dets:
            best = max(expected_dets, key=lambda d: d["conf"])
            top_label = best["label"]
            top_conf  = best["conf"]
            if top_conf >= CONFIDENCE_THRESHOLD:
                status = "verified"
        elif other_dets:
            best = max(other_dets, key=lambda d: d["conf"])
            if best["conf"] >= 0.90:
                top_label = best["label"]
                top_conf  = best["conf"]
                status = "wrong"

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        match = det["label"] == expected
        color = (50, 220, 100) if match else (50, 80, 240)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"{det['label']} {det['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, tag, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, status, top_label, top_conf


def _update_session(status, top_label, top_conf):
    if session.complete:
        return
    if session.step_status in [Session.WAITING, Session.WRONG]:
        if status == "verified":
            session.mark_verified(top_conf)
            if top_conf >= INSTANT_VERIFY_THRESHOLD:
                session.advance()
        elif status == "wrong":
            session.mark_wrong(top_label or "")
        elif status == "waiting" and session.step_status == Session.WRONG:
            session.step_status = Session.WAITING
    if session.step_status == Session.VERIFIED:
        if time.time() - session.verify_time >= VERIFIED_HOLD_SECONDS:
            session.advance()


def mjpeg_generator():
    camera.add_client()
    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.08)
                continue
            if not session.complete:
                expected = session.current_object
                frame, status, top_label, top_conf = detect_and_annotate(frame, expected)
                _update_session(status, top_label, top_conf)
                if session.is_recording and session.video_writer:
                    rec_frame = cv2.resize(frame, (1280, 720))
                    session.video_writer.write(rec_frame)
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
            time.sleep(1 / 30)
    finally:
        camera.remove_client()


# ── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(title="Prevo Audit AI Agent - Object Verification System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "YOLOv8n",
        "objects": DISPLAY_NAMES,
        "camera": "active" if camera.is_active else "standby",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.get("/video")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/reset")
def reset_session():
    if session.is_recording:
        session.stop_recording()
    session.reset()
    return {"status": "ok"}


@app.post("/start_recording")
def start_rec():
    session.start_recording()
    return {"status": "ok", "file": session.recording_filename}


@app.post("/stop_recording")
def stop_rec():
    session.stop_recording()
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            payload = {
                "current_index":  session.current_index,
                "current_object": session.current_display if not session.complete else "",
                "step_status":    session.step_status,
                "complete":       session.complete,
                "is_recording":   session.is_recording,
                "results":        session.results,
            }
            await ws.send_json(payload)
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass


# ── Browser Camera Frame Detection ──────────────────────────────────────

class FrameRequest(BaseModel):
    image: str


@app.post("/detect_frame")
async def detect_frame(request: FrameRequest):
    try:
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return Response(content=b"", status_code=400)

        if not session.complete:
            expected = session.current_object
            frame, status, top_label, top_conf = detect_and_annotate(frame, expected)
            _update_session(status, top_label, top_conf)
            if session.is_recording and session.video_writer:
                rec_frame = cv2.resize(frame, (1280, 720))
                session.video_writer.write(rec_frame)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(content=jpeg.tobytes(), media_type="image/jpeg")
    except Exception as e:
        print(f"[Prevo Audit AI Agent] detect_frame error: {e}")
        return Response(content=b"", status_code=500)


# ── POST-based detection (backward compatible) ──────────────────────────

class DetectRequest(BaseModel):
    image: str


@app.post("/detect")
async def detect(request: DetectRequest):
    try:
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"detections": []}

        results = model(img, verbose=False)[0]
        detections = []
        for box in results.boxes:
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = model.names[cls]
            if label == "person":
                continue
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            display = label
            if label == "cell phone": display = "Mobile Phone"
            elif label == "laptop": display = "Laptop"
            elif label == "bottle": display = "Bottle"
            detections.append({
                "class_name": label,
                "display_name": display,
                "confidence": round(conf, 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "valid": True,
            })
        return {"detections": detections}
    except Exception as e:
        print(f"[Prevo Audit AI Agent] Detection error: {e}")
        return {"detections": []}


if __name__ == "__main__":
    import uvicorn
    print("[Prevo Audit AI Agent] Starting server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
