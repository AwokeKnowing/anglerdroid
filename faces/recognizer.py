"""Face detection and recognition core.

Auto-selects best available backend:
1. face_recognition (dlib) - best quality if installed
2. OpenCV DNN - fallback with decent quality
3. OpenCV Haar cascades - minimal fallback
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

try:
    import face_recognition
    _HAS_FACE_RECOGNITION = True
except ImportError:
    _HAS_FACE_RECOGNITION = False

try:
    import cv2
    _HAS_OPENCV = True
except ImportError:
    _HAS_OPENCV = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


class FaceRecognizer:
    """Face detection and recognition with multiple backend support."""
    
    def __init__(self, gallery_path: Optional[str] = None, backend: str = "auto"):
        """Initialize face recognizer.
        
        Args:
            gallery_path: Path to face gallery storage (default: ~/.kevin/faces)
            backend: "auto", "face_recognition", "opencv-dnn", "opencv-haar"
        """
        if gallery_path is None:
            gallery_path = os.path.expanduser("~/.kevin/faces")
        
        self.gallery_path = Path(gallery_path)
        self.gallery_path.mkdir(parents=True, exist_ok=True)
        
        if backend == "auto":
            if _HAS_FACE_RECOGNITION:
                self.backend = "face_recognition"
            elif _HAS_OPENCV:
                self.backend = "opencv-dnn"
            else:
                raise RuntimeError("No face recognition backend available. "
                                   "Install opencv-python or face_recognition")
        else:
            self.backend = backend
        
        self.db = self._load_database()
        
        if self.backend == "opencv-dnn":
            self._init_opencv_dnn()
        elif self.backend == "opencv-haar":
            self._init_opencv_haar()
        
        print(f"FaceRecognizer: using {self.backend} backend")
        print(f"FaceRecognizer: gallery at {self.gallery_path}")
        print(f"FaceRecognizer: loaded {len(self.db)} known faces")
    
    def _init_opencv_dnn(self):
        """Initialize face detector — YuNet ONNX on OpenCV 5 (no Caffe)."""
        model_dir = self.gallery_path / "models"
        model_dir.mkdir(exist_ok=True)
        self.face_net = None
        self.yunet = None

        yunet = model_dir / "face_detection_yunet_2023mar.onnx"
        if yunet.exists() and hasattr(cv2, "FaceDetectorYN"):
            # Input size set per-frame in detect
            self.yunet = cv2.FaceDetectorYN.create(
                str(yunet), "", (320, 320), 0.6, 0.3, 5000
            )
            print(f"FaceRecognizer: YuNet loaded from {yunet}")
            return

        prototxt = model_dir / "deploy.prototxt"
        caffemodel = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        if prototxt.exists() and caffemodel.exists() and hasattr(cv2.dnn, "readNetFromCaffe"):
            self.face_net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
            return

        print("Warning: No YuNet/Caffe face model usable; detection may fail.")
    
    def _init_opencv_haar(self):
        """Initialize OpenCV Haar cascade face detector (if available)."""
        self.face_cascade = None
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def _load_database(self) -> Dict[str, Dict]:
        """Load face database from disk."""
        db_path = self.gallery_path / "database.pkl"
        if db_path.exists():
            with open(db_path, "rb") as f:
                return pickle.load(f)
        return {}
    
    def _save_database(self):
        """Save face database to disk."""
        db_path = self.gallery_path / "database.pkl"
        with open(db_path, "wb") as f:
            pickle.dump(self.db, f)
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image.
        
        Args:
            image: BGR image (OpenCV format)
        
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        if self.backend == "face_recognition":
            return self._detect_face_recognition(image)
        elif self.backend == "opencv-dnn":
            return self._detect_opencv_dnn(image)
        elif self.backend == "opencv-haar":
            return self._detect_opencv_haar(image)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _detect_face_recognition(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using face_recognition library."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        return [(left, top, right - left, bottom - top) 
                for top, right, bottom, left in boxes]
    
    def _detect_opencv_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using YuNet (preferred) or legacy SSD."""
        h, w = image.shape[:2]
        if getattr(self, "yunet", None) is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(image)
            boxes = []
            if faces is not None:
                for f in faces:
                    x, y, bw, bh = [int(v) for v in f[:4]]
                    # clamp
                    x = max(0, x); y = max(0, y)
                    bw = max(1, min(bw, w - x)); bh = max(1, min(bh, h - y))
                    boxes.append((x, y, bw, bh))
            return boxes

        if self.face_net is None:
            return self._detect_opencv_haar(image)

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 117, 123))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
    
    def _detect_opencv_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar cascades."""
        if getattr(self, "face_cascade", None) is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    
    def extract_embedding(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract face embedding from image.
        
        Args:
            image: BGR image
            box: (x, y, w, h) face bounding box
        
        Returns:
            Face embedding vector
        """
        if self.backend == "face_recognition":
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x, y, w, h = box
            face_location = (y, x + w, y + h, x)
            encodings = face_recognition.face_encodings(rgb, [face_location])
            return encodings[0] if encodings else None
        else:
            x, y, w, h = box
            face = image[y:y+h, x:x+w]
            if face.size == 0:
                return None
            face_resized = cv2.resize(face, (128, 128))
            return face_resized.flatten().astype(np.float32) / 255.0
    
    def enroll(self, name: str, image: np.ndarray) -> int:
        """Enroll a person's face.
        
        Args:
            name: Person's display name
            image: BGR image containing face
        
        Returns:
            Number of faces enrolled (0 if none found)
        """
        boxes = self.detect_faces(image)
        if not boxes:
            print(f"No faces detected in image for {name}")
            return 0

        # Prefer largest face (primary subject) when multiple detections
        if len(boxes) > 1:
            boxes = [max(boxes, key=lambda b: b[2] * b[3])]
        
        embeddings = []
        for box in boxes:
            emb = self.extract_embedding(image, box)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            print(f"No face embeddings extracted for {name}")
            return 0
        
        if name not in self.db:
            self.db[name] = {"embeddings": []}
        
        self.db[name]["embeddings"].extend(embeddings)
        self._save_database()
        
        person_dir = self.gallery_path / name.replace(" ", "_").lower()
        person_dir.mkdir(exist_ok=True)
        
        img_idx = len(list(person_dir.glob("*.jpg")))
        img_path = person_dir / f"{img_idx:03d}.jpg"
        cv2.imwrite(str(img_path), image)
        
        print(f"Enrolled {len(embeddings)} face(s) for {name} (total: {len(self.db[name]['embeddings'])})")
        return len(embeddings)
    
    def recognize(self, image: np.ndarray, threshold: float = 0.6) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Recognize faces in image.
        
        Args:
            image: BGR image
            threshold: Recognition confidence threshold (lower = stricter)
        
        Returns:
            List of (name, confidence, box) tuples
        """
        boxes = self.detect_faces(image)
        results = []
        
        for box in boxes:
            emb = self.extract_embedding(image, box)
            if emb is None:
                continue
            
            best_name = "unknown"
            best_distance = float('inf')
            
            for name, data in self.db.items():
                for stored_emb in data["embeddings"]:
                    if self.backend == "face_recognition":
                        distance = np.linalg.norm(emb - stored_emb)
                    else:
                        distance = 1.0 - np.dot(emb, stored_emb) / (
                            np.linalg.norm(emb) * np.linalg.norm(stored_emb) + 1e-8
                        )
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_name = name
            
            confidence = max(0.0, 1.0 - best_distance)
            
            if confidence >= threshold:
                results.append((best_name, confidence, box))
            else:
                results.append(("unknown", confidence, box))
        
        return results
    
    def list_people(self) -> List[Tuple[str, int]]:
        """List enrolled people.
        
        Returns:
            List of (name, count) tuples where count is number of embeddings
        """
        return [(name, len(data["embeddings"])) for name, data in self.db.items()]
    
    def remove_person(self, name: str) -> bool:
        """Remove a person from the database.
        
        Args:
            name: Person's display name
        
        Returns:
            True if removed, False if not found
        """
        if name in self.db:
            del self.db[name]
            self._save_database()
            
            person_dir = self.gallery_path / name.replace(" ", "_").lower()
            if person_dir.exists():
                import shutil
                shutil.rmtree(person_dir)
            
            print(f"Removed {name} from database")
            return True
        return False
