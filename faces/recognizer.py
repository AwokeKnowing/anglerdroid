"""Face detection and recognition core.

Auto-selects best available backend:
1. insightface - SOTA ArcFace embeddings (buffalo_l or buffalo_s)
2. face_recognition (dlib) - good quality if installed
3. OpenCV DNN - fallback with decent quality
4. OpenCV Haar cascades - minimal fallback
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

try:
    import onnxruntime as ort
    _HAS_ONNXRUNTIME = True
except ImportError:
    _HAS_ONNXRUNTIME = False


def _align_face_insightface(img: np.ndarray, landmarks: np.ndarray, image_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """Align face using 5-point landmarks (InsightFace standard).
    
    Args:
        img: BGR image
        landmarks: 5x2 array of (x, y) landmark coordinates
        image_size: Target size for aligned face (default 112x112)
    
    Returns:
        Aligned face image
    """
    if not _HAS_OPENCV:
        raise RuntimeError("OpenCV required for landmark alignment")
    
    # Standard reference landmarks for 112x112 face (InsightFace ARCFACE standard)
    src = np.array([
        [30.2946 + 8.0, 51.6963],
        [65.5318 + 8.0, 51.5014],
        [48.0252 + 8.0, 71.7366],
        [33.5493 + 8.0, 92.3655],
        [62.7299 + 8.0, 92.2041]
    ], dtype=np.float32)
    
    if image_size != (112, 112):
        # Scale reference landmarks if target size differs
        scale_x = image_size[0] / 112.0
        scale_y = image_size[1] / 112.0
        src[:, 0] *= scale_x
        src[:, 1] *= scale_y
    
    # Compute similarity transform
    tform = cv2.estimateAffinePartial2D(landmarks, src, method=cv2.LMEDS)[0]
    
    # Warp image
    aligned = cv2.warpAffine(img, tform, image_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aligned


class FaceRecognizer:
    """Face detection and recognition with multiple backend support."""
    
    def __init__(self, gallery_path: Optional[str] = None, backend: str = "auto", model_pack: str = "buffalo_l"):
        """Initialize face recognizer.
        
        Args:
            gallery_path: Path to face gallery storage (default: ~/.kevin/faces)
            backend: "auto", "insightface", "face_recognition", "opencv-dnn", "opencv-haar"
            model_pack: For insightface backend: "buffalo_l" (512-D, best) or "buffalo_s" (mobile)
        """
        if gallery_path is None:
            gallery_path = os.path.expanduser("~/.kevin/faces")
        
        self.gallery_path = Path(gallery_path)
        self.gallery_path.mkdir(parents=True, exist_ok=True)
        self.model_pack = model_pack
        
        if backend == "auto":
            if _HAS_ONNXRUNTIME and _HAS_OPENCV:
                self.backend = "insightface"
            elif _HAS_FACE_RECOGNITION:
                self.backend = "face_recognition"
            elif _HAS_OPENCV:
                self.backend = "opencv-dnn"
            else:
                raise RuntimeError("No face recognition backend available. "
                                   "Install opencv-python + onnxruntime or face_recognition")
        else:
            self.backend = backend
        
        self.db = self._load_database()
        
        if self.backend == "insightface":
            self._init_insightface()
        elif self.backend == "opencv-dnn":
            self._init_opencv_dnn()
        elif self.backend == "opencv-haar":
            self._init_opencv_haar()
        
        print(f"FaceRecognizer: using {self.backend} backend")
        if self.backend == "insightface":
            print(f"FaceRecognizer: model pack {self.model_pack}")
        print(f"FaceRecognizer: gallery at {self.gallery_path}")
        print(f"FaceRecognizer: loaded {len(self.db)} known faces")
    
    def _init_opencv_dnn(self):
        """Initialize face detector — YuNet ONNX on OpenCV 5 (no Caffe)."""
        model_dir = self.gallery_path / "models"
        model_dir.mkdir(exist_ok=True)
        self.face_net = None
        self.yunet = None
        self.sface = None  # SFace fallback for opencv-dnn backend

        yunet = model_dir / "face_detection_yunet_2023mar.onnx"
        if yunet.exists() and hasattr(cv2, "FaceDetectorYN"):
            # Input size set per-frame in detect
            # Use 0.55 threshold from main (stricter than original 0.45)
            self.yunet = cv2.FaceDetectorYN.create(
                str(yunet), "", (320, 320), 0.55, 0.3, 5000
            )
            print(f"FaceRecognizer: YuNet loaded from {yunet}")
            
            # Try to load SFace for better embeddings (fallback if InsightFace not available)
            sface = model_dir / "face_recognition_sface_2021dec.onnx"
            if sface.exists() and hasattr(cv2, "FaceRecognizerSF"):
                self.sface = cv2.FaceRecognizerSF.create(str(sface), "")
                print(f"FaceRecognizer: SFace loaded from {sface} (fallback embeddings)")
            else:
                print("FaceRecognizer: SFace missing — will use weak pixel-based embeddings")
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
    
    def _init_insightface(self):
        """Initialize InsightFace models (detector + recognition)."""
        if not _HAS_ONNXRUNTIME:
            raise RuntimeError("onnxruntime required for insightface backend. Install onnxruntime-gpu")
        
        model_dir = self.gallery_path / "models"
        model_dir.mkdir(exist_ok=True)
        
        # Keep YuNet for detection (proven reliable at low res)
        yunet_path = model_dir / "face_detection_yunet_2023mar.onnx"
        if yunet_path.exists() and hasattr(cv2, "FaceDetectorYN"):
            self.yunet = cv2.FaceDetectorYN.create(
                str(yunet_path), "", (320, 320), 0.45, 0.3, 5000
            )
            print(f"FaceRecognizer: YuNet detector loaded from {yunet_path}")
        else:
            print(f"Warning: YuNet model not found at {yunet_path}")
            print("Download from: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
            self.yunet = None
        
        # Load InsightFace recognition model
        if self.model_pack == "buffalo_l":
            rec_model = "w600k_r50.onnx"  # 512-D embeddings
            self.embedding_dim = 512
        elif self.model_pack == "buffalo_s":
            rec_model = "w600k_mbf.onnx"  # MobileFaceNet, lighter
            self.embedding_dim = 512
        else:
            raise ValueError(f"Unknown model_pack: {self.model_pack}. Use buffalo_l or buffalo_s")
        
        rec_path = model_dir / rec_model
        if not rec_path.exists():
            print(f"Warning: InsightFace model not found at {rec_path}")
            print(f"Please download {self.model_pack} models from InsightFace model zoo")
            print(f"Place {rec_model} in {model_dir}/")
            self.rec_session = None
        else:
            # Use GPU if available, otherwise CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
            self.rec_session = ort.InferenceSession(str(rec_path), providers=providers)
            print(f"FaceRecognizer: {self.model_pack} recognition model loaded ({self.embedding_dim}-D)")
            print(f"FaceRecognizer: ONNX providers: {self.rec_session.get_providers()}")
    
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
        if self.backend == "insightface":
            return self._detect_insightface(image)
        elif self.backend == "face_recognition":
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
    
    def _detect_insightface(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using YuNet (same as opencv-dnn)."""
        return self._detect_opencv_dnn(image)
    
    def detect_faces_with_landmarks(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], Optional[np.ndarray]]]:
        """Detect faces and extract 5-point landmarks (for InsightFace alignment).
        
        Args:
            image: BGR image
        
        Returns:
            List of ((x, y, w, h), landmarks) tuples where landmarks is 5x2 array or None
        """
        if self.backend != "insightface" or not hasattr(self, 'yunet') or self.yunet is None:
            # Fallback: return boxes without landmarks
            boxes = self.detect_faces(image)
            return [(box, None) for box in boxes]
        
        h, w = image.shape[:2]
        self.yunet.setInputSize((w, h))
        _, faces = self.yunet.detect(image)
        
        results = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh = [int(v) for v in f[:4]]
                # Clamp box
                x = max(0, x); y = max(0, y)
                bw = max(1, min(bw, w - x)); bh = max(1, min(bh, h - y))
                box = (x, y, bw, bh)
                
                # Extract 5 landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
                if len(f) >= 14:
                    landmarks = np.array([
                        [f[4], f[5]],   # right eye
                        [f[6], f[7]],   # left eye
                        [f[8], f[9]],   # nose
                        [f[10], f[11]], # right mouth
                        [f[12], f[13]]  # left mouth
                    ], dtype=np.float32)
                else:
                    landmarks = None
                
                results.append((box, landmarks))
        
        return results
    
    def _detect_opencv_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using YuNet (preferred) or legacy SSD."""
        h, w = image.shape[:2]
        if getattr(self, "yunet", None) is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(image)
            boxes = []
            if faces is not None:
                # Store raw YuNet faces for SFace alignment
                self._last_yunet_faces = faces
                for f in faces:
                    x, y, bw, bh = [int(v) for v in f[:4]]
                    # clamp
                    x = max(0, x); y = max(0, y)
                    bw = max(1, min(bw, w - x)); bh = max(1, min(bh, h - y))
                    boxes.append((x, y, bw, bh))
            else:
                self._last_yunet_faces = None
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
    

    @staticmethod
    def padded_crop(image: np.ndarray, box: Tuple[int, int, int, int],
                    pad: float = 0.25) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Crop face with margin; returns (crop_bgr, clamped_box)."""
        h, w = image.shape[:2]
        x, y, bw, bh = box
        px = int(round(bw * pad))
        py = int(round(bh * pad))
        x0 = max(0, x - px)
        y0 = max(0, y - py)
        x1 = min(w, x + bw + px)
        y1 = min(h, y + bh + py)
        crop = image[y0:y1, x0:x1].copy()
        return crop, (x0, y0, x1 - x0, y1 - y0)

    def extract_embedding(self, image: np.ndarray, box: Tuple[int, int, int, int], landmarks: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Extract face embedding from image.
        
        Args:
            image: BGR image
            box: (x, y, w, h) face bounding box
            landmarks: Optional 5x2 landmarks array (required for insightface)
        
        Returns:
            Face embedding vector (normalized)
        """
        if self.backend == "insightface":
            return self._extract_insightface(image, box, landmarks)
        elif self.backend == "face_recognition":
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x, y, w, h = box
            face_location = (y, x + w, y + h, x)
            encodings = face_recognition.face_encodings(rgb, [face_location])
            return encodings[0] if encodings else None
        else:
            # opencv-dnn backend: prefer SFace if available, else pixel embeddings
            if getattr(self, "sface", None) is not None:
                face_row = None
                raw = getattr(self, "_last_yunet_faces", None)
                if raw is not None:
                    x, y, bw, bh = box
                    best_i, best_iou = None, -1.0
                    for i, f in enumerate(raw):
                        fx, fy, fw, fh = [float(v) for v in f[:4]]
                        # IoU with requested box
                        xa, ya = max(x, fx), max(y, fy)
                        xb, yb = min(x + bw, fx + fw), min(y + bh, fy + fh)
                        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
                        union = bw * bh + fw * fh - inter + 1e-6
                        iou = inter / union
                        if iou > best_iou:
                            best_iou, best_i = iou, i
                    if best_i is not None and best_iou > 0.1:
                        face_row = raw[best_i]
                if face_row is None:
                    # Synthetic YuNet-like row: box + empty landmarks + score
                    x, y, bw, bh = box
                    face_row = np.array(
                        [x, y, bw, bh, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0],
                        dtype=np.float32,
                    )
                try:
                    aligned = self.sface.alignCrop(image, face_row)
                    feat = self.sface.feature(aligned)
                    vec = np.asarray(feat, dtype=np.float32).reshape(-1)
                    n = float(np.linalg.norm(vec)) + 1e-8
                    return vec / n
                except Exception as e:
                    print(f"FaceRecognizer: SFace embed failed: {e}")
            
            # Fallback: weak pixel-based embeddings
            face, _ = self.padded_crop(image, box, pad=0.25)
            if face is None or face.size == 0:
                return None
            face_resized = cv2.resize(face, (128, 128))
            # L2-normalize so cosine match is stable across lighting
            vec = face_resized.flatten().astype(np.float32) / 255.0
            n = float(np.linalg.norm(vec)) + 1e-8
            return vec / n
    
    def _extract_insightface(self, image: np.ndarray, box: Tuple[int, int, int, int], landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Extract InsightFace embedding with landmark alignment.
        
        Args:
            image: BGR image
            box: (x, y, w, h) face box
            landmarks: 5x2 landmarks (required for proper alignment)
        
        Returns:
            512-D normalized embedding or None
        """
        if not hasattr(self, 'rec_session') or self.rec_session is None:
            print("Warning: InsightFace recognition model not loaded")
            return None
        
        # Align face using landmarks (critical for InsightFace quality)
        if landmarks is not None and landmarks.shape == (5, 2):
            aligned = _align_face_insightface(image, landmarks, image_size=(112, 112))
        else:
            # Fallback: padded crop + resize (worse quality)
            print("Warning: No landmarks provided, using padded crop (quality degradation)")
            face, _ = self.padded_crop(image, box, pad=0.2)
            if face is None or face.size == 0:
                return None
            aligned = cv2.resize(face, (112, 112))
        
        # Prepare input: (1, 3, 112, 112) CHW RGB normalized to [0, 1]
        blob = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = blob.astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        
        # Run inference
        input_name = self.rec_session.get_inputs()[0].name
        outputs = self.rec_session.run(None, {input_name: blob})
        embedding = outputs[0].flatten()
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def enroll(self, name: str, image: np.ndarray) -> int:
        """Enroll a person's face.
        
        Args:
            name: Person's display name
            image: BGR image containing face
        
        Returns:
            Number of faces enrolled (0 if none found)
        """
        # Use landmark detection for InsightFace
        if self.backend == "insightface":
            detections = self.detect_faces_with_landmarks(image)
            if not detections:
                print(f"No faces detected in image for {name}")
                return 0
            
            # Prefer largest face when multiple detections
            if len(detections) > 1:
                detections = [max(detections, key=lambda d: d[0][2] * d[0][3])]
            
            embeddings = []
            for box, landmarks in detections:
                emb = self.extract_embedding(image, box, landmarks)
                if emb is not None:
                    embeddings.append(emb)
        else:
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

        # Persist TIGHT face crops (with pad), not the full source photo.
        if self.backend == "insightface":
            for box, _ in detections:
                crop, _ = self.padded_crop(image, box, pad=0.25)
                if crop is None or crop.size == 0:
                    continue
                img_idx = len(list(person_dir.glob("*.jpg")))
                img_path = person_dir / f"{img_idx:03d}.jpg"
                cv2.imwrite(str(img_path), crop)
        else:
            for box in boxes:
                crop, _ = self.padded_crop(image, box, pad=0.25)
                if crop is None or crop.size == 0:
                    continue
                img_idx = len(list(person_dir.glob("*.jpg")))
                img_path = person_dir / f"{img_idx:03d}.jpg"
                cv2.imwrite(str(img_path), crop)

        print(f"Enrolled {len(embeddings)} face(s) for {name} (total: {len(self.db[name]['embeddings'])})")
        return len(embeddings)
    
    def recognize(self, image: np.ndarray, threshold: float = 0.6, margin: float = 0.05, 
                  log_scores: bool = False) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Recognize faces in image with stricter matching.
        
        Args:
            image: BGR image
            threshold: Recognition confidence threshold (cosine similarity, higher = stricter)
            margin: Second-best margin (best - second_best must exceed margin for positive ID)
            log_scores: Log id:confidence for debugging
        
        Returns:
            List of (name, confidence, box) tuples
        """
        # Use landmark detection for InsightFace
        if self.backend == "insightface":
            detections = self.detect_faces_with_landmarks(image)
        else:
            boxes = self.detect_faces(image)
            detections = [(box, None) for box in boxes]
        
        results = []
        
        for box, landmarks in detections:
            emb = self.extract_embedding(image, box, landmarks)
            if emb is None:
                continue
            
            best_name = "unknown"
            best_score = -1.0
            second_best_score = -1.0
            scores = {}  # name -> best_score
            
            for name, data in self.db.items():
                name_best = -1.0
                for stored_emb in data["embeddings"]:
                    if self.backend == "face_recognition":
                        # face_recognition uses Euclidean distance
                        distance = np.linalg.norm(emb - stored_emb)
                        score = max(0.0, 1.0 - distance)
                    else:
                        # InsightFace and others use cosine similarity
                        score = float(np.dot(emb, stored_emb) / (
                            np.linalg.norm(emb) * np.linalg.norm(stored_emb) + 1e-8
                        ))
                    
                    if score > name_best:
                        name_best = score
                
                scores[name] = name_best
                
                if name_best > best_score:
                    second_best_score = best_score
                    best_score = name_best
                    best_name = name
                elif name_best > second_best_score:
                    second_best_score = name_best
            
            # Apply threshold + margin criteria
            confidence = best_score
            margin_check = (best_score - second_best_score) >= margin if second_best_score > 0 else True
            
            if confidence >= threshold and margin_check:
                identified_name = best_name
            else:
                identified_name = "unknown"
            
            if log_scores:
                print(f"Face recognition scores: {identified_name}:{confidence:.3f} (margin={best_score - second_best_score:.3f})")
                for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"  {name}: {score:.3f}")
            
            results.append((identified_name, confidence, box))
        
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
