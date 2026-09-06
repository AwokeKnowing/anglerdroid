# Face Recognition & Conversation System

Conversational face recognition for Kevin with enrollment, greetings, and persistent storage.

⚠️ **OFFLINE-SAFE**: This system does NOT control robot driving. Hardware motion stays stopped after the couch crash.

## Features

- **Face enrollment**: Store known faces with display names
- **Face recognition**: Identify people from camera or images
- **Persistent storage**: Embeddings saved to `~/.kevin/faces/`
- **Conversational greetings**: Contextual greetings based on time of day
- **Unknown handling**: Ask for names and enroll new people
- **TTS integration**: Soft volume (10%) using existing speech_io pattern
- **Multiple backends**: face_recognition (dlib), OpenCV DNN, or Haar cascades

## Installation

### Minimal (OpenCV fallback)
```bash
pip install numpy opencv-python pillow
```

### Recommended (best quality)
```bash
pip install numpy opencv-python pillow face_recognition
```

Note: `face_recognition` requires dlib, which may need build tools. If installation fails, the system falls back to OpenCV automatically.

### Optional (TTS)
```bash
# For espeak TTS (Linux only)
sudo apt-get install espeak

# Kokoro/Gemini TTS use existing ui.py integration
```

## Quick Start

### 1. Enroll Faces

```bash
# Enroll from single image
python -m faces.cli enroll "John Doe" photo.jpg

# Enroll from multiple images (better accuracy)
python -m faces.cli enroll "Jane Smith" photo1.jpg photo2.jpg photo3.jpg

# List enrolled people
python -m faces.cli list
```

### 2. Recognize Faces

```bash
# Recognize from image
python -m faces.cli recognize photo.jpg

# Show annotated image
python -m faces.cli recognize photo.jpg --show

# Adjust threshold (lower = stricter)
python -m faces.cli recognize photo.jpg --threshold 0.5
```

### 3. Live Demo

```bash
# Webcam with conversation
python -m faces.demo webcam

# Process directory of images
python -m faces.demo images /path/to/photos/

# With espeak TTS
python -m faces.demo webcam --tts espeak --volume 0.1
```

## CLI Reference

### Enrollment
```bash
python -m faces.cli enroll "Name" image1.jpg [image2.jpg ...]
```
Enrolls a person from one or more images. Multiple images improve accuracy.

### Recognition
```bash
python -m faces.cli recognize image.jpg [--threshold 0.6] [--show]
```
Recognizes faces in an image. Use `--show` to display annotated image.

### List People
```bash
python -m faces.cli list
```
Shows all enrolled people and face counts.

### Remove Person
```bash
python -m faces.cli remove "Name"
```
Removes a person from the database.

### Webcam Demo
```bash
python -m faces.demo webcam [--camera 0] [--tts stub] [--volume 0.1]
```
Live webcam demo with conversation. Press 'e' to enroll, 'q' to quit.

## Storage Structure

```
~/.kevin/faces/
├── database.pkl              # Face embeddings and metadata
├── john_doe/                 # Per-person directories
│   ├── 000.jpg              # Enrollment images
│   ├── 001.jpg
│   └── 002.jpg
├── jane_smith/
│   └── 000.jpg
└── models/                   # Optional OpenCV DNN models
    ├── deploy.prototxt
    └── res10_300x300_ssd_iter_140000.caffemodel
```

For sim demos, use `--gallery faces/gallery/` to store in the project directory.

## Backend Selection

The system auto-selects the best available backend:

1. **face_recognition (dlib)** - Best quality, requires dlib
   - 128-dim face embeddings
   - Robust to lighting and pose
   - Requires build tools for installation

2. **OpenCV DNN** - Good quality, pure Python
   - ResNet-based face detection
   - Simple cosine similarity matching
   - Requires DNN model files (downloaded separately)

3. **OpenCV Haar cascades** - Minimal fallback
   - Built into OpenCV
   - Less accurate, good for testing
   - No external downloads required

Status check:
```python
from faces import FACE_RECOGNITION_AVAILABLE, OPENCV_AVAILABLE, BACKEND
print(f"Backend: {BACKEND}")
```

## Conversation Flow

### Greeting Known People

When a known face is detected:
1. Check cooldown (default: 5 minutes since last greeting)
2. Select contextual greeting based on time of day:
   - **Morning (5am-12pm)**: "Good morning, {name}!"
   - **Afternoon (12pm-6pm)**: "Hi {name}!"
   - **Evening (6pm-5am)**: "Good evening, {name}!"
3. Speak greeting at 10% volume
4. Log interaction

### Handling Unknown Faces

When an unknown face is detected:
1. Check cooldown (avoid spamming)
2. Speak prompt: "Hi! I don't think we've met. What's your name?"
3. Wait for user input (manual enrollment via CLI/demo)
4. Enroll with provided name
5. Confirm: "Nice to meet you, {name}! I'll remember you."

### TTS Integration

The system uses the existing speech_io pattern from `ui.py`:
- **Stub mode**: Prints to console (offline demos)
- **espeak mode**: System TTS (Linux)
- **Kokoro mode**: High-quality ONNX TTS (requires brain_zmq)
- **Gemini mode**: Cloud TTS (requires ui.py integration)

Volume is soft (10% default) to avoid startling people.

## Integration with House-Bot (Future)

The face system is designed to integrate with house-bot autonomy:

```
┌─────────────┐
│ Camera Feed │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Face Recog │  ← faces.recognizer
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Perception  │  ← Name + location → semantic map
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Utterance   │  ← faces.conversation → TTS
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Local Executive│ ← Goals: "approach John", "greet Jane"
└─────────────┘
```

### Example Integration

```python
from faces.recognizer import FaceRecognizer
from faces.conversation import ConversationManager

# In main.py perception loop
recognizer = FaceRecognizer()
conversation = ConversationManager(recognizer, speak_fn=tools.speak)

# Every frame
results = conversation.process_frame(camera_frame)

for greeting in results["greetings"]:
    # Already spoken by conversation manager
    # Optionally: update semantic map with person location
    name = greeting["name"]
    x, y, w, h = greeting["box"]
    # semantic_map.add_person(name, (x, y))

for unknown in results["unknowns"]:
    # Prompt already spoken
    # Could trigger manual enrollment via UI
    pass
```

### Semantic Map Extension

Future house-bot could maintain person locations:
```python
# In navigator.py or similar
class SemanticMap:
    def add_person(self, name: str, bbox: tuple, frame_pose: tuple):
        """Add person to map with world coordinates."""
        world_pos = self.bbox_to_world(bbox, frame_pose)
        self.people[name] = {
            "position": world_pos,
            "last_seen": time.time(),
            "confidence": 0.8
        }
    
    def get_approach_goal(self, name: str) -> tuple:
        """Generate goal position to approach person."""
        if name in self.people:
            px, py = self.people[name]["position"]
            # Stand 1.5m in front
            return (px - 1.5, py)
        return None
```

### LocalExecutive Goals

Face recognition could generate high-level goals:
```python
# In local_executive.py
if known_person_detected:
    goal = {
        "type": "greet_person",
        "name": person_name,
        "position": person_location,
        "priority": "high"
    }
    self.goal_queue.append(goal)
```

## Safety Notes

✅ **OFFLINE-SAFE**: No robot motion commands  
✅ **NO live driving**: Hardware stays stopped  
✅ **NO auto-navigation**: Face system only provides perception  
✅ **Soft TTS**: 10% volume, non-intrusive  
✅ **Privacy**: Local storage, no cloud uploads (except Gemini TTS if enabled)

The face system is **read-only** from a robot control perspective. It provides:
- Perception data (who is where)
- Conversational output (greetings, prompts)
- NOT motion commands or navigation goals

Robot driving will remain stopped until policies are proven in `sim/`.

## Testing

### Unit Tests
```bash
python -m faces.test
```

### Integration Tests
```bash
# Enroll test faces
python -m faces.cli enroll "Test User" test_image.jpg

# Verify recognition
python -m faces.cli recognize test_image.jpg

# Clean up
python -m faces.cli remove "Test User"
```

### Demo Test
```bash
# Process sample images
mkdir -p test_photos
# Add some photos with faces
python -m faces.demo images test_photos/
```

## Performance

- **Enrollment**: ~100-500ms per face (dlib), ~50-200ms (OpenCV)
- **Recognition**: ~50-300ms per frame depending on backend
- **Memory**: ~1KB per enrolled face embedding
- **Storage**: ~50-200KB per person (embeddings + images)

Runs efficiently on Jetson Orin NX:
- face_recognition: May need dlib optimization
- OpenCV DNN: Good performance with GPU
- Haar cascades: Very fast, CPU only

## Troubleshooting

### face_recognition won't install
```bash
# Fall back to OpenCV (still good quality)
pip install opencv-python

# System will auto-detect and use OpenCV backend
```

### No faces detected
- Ensure good lighting
- Face should be frontal, not too small
- Try adjusting `--threshold` lower (e.g. 0.4)
- Check that camera/image is not corrupted

### False recognitions
- Increase `--threshold` (e.g. 0.7 or 0.8)
- Enroll more images per person (3-5 recommended)
- Use face_recognition backend for better accuracy

### Webcam not opening
```bash
# Try different camera index
python -m faces.demo webcam --camera 1

# Check available cameras
ls /dev/video*
```

## Future Enhancements

- [ ] Multi-camera fusion (RealSense RGB + webcam)
- [ ] Age/emotion estimation
- [ ] Track person ID across frames (re-identification)
- [ ] Integrate with semantic SLAM map
- [ ] Voice enrollment ("Hi, I'm John")
- [ ] Face clustering (find all photos of same person)

## Dependencies

**Required**:
- `numpy` - Array operations

**Recommended**:
- `opencv-python` - Face detection and image I/O
- `pillow` - Image loading fallback
- `face_recognition` - High-quality face embeddings (optional)

**Optional**:
- `espeak` - System TTS (Linux)

Compatible with Python 3.8+, tested on Ubuntu 20.04/22.04 and Jetson Orin NX.
