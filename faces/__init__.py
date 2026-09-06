"""Face recognition and enrollment system for Kevin.

Supports face_recognition (dlib-based) if available, otherwise falls back
to OpenCV DNN face detection with simple feature extraction.

Optional dependencies:
    pip install opencv-python face_recognition pillow

Storage: ~/.kevin/faces/ (or faces/gallery/ for sim demos)
"""

__version__ = "0.1.0"

try:
    import face_recognition as _fr
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    import cv2 as _cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Report backend availability
if FACE_RECOGNITION_AVAILABLE:
    BACKEND = "face_recognition (dlib)"
elif OPENCV_AVAILABLE:
    BACKEND = "opencv-dnn"
else:
    BACKEND = "none (install opencv-python or face_recognition)"


# People-behavior stubs (offline; do not arm on hardware from here)
from faces.people_behavior import (  # noqa: E402
    GreetHours,
    PeopleAction,
    PeopleBehaviorStub,
    create_people_behavior,
)
