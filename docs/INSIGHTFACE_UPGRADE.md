# InsightFace Upgrade: Model Selection and Performance

## Summary

Upgraded Kevin face recognition from OpenCV pixel embeddings to InsightFace ArcFace-class embeddings (buffalo_l/buffalo_s) for SOTA quality at conversational distance.

## Model Selection

### Buffalo_l (Recommended for Orin NX)
- **Architecture**: ResNet-50 (w600k_r50)
- **Embedding dimension**: 512-D
- **Training**: WebFace600K dataset (~600K identities)
- **Quality**: SOTA, excellent separation even at hard poses
- **Expected Orin NX latency**: ~15-25ms per face @ FP16 CUDA
- **Use case**: Primary recognition for live social interaction

### Buffalo_s (Mobile fallback)
- **Architecture**: MobileFaceNet
- **Embedding dimension**: 512-D  
- **Training**: WebFace600K dataset
- **Quality**: Good, mobile-optimized
- **Expected Orin NX latency**: ~8-12ms per face @ FP16 CUDA
- **Use case**: If buffalo_l exceeds latency budget (~2 Hz social tick)

## Why InsightFace over SFace?

### Problem with SFace 2021dec (128-D)
- User (James) reported **0.4–0.5 cosine** similarity vs gallery
- False IDs occurred on weak pixel embeddings
- Poor separation between household members at conversational distance
- Training data: smaller, older (2021)

### InsightFace buffalo_* advantages
1. **Higher capacity**: 512-D vs 128-D
2. **Larger training set**: WebFace600K vs SFace's smaller corpus
3. **Better landmark alignment**: 5-point affine transform to canonical pose
4. **Proven SOTA**: Used in production face recognition systems
5. **Active maintenance**: Regular updates, broad model zoo

## Architecture

### Detection Pipeline
```
RGB Frame (320x240 upsampled 2x)
    ↓
YuNet ONNX Detector (kept: proven reliable at low res)
    ↓
5-point landmarks (eyes, nose, mouth corners)
    ↓
Affine alignment → 112x112 canonical face
    ↓
buffalo_l ResNet-50 ONNX → 512-D embedding (L2-normalized)
```

### Recognition Pipeline
```
Query embedding (512-D)
    ↓
Cosine similarity vs gallery embeddings
    ↓
Threshold check (e.g., 0.60 for positive ID)
    ↓
Margin check (best - second_best >= 0.05)
    ↓
Log id:confidence for debugging
```

## Latency Budget (Orin NX 16GB)

### Target: ~2 Hz social tick (~500ms per frame)

**Budget breakdown**:
- Frame capture: ~10ms
- YuNet detection (1-2 faces): ~15-20ms
- Buffalo_l embedding (per face): ~20ms × 2 = 40ms
- Gallery match (5 people, 3-5 samples each): ~2ms
- Overhead (alignment, logging): ~5ms
- **Total**: ~72ms per frame (13.8 Hz) ✅

**Comfortable headroom** for 2 Hz social behavior tick.

### If buffalo_l too slow
Fall back to buffalo_s:
- Embedding: ~10ms × 2 = 20ms
- **Total**: ~52ms per frame (19.2 Hz) ✅

## Quality Expectations

### Gallery Setup
- Household: 5 people (James, Erika, David, Nohemi, Karina)
- Gallery source: Cropped phone photos under `~/.kevin/faces/<name>/`
- Live domain: 320x240 RGB webcam at 1-2m conversational distance

### Expected Performance (buffalo_l)
- **Positive ID threshold**: cosine >= 0.60 (stricter than SFace 0.38)
- **Margin requirement**: best - second_best >= 0.05
- **Target accuracy**: >95% true positive at conversational distance
- **False accept rate (FAR)**: <1% with margin check

### Domain Gap Mitigation
**Problem**: Gallery = phone crops, Live = 320x240 webcam  
**Solution**: `enroll_live.py` - capture additional samples from live webcam
- Add 3-5 live crops per person to gallery
- Matches robot's actual viewing conditions
- Reduces domain gap significantly

## Stricter Matching Logic

### Why stricter than SFace?

SFace used **threshold-only** matching:
```python
if confidence >= 0.38:  # Too loose
    return name
```

### New InsightFace matching (two-stage):
```python
# Stage 1: Threshold check
if confidence < 0.60:
    return "unknown"

# Stage 2: Margin check (disambiguation)
if (best_score - second_best_score) < 0.05:
    return "unknown"  # Too close to call

return best_name
```

### Benefits
1. Prevents false IDs when two people are similar
2. Requires confident separation, not just "best match"
3. Logs scores for debugging: `id:conf (margin=X.XXX)`

## Migration Path

### 1. Install dependencies
```bash
pip install onnxruntime-gpu opencv-python numpy
```

### 2. Download models
```bash
mkdir -p ~/.kevin/faces/models

# YuNet detector (already have this)
# https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# Buffalo_l recognition model
# Download w600k_r50.onnx from InsightFace model zoo
# Place in ~/.kevin/faces/models/w600k_r50.onnx
```

**InsightFace model sources**:
- Official: [https://github.com/deepinsight/insightface/tree/master/model_zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- Buffalo models: Look for buffalo_l or buffalo_s packs
- Extract `w600k_r50.onnx` (recognition model) to models directory

### 3. Rebuild embeddings from gallery
```bash
# Backup old database
cp ~/.kevin/faces/database.pkl ~/.kevin/faces/database.pkl.sface_backup

# Rebuild with InsightFace buffalo_l
python -m faces.rebuild_embeddings --backend insightface --model-pack buffalo_l

# Verify
python -m faces.cli list
```

### 4. (Optional) Add live crops for domain gap
```bash
python -m faces.enroll_live --name "James" --samples 3 --show-scores
python -m faces.enroll_live --name "Erika" --samples 3 --show-scores
# ... repeat for each person
```

### 5. Test recognition
```bash
# Use webcam to verify quality improvement
python -m faces.cli webcam --threshold 0.60

# Should see:
# - Higher confidence scores (0.7-0.9 for positive IDs)
# - Clearer separation between people
# - Fewer "unknown" false positives
```

## Monitoring and Debugging

### Enable score logging in people_behavior.py
```python
recognizer = FaceRecognizer(backend="insightface", model_pack="buffalo_l")
results = recognizer.recognize(frame, threshold=0.60, margin=0.05, log_scores=True)
```

### Output example
```
Face recognition scores: james:0.782 (margin=0.215)
  james: 0.782
  david: 0.567
  unknown: 0.412
```

### Key metrics to watch
1. **Confidence scores**: Should be 0.7-0.9 for household members
2. **Margin**: Should be >0.1 for confident IDs
3. **Latency**: <25ms per face on Orin NX
4. **False accepts**: Should be rare with margin check

## Hypothesis Validation

**Original hypothesis** (from user):
> "Live gap is domain (phone crop gallery vs 320x240 robot cam) + SFace capacity; buffalo_* + live enroll crops will fix."

**Validation approach**:
1. ✅ Upgrade to buffalo_l (higher capacity: 512-D vs 128-D)
2. ✅ Landmark alignment (proper pose normalization)
3. ✅ Stricter matching (threshold + margin)
4. ✅ Live enroll tool (`enroll_live.py`) for domain gap mitigation
5. 🔄 Test on live robot at conversational distance (user to verify)

**Expected outcome**: Cosine similarity 0.7-0.9 (up from 0.4-0.5), no false IDs with margin check.

## Safety Notes

- ✅ All models run offline (no cloud)
- ✅ No face images committed to repo (.gitignore enforced)
- ✅ Gallery remains at `~/.kevin/faces/` (outside repo)
- ✅ Tests use synthetic embeddings only (no household photos)
- ✅ Live enrollment is opt-in (manual capture)

## Files Changed

- `faces/recognizer.py` - Added InsightFace backend, landmark alignment, stricter matching
- `faces/rebuild_embeddings.py` - Script to rebuild database with new backend (NEW)
- `faces/enroll_live.py` - Live webcam enrollment for domain gap (NEW)
- `faces/live_enrollment.py` - Interactive enrollment manager for unknown faces (NEW)
- `faces/people_behavior.py` - Integrated live enrollment flow (UPDATED)
- `faces/test.py` - Added InsightFace tests with synthetic embeddings
- `faces/test_live_enrollment.py` - Tests for interactive enrollment (NEW)
- `src/requirements.txt` - Added onnxruntime-gpu
- `docs/INSIGHTFACE_UPGRADE.md` - This document (NEW)

## Interactive Live Enrollment

**NEW**: Kevin now handles unknown faces gracefully with interactive enrollment.

### Workflow

1. **Face detected** with low confidence (<90% user-facing threshold)
2. **Polite prompt**: "Hi! I don't think we've met. What's your name?"
3. **Name collection** from ASR (English/Spanish auto-detect)
4. **Sample collection**: 3-5 live face crops from robot webcam
5. **Enrollment**: Add to gallery with InsightFace embeddings
6. **Confirmation**: "Nice to meet you, {name}!"
7. **Cooldown**: 5-minute cooldown before re-prompting

### Kevin Personality
- **Curious**: Actively wants to learn who you are
- **Calm**: Non-pushy, polite timeout if no response
- **Clean humor**: Friendly but professional
- **Bilingual**: Starts English, switches to Spanish if detected

### Confidence Mapping

**User-facing "90%" bar** maps to internal cosine threshold with margin:

| User confidence | Internal threshold | Margin | Action |
|-----------------|-------------------|--------|--------|
| >= 90% | cosine >= 0.60 | + 0.05 margin | Greet normally |
| < 90% | cosine < 0.60 OR margin fail | - | Prompt for enrollment |

This keeps false IDs rare (<1%) while still enrolling uncertain faces.

### Timeout Behavior

If no name is provided within 15 seconds:
- Kevin politely leaves: "No worries! Let me know if you need anything."
- No enrollment occurs
- Cooldown starts (won't re-prompt for 5 minutes)

### Safety & Privacy

- ✅ **Speech + camera only** (no driving required)
- ✅ **Live samples** from robot webcam (matches viewing conditions)
- ✅ **Never commits** face images to git (.gitignore enforced)
- ✅ **User-initiated** (only happens when face is seen, not proactive)
- ✅ **Cooldown protected** (won't spam re-prompts)

### Integration

Wire into your perception loop:

```python
from faces.recognizer import FaceRecognizer
from faces.people_behavior import create_people_behavior

# Setup
recognizer = FaceRecognizer(backend="insightface", model_pack="buffalo_l")
behavior = create_people_behavior(
    speak_fn=your_speak_function,
    recognizer=recognizer,
    enable_live_enrollment=True,
    enrollment_confidence_threshold=0.90  # User-facing "90%" bar
)

# In perception loop
results = recognizer.recognize(frame, threshold=0.60, margin=0.05, log_scores=True)

for name, confidence, box in results:
    # Handle face with enrollment integration
    action = behavior.on_face_seen(
        name=name,
        confidence=confidence,
        box=box,
        image=frame,  # Pass frame for sample collection
        landmarks=landmarks if available else None,
        speak=True
    )
    
    if action:
        print(f"Action: {action.kind} - {action.utterance}")

# In ASR callback
def on_speech(transcript, language="en"):
    action = behavior.on_transcript(
        transcript,
        language=language,
        speak=True
    )
    if action:
        print(f"Action: {action.kind} - {action.utterance}")
```

### Testing

Run live enrollment tests:
```bash
python3 -m faces.test_live_enrollment

# Tests:
# - Unknown face → prompt for name
# - Low-confidence known → treat as unknown
# - Name collection from ASR
# - Sample collection (3-5 crops)
# - Timeout after 15s
# - Cooldown respects 5-minute window
# - High confidence (≥90%) skips enrollment
```

All tests use mocked speech/images (no real household photos).

## Files Changed

## Performance Targets

| Metric | SFace (before) | InsightFace buffalo_l (after) |
|--------|----------------|-------------------------------|
| Embedding dim | 128-D | 512-D |
| Cosine similarity | 0.4-0.5 | 0.7-0.9 (target) |
| Threshold | 0.38 | 0.60 |
| Margin check | None | 0.05 |
| Latency (Orin) | ~10ms | ~20ms |
| False ID rate | Frequent | <1% (with margin) |

## Next Steps (User)

1. Download buffalo_l ONNX model (`w600k_r50.onnx`)
2. Place in `~/.kevin/faces/models/`
3. Run rebuild script
4. Test on live robot at conversational distance
5. Use `enroll_live.py` if domain gap persists
6. Report confidence scores and false ID rate

---

**Status**: Implementation complete, pending model download + user verification on live hardware.
