# Bug Fixes & Integration Complete - PR #7

## Summary

Fixed critical confidence threshold bug and completed integration into `people_live.py`. All issues from user feedback addressed.

## 🔴 CRITICAL BUG FIXED: Confidence Threshold Semantics

### The Problem
Original implementation compared **raw cosine similarity** directly to 0.90:
```python
# WRONG (would trigger enrollment on almost every face)
if confidence < 0.90:  # But ArcFace cosine is ~0.45-0.75 for known!
    trigger_enrollment()
```

**Why this was broken**:
- ArcFace cosine similarity for known matches: **0.45–0.75** (typical range)
- Rarely exceeds 0.90 in practice
- Would trigger enrollment for essentially **every** face, even known people

### The Fix
Enrollment now triggers ONLY when `recognize()` returns `"unknown"`:
```python
# CORRECT (triggers only when recognition rejects)
if name == "unknown":  # Failed threshold + margin checks
    trigger_enrollment()
```

**How acceptance works**:
- `recognize(threshold=0.60, margin=0.05)` returns:
  - `name="Alice"` → **ACCEPTED** (cosine ≥ 0.60 AND margin ≥ 0.05)
  - `name="unknown"` → **REJECTED** (cosine < 0.60 OR margin < 0.05)
- Only rejected faces trigger enrollment

### Files Changed
- `faces/live_enrollment.py` - Removed `confidence_threshold` param, changed `should_enroll()` to check `name == "unknown"`
- `faces/people_behavior.py` - Removed `enrollment_confidence_threshold` param, updated `on_face_seen()`
- `faces/test_live_enrollment.py` - Fixed all 7 tests to use unknown→enroll vs accepted→greet semantics

## ✅ Integration into `people_live.py`

### What Was Wired
**Before**: `people_live.py` only fed known faces to SocialFSM; unknowns silently ignored  
**After**: Complete live enrollment flow integrated

### Changes to `src/people_live.py`

**1. Initialization (`_ensure`)**:
```python
# Try InsightFace first (auto-falls back if models missing)
rec = FaceRecognizer(backend="auto", model_pack="buffalo_l")

# Create behavior with enrollment enabled
self._pb = create_people_behavior(
    speak_fn=_speak,
    recognizer=rec,
    enable_live_enrollment=True,  # NEW
)
```

**2. Face detection (`_tick_faces`)**:
```python
# Recognize with threshold + margin (returns "unknown" when rejected)
faces = self._cm.recognizer.recognize(img, threshold=0.60, margin=0.05)

# Handle known faces via SocialFSM (existing)
for name, confidence, box in faces:
    if name != "unknown":
        action = self._fsm.on_face_seen(name, confidence, box, now, depth_map)
        # Apply FSM actions (greet/approach/leave)

# Handle unknown faces with live enrollment (NEW)
detections = self._cm.recognizer.detect_faces_with_landmarks(img)
for name, confidence, box in faces:
    # Get landmarks for InsightFace alignment
    landmarks = match_landmarks(box, detections)
    
    # Feed to enrollment manager
    action = self._pb.on_face_seen(
        name=name,  # "unknown" triggers enrollment
        confidence=confidence,
        box=box,
        image=img,  # Pass for sample collection
        landmarks=landmarks,
        now=now,
        speak=True
    )
```

**3. ASR handling (`_tick_listen`)**:
```python
# Check if enrollment session is waiting for name (NEW)
if self._pb.enrollment_manager.is_session_active():
    session = self._pb.enrollment_manager.active_session
    if session.person_name is None:
        # Feed transcript to enrollment manager
        action = self._pb.on_transcript(text, now=now, language="en")
        # Handles name responses: "My name is X", "I'm X", etc.
        return

# Existing: wake/commands/empathy
action = self._pb.on_transcript(text, now=now, language="en")
```

## 📦 Model Download Helper

Created `download_models.py`:
- Check for existing YuNet + buffalo models
- Download YuNet automatically from OpenCV Zoo
- Show manual download instructions for buffalo models
- Verify SHA256 if provided
- `chmod +x` for easy execution

**Usage**:
```bash
./download_models.py --model-pack buffalo_l
```

## 🛡️ Dimension Mismatch Protection

Updated `faces/rebuild_embeddings.py`:
- Check existing database for embedding dimension before rebuild
- Warn if dimensions mismatch (e.g., SFace 128-D → InsightFace 512-D)
- Refuse to proceed without backup if mismatch detected
- Report embedding dimension after rebuild

**Why critical**: Mixing 128-D and 512-D embeddings causes recognition failures

## ✅ All Tests Pass

**Main tests** (`python3 -m faces.test`):
- 18 tests, all pass with graceful dependency skipping

**Live enrollment tests** (`python3 -m faces.test_live_enrollment`):
- 7 tests, all pass:
  1. ✅ Unknown face → enrollment prompt
  2. ✅ Rejected face → treat as unknown (NEW semantics)
  3. ✅ Name collection from ASR
  4. ✅ Sample collection (3-5 crops)
  5. ✅ Timeout after 15s → polite leave
  6. ✅ Cooldown respects 5-minute window
  7. ✅ Accepted face → greet normally (NEW semantics)

## 🚀 Verification on Device (User Next Steps)

### 1. Download Models
```bash
cd /workspace
./download_models.py --model-pack buffalo_l

# Manual: Download w600k_r50.onnx from InsightFace model zoo
# Place in ~/.kevin/faces/models/w600k_r50.onnx
```

### 2. Rebuild Embeddings
```bash
# CRITICAL: Old SFace 128-D embeddings incompatible with InsightFace 512-D
python -m faces.rebuild_embeddings --backend insightface --model-pack buffalo_l

# Expected output:
# - Dimension check (warns if mixing 128-D and 512-D)
# - Backs up old database.pkl
# - Re-embeds all gallery crops with 512-D
# - Reports: "Embedding dimension: 512-D"
```

### 3. Test Offline (No Hardware)
```bash
# Test recognition with webcam
python -m faces.cli webcam --backend insightface --threshold 0.60 --margin 0.05

# Should see:
# - Known faces: name + 0.6-0.8 cosine
# - Unknown faces: "unknown" + lower cosine
```

### 4. Test on Live Robot (Drive DISARMED)
```bash
# Start people_live
# It will auto-detect InsightFace if models present
# Watch logs for:

# Known face (accepted):
people_live: tick#X faces=1 fsm_actions=1 enrollment_actions=0 unknowns=0
people_live: greet action=greet person=James

# Unknown face (rejected):
people_live: tick#X faces=1 fsm_actions=0 enrollment_actions=1 unknowns=1
people_live: enrollment enrollment_prompt for box=(...)
🔊 [people]: Hi! I don't think we've met. What's your name?

# ASR name response:
people_live: heard#Y kind=enrollment_name_received utter='Thanks, Bob!...'
people_live: enrollment enrollment_collecting for box=(...)
# (Collects 3-5 samples over ~2s)
people_live: enrollment enrollment_complete for box=(...)
🔊 [people]: Nice to meet you, Bob! I'll remember you.

# Timeout (no response after 15s):
people_live: enrollment enrollment_timeout for box=(...)
🔊 [people]: No worries! Let me know if you need anything.
```

### 5. Verify Thresholds
Check `recognize()` scores in logs:
```
Face recognition scores: james:0.782 (margin=0.215)
  james: 0.782  ← Should be 0.6-0.8 for known
  david: 0.567  ← Second-best
  unknown: 0.412
```

**If too many unknowns** (false rejects):
- Lower threshold: `--threshold 0.55`
- Lower margin: `--margin 0.03`

**If false IDs** (false accepts):
- Raise threshold: `--threshold 0.65`
- Raise margin: `--margin 0.08`

## 📊 Confidence Semantics (Final)

| Layer | Value | Interpretation |
|-------|-------|----------------|
| **Raw cosine** | 0.45–0.75 | Typical ArcFace match |
| **recognize() result** | `name="Alice"` | **ACCEPTED** (≥0.60 cosine + ≥0.05 margin) |
| **recognize() result** | `name="unknown"` | **REJECTED** (failed threshold or margin) |
| **User-facing** | "~90% certainty" | Means accepted by threshold+margin |
| **Enrollment trigger** | `name=="unknown"` | ONLY when recognition rejects |

**NEVER** compare raw cosine to 0.90.

## ⚠️ Important Notes

1. **Drive remains DISARMED**: Enrollment is speech + camera only
2. **No face images in git**: `.gitignore` enforced
3. **Dimension mismatch**: Old SFace database MUST be rebuilt
4. **Backend auto-selection**: `backend="auto"` tries InsightFace → face_recognition → opencv-dnn
5. **SocialFSM unchanged**: Known-person greets still via FSM (no double-speak)

## 📂 Files Changed (This PR)

**Bugs fixed**:
- `faces/live_enrollment.py` - Removed confidence_threshold, fixed should_enroll()
- `faces/people_behavior.py` - Removed enrollment_confidence_threshold, updated on_face_seen()
- `faces/test_live_enrollment.py` - Fixed tests for unknown→enroll semantics

**Integration**:
- `src/people_live.py` - Wired enrollment into _tick_faces + _tick_listen

**Tools**:
- `download_models.py` - Model download helper (NEW)
- `faces/rebuild_embeddings.py` - Added dimension mismatch protection

**Documentation**:
- `docs/INSIGHTFACE_UPGRADE.md` - Updated confidence semantics

## ✅ Success Criteria (All Met)

1. ✅ Enrollment triggers on `name=="unknown"` only (not raw cosine)
2. ✅ Accepted IDs greet via SocialFSM as before
3. ✅ `people_live.py` wired with full enrollment flow
4. ✅ Drive never armed by this change (speech + camera only)
5. ✅ Model download helper with instructions
6. ✅ Dimension mismatch protection in rebuild script
7. ✅ All tests pass
8. ✅ Documentation updated

---

**PR**: [#7](https://github.com/AwokeKnowing/anglerdroid/pull/7) (draft)  
**Branch**: `cursor/upgrade-insightface-embeddings-b8e3`  
**Commits**: 8 total (~3,000 lines)  
**Status**: ✅ Ready for device verification (drive remains DISARMED)

**Next**: User downloads models, rebuilds embeddings, tests on live robot.
