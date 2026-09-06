# Face Gallery Privacy Protection

## Critical Privacy Constraint

**NEVER commit face gallery images or embeddings to the public anglerdroid repository.**

All face recognition data lives **only on Kevin's device** at `~/.kevin/faces` (and backup at `~/.kevin/faces_fullphoto_backup`). This data is personal and private to the household.

---

## Gallery Storage Locations (On Device Only)

### Primary Gallery
**Path**: `~/.kevin/faces/`

**Contents**:
- `database.pkl` — face embeddings database (NEVER commit)
- `{person_name}/` — directories per person containing face crops (NEVER commit)
- `models/` — YuNet face detection model (safe to commit if needed, but already in repo references)

### Full Photo Backup
**Path**: `~/.kevin/faces_fullphoto_backup/`

**Contents**:
- Original uncropped enrollment photos before face extraction
- NEVER commit these files

### Incoming Enrollment
**Path**: `~/gallery_incoming/`

**Contents**:
- Temporary storage for enrollment photos before processing
- NEVER commit these files

---

## Code Paths (Safe)

All code uses `os.path.expanduser("~/.kevin/faces")` which expands to the user's home directory on Kevin's device. **No relative paths to repo-local gallery directories.**

### Verified Safe Paths

✅ **`faces/recognizer.py`**:
```python
if gallery_path is None:
    gallery_path = os.path.expanduser("~/.kevin/faces")
```

✅ **`faces/cli.py`**, **`faces/demo.py`**:
```python
parser.add_argument("--gallery", default=None,
                   help="Gallery path (default: ~/.kevin/faces)")
```

✅ **`src/people_live.py`**:
```python
# Uses ~/.kevin/faces gallery (comment only; code imports FaceRecognizer with default)
```

**No repo-relative paths** like `faces/gallery/` or `./gallery_incoming/` in production code.

---

## .gitignore Protection (Comprehensive)

The `.gitignore` includes comprehensive patterns to prevent accidental commits:

```gitignore
# Face gallery lives only on Kevin at ~/.kevin/faces (not in repo)
.kevin/faces/
.kevin/faces_fullphoto_backup/
gallery_incoming/

# Face-related files anywhere in repo (safety net)
**/faces/**/*.jpg
**/faces/**/*.jpeg
**/faces/**/*.png
**/faces/**/*.pkl
**/faces/**/database.pkl
**/gallery_incoming/**/*.jpg

# Enrollment dumps (any .jpg in root or subdirs)
/*.jpg
/*.jpeg
/*.png
/enroll_*.jpg
/face_*.jpg
/crop_*.jpg

# Face embeddings database
database.pkl
**/database.pkl
```

---

## What IS Safe to Commit

✅ **Code only**:
- `faces/*.py` — Python modules (recognizer, conversation, people_behavior, CLI, tests)
- `src/social_fsm.py` — social FSM logic
- `src/people_live.py` — integration layer
- `docs/*.md` — documentation

✅ **Test fixtures** (if needed):
- Generated/synthetic test images (e.g., solid color squares, patterns)
- **NEVER real person photos**

✅ **Models** (optional, if auto-download fails):
- Public pre-trained models like YuNet ONNX (face detection, no identity data)
- These are publicly available and contain no household identity information

---

## What is NEVER Safe to Commit

❌ **Gallery images**: `~/.kevin/faces/{person_name}/*.jpg`  
❌ **Embeddings database**: `~/.kevin/faces/database.pkl`  
❌ **Full photo backup**: `~/.kevin/faces_fullphoto_backup/*.jpg`  
❌ **Incoming enrollment**: `~/gallery_incoming/*.jpg`  
❌ **Any real person photos** from the household

---

## Verification Checklist

Before every commit to the public anglerdroid repo:

1. ✅ Run: `git status --porcelain | grep -E '\.(jpg|jpeg|png|pkl)$'`  
   - Should return **nothing** (or only safe test fixtures)

2. ✅ Run: `git log --name-only --oneline HEAD~10..HEAD | grep -E '\.(jpg|jpeg|png|pkl)$'`  
   - Should return **nothing** (or only safe test fixtures)

3. ✅ Check: All gallery paths use `~/.kevin/faces` (expanduser), not repo-relative paths

4. ✅ Check: `.gitignore` contains comprehensive face gallery patterns

---

## Household Privacy

Kevin serves **one household**. Face embeddings and photos are personal identity data that must remain **on-device only**.

- Users trust Kevin to recognize them without uploading their faces to GitHub
- Public repo = public internet = anyone can download
- Face embeddings can potentially be used for re-identification
- Full photos reveal household members, home layout, personal context

**When in doubt, do NOT commit any file that came from a camera pointed at a real person.**

---

## Summary

**Simple rule**: If it came from a camera or contains face embeddings → **NEVER commit**.

Only commit:
- Python code
- Documentation
- Public pre-trained models (if necessary)
- Synthetic test fixtures (colored squares, not faces)

Gallery lives at `~/.kevin/faces` on Kevin's device **only**.
