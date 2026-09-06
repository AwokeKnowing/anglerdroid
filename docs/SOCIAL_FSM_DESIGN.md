# Kevin's Social Conversation FSM — Design & HRI Principles

## Goal
Wander around encouraging/empathizing/cheering people up **without annoying them**.  
Behave like a respectful companion, not a clingy toy.

## HRI Research Foundation

### 1. Engagement-Aware Proxemics (RO-MAN 2025 / JSME)
**Principle**: Adapt stop distance based on engagement cues.

**Implementation**:
- **Face toward robot** (frontal face detected) = closer OK → target 1.2 m (lower end of conversational)
- **Face away / profile** (side angle, no direct gaze) = stay farther → target 1.5 m or don't approach
- **Default**: 1.35 m (middle of 1.2–1.5 m conversational range)

**Rationale**: People signal receptiveness through body orientation. Approaching someone who's turned away feels intrusive.

**Future enhancement** (not in MVP): Use head pose estimation to detect gaze direction and adjust dynamically during conversation.

---

### 2. Kendon Greeting Phases (Int J Social Robotics 2025)
**Principle**: Progressive greeting stages match human social norms.

**Classic phases**:
1. **Initiation** — notice person, decide to approach
2. **Distance salutation** — brief acknowledgement from afar (optional, if far)
3. **Approach** — move to conversational distance
4. **Close salutation** — full greeting at conversational distance

**Implementation**:
- **NOTICE** state: brief observation (0.5 s) — no speech
- **APPROACH** state: silent movement toward conversational distance
  - If person is very far (>3 m), could add optional "Hi!" wave (not yet implemented)
  - Avoids yelling full conversation from across the room
- **GREET** state: short name greeting at conversational distance
  - "Hi Erika!" or "¡Hola Nohemi!" — warm but brief
  - Only when physically close enough for comfortable conversation

**Rationale**: Shouting "GOOD MORNING JAMES, HOW ARE YOU DOING TODAY?" from 5 meters away is jarring. Humans naturally wait until they're closer for substantive greetings.

---

### 3. Designing for Exit (WeRobot 2022)
**Principle**: Users must be able to end interaction easily and immediately.

**Implementation**:
- **"Go away" / "leave me alone"** → immediate retreat (2 m back), **10 min cooldown** (2x normal)
- **"Not now" / "I'm busy"** → immediate leave + resume wander, **10 min cooldown**
- **15 second silence** after greeting → polite leave, 5 min cooldown
- **"Stop" / "halt"** → immediately clear all goals, stay in place

**Rationale**: Forced interaction is annoying. Kevin must respect disengagement signals more strongly than engagement signals.

**Cooldown escalation**:
- Normal no-engage: 5 minutes
- After conversation: 1.5 minutes (shorter; they engaged positively)
- **Explicit dismissal** ("go away"): **10 minutes** (respect the strong signal)

---

### 4. Measured Performance (Live Logs)
**Baseline** (pre-FSM, from actual Kevin Jetson Orin logs):
- Main control loop: **29.9 fps** (~20 ms process time, budget 33.3 ms)
- Atlas JPEG encode (TurboJPEG): **~10 ms @ quality 70** → **~18 KB** output
- WebSocket viewer send: **~12 fps** by design (intentional throttle; humans don't need 30 fps)

**Design constraint**: Do NOT "fix" FPS by cranking JPEG quality or encoding every frame. The 30 Hz loop is preserved; the viewer throttle is intentional for Wi-Fi/CPU efficiency.

**FSM impact**: All social processing runs in `people_live` thread (~1.67 Hz face ticks). **Zero added load on 30 Hz control loop.**

---

### 5. Drive Arm Safety Latch
**Current state**: Drive is **DISARMED** by default (`~/.kevin/drive_arm` file does not exist).

**Gating rule**:
- **Disarmed** (no file): FSM runs speech-only; all motion goal hints are logged but NOT applied to LocalExecutive
- **Armed** (file exists): Motion goals (approach, retreat, wander) are applied to LocalExecutive → MPPI → SafetyGuard

**Implementation**:
```python
# people_live.py, _apply_goal_hint()
armed = os.path.exists(DRIVE_ARM_FILE)
if not armed:
    print("goal hint skipped (drive disarmed)")
    return
# ... apply goal to LocalExecutive
```

**Testing**: All FSM state transitions, greetings, empathy responses, and command parsing work **with drive disarmed**. Only the motion execution is gated.

---

## FSM State Diagram

```
                    ┌─────────────┐
                    │ IDLE_WANDER │ (wandering; no person noticed)
                    └──────┬──────┘
                           │ face_seen(known_person)
                           ▼
                    ┌─────────────┐
                    │   NOTICE    │ (observe 0.5s; decide approach)
                    └──────┬──────┘
                           │
                ┌──────────┴──────────┐
                │                     │
        too_close / already_good      far (>1.5m)
                │                     │
                ▼                     ▼
         ┌─────────────┐      ┌─────────────┐
         │    GREET    │◀─────│  APPROACH   │ (move to ~1.35m, silent)
         └──────┬──────┘      └─────────────┘
                │ (greeting spoken)
                ▼
         ┌─────────────┐
         │ WAIT_ENGAGE │ (listen 15s for reply)
         └──────┬──────┘
                │
      ┌─────────┴─────────┐
      │                   │
  speech_heard        timeout (15s) / "go away"
      │                   │
      ▼                   ▼
┌─────────────┐    ┌─────────────┐
│  CONVERSE   │───▶│    LEAVE    │ (resume wander; cooldown)
└─────────────┘    └─────────────┘
  (empathy turns)      │
      │                │
      └────────────────┘
   (silence timeout)
```

---

## Distance & Timing Parameters

### Hall's Proxemics (applied)
- **Intimate**: 0–0.5 m — do NOT enter uninvited
- **Personal/Conversational**: **1.2–1.5 m** — target stop zone
- **Social**: 1.5–3.6 m — approach from here
- **Public**: >3.6 m — may skip or add distance salutation

### FSM Constants
```python
CONVERSATIONAL_DIST_MIN = 1.2  # meters (4 ft)
CONVERSATIONAL_DIST_MAX = 1.5  # meters (5 ft)
APPROACH_STOP_DIST = 1.35      # target (middle of range)
INTIMATE_DIST = 0.5            # too close; back off
```

### Timing
```python
NOTICE_DECIDE_DELAY = 0.5      # brief observation before approach
WAIT_ENGAGE_TIMEOUT = 15.0     # silence → leave
CONVERSE_TURN_MAX = 8.0        # keep conversational turns short
```

### Cooldowns
```python
COOLDOWN_NO_ENGAGE = 300.0     # 5 min after leave-without-chat
COOLDOWN_ENGAGED = 90.0        # 1.5 min after conversation
COOLDOWN_DISMISSED = 600.0     # 10 min after explicit "go away"
```

---

## Distance Estimation

### 1. Depth-based (preferred)
When RGB-D camera depth is available:
- Sample median of valid depth pixels in face box center region
- Robust to noise; ~0.1 ms compute time

### 2. Box heuristic (fallback)
When depth unavailable, estimate from face bounding box width:
- **>80 px** → closer than conversational (~0.8 m)
- **40–70 px** → good conversational range (~1.2–1.5 m)
- **<30 px** → farther than conversational (~3 m)

Calibrated for 320px wide RGB frame, YuNet face detection after 2x upscale.

**Accuracy**: ±0.3 m typically sufficient for social approach (not precision docking).

---

## Kevin's Personality (Voice Lock)

**Core traits**: Curious, calm, earnest helper — like Hero / Astro Boy  
**Humor style**: Clean wholesome (puns, gentle self-deprecation as a learning robot, playful observations)  
**Priority**: Encouraging and empathizing; jokes secondary to kindness  
**Tone**: Never manic, sarcastic-mean, edgy, dark, sexual, or insulting  
**Length**: Keep lines short (1–2 sentences)

**Applied across**:
- Greetings ("Hi Erika!", "¡Hola Nohemi!")
- Wait/chat empathy ("You're doing great!", "I'm learning too.")
- Leave messages ("I'll let you get back to it!", "My wheels are excited!")
- Command acks ("Rolling over now!", "Hmm. I'm still learning that one!")

**Spanish for Nohemi/Karina**: Same earnest, gentle tone in Spanish

---

## Empathy & Encouragement

### English Templates (Curious, Earnest, Optionally Playful)
```python
EMPATHY_TEMPLATES = [
    "I hear you!",
    "That makes sense to me.",
    "You're doing great!",
    "I'm learning too. Keep going!",
    "That sounds tricky.",
    "I'm here if you need me!",
    "You've got this!",
    "That's really interesting!",
]

EMPATHY_TOUGH = [
    "That sounds hard. You're doing your best!",
    "I'm still learning, but I think you're brave.",
    "Tough day? I'm here.",
]

EMPATHY_HAPPY = [
    "That's wonderful! I'm happy for you!",
    "That sounds great!",
    "That made my sensors warm. Keep it up!",
]
```

### Spanish Templates (Same Earnest, Gentle Tone)
```python
EMPATHY_TEMPLATES_ES = [
    "¡Te escucho!",
    "Tiene sentido para mí.",
    "¡Lo estás haciendo muy bien!",
    "Yo también estoy aprendiendo. ¡Sigue así!",
    "Eso suena difícil.",
    "¡Estoy aquí si me necesitas!",
]

EMPATHY_TOUGH_ES = [
    "Eso suena difícil. ¡Lo estás haciendo bien!",
    "Todavía estoy aprendiendo, pero creo que eres valiente.",
]

EMPATHY_HAPPY_ES = [
    "¡Qué maravilla! ¡Me alegro por ti!",
    "Eso me calentó los sensores. ¡Sigue así!",
]
```

### Leave Messages (Polite + Optional Tiny Clean Joke, Never Guilt-Trip)
```python
LEAVE_MESSAGES = [
    "",  # silent leave is fine
    "I'll let you get back to it!",
    "Catch you later!",
    "I'll keep wandering. Call if you need me!",
    "Off to explore. My wheels are excited!",
]
```

### Selection Strategy
- **Keyword-based**: "hard"/"difficult"/"tired" → supportive ("That sounds hard. You're doing your best!")
- **Positive**: "good"/"happy"/"great" → encouraging ("That's wonderful! I'm happy for you!")
- **Default**: generic acknowledgement ("I hear you!", "That makes sense to me.")

**Keep turns short** (1–2 sentences) — Kevin is a companion, not a therapist. Long monologues are exhausting.

---

## Commands ("Hey Kevin, ...")

All command acknowledgements use curious, calm, earnest tone with optional playful flair.

### Movement
- `stop / halt / freeze` → "Stopping!" / "Okay, holding still." / "Wheels stopping now!"
- `come here / come to me` → "Coming over!" / "On my way!" / "Rolling over now!"
- `go away / leave me alone / give me space` → "Sorry! Moving away." / "Understood. Backing up." / "No problem. I'll give you space!"
- `wander / explore / look around` → "Back to wandering!" / "Time to explore!" / "Off I go!"

### Directional Help (existing)
- `where is the kitchen / bathroom / bedroom?` → point direction
- `which way is left / right / ahead?` → point direction

### Unknown Commands (Gentle Self-Deprecation)
- "I heard you, but I'm not sure how to do that yet."
- "Hmm. I'm still learning that one!"
- "That's a new one for me. Still learning!"

### Future (not yet implemented)
- `follow me` → trail behind at 2 m
- `say hi to [name]` → approach that person if seen
- `go to the charger` → navigate to known landmark

---

## Safety Guarantees

**All existing safety layers preserved**:

1. **SafetyGuard** (vision.py): forward/backward/angular scaling from top-down depth — untouched
2. **Top-down depth immobilize**: no valid RS1 depth → fwd_scale = 0 — untouched
3. **Soft keepouts** (keepouts.py): paint no-go zones into ego obs — active
4. **MPPI costmap planner**: all goals → local_executive → MPPI → wheelbase — unchanged

**Social FSM never bypasses safety**:
- All approach goals are world-frame (x, y) hints to `local_executive.set_goal_xy()`
- LocalExecutive → MPPI → SafetyGuard → wheelbase (normal pipeline)
- If path is blocked, MPPI won't move; FSM doesn't force motion

**Drive arm latch**:
- `~/.kevin/drive_arm` file must exist for ANY motion goals to apply
- Speech, greetings, empathy work regardless of arm state
- Allows safe testing of social behavior without wheels

---

## Performance Design

### Main Loop (30 Hz)
**Target**: 30 fps (33.3 ms budget per frame)  
**Measured**: 29.9 fps (~20 ms actual) — **healthy margin**

**Components** (from vision.py breakdown):
- Camera grab: ~X ms
- RS1 top-down depth: ~X ms
- RS2 forward depth: ~X ms
- Obstacle map: ~X ms
- Odometry: ~X ms
- Global map: ~X ms
- SafetyGuard: ~X ms
- **Social FSM**: **0 ms** (runs in separate thread)

### People Thread (~1.67 Hz)
**Face tick** (every 0.6 s):
- Face detection (YuNet): ~50–200 ms (CPU-bound, acceptable at low rate)
- Face recognition (gallery match): ~10–50 ms
- **FSM state update**: <0.1 ms per face
- **Distance estimation**: ~0.1 ms (depth) or <0.01 ms (box)
- **Goal hint apply**: <0.5 ms (throttled, mailbox write)

**Listen tick** (every 12 s):
- Record audio (parecord): 2.5 s blocking (by design)
- ASR (faster-whisper): ~500–2000 ms (CPU-bound, rare)
- Command parse: <1 ms

**Total added overhead**: <1 ms per face tick, already off critical path.

### Atlas Viewer
**Encoding** (TurboJPEG SIMD):
- Quality: **70** (do not raise; CPU efficiency matters)
- Encode time: **~10 ms** per frame
- Output size: **~18 KB** (960×960 RGB atlas)

**WebSocket send**: **~12 fps** (intentional throttle)  
**Rationale**: Humans don't need 30 fps for viewing; Wi-Fi bandwidth is precious; CPU encode cost matters.

**Do NOT "fix" FPS** by cranking quality to 95 or encoding every frame. The measured 29.9 fps main loop is excellent; the 12 fps viewer send is by design.

---

## Open Questions / Future Work

1. **Head pose / gaze estimation**: Adapt conversational distance dynamically based on where person is looking (engagement-aware proxemics, phase 2).

2. **Multi-person conversation**: FSM currently handles one `current_person` at a time. For household with 3+ people, need turn-taking / group conversation logic.

3. **Persistent memory**: Cooldowns are session-local (lost on restart). Save to `~/.kevin/social_memory.json` for cross-session cooldowns and preferences.

4. **LLM empathy** (optional): Replace template bank with small on-device LLM (e.g., Llama 3.2 1B) for richer responses. Keep templates as fallback when offline or CPU-constrained.

5. **Arc approach**: Approach on a curved path (not straight line) feels less aggressive (HRI literature). Needs path planner enhancement.

6. **Idle behaviors**: When no one is around, occasionally say something cheerful or curious (not just silent wander). Avoid creepy silence.

---

## Testing Strategy

### Unit Tests (27 passing)
- **src/test_social_fsm.py** (15 tests): FSM transitions, distance, cooldowns, commands
- **faces/test_people_behavior.py** (12 tests): command parsing, greet hours, name-call

### Drive Disarmed Tests
All tests run with `drive_armed=False`:
- FSM state transitions work
- Greetings and empathy responses work
- Command parsing returns correct `goal_hint` dicts
- Goal application is logged but skipped when disarmed

### Live Testing (with drive armed)
1. **Approach test**: Stand 3 m away, face Kevin → should approach to ~1.35 m and greet
2. **Engagement test**: Reply after greeting → should respond with empathy and stay
3. **Timeout test**: Don't reply after greeting → should leave after 15 s
4. **Go away test**: Say "go away" → should retreat and not re-approach for 10 min
5. **Spanish test**: Nohemi/Karina get Spanish greetings

---

## References

1. **Engagement-aware proxemics**: RO-MAN 2025, JSME (adaptive stop distance based on gaze/orientation)
2. **Kendon greeting phases**: Int J Social Robotics 2025 (progressive greeting stages)
3. **Designing for Exit**: WeRobot 2022 (user must be able to end interaction easily)
4. **Hall's proxemics**: Edward T. Hall, "The Hidden Dimension" (1966) — conversational distance 1.2–3.6 m
5. **HRI engagement**: Jibo, ElliQ, PARO literature (approach on arc, announce, yield turn, leave on silence)

---

## Summary

Kevin's social FSM is designed around **respect and non-annoyance**:
- Approach to comfortable conversational distance (4-5 ft)
- Brief, warm greetings (not yelling from across the room)
- Wait for engagement; leave gracefully if ignored
- Short empathetic turns during conversation
- Immediate response to "go away" with long cooldown
- 30 Hz control loop preserved (all social work off critical path)
- Drive arm latch for safe testing

**Goal achieved**: Respectful companion, not clingy toy.
