# Kevin sim server on i777 (M1)

One sim body that behaves like the real Kevin, viewable and commandable by several people at once, locally or through the tunnel.

```
emu.world (MuJoCo, casita)  --shm RS1/RS2 + UDP CAN-->  nav.run :8090 (same brain/safety/wheelbase as the Orin, KEVIN_HAL=sim)
                                                            ^ POST /api/command          | GET /api/state
                                       emu/viewer/viewer.py :8095  (hub: merged state @15 Hz)
                                       |  WS /ws  (state broadcast, shared chat log, chat in)   /state.json polling fallback
                         browsers (N, own orbit camera each)  <-- cloudflared --> https://<tunnel>/
chat:  text -> nav/agent.py: exact goals.parse grammar first (stop ~0 ms) else local LLM (ollama, OpenAI API) -> validated commands
       'ask <instruction>' -> nav/vln.py -> LightNav-0 :8050 (WS login/reset/next)
```

## Start
```
scripts/kevin-server-up.sh                     # ollama (if down) + world + nav + viewer; checks LightNav on :8050
~/.local/bin/cloudflared tunnel --no-autoupdate --url http://localhost:8095    # remote (quick tunnel, URL changes on restart)
```
Env: `KEVIN_LLM_URL` (default `http://localhost:11434/v1`), `KEVIN_LLM_MODEL` (default `qwen2.5:7b-instruct`), `KEVIN_LLM_KEY` (optional, for a
hosted OpenAI-compatible endpoint), `KEVIN_VLN_URL` (default `ws://localhost:8050`), `ENV`, `KEVIN_EMU_NOISE`. Logs in `~/.kevin/emu/`.
ollama is a user-space install in `~/ollama` (no sudo); models in `~/.ollama`.
LightNav-0 is long-lived (≈12 GB GPU) and started separately:
`~/lightnav/.venv/bin/lightnav-serve --task vln --model_path ~/lightnav/ckpt --backend vllm_local --gpu_memory_utilization 0.4`.

## Multi-client rules (M1)
- Every client sees the same authoritative state (one hub, one `seq`); the orbit camera is per-browser only (server camera `POST /cam` removed).
- Anyone connected may command; last command wins (a new chat line cancels any queued multi-step plan). `stop` from anyone is applied
  immediately on the WS reader thread and never touches the LLM.
- **TODO (M2): no auth on the tunnel yet.** Anyone with the URL can drive the sim. Add tokens before sharing the URL or pointing it at the real robot.

## Command heartbeat (nav/hw.py, shared with the Orin)
`Hardware.drive()` stores the last filtered+capped (v, w). A 20 Hz thread re-sends it only while it is < 0.3 s old, then sends one zero and
goes silent. `src/wheelbase.py`'s 0.5 s stale-command watchdog and the ODrive watchdog are unchanged underneath. Disable with `KEVIN_HB=0`.
Unit test: `python tests/test_heartbeat.py`.

## Tests
- `python tests/test_heartbeat.py` – heartbeat gating (fake wheelbase).
- `python tests/test_agent_phrasings.py` – 20 English phrasings + 5 nonsense against the live LLM (needs ollama).
- `python tests/ws_clients.py ws://localhost:8095/ws 3` (or `wss://<tunnel>/ws`) – N headless clients: same state, shared chat, skew.
