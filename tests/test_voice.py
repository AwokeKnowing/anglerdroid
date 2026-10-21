"""nav/voice.py: wake-word match (EN/ES, accents) and the webhook payload (no models loaded)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import voice as V


def test_addressed():
    for t in ["Kevin, what time is it?", "Oye Kevin, ¿qué hora es?", "KEVÍN ven aquí", "hey kevin.", "Kelvin, stop", "Oye qué vin, cómo estás"]:
        assert V.addressed(t), t
    for t in ["", "what time is it", "Kevinator is a word", "the kevlar vest", "que vino ayer"]:
        assert not V.addressed(t), t


def test_payload():
    p = V.voice_payload("Kevin, ¿qué hora es?", "es", 0.97)
    assert p["kind"] == "voice" and p["robot"] == "kevin001" and p["lang"] == "es" and "speaker" not in p
    assert p["reply"]["lang"] == "es" and "/api/say" in p["reply"]["command"] and '"lang": "es"' in p["reply"]["command"]
    json.dumps(p)
    assert V.voice_payload("Kevin hallo", "de", 0.5)["reply"]["lang"] == "en"



def test_muted_by_default_plays_nothing(tmp_path=None):
    import subprocess, tempfile
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    import robot_params as RP
    assert RP.resolve(env={})["audio_enabled"]["value"] is False             # robot.json audio.enabled false
    assert RP.resolve(env={"KEVIN_AUDIO_ENABLED": "1"})["audio_enabled"]["value"] is True
    os.environ.pop("KEVIN_AUDIO_ENABLED", None)
    assert V.muted()
    os.environ["KEVIN_MUTE"] = "1"; assert V.muted(); os.environ.pop("KEVIN_MUTE")
    os.environ["KEVIN_AUDIO_ENABLED"] = "1"; assert not V.muted()
    os.environ["KEVIN_MUTE"] = "1"; assert V.muted(); os.environ.pop("KEVIN_MUTE"); os.environ.pop("KEVIN_AUDIO_ENABLED")
    os.environ["KEVIN_AUDIO_ENABLED"] = "true"; assert V.muted(); os.environ.pop("KEVIN_AUDIO_ENABLED")   # only exactly "1" enables
    calls = []; real = subprocess.run
    subprocess.run = lambda *a, **k: calls.append(a) or real(["true"])
    try:
        V.MUTED_DIR = tempfile.mkdtemp(); w = os.path.join(V.MUTED_DIR, "x.wav"); V._wav(w, np.zeros(2400, np.float32), 24000)
        assert V.play(w, 0.1) and not any("paplay" in str(c) for c in calls)
        assert os.listdir(V.MUTED_DIR) and len(os.listdir(V.MUTED_DIR)) == 2
    finally:
        subprocess.run = real


def test_mast_upright():
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from cameras import mast_upright
    f = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert (mast_upright(f) == f[::-1, ::-1]).all()                          # 180 deg, same as WebCam._ingest_bgr


if __name__ == "__main__":
    test_addressed(); test_payload(); test_muted_by_default_plays_nothing(); test_mast_upright(); print("ok")
