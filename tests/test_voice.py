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


if __name__ == "__main__":
    test_addressed(); test_payload(); print("ok")
