"""Unit tests for offline people-behavior stubs (no hardware, no drive)."""

from faces.people_behavior import (
    DirectionalHelp,
    GreetHours,
    NameCallResponder,
    PeopleBehaviorStub,
    create_people_behavior,
)


def test_greet_hours_day_window():
    gh = GreetHours(start_hour=8, end_hour=22)
    assert gh.allows(8) is True
    assert gh.allows(12) is True
    assert gh.allows(21) is True
    assert gh.allows(22) is False
    assert gh.allows(3) is False
    assert gh.allows(7) is False
    print("✅ test_greet_hours_day_window passed")
    return True


def test_greet_hours_wrap_midnight():
    gh = GreetHours(start_hour=22, end_hour=8)
    assert gh.allows(23) is True
    assert gh.allows(0) is True
    assert gh.allows(7) is True
    assert gh.allows(8) is False
    assert gh.allows(12) is False
    print("✅ test_greet_hours_wrap_midnight passed")
    return True


def test_name_call_match():
    nc = NameCallResponder()
    assert nc.match("hey Kevin") == ""
    assert nc.match("Kevin, what's up") == "what's up"
    assert nc.match("hello there") is None
    assert nc.match("") is None
    action = nc.respond("Hey Kevin")
    assert action is not None
    assert action.kind == "name_call"
    assert action.utterance
    print("✅ test_name_call_match passed")
    return True


def test_directional_help_landmark():
    dh = DirectionalHelp()
    action = dh.respond("where is the kitchen?")
    assert action is not None
    assert action.kind == "directional_help"
    assert "kitchen" in action.utterance.lower()
    assert action.goal_hint is not None
    assert action.goal_hint["type"] == "relative_bearing"
    assert action.goal_hint["label"] == "kitchen"
    # goal_hint must never be auto-applied — stub only carries the dict
    print("✅ test_directional_help_landmark passed")
    return True


def test_directional_help_with_name_call():
    spoken = []
    stub = create_people_behavior(
        speak_fn=lambda t: spoken.append(t),
        cooldown_seconds=300.0,
    )
    action = stub.on_transcript("Kevin, where is the bathroom?", now=1000.0)
    assert action is not None
    assert action.kind == "directional_help"
    assert "bathroom" in action.utterance.lower()
    assert spoken and "bathroom" in spoken[0].lower()
    print("✅ test_directional_help_with_name_call passed")
    return True


def test_face_greet_hours_gate():
    spoken = []
    stub = PeopleBehaviorStub(
        speak_fn=lambda t: spoken.append(t),
        greet_hours=GreetHours(8, 22),
        cooldown_seconds=60.0,
    )
    # Outside window → no speak
    action = stub.on_face_seen("James", hour=3, now=100.0)
    assert action is not None
    assert action.meta.get("skipped") == "outside_greet_hours"
    assert action.utterance == ""
    assert spoken == []

    # Inside window → greet
    action = stub.on_face_seen("James", hour=9, now=200.0)
    assert action.kind == "greet"
    assert "James" in action.utterance
    assert len(spoken) == 1

    # Cooldown
    action2 = stub.on_face_seen("James", hour=9, now=210.0)
    assert action2.meta.get("skipped") == "cooldown"
    assert len(spoken) == 1
    print("✅ test_face_greet_hours_gate passed")
    return True


def test_hardware_flag_stays_off():
    stub = create_people_behavior()
    assert stub.enabled_on_hardware is False
    print("✅ test_hardware_flag_stays_off passed")
    return True


def run_all_tests():
    tests = [
        test_greet_hours_day_window,
        test_greet_hours_wrap_midnight,
        test_name_call_match,
        test_directional_help_landmark,
        test_directional_help_with_name_call,
        test_face_greet_hours_gate,
        test_hardware_flag_stays_off,
    ]
    failed = []
    for t in tests:
        try:
            if not t():
                failed.append(t.__name__)
        except Exception as e:
            print(f"❌ {t.__name__} crashed: {e}")
            failed.append(t.__name__)
    print(f"\n{'='*60}")
    if failed:
        print(f"❌ {len(failed)} people-behavior test(s) failed: {failed}")
        return False
    print(f"✅ All {len(tests)} people-behavior tests passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all_tests() else 1)
