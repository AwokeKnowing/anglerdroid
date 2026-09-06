"""Unit tests for social_fsm.py (Kevin's social conversation FSM).

Tests cover:
- State transitions (IDLE → NOTICE → APPROACH → GREET → WAIT → CONVERSE → LEAVE)
- Distance estimation (depth-based and box heuristic)
- Conversational distance rules (4-5 ft / 1.2-1.5 m)
- 15 second engagement timeout
- Cooldown periods (5 min no-engage, 1.5 min after conversation)
- Spanish preference for Nohemi/Karina
- Command parsing (stop, come here, go away, wander)

No live hardware required; all tests use simulated time and mock data.
"""

import math
import sys
import os

# Add src to path for imports
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from social_fsm import (
    SocialFSM,
    SocialState,
    CONVERSATIONAL_DIST_MIN,
    CONVERSATIONAL_DIST_MAX,
    APPROACH_STOP_DIST,
    WAIT_ENGAGE_TIMEOUT,
    COOLDOWN_NO_ENGAGE,
    COOLDOWN_ENGAGED,
    COOLDOWN_DISMISSED,
    FACE_BOX_CLOSE_PX,
    FACE_BOX_FAR_PX,
    FACE_BOX_GOOD_MIN,
    FACE_BOX_GOOD_MAX,
)


def test_distance_estimation_box_heuristic():
    """Test face box size → distance estimation."""
    fsm = SocialFSM(drive_armed=False)
    
    # Close: box width > 80 px
    dist = fsm.estimate_distance_from_box((0, 0, 90, 90))
    assert dist < CONVERSATIONAL_DIST_MIN, f"Close box should be <{CONVERSATIONAL_DIST_MIN}m, got {dist}"
    
    # Far: box width < 30 px
    dist = fsm.estimate_distance_from_box((0, 0, 25, 25))
    assert dist > CONVERSATIONAL_DIST_MAX, f"Far box should be >{CONVERSATIONAL_DIST_MAX}m, got {dist}"
    
    # Good range: 40-70 px → ~1.2-1.5 m
    dist = fsm.estimate_distance_from_box((0, 0, 50, 50))
    assert CONVERSATIONAL_DIST_MIN <= dist <= CONVERSATIONAL_DIST_MAX * 1.5, \
        f"Good box should be ~{CONVERSATIONAL_DIST_MIN}-{CONVERSATIONAL_DIST_MAX}m, got {dist}"
    
    print("✅ test_distance_estimation_box_heuristic passed")
    return True


def test_fsm_idle_to_notice():
    """Test IDLE_WANDER → NOTICE transition."""
    fsm = SocialFSM(drive_armed=False)
    
    # First sight of person → NOTICE
    box = (100, 100, 50, 50)  # good distance
    action = fsm.on_face_seen("James", 0.9, box, now=100.0)
    
    assert action is not None, "Should return action on first sight"
    assert action["action_type"] == "notice", f"Expected 'notice', got {action['action_type']}"
    assert action["person"] == "James"
    assert fsm.encounters["James"].state == SocialState.NOTICE
    
    print("✅ test_fsm_idle_to_notice passed")
    return True


def test_fsm_approach_when_far():
    """Test NOTICE → APPROACH when person is far."""
    fsm = SocialFSM(drive_armed=True)  # armed for approach
    
    # Far person: box width 25 px → ~3 m
    box = (150, 100, 25, 25)
    fsm.on_face_seen("Erika", 0.9, box, now=100.0)
    
    # After notice delay, should transition to APPROACH
    action = fsm.on_face_seen("Erika", 0.9, box, now=100.6)
    
    assert action is not None
    assert action["action_type"] == "approach", f"Expected 'approach', got {action['action_type']}"
    assert fsm.encounters["Erika"].state == SocialState.APPROACH
    assert action["goal_hint"] is not None
    assert action["goal_hint"]["type"] == "approach_person"
    
    print("✅ test_fsm_approach_when_far passed")
    return True


def test_fsm_greet_at_conversational_distance():
    """Test NOTICE → GREET when already at good distance."""
    fsm = SocialFSM(drive_armed=False)
    
    # Good distance: box width 50 px → ~1.35 m
    box = (150, 100, 50, 50)
    fsm.on_face_seen("James", 0.9, box, now=100.0)
    
    # After notice delay, should transition to GREET
    action = fsm.on_face_seen("James", 0.9, box, now=100.6)
    
    assert action is not None
    assert action["action_type"] == "greet", f"Expected 'greet', got {action['action_type']}"
    assert fsm.encounters["James"].state == SocialState.GREET
    assert "James" in action["utterance"], f"Greeting should include name, got {action['utterance']}"
    
    print("✅ test_fsm_greet_at_conversational_distance passed")
    return True


def test_fsm_wait_engage_timeout():
    """Test WAIT_ENGAGE → LEAVE after 15 second timeout."""
    fsm = SocialFSM(drive_armed=False)
    
    # Greet James
    box = (150, 100, 50, 50)
    fsm.on_face_seen("James", 0.9, box, now=100.0)
    fsm.on_face_seen("James", 0.9, box, now=100.6)  # greet
    
    # Wait for engagement timeout
    # After greeting, state should be GREET, then WAIT_ENGAGE
    fsm.encounters["James"].state = SocialState.WAIT_ENGAGE
    
    # Before timeout: no leave
    action = fsm.on_face_seen("James", 0.9, box, now=100.6 + WAIT_ENGAGE_TIMEOUT - 1.0)
    assert action is None or action["action_type"] != "leave", "Should not leave before timeout"
    
    # After timeout: should leave
    action = fsm.on_face_seen("James", 0.9, box, now=100.6 + WAIT_ENGAGE_TIMEOUT + 1.0)
    assert action is not None
    assert action["action_type"] == "leave", f"Expected 'leave' after timeout, got {action['action_type']}"
    assert fsm.is_on_cooldown("James", now=100.6 + WAIT_ENGAGE_TIMEOUT + 1.0)
    
    print("✅ test_fsm_wait_engage_timeout passed")
    return True


def test_fsm_engagement_starts_conversation():
    """Test WAIT_ENGAGE → CONVERSE on speech heard."""
    from social_fsm import PersonEncounter
    
    fsm = SocialFSM(drive_armed=False)
    
    # Setup: person at WAIT_ENGAGE state
    enc = PersonEncounter(name="Erika")
    enc.state = SocialState.WAIT_ENGAGE
    enc.greet_time = 100.0
    enc.last_heard_time = 0.0
    enc.last_spoke_time = 100.0
    fsm.encounters["Erika"] = enc
    fsm.current_person = "Erika"
    
    # Hear speech → engagement
    action = fsm.on_speech_heard("Erika", "How are you?", now=101.0)
    
    assert action is not None
    assert action["action_type"] == "empathy", f"Expected 'empathy', got {action['action_type']}"
    assert action["utterance"], "Should have empathy response"
    enc = fsm.encounters["Erika"]
    assert enc.state == SocialState.CONVERSE
    assert enc.engaged_this_session is True
    
    print("✅ test_fsm_engagement_starts_conversation passed")
    return True


def test_cooldown_no_engage_vs_engaged():
    """Test cooldown periods: 5 min after no-engage, 1.5 min after conversation."""
    fsm = SocialFSM(drive_armed=False)
    
    # Scenario 1: leave without engagement
    box = (150, 100, 50, 50)
    fsm.on_face_seen("James", 0.9, box, now=100.0)
    fsm.encounters["James"].state = SocialState.WAIT_ENGAGE
    fsm.encounters["James"].greet_time = 100.0
    
    # Timeout → leave
    action = fsm.on_face_seen("James", 0.9, box, now=100.0 + WAIT_ENGAGE_TIMEOUT + 1.0)
    assert action["action_type"] == "leave"
    
    # Should be on cooldown for COOLDOWN_NO_ENGAGE
    now_check = 100.0 + WAIT_ENGAGE_TIMEOUT + 1.0
    assert fsm.is_on_cooldown("James", now_check + COOLDOWN_NO_ENGAGE - 10.0)
    assert not fsm.is_on_cooldown("James", now_check + COOLDOWN_NO_ENGAGE + 10.0)
    
    # Scenario 2: engaged conversation
    from social_fsm import PersonEncounter
    
    fsm.reset_encounter("Erika")
    enc2 = PersonEncounter(name="Erika")
    enc2.state = SocialState.CONVERSE
    enc2.last_heard_time = 200.0
    enc2.engaged_this_session = True
    fsm.encounters["Erika"] = enc2
    fsm.current_person = "Erika"
    
    # Conversation timeout → leave
    action = fsm.on_face_seen("Erika", 0.9, box, now=200.0 + WAIT_ENGAGE_TIMEOUT + 1.0)
    assert action["action_type"] == "leave"
    
    # Should be on cooldown for COOLDOWN_ENGAGED (shorter)
    now_check2 = 200.0 + WAIT_ENGAGE_TIMEOUT + 1.0
    assert fsm.is_on_cooldown("Erika", now_check2 + COOLDOWN_ENGAGED - 10.0)
    assert not fsm.is_on_cooldown("Erika", now_check2 + COOLDOWN_ENGAGED + 10.0)
    
    print("✅ test_cooldown_no_engage_vs_engaged passed")
    return True


def test_spanish_preference():
    """Test Spanish greetings for Nohemi/Karina."""
    fsm = SocialFSM(drive_armed=False)
    
    # Nohemi should get Spanish greeting
    box = (150, 100, 50, 50)
    fsm.on_face_seen("Nohemi", 0.9, box, now=100.0)
    action = fsm.on_face_seen("Nohemi", 0.9, box, now=100.6)
    
    assert action is not None
    assert action["action_type"] == "greet"
    # Check for Spanish words (¡, Buenos, Hola, Buenas)
    utterance = action["utterance"]
    assert any(word in utterance for word in ["¡", "Buenos", "Hola", "Buenas"]), \
        f"Expected Spanish greeting, got {utterance}"
    
    # James should get English greeting
    fsm.reset_encounter("James")
    fsm.on_face_seen("James", 0.9, box, now=200.0)
    action = fsm.on_face_seen("James", 0.9, box, now=200.6)
    
    assert action is not None
    utterance = action["utterance"]
    assert any(word in utterance for word in ["Good morning", "Hi", "Good evening"]), \
        f"Expected English greeting, got {utterance}"
    
    print("✅ test_spanish_preference passed")
    return True


def test_command_stop():
    """Test 'Hey Kevin, stop' command."""
    fsm = SocialFSM(drive_armed=True)
    
    action = fsm.on_command("stop", now=100.0)
    
    assert action is not None
    assert action["action_type"] == "command"
    assert action["command"] == "stop"
    assert action["goal_hint"] is not None
    assert action["goal_hint"]["type"] == "clear"
    assert "stop" in action["utterance"].lower() or "stopping" in action["utterance"].lower()
    
    print("✅ test_command_stop passed")
    return True


def test_command_come_here():
    """Test 'Hey Kevin, come here' command."""
    fsm = SocialFSM(drive_armed=True)
    
    action = fsm.on_command("come here", now=100.0)
    
    assert action is not None
    assert action["command"] == "come_here"
    assert action["goal_hint"]["type"] == "approach_speaker"
    
    print("✅ test_command_come_here passed")
    return True


def test_command_go_away():
    """Test 'Hey Kevin, go away' command."""
    fsm = SocialFSM(drive_armed=True)
    
    action = fsm.on_command("go away", now=100.0)
    
    assert action is not None
    assert action["command"] == "go_away"
    assert action["goal_hint"]["type"] == "retreat"
    assert action["goal_hint"]["distance_m"] > 0
    
    print("✅ test_command_go_away passed")
    return True


def test_command_wander():
    """Test 'Hey Kevin, wander around' command."""
    fsm = SocialFSM(drive_armed=True)
    
    action = fsm.on_command("wander around", now=100.0)
    
    assert action is not None
    assert action["command"] == "wander"
    assert action["goal_hint"]["type"] == "resume_wander"
    
    print("✅ test_command_wander passed")
    return True


def test_command_cooldown():
    """Test command cooldown to prevent spam."""
    fsm = SocialFSM(drive_armed=True)
    
    # First command works
    action1 = fsm.on_command("stop", now=100.0)
    assert action1 is not None
    
    # Immediate second command blocked by cooldown
    action2 = fsm.on_command("wander", now=100.5)
    assert action2 is None, "Second command should be blocked by cooldown"
    
    # After cooldown, command works again
    action3 = fsm.on_command("wander", now=102.5)
    assert action3 is not None
    
    print("✅ test_command_cooldown passed")
    return True


def test_approach_stop_distance_target():
    """Test approach targets conversational stop distance (~1.35 m)."""
    fsm = SocialFSM(drive_armed=True)
    
    # Far person
    box = (150, 100, 25, 25)  # ~3 m
    fsm.on_face_seen("James", 0.9, box, now=100.0)
    action = fsm.on_face_seen("James", 0.9, box, now=100.6)
    
    assert action is not None
    goal_hint = action.get("goal_hint")
    assert goal_hint is not None
    target_dist = goal_hint.get("target_distance_m")
    
    # Target should be near APPROACH_STOP_DIST (1.35 m)
    assert target_dist is not None
    assert CONVERSATIONAL_DIST_MIN <= target_dist <= CONVERSATIONAL_DIST_MAX * 1.2, \
        f"Target distance {target_dist} should be in conversational range"
    
    print("✅ test_approach_stop_distance_target passed")
    return True


def test_drive_disarmed_skips_approach():
    """Test that drive_armed=False skips approach goals (speech-only)."""
    fsm = SocialFSM(drive_armed=False)
    
    # Far person; with armed=False, should skip approach
    box = (150, 100, 25, 25)
    fsm.on_face_seen("James", 0.9, box, now=100.0)
    action = fsm.on_face_seen("James", 0.9, box, now=100.6)
    
    # Should go straight to GREET (no approach goal)
    assert action is not None
    assert action["action_type"] == "greet", \
        f"With drive disarmed, should greet immediately, got {action['action_type']}"
    
    print("✅ test_drive_disarmed_skips_approach passed")
    return True


def test_dismissal_extended_cooldown():
    """Test 'go away' / 'not now' triggers 10 min cooldown (Designing for Exit)."""
    fsm = SocialFSM(drive_armed=True)
    
    # Setup: conversing with James
    from social_fsm import PersonEncounter
    enc = PersonEncounter(name="James")
    enc.state = SocialState.CONVERSE
    fsm.encounters["James"] = enc
    fsm.current_person = "James"
    
    # User says "go away"
    action = fsm.on_command("go away", now=100.0)
    
    assert action is not None
    assert action["command"] == "go_away"
    assert action.get("cooldown_extended") is True
    
    # Should be on extended cooldown (10 min)
    assert fsm.is_on_cooldown("James", now=100.0 + COOLDOWN_DISMISSED - 10.0)
    assert not fsm.is_on_cooldown("James", now=100.0 + COOLDOWN_DISMISSED + 10.0)
    
    # "not now" should also trigger extended cooldown
    fsm.reset_encounter("Erika")
    enc2 = PersonEncounter(name="Erika")
    fsm.encounters["Erika"] = enc2
    fsm.current_person = "Erika"
    
    action2 = fsm.on_command("not now, I'm busy", now=200.0)
    assert action2 is not None
    assert fsm.is_on_cooldown("Erika", now=200.0 + COOLDOWN_DISMISSED - 10.0)
    
    print("✅ test_dismissal_extended_cooldown passed")
    return True


def run_all_tests():
    """Run all social FSM tests."""
    tests = [
        test_distance_estimation_box_heuristic,
        test_fsm_idle_to_notice,
        test_fsm_approach_when_far,
        test_fsm_greet_at_conversational_distance,
        test_fsm_wait_engage_timeout,
        test_fsm_engagement_starts_conversation,
        test_cooldown_no_engage_vs_engaged,
        test_spanish_preference,
        test_command_stop,
        test_command_come_here,
        test_command_go_away,
        test_command_wander,
        test_command_cooldown,
        test_approach_stop_distance_target,
        test_drive_disarmed_skips_approach,
        test_dismissal_extended_cooldown,
    ]
    
    failed = []
    for t in tests:
        try:
            if not t():
                failed.append(t.__name__)
        except Exception as e:
            print(f"❌ {t.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed.append(t.__name__)
    
    print(f"\n{'='*60}")
    if failed:
        print(f"❌ {len(failed)} social_fsm test(s) failed: {failed}")
        return False
    print(f"✅ All {len(tests)} social_fsm tests passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all_tests() else 1)
