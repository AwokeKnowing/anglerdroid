"""test_speech_io.py — ReSpeaker sink picker + /say helpers."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from speech_io import pick_respeaker_sink, pick_respeaker_src, DEFAULT_SINK, DEFAULT_SRC  # noqa: E402

PACTL = """\
45	alsa_output.platform-sound.analog-stereo	PipeWire	s32le 2ch 48000Hz	SUSPENDED
53	alsa_output.platform-3510000.hda.hdmi-stereo	PipeWire	s32le 2ch 48000Hz	SUSPENDED
87	alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo	PipeWire	s24le 2ch 16000Hz	SUSPENDED
"""

PACTL_SRC = """\
44	alsa_input.platform-sound.analog-stereo	PipeWire	s16le 2ch 48000Hz	SUSPENDED
87	alsa_output.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-stereo.monitor	PipeWire	s24le 2ch 16000Hz	RUNNING
88	alsa_input.usb-SEEED_ReSpeaker_4_Mic_Array__UAC1.0_-00.analog-surround-21	PipeWire	s16le 1ch 16000Hz	SUSPENDED
"""


def test_picks_respeaker_not_onboard():
    name = pick_respeaker_sink(PACTL)
    assert "SEEED_ReSpeaker" in name, name
    assert "platform-sound" not in name


def test_empty_falls_back():
    assert pick_respeaker_sink("") == DEFAULT_SINK
    assert pick_respeaker_sink("1\tonboard\n") == DEFAULT_SINK


def test_picks_respeaker_source():
    name = pick_respeaker_src(PACTL_SRC)
    assert "SEEED_ReSpeaker" in name, name
    assert ".monitor" not in name
    assert "alsa_input" in name
    assert pick_respeaker_src("") == DEFAULT_SRC


if __name__ == "__main__":
    test_picks_respeaker_not_onboard()
    test_empty_falls_back()
    test_picks_respeaker_source()
    print("ok")
