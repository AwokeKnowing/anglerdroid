"""
ui.py - Audio in/out and webview. Server thread for browser (localhost) and agent;
main app reads user text and pending tool-calls from AI.
Minimal stub: no LiveKit/WebSocket yet; interface only.
"""

import threading
import time
from typing import List, Any, Optional

# Stub: no browser/audio deps required for minimal run
# When integrated: browser at localhost, WebSocket to this app; agent (e.g. LiveKit + Gemini/Grok) runs in thread.


class UI:
    """
    Single thread runs "server" (future: LiveKit agent or WebSocket server).
    Main loop calls get_user_text() and get_pending_tool_calls() each tick.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._last_user_text: str = ""
        self._pending_tool_calls: List[dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._server_loop, daemon=True)
        self._thread.start()
        print("ui: server thread started (stub)")

    def _server_loop(self):
        while self._running:
            # Stub: no real server. Future: accept WebSocket, run agent, push tool calls.
            time.sleep(0.1)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("ui: stopped")

    def set_user_text(self, text: str):
        with self._lock:
            self._last_user_text = text

    def get_user_text(self) -> str:
        with self._lock:
            out = self._last_user_text
            self._last_user_text = ""
            return out

    def push_tool_calls(self, calls: List[dict]):
        """Called by agent thread when AI issues tool calls."""
        with self._lock:
            self._pending_tool_calls.extend(calls)

    def get_pending_tool_calls(self) -> List[dict]:
        with self._lock:
            out = self._pending_tool_calls.copy()
            self._pending_tool_calls.clear()
            return out

    def submit_response(self, tool_call_id: str, result: Any):
        """Optional: send tool result back to agent for next turn."""
        pass
