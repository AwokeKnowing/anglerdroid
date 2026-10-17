"""Emulated `inputs` (gamepad) package: the emulator has no USB gamepad attached."""
class UnpluggedError(RuntimeError): pass
class GamePad:
    def __init__(self, manager, path, *a, **k): raise UnpluggedError("no gamepad in emulator")
class _Devices:
    def __init__(self): self.gamepads = []; self.keyboards = []; self.mice = []
devices = _Devices()
def get_gamepad(): raise UnpluggedError("No gamepad found.")
