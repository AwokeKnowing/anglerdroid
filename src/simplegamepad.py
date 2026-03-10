"""Simple gamepad/joystick class."""

import threading
import time
import math
import inputs



EVENT_ABB = (
    # D-PAD, aka HAT
    ('Absolute-ABS_HAT0X', 'HX'), #
    ('Absolute-ABS_HAT0Y', 'HY'), #

    # joystick
    ('Absolute-ABS_X', 'X'),      #
    ('Absolute-ABS_Y', 'Y'),      #
    ('Absolute-ABS_RX', 'RX'),    #
    ('Absolute-ABS_RY', 'RY'),    #

    # LR triggers analog
    ('Absolute-ABS_Z', 'Z'),      #
    ('Absolute-ABS_RZ', 'RZ'),    #

    # Face Buttons
    ('Key-BTN_NORTH', 'N'),       #
    ('Key-BTN_EAST', 'E'),        #
    ('Key-BTN_SOUTH', 'S'),       #
    ('Key-BTN_WEST', 'W'),        #

    # Other buttons
    ('Key-BTN_THUMBL', 'THL'),    #
    ('Key-BTN_THUMBR', 'THR'),    #
    ('Key-BTN_TL', 'TL'),         #
    ('Key-BTN_TR', 'TR'),         #
    ('Key-BTN_TL2', 'TL2'),
    ('Key-BTN_TR2', 'TR3'),
    ('Key-BTN_MODE', 'M'),        #
    ('Key-BTN_START', 'ST'),      #
    ('Key-BTN_SELECT', 'SL'),     #

    # PiHUT SNES style controller buttons
    ('Key-BTN_TRIGGER', 'N'),
    ('Key-BTN_THUMB', 'E'),
    ('Key-BTN_THUMB2', 'S'),
    ('Key-BTN_TOP', 'W'),
    ('Key-BTN_BASE3', 'SL'),
    ('Key-BTN_BASE4', 'ST'),
    ('Key-BTN_TOP2', 'TL'),
    ('Key-BTN_PINKIE', 'TR')
)


# This is to reduce noise from the PlayStation controllers
# For the Xbox controller, you can set this to 0
MIN_ABS_DIFFERENCE = 5


class SimpleGamepad(object):
    """Simple gamepad class."""
    def __init__(self, btnCallback=None, gamepad=None, abbrevs=EVENT_ABB):
        self.print_state=False
        self.btnCallback = btnCallback
        self.deadzones={'X':10,'Y':10,'RX':10, 'RY':10}
        self.apply_deadzone=[]
        self.btn_state = {}
        self.old_btn_state = {}
        self.abs_state = {}
        self.old_abs_state = {}
        self.abbrevs = dict(abbrevs)
        for key, value in self.abbrevs.items():
            if key.startswith('Absolute'):
                self.abs_state[value] = 0
                self.old_abs_state[value] = 0
            if key.startswith('Key'):
                self.btn_state[value] = 0
                self.old_btn_state[value] = 0
        self._other = 0
        self.gamepad = gamepad
        if not gamepad:
            self._get_gamepad()

        self.gamepadThread = threading.Thread(target=monitorGamepad)
        self.gamepadThread.daemon = True
        self.gamepadThread.start()

    def _get_gamepad(self):
        """Get a gamepad object."""
        try:
            self.gamepad = inputs.devices.gamepads[0]
        except IndexError:
            raise inputs.UnpluggedError("No gamepad found.")

    def handle_unknown_event(self, event, key):
        """Deal with unknown events."""
        if event.ev_type == 'Key':
            new_abbv = 'B' + str(self._other)
            self.btn_state[new_abbv] = 0
            self.old_btn_state[new_abbv] = 0
        elif event.ev_type == 'Absolute':
            new_abbv = 'A' + str(self._other)
            self.abs_state[new_abbv] = 0
            self.old_abs_state[new_abbv] = 0
        else:
            return None

        self.abbrevs[key] = new_abbv
        self._other += 1

        return self.abbrevs[key]

    def process_event(self, event):
        """Process the event into a state."""
        if event.ev_type == 'Sync':
            return
        if event.ev_type == 'Misc':
            return
        key = event.ev_type + '-' + event.code
        try:
            abbv = self.abbrevs[key]
        except KeyError:
            abbv = self.handle_unknown_event(event, key)
            if not abbv:
                return
        if event.ev_type == 'Key':
            self.old_btn_state[abbv] = self.btn_state[abbv]
            self.btn_state[abbv] = event.state
            if self.btnCallback is not None:
                 self.btnCallback(abbv, self.btn_state.copy())

        if event.ev_type == 'Absolute':
            self.old_abs_state[abbv] = self.abs_state[abbv]
            state = event.state
            if abbv in self.deadzones and abs(state) < self.deadzones[abbv]:
                state=0
            
            self.abs_state[abbv] = state

        if self.print_state:
            self.output_state(event.ev_type, abbv)

    def format_state(self):
        """Format the state."""
        output_string = ""
        for key, value in self.abs_state.items():
            output_string += key + ':' + '{:>4}'.format(str(value) + ' ')

        for key, value in self.btn_state.items():
            output_string += key + ':' + str(value) + ' '

        return output_string

    def output_state(self, ev_type, abbv):
        """Print out the output state."""
        if ev_type == 'Key':
            if self.btn_state[abbv] != self.old_btn_state[abbv]:
                print(self.format_state())
                return

        if abbv[0] == 'H':
            print(self.format_state())
            return

        difference = self.abs_state[abbv] - self.old_abs_state[abbv]
        if (abs(difference)) > MIN_ABS_DIFFERENCE:
            print(self.format_state())

    def process_events(self):
        """Process available events."""

        try:
            #events = self.gamepad.read()
            events=gamepadEvents()
        except EOFError:
            events = []
        events = [event for event in events if event.ev_type!='Sync']
        for event in events:
            self.process_event(event)

        if self.print_state:
            print(len(events),"gamepad events")

        return len(events) > 0

    def diffDrive(self,abbv_x='RX',abbv_y='RY'):
        self.process_events()
        wheel_vels= joystickToDiff(self.abs_state[abbv_x], -self.abs_state[abbv_y], -32768, 32767, -1, 1)
        
        return wheel_vels



eventList = []
_eventLock = threading.Lock()
_monitorError = None

def monitorGamepad():
    global _monitorError
    while True:
        try:
            _monitorError = None
            for e in inputs.get_gamepad():
                with _eventLock:
                    eventList.append(e)
        except inputs.UnpluggedError:
            _monitorError = "Gamepad unplugged"
            time.sleep(0.5)
        except Exception as ex:
            _monitorError = str(ex)
            print(f"[gamepad] monitorGamepad error: {ex}")
            time.sleep(1.0)

def gamepadEvents():
    with _eventLock:
        copy = eventList[:]
        eventList.clear()
    return copy
    
def haveGamepad():
    return len(inputs.devices.gamepads)>0


def joystickToDiff(x, y, minJoystick, maxJoystick, minSpeed, maxSpeed):	
    # If x and y are 0, then there is not much to calculate...
	if x == 0 and y == 0:
		return {'right':0, 'left':0}
    

	# First Compute the angle in deg
	# First hypotenuse
	z = math.sqrt(x * x + y * y)

	# angle in radians
	rad = math.acos(math.fabs(x) / z)

	# and in degrees
	angle = rad * 180 / math.pi

	# Now angle indicates the measure of turn
	# Along a straight line, with an angle o, the turn co-efficient is same
	# this applies for angles between 0-90, with angle 0 the coeff is -1
	# with angle 45, the co-efficient is 0 and with angle 90, it is 1

	tcoeff = -1 + (angle / 90) * 2
	turn = tcoeff * math.fabs(math.fabs(y) - math.fabs(x))
	turn = round(turn * 100, 0) / 100

	# And max of y or x is the movement
	mov = max(math.fabs(y), math.fabs(x))

	# First and third quadrant
	if (x >= 0 and y >= 0) or (x < 0 and y < 0):
		rawLeft = mov
		rawRight = turn
	else:
		rawRight = mov
		rawLeft = turn

	# Reverse polarity
	if y < 0:
		rawLeft = 0 - rawLeft
		rawRight = 0 - rawRight

	# minJoystick, maxJoystick, minSpeed, maxSpeed
	# Map the values onto the defined rang
	rightOut = map(rawRight, minJoystick, maxJoystick, minSpeed, maxSpeed)
	leftOut = map(rawLeft, minJoystick, maxJoystick, minSpeed, maxSpeed)

	return {'right':rightOut, 'left':leftOut}


def map(v, in_min, in_max, out_min, out_max):
	# Check that the value is at least in_min
	if v < in_min:
		v = in_min
	# Check that the value is at most in_max
	if v > in_max:
		v = in_max
	return (v - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def main():  
    """Process all events forever."""
    #TODO add a eventlist max and clear to avoid endlessly growing before use
    gamepad = SimpleGamepad()
    gamepad.print_state=True
    while 1:
        time.sleep(.1)
        print("read all events")
        gamepad.process_events()
     

if __name__ == "__main__":  
    main()
