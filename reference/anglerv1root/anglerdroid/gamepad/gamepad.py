import time

from .simplegamepad import SimpleGamepad
        
def start(config, whiteFiber, brainSleeping):
    print("gamepad: starting", flush=True)
    axon = whiteFiber.axon(
        get_topics=[
            
        ],
        put_topics=[
            "/gamepad/diffdrive/leftrightvels",
            "/gamepad/buttons/press"
        ]
    )

    def onButtonPress(key, states):
        nonlocal axon
        if states[key] == 1:
            axon['/gamepad/buttons/press'].put((key, states))
        #print("BUTTON PRESS: ", key, states[key])

    
    gamepad = SimpleGamepad(btnCallback=onButtonPress)
    gamepad.print_state=False # config

    print("gamepad: ready", flush=True)    
    while not brainSleeping.isSet():
        time.sleep(.1)
        vels_norm=gamepad.diffDrive()
        axon["/gamepad/diffdrive/leftrightvels"].put((vels_norm['left'],vels_norm['right']))

    print("gamepad: stopped", flush=True)
        
     