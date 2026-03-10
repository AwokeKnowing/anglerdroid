import time
import cv2

def display_images(images):
    if len(images) > 0:
        for title, img in images:
            cv2.imshow(title, img)
        key = cv2.pollKey() & 0xFF
        if key != 255:
            return key
    return None
        
def start(config, whiteFiber, brainSleeping):
    cv2.startWindowThread()
    cv2.pollKey()
    
    print("display: starting", flush=True)
    axon = whiteFiber.axon(
        get_topics=[
            "/display/imshow"
        ],
        put_topics=[
            "/keyboard/press" #because opencv requires read keyboard
        ]
    )
    
    print("display: ready", flush=True)    
    while not brainSleeping.isSet():
        images = axon["/display/imshow"].get_all()
        keyCode = display_images(images)
        if keyCode is not None:
            axon['/keyboard/press'].put(chr(keyCode))
            print(keyCode)

        time.sleep(.01)
        
    print("display: stopped", flush=True)
        
     