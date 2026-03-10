import time

from anglerdroid.vision.anglerdroidvision import AnglerDroidVision




def start(config, whiteFiber, brainSleeping):
    print("vision: starting", flush=True)

    axon = whiteFiber.axon(
        get_topics = [

        ],
        put_topics = [
            "/vision/images/topdown",
            "/display/imshow"
        ]
    )

   
    rs_forward_sn=config['vision.realsense_forward_serial']
    # Create a context object for the Intel RealSense camera

    #config['vision.realsense_topdown_serial']
    #config['vision.realsense_forward_serial']
    

    print("vision: ready", flush=True)
    with AnglerDroidVision(rsTopdownSerial="815412070676", 
                           rsForwardSerial="944622074292",
                           webForwardDeviceId='/dev/video12',
                           axon=axon) as a7vis:
        
        while not brainSleeping.isSet():
            time.sleep(.0001)
            a7vis.update() 
            if a7vis.state.obstacles_img is None:
                continue
            axon["/vision/images/topdown"].put(a7vis.state.obstacles_img)         
    
    print("vision: stopped", flush=True)

print("in vision")

if "__name__" == "__main__":
    print("starting vision test")
    with AnglerDroidVision(rsTopdownSerial="815412070676", 
                        rsForwardSerial="944622074292",
                        webForwardDeviceId='/dev/video12',
                        axon=None) as a7vis:
    
        try:
            while True:
                time.sleep(.0001)
                a7vis.update() 
                if a7vis.state.obstacles_img is None:
                    continue
                
        except KeyboardInterrupt:
            pass
            
    print("vision test ended")
