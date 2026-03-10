import time
import cv2
import threading

from .pathfinder import DynamicPathfinder

        
def start(config, whiteFiber, brainSleeping):
    
    print("planmotion: starting", flush=True)
    axon = whiteFiber.axon(
        get_topics=[
            "/vision/images/topdown",
            "/odometry/wheels/leftrightvels",
            "/plan/motion/goalxy"
        ],
        put_topics=[
            "/plan/motion/diffdrive/leftrightvels",
            "/display/imshow"
        ]
    )
    
    wheel_space = .34 *1.8 # in meters
    pixel_size = .025 * 1.0
    wheel_diameter=.172 *.5

    
    botx=96
    boty=155

    pathfinder = DynamicPathfinder((botx, boty), 
                                   wheel_space, 
                                   wheel_diameter, 
                                   pixel_size,
                                   lookahead_sec=6,
                                   max_rev_per_sec=2.8)
    

    left = 0.0
    right = 0.0

    goalx = 96
    goaly = 0


    print("planmotion: ready", flush=True)    
    
    while not brainSleeping.isSet():
        time.sleep(.1)
        #time.sleep(1.5)
        newGoal =  axon["/plan/motion/goalxy"].get_all()
        if len(newGoal) > 0:
            goalx, goaly = newGoal[-1]
        
        wheelVels = axon["/odometry/wheels/leftrightvels"].get_all()          
        if len(wheelVels)>0:
            left,right = wheelVels[-1]
            if left or right:
                print("planmotion: got wheel odometry",left,right, flush=True)

        obstacles = axon["/vision/images/topdown"].get_all()
        if len(obstacles) == 0:
            continue
        else:
            obstacles= obstacles[-1]
            axon['/display/imshow'].put(("obstacles here",obstacles))
            #show=obstacles.copy()

          
        path = pathfinder.next_wheel_vels(obstacles, (goalx, goaly),(left, right)) 
        new_left,new_right=path['vels']
        print("planmotion: suggests new vels ",new_left,new_right, flush=True)
        show=obstacles.copy()
        pathfinder.draw_path(show, path['points'],(255,))
        cv2.circle(show, (goalx, goaly), int((wheel_space/pixel_size)/2), (128,), -1)
        cv2.circle(show, (botx, boty), int((wheel_space/pixel_size)/2), (64,), -1)
        #cv2.imshow("path", show)
        
        axon['/display/imshow'].put(("path", show))
        axon['/display/imshow'].put(("paths", path['image']))
        axon["/plan/motion/diffdrive/leftrightvels"].put((new_left, new_right))

    print("planmotion: stopped", flush=True)
        
     