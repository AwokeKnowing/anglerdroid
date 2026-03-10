#! python3.7

import argparse
import multiprocessing as mp
import threading
import itertools
import time

import cowsay

import anglerdroid as a7


class Brain:
    def __init__(self,configJson):
        self.configJson=configJson

        self.topic_defs = {
            "/hearing/statement":                   "string",
            "/language/chat/in/statement":          "string",
            "/language/chat/out/statement":         "string",
            "/voice/statement":                     "string",
            "/gamepad/buttons/press":               "tuple(key, states)",
            "/keyboard/press":                      "string",
            "/gamepad/diffdrive/leftrightvels":     "tuple(left,right)",
            "/wheels/diffdrive/leftrightvels":      "tuple(left,right)",
            "/odometry/wheels/leftrightvels":       "tuple(left,right)",
            "/vision/images/topdown":               "BGRImage",
            "/plan/motion/diffdrive/leftrightvels": "tuple(left,right)",
            "/plan/motion/goalxy":                  "tuple(x,y)",
            "/display/imshow":                      "tuple(title,img,wait)",
            
            
        }

        self.input = {
            "hearing": a7.hearing.start,
            "gamepad": a7.gamepad.start,
            "vision":  a7.vision.start,
        }
        self.output = {
            "display": a7.display.start,
            "voice": a7.voice.start,
            "wheels": a7.wheels.start,
        }
        self.processors = {
            "language": a7.language.start,
            "planmotion": a7.planmotion.start
        }

        self.handlers = {}

        self.brainSleeping = threading.Event()

    def wake(self):
        print("brain: waking up", flush=True)
        with a7.Configuration(self.configJson) as robotConfig:
            self.is_being_addressed=False
            self.robot_asked_question=False

            self.whiteFiber = a7.WhiteFiber(self.topic_defs.keys())

            axon=self.whiteFiber.axon(
                get_topics=[
                    "/hearing/statement",
                    "/language/chat/out/statement",
                    "/gamepad/diffdrive/leftrightvels",
                    "/gamepad/buttons/press",
                    "/keyboard/press",
                    "/plan/motion/diffdrive/leftrightvels",
                ],
                put_topics=[
                    "/language/chat/in/statement",
                    "/voice/statement",
                    "/wheels/diffdrive/leftrightvels",
                    "/plan/motion/goalxy"
                ]

            )
            

            # Create a process for the worker functions
            for pname, pstart in itertools.chain(self.input.items(), self.output.items(), self.processors.items()):
                self.handlers[pname] = threading.Thread(target=pstart, args=(robotConfig,self.whiteFiber,self.brainSleeping))
                self.handlers[pname].daemon = True
                self.handlers[pname].start()

            cowsay.tux("Hello!")
            goalx = 96
            goaly = 0

            auto_nav = False
            gamepad_to_vel = .5

            # Read messages from the queue
            while not self.brainSleeping.isSet():
                try:
                    message =       axon["/hearing/statement"].get()
                    wheelVelsPad =  axon["/gamepad/diffdrive/leftrightvels"].get()
                    wheelVelsPlan = axon["/plan/motion/diffdrive/leftrightvels"].get()
                    chatToSay =     axon["/language/chat/out/statement"].get()
                    keyPress =      axon["/keyboard/press"].get()
                    buttonPress =   axon["/gamepad/buttons/press"].get()

                    if buttonPress is not None:
                        print("brain: BUTTON: ",buttonPress[0], flush=True)

                        if buttonPress[0]=="ST":
                            auto_nav = not auto_nav
                            if auto_nav:
                                print("brain: AUTONOMOUS NAVIGAION ENABLED", flush=True)
                            else:
                                print("brain: MANUAL NAVIGATION ENABLED", flush=True)
                        

                    wheelVels = None

                    if wheelVelsPad is not None:
                        wheelVelsPad = (wheelVelsPad[0] * gamepad_to_vel, wheelVelsPad[1] * gamepad_to_vel)
                        
                        wheelVels = wheelVelsPad


                    if auto_nav:
                        wheelVels = wheelVelsPlan

                        #in auto nav, if gamepad is doing something, use that instead
                        if wheelVelsPad is not None and abs(wheelVelsPad[0])>0 and abs(wheelVelsPad[1])>0:
                            wheelVels = wheelVelsPad

                    
                    
                    if wheelVels is not None:
                        left,right = wheelVels
                        if left or right:
                            print("brain: commanding wheels ",wheelVels, "auto_nav", auto_nav, flush=True)

                        axon["/wheels/diffdrive/leftrightvels"].put(wheelVels)
                    


                        
                    
                    # Do something with the message
                    if message:
                        if len(message)>0 and message != " " and message != ".":
                            cowsay.cow(message)
                        if message.lower().startswith("turn yourself off") and len(message)<19:
                            break

                        words = ''.join([c for c in message if c not in '¡!¿?,.']).lower().split()
                        print("words", words)
                        words_to_robot = self.get_words_addressed_to_robot(words)
                        if len(words_to_robot) > 1 or self.robot_asked_question:
                            print("brain: Thinking...", flush=True)
                            axon["/language/chat/in/statement"].put(message)
                            

                    if chatToSay:
                        self.robot_asked_question = chatToSay[-1]=="?"
                        cowsay.tux(chatToSay)
                        axon['/voice/statement'].put(chatToSay)


                    if keyPress is not None:
                        if keyPress == 'w': goaly = goaly - 10
                        if keyPress == 's': goaly = goaly + 10
                        if keyPress == 'a': goalx = goalx - 10
                        if keyPress == 'd': goalx = goalx + 10
                        
                        axon["/plan/motion/goalxy"].put((goalx,goaly))
                    time.sleep(.0001)
                        
                except KeyboardInterrupt:
                    cowsay.tux("Goodbye!")
                    break
            
            self.shutdown()
            cowsay.tux("Zzzz")
                    
    def shutdown(self):
        self.brainSleeping.set()
        time.sleep(5)
        for name,handler in self.handlers.items():
            if handler.isAlive():
                print("brain: waiting for "+ name, flush=True)
                handler.join(timeout=20)
                if handler.isAlive():
                    print("brain: force stopping "+name, flush=True)

    def get_words_addressed_to_robot(self, words, name='kevin', before=8):
        try:
            name_index = words.index(name)
            start_index = max(0, name_index - before)
            return words[start_index:]
        except ValueError:
            return []
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("brain: sleeping", flush=True)


if __name__ == "__main__":
    Brain().wake()