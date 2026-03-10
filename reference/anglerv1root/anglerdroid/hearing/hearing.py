#!/usr/bin/env python3

import threading
import speech_recognition as sr
import time
import sounddevice  #gets rid of all the ALSA messages printed to console (just importing it does that)
import copy
import io


def start(config, whiteFiber, brainSleeping):
    print("hearing: starting", flush=True)
    source=None
    mic=None

    axon = whiteFiber.axon(
        get_topics = [

        ],
        put_topics = [
            "/hearing/statement"
        ]
    )
    
    # obtain audio from the microphone
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = config['ears.mic_energy_min']
    recognizer.dynamic_energy_ratio = 1.7
    recognizer.dynamic_energy_adjustment_damping = 0.5
    recognizer.dynamic_energy_threshold=False

    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    #r.dynamic_energy_threshold = False
    #r.pause_threshold=2

    # Prevents permanent application hang and crash by using the wrong Microphone
    mic_name = config['ears.mic_device']
    if not mic_name or mic_name == 'list':
        print("hearing: Available microphone devices are: ", flush=True)
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"hearing: Microphone with name \"{name}\" found", flush=True)   
        return
    else:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                mic = sr.Microphone(sample_rate=16000, device_index=index)
                
                print ("hearing: mic",repr(source), flush=True)
                break

    #with mic as source: r.adjust_for_ambient_noise(source)

    def map_mic_energy(value):
        old_min = 150
        old_max = 800
        new_min = 100
        new_max = 10

        # Calculate the normalized value within the old range
        normalized_value = (value - old_min) / (old_max - old_min)

        # Map the normalized value to the new range
        new_value = new_min + (normalized_value * (new_max - new_min))

        return int(new_value)


    def transcribe(axon, recognizer, audio):
        try:
            audio2 = copy.deepcopy(audio)
            print("hearing: transcribing", flush=True)
            text = recognizer.recognize_google(audio2,key=None,language="en-US",pfilter=1,show_all=False,with_confidence=False)
            if len(text):
                axon["/hearing/statement"].put(text)
        except sr.UnknownValueError:
            recognizer.energy_threshold = min(800, recognizer.energy_threshold+100)
            
            #print("didn't understand")
            
        except sr.RequestError as e:
            print("hearing: Could not request results from Speech Recognition service; {0}".format(e), flush=True)    

    """    
        stop_hearing = recognizer.listen_in_background(
            mic, 
            lambda r, audio:transcribe(axon,r,audio),
            phrase_time_limit=15)
            
        while not brainSleeping.isSet():
            print("Kevin is listening...")    
            time.sleep(1)

        stop_hearing(True)
        print("stopped hearing")
    """
    #mic energy should be 200 (silence) to 800 (air conditioner)

    print("hearing: ready", flush=True)
    while not brainSleeping.isSet():
        try:
            #print("Kevin is listening...")
            print("hearing: mic level", str(map_mic_energy(recognizer.energy_threshold))+'%' , flush=True)
            with mic as source:audio = recognizer.listen(source, timeout=2,phrase_time_limit=15)

            transcriber = threading.Thread(target=transcribe, args=(axon, recognizer, audio),daemon=True)
            transcriber.start()
            
        
        except sr.WaitTimeoutError:
            #with mic as source: recognizer.adjust_for_ambient_noise(source)
            #print("heard nothing")
            recognizer.energy_threshold = max(150, recognizer.energy_threshold-20)

        time.sleep(.01)

    print("hearing: stopped", flush=True)
