import threading
import time

class IdleTimer:
    def __init__(self, seconds=5, callback=None, sensitivity=0.1):
        self.threshold = seconds
        self.sensitivity = sensitivity
        self.last_update = time.time()
        self.timer_event = threading.Event()
        self.timer_thread = threading.Thread(target=self.idle_check)
        self.callback = callback
        self.timer_thread.daemon = True
        self.callback_triggered=False
        self.enable()

    def activity(self):
        print("idletimer: activity", flush=True)
        self.last_update = time.time()
        self.callback_triggered = False

    def idle_check(self):
        while not self.timer_event.is_set():
            time.sleep(self.sensitivity)
            if self.callback_triggered:
                continue
            
            elapsed_time = time.time() - self.last_update
            #print("idle", elapsed_time)
            if elapsed_time >= self.threshold:
                if self.callback is not None:
                    self.callback()
                self.callback_triggered=True
            

    def enable(self, seconds=None):
        print("idletimer: enable", flush=True)
        self.activity()
        if seconds is not None:
            self.threshold = seconds
        
        self.timer_event.clear()
        if self.timer_thread is None or not self.timer_thread.is_alive():
            self.timer_thread = threading.Thread(target=self.idle_check)
            self.timer_thread.daemon = True
            self.timer_thread.start()

    def disable(self):
        print("idletimer: disable", flush=True)
        self.timer_event.set()
        if self.timer_thread:
            self.timer_thread.join()
            self.timer_thread = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()

    def __enter__(self):
        return self

if __name__ == '__main__':
    def idle_callback():
        print("Idle threshold reached. Performing cleanup...")

    # Example usage of IdleTimer as a context manager
    with IdleTimer(seconds=2, callback=idle_callback) as t:
        print("Idle timer started.")

        # Perform an activity after 1 second
        time.sleep(1)
        t.activity()

        # Sleep for 3 seconds to trigger the callback
        time.sleep(3)

    print("Idle timer ended.")