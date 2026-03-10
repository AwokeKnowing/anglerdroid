import threading
import time
import queue
from queue import Queue

class WhiteFiber:
    def __init__(self, topics):
        self.topics = topics
        self.put_queues = {topic: Queue() for topic in topics}
        self.get_queues = {topic:[] for topic in topics}      
        self.broadcast_thread = None
        
    def axon(self, get_topics=None, put_topics=None, maxsize=1000):
        if get_topics is None:
            get_topics = []

        if put_topics is None:
            put_topics = []
        
        # new queues broadcast messages to the new subscriber
        get_queues = {topic: Queue(maxsize=1000) for topic in get_topics}

        # selection of put queues to listen to
        put_queues = {topic: self.put_queues[topic] for topic in put_topics}

        # store the new queues for publishing
        for topic, q in get_queues.items():
            self.get_queues[topic].append(q)

        # might need to store them?        
        subscriber = Subscriber(get_topics, put_topics, get_queues, put_queues)

        # start up the thread 
        if self.broadcast_thread is None:
            self.broadcast_thread = threading.Thread(target=self._broadcast,daemon=True)
            self.broadcast_thread.start()

        return subscriber
    
    def _broadcast(self):
        while True:
            for topic, queue in self.put_queues.items():
                #if topic == "/plan/motion/diffdrive/leftrightvels":
                #    print("here")
                while not queue.empty():
                    data = queue.get()
                    for get_queue in self.get_queues[topic]:
                        #print("broadcasting", topic, data)
                        get_queue.put(data)

            time.sleep(.001)


class Subscriber:
    def __init__(self, get_topics, put_topics, get_queues, put_queues):
        self.get_topics = get_topics
        self.put_topics = put_topics
        self.get_queues = get_queues
        self.put_queues = put_queues
        self.queues={}
        self.topics = set()
        self.topics.update(self.get_topics)
        self.topics.update(self.put_topics)

        for topic in self.topics:
            get_q, put_q = None, None
            if topic in self.get_topics:
                get_q=self.get_queues[topic]
            if topic in self.put_topics:
                put_q=self.put_queues[topic]
            self.queues[topic] = TopicProxy(get_q, put_q)

    def __getitem__(self, topic):
        return self.queues[topic]
        

class TopicProxy:
    # todo: set maxsize here
    def __init__(self,get_q, put_q):
        self.get_q = get_q
        self.put_q = put_q

    def put(self, data):
        #print("putting",data)
        return self.put_q.put(data)

    def get(self, block=False,throwOnEmpty=False):
        if block:
            return self.get_q.get()
        
        if throwOnEmpty:
            return self.get_q.get(False)
        else:
            try:
                return self.get_q.get(False)
            except queue.Empty:
                return None
            
    def get_all(self):
        items=[]
        while True:
            item = self.get()
            if item is None:
                break
            items.append(item)

        return items
        


    def get_nowait(self):
        return self.get_q.get_nowait()         
   

if __name__ == '__main__':
    topics = [
        '/a/b/c',
        '/my/topic/thing',
        '/my/other',
        '/my/something/else'
    ]

    fiber = WhiteFiber(topics)

    # Subscriber 1
    sub1 = fiber.axon(['/a/b/c'],['/a/b/c'])
    sub1['/a/b/c'].put('hi')

    print("sub2")
    # Subscriber 2
    sub2 = fiber.axon(['/my/other', '/my/topic/thing'],['/my/other', '/my/topic/thing'])
    sub2['/my/other'].put(4)
    sub2['/my/topic/thing'].put('test')

    print("sub3")
    # Subscriber 3
    sub3 = fiber.axon(['/my/other'],['/my/other'])
    sub3['/my/other'].put(7)

    print(sub1['/a/b/c'].get())  # Output: hi
    print(sub2['/my/other'].get())  # Output: 4
    print(sub2['/my/topic/thing'].get())  # Output: test
    print(sub3['/my/other'].get())  # Output: 7
