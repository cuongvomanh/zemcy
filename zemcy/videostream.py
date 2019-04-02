# -*- coding: utf-8 -*-
from threading import Thread
import sys
is_py2 = sys.version[0] == '2'
if is_py2:
    import Queue
else:
    from queue import Queue
import time
import cv2

class StreamStat:
    def __init__(self):
        self.frmCount = 0
        self.frmDrop = 0
        self.frmGot = 0

class QueuedStream:
    '''Threaded video decoding stream
    '''
    def __init__(self, uri, queueSize=8, fps=15):
        '''init
        fps: frame per second for offline file, 0 for live stream'''
        try:
            uri = int(uri)
        except:
            pass

        self.fps = fps
        if len(str(uri))==0 or uri==0:
            self.stream = cv2.VideoCapture(-1)
            self.fps = 0
        else:
            self.stream = cv2.VideoCapture(uri)
            if uri.startswith('rtsp://'):
                self.fps = 0
        self.stopped = False
        self.queue = Queue(maxsize=queueSize)
        self.opened = self.stream.isOpened()
        self.stat = StreamStat()
        self.dropObs = []

    def start(self):
        '''start a thread to read frames from the file video stream'''
        self.th = Thread(target=self._thread_func, args=())
        self.th.daemon = True
        self.th.start()
        return self

    def read(self):
        '''blocking get next frame in the queue'''
        try:
            frame = self.queue.get(True, 5)
            if frame is None:
                return (False, None)
            self.stat.frmGot = self.stat.frmGot + 1
            return (True, frame)
        except:
            return (False, None)

    def hasMore(self):
        '''return True if there are still frames in the queue'''
        return self.queue.qsize() > 0

    def stop(self):
        '''Stop'''
        if not self.stopped:
            self.stopped = True
            self.th.join()
            self.stream.release()

    def isOpened(self):
        '''isOpened'''
        return self.isOpened

    def release(self):
        '''release'''
        self.stop()

    def addDropObserver(self, observer):
        '''Register a dropped frame observer'''
        self.dropObs.append(observer)

    def removeDropObserver(self, observer):
        '''Remove dropped frame observer'''
        self.dropObs.remove(observer)

    def _thread_func(self):
        '''keep looping infinitely'''
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                return
            if self.queue.full():
                f = self.queue.get(False)
                self.stat.frmDrop = self.stat.frmDrop + 1
                for obs in self.dropObs:
                    obs.frameDropped(f)
            self.queue.put(frame)
            self.stat.frmCount = self.stat.frmCount + 1
            if self.fps > 0:
                time.sleep(1.0 / self.fps)