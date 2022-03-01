import time
import sys


class progressBar():
    def __init__(self, lst):
        self.lst = lst

    def start(self):
        self.t0 = time.time()
    
    def update(self, i, msg=""):
        percent = (i + 1) / abs(((self.lst.start - self.lst.stop) / self.lst.step))
        percent = percent * 100
        k = 0
        bar = ""
        for j in range(0, int((percent * 25) / 100)):
            bar += "="
            k = j
        if (k < 24):
            bar += ">"
        t = time.time() - self.t0
        eta = ((100 * t) / percent) - t
        if (eta < 0):
            eta = 0
        sys.stdout.write(f'ETA: {eta:.2f}s [{percent:3.0f}%][{bar:25}] {i + 1}/{self.lst.stop} | elapsed time {t:.2f}s {msg}\r')
        sys.stdout.flush()
        bar = ""
        if (i + 1 == self.lst.stop):
            print()