import time


class ThrottledVar:
    def __init__(self, var, min_interval):
        self.var = var
        self.min_interval = min_interval
        self.last_time = 0

    def trace(self, mode, callback):
        def throttled_callback(*args):
            current_time = time.time()
            if current_time - self.last_time >= self.min_interval:
                self.last_time = current_time
                callback(*args)
        self.var.trace(mode, throttled_callback)

    def trace_add(self, mode, callback):
        def throttled_callback(*args):
            current_time = time.time()
            if current_time - self.last_time >= self.min_interval:
                self.last_time = current_time
                callback(*args)
        self.var.trace_add(mode, throttled_callback)

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(value)
