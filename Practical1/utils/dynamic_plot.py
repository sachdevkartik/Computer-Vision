# Credit:
# https://exceptionshub.com/dynamically-updating-plot-in-matplotlib.html


import matplotlib.pyplot as plt

class DynamicUpdate():
    #Suppose we know the x range
    min_x = 0
    max_x = 10

    def __init__(self):
        plt.ion()

    def set_min(self, mi):
        self.min_x = mi

    def set_max(self, ma):
        self.max_x = ma

    def new_line(self, title=None):
        self.lines, = self.ax.plot([],[], label=title)
        if title:
            self.add_legend()

    def add_legend(self):
        self.ax.legend(loc="best")

    def on_launch(self):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        # self.new_line()
        # self.lines, = self.ax.plot([],[])
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        # self.ax.grid()

    def on_running(self, xdata, ydata):
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        # Update the axis
        self.max_x = max(self.max_x, len(ydata))
        self.ax.set_xlim(self.min_x, self.max_x)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def show(self):
        import time
        while True:
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            time.sleep(0.05)

    #Example
    def __call__(self):
        import numpy as np
        import time
        self.on_launch()
        xdata = []
        ydata = []
        for x in np.arange(0,10,0.5):
            xdata.append(x)
            ydata.append(np.exp(-x**2)+10*np.exp(-(x-7)**2))
            self.on_running(xdata, ydata)
            time.sleep(1)
        return xdata, ydata

