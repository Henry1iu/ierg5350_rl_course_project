import os
import time

import numpy as np
import matplotlib.pyplot as plt

# from NetFT import Sensor as FT_Sensor
from ATI_Sensor import Sensor as FT_Sensor

DISPLAY_WINDOW_NAME = "The data streamed form F/T sensor"
DISPLAY_DATALENGTH = 200

NETBOX_IP = "192.168.1.108"


# Dynamic update of the figure
class DynamicUpdate(object):
    def __init__(self, min_x, max_x):
        self.min_x = min_x
        self.max_x = max_x
        self.figure, self.ax = plt.subplots()
        self.ax.set(xlabel='time (s)', ylabel='Sensor Reading', title=DISPLAY_WINDOW_NAME)
        self.ax.grid()

        self.line1, = self.ax.plot([], [], 'r', label="force[0]")
        self.line2, = self.ax.plot([], [], 'g', label="force[1]")
        self.line3, = self.ax.plot([], [], 'b', label="force[2]")
        self.line4, = self.ax.plot([], [], 'c', label="torque[0]")
        self.line5, = self.ax.plot([], [], 'y', label="torque[1]")
        self.line6, = self.ax.plot([], [], 'm', label="torque[2]")

        # Auto-scale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)

        plt.legend()
        plt.show()

    def update(self, xdata, ydata):
        # Update data (with the new _and_ the old points)
        self.line1.set_xdata(xdata)
        self.line1.set_ydata(ydata[:, 0])

        self.line2.set_xdata(xdata)
        self.line2.set_ydata(ydata[:, 1])

        self.line3.set_xdata(xdata)
        self.line3.set_ydata(ydata[:, 2])

        self.line4.set_xdata(xdata)
        self.line4.set_ydata(ydata[:, 3])

        self.line5.set_xdata(xdata)
        self.line5.set_ydata(ydata[:, 4])

        self.line6.set_xdata(xdata)
        self.line6.set_ydata(ydata[:, 5])

        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()

        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


class Container(object):
    def __init__(self, container_size=DISPLAY_DATALENGTH, data_dimension=6):
        self.container_size = container_size
        self.data_dimension = data_dimension

        self._x = np.arange(0, self.container_size)
        self._y = None

        self.length = 0

    def update(self, y_new):
        length, dimension = y_new.shape
        assert length <= self.container_size, "The data is too large in the length!"
        assert dimension == self.data_dimension, "The data dimension doesn't fit!"

        if self.length == 0:
            self._y = y_new
            self.length += length

        if self.length + length > self.container_size:
            self._y = np.concatenate((self._y[length:, :], y_new), axis=0)
            self.length = self.container_size

        else:
            self._y = np.concatenate((self._y, y_new), axis=0)
            self.length += length


    @property
    def x(self):
        return self._x

    @property
    def y(self):
        if not isinstance(self._y, np.ndarray):
            return np.zeros((self.container_size, self.data_dimension))
        else:
            if self.length < self.container_size:
                return np.concatenate((self._y, np.zeros((self.container_size - self.length, self.data_dimension))), axis=0)
            else:
                return self._y


if __name__ == "__main__":
    # init window
    plt.ion()
    dynamic_update = DynamicUpdate(0, DISPLAY_DATALENGTH)

    # init data container
    data_container = Container()
    x = data_container.x

    # init sensor device
    ft_sensor = FT_Sensor(NETBOX_IP)
    ft_sensor.tare(500)
    ft_sensor.startStreaming()
    ft_sensor.recieve()

    # collect and display data
    while True:
        # receiving new data
        print("[Info]: Receiving new data...")
        data = ft_sensor.measurement()
        data = np.array(data, ndmin=2) / 1000000.0

        # store in the container
        print("[Info]: updating the data in the container...")
        data_container.update(data)

        # update the figure
        print("[Info]: updating the data in the figure...")
        dynamic_update.update(x, data_container.y)

        time.sleep(1/30)


