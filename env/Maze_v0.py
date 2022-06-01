import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

class Maze_v0(tk.Tk, object):

    def __init__(self, unit=40, length=20):

        super(Maze_v0, self).__init__()

        self.unit = unit
        self.length = length

        self.step = 0
        self.action_space = ['l', 'r']

        self.n_actions = len(self.action_space)
        self.n_features = 4
        self.title('maze_v0')

        self.geometry('{0}x{1}'.format(self.length * self.unit, 1 * self.unit))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=1 * self.unit,
                                width=self.length * self.unit)

        # create grids
        for c in range(0, self.length * self.unit, self.unit):
            x0, y0, x1, y1 = c, 0, c, self.length * self.unit
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, 1 * self.unit, self.unit):
            x0, y0, x1, y1 = 0, r, 1 * self.unit, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])


        # create oval
        oval_center = np.array([20, 20 + self.unit * (self.length - 1)])
        self.oval = self.canvas.create_oval(
            oval_center[1] - 15, oval_center[0] - 15,
            oval_center[1] + 15, oval_center[0] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        self.step = 0
        # time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])

        if action == 0:  # left
            if s[0] > self.unit:
                base_action[0] -= self.unit
        elif action == 1:  # right
            if s[0] < (self.length - 1) * self.unit:
                base_action[0] += self.unit


        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif s_ == s:
            reward = -1
            done = False
        else:
            reward = 0
            done = False
        self.step += 1
        # time.sleep(0.15)
        return s_, reward, done, self.step

    def render(self):
        # time.sleep(0.1)
        self.update()