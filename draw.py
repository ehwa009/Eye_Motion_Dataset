import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np

from matplotlib import animation

class Display:

    def __init__(self, x_lim, y_lim, lw=1):
        self.fig = plt.figure()
        ax = plt.axes(xlim=x_lim, ylim=y_lim)

        # left eyebrow
        [self.leb_line1] = ax.plot([], [], lw=lw, c='green')

        # right eyebrow
        [self.reb_line1] = ax.plot([], [], lw=lw, c='green')

        # left eye region
        [self.ler_line1] = ax.plot([], [], lw=lw, c='green')

        # right eye region
        [self.rer_line1] = ax.plot([], [], lw=lw, c='green')

        # left and right pupil
        self.lep_dot = ax.scatter(0, 0, lw=0.1, c='black')
        self.rep_dot = ax.scatter(0, 0, lw=0.1, c='black')

    def init_line(self):
        self.leb_line1.set_data([], [])
        self.reb_line1.set_data([], [])
        self.ler_line1.set_data([], [])
        self.rer_line1.set_data([], [])
        self.lep_dot.set_offsets(np.array([0,0]))
        self.rep_dot.set_offsets(np.array([0,0]))

        return([self.leb_line1, self.reb_line1, self.ler_line1, self.rer_line1, self.lep_dot, self.rep_dot])

    def animate_frame(self, landmarks):
        self.leb_line1.set_data(landmarks[38:48:2], landmarks[39:48:2])
        self.reb_line1.set_data(landmarks[28:38:2], landmarks[29:38:2])
        self.ler_line1.set_data(landmarks[4:16:2], landmarks[5:16:2])
        self.rer_line1.set_data(landmarks[16:28:2], landmarks[17:28:2])
        self.lep_dot.set_offsets(np.array([landmarks[0], landmarks[1]]))
        self.rep_dot.set_offsets(np.array([landmarks[2], landmarks[3]]))

        return([self.leb_line1, self.reb_line1, self.ler_line1, self.rer_line1, self.lep_dot, self.rep_dot]) 

    def animate(self, frame_to_play, interval):
        anim = animation.FuncAnimation(self.fig, 
                                    self.animate_frame,
                                    init_func=self.init_line, 
                                    frames=frame_to_play,
                                    interval=interval, blit=True)

        return anim


if __name__ == '__main__':
    d = Display((-800, -200), (-400, 0))
    with open('./facial_keypoints/Xo9J_G1cTsk.pickle', 'rb') as f:
        sk = pickle.load(f)
    sk = np.array(sk)
    sk *= -1
    anim = d.animate(sk, 50)
    plt.show() 
    exit(-1)   


   
                

