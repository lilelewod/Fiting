import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from .geometry import set_axes_equal
# matplotlib.use('TkAgg')


class Plotting:
    def __init__(self):
        self.data = np.empty((0, 3))
        self.model = np.empty((0, 3))
        self.scores = [1, 4, 9, 16]
        self.times = [1, 2, 3, 4]
        self.episodes = [1, 2, 3, 4]
        self.optimizer_name = 'optimizer'
        self.model_labels = None
        self.model_image = None
        self.runner_id = 0
        self.rollout_id = 0


class Plotter:

    def __init__(self, compared_file, dimension=3, log_dir='', rotate=False):
        self.pipe = None
        self.timer = None
        self.dimension = dimension
        self.log_dir = log_dir
        self.rotate = rotate
        if compared_file is None:
            self.compared = None
            return
        try:
            with open(compared_file, 'rb') as f:
                self.compared = pickle.load(f)
        except (EOFError, FileNotFoundError) as e:
            self.compared = None
            print(e)

    def update(self, pipe):
        if pipe.poll():
            plotting = pipe.recv()
            if plotting == 'close':
                self.close()
            else:
                self.plot(plotting)
    
    def close(self):
        self.timer.stop()
        self.pipe.close()
        print('plotter is closed')

    def __call__(self, pipe=None):
        self.pipe = pipe
        print("backend is " + matplotlib.get_backend())

        if self.dimension == 3:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            set_axes_equal(self.ax)
        elif self.dimension == 2:
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal')
            # self.ax.set_axis_off()
        else:
            assert False
        self.fig2 = plt.figure()
        self.ax2 = self.fig2.add_subplot(222)
        # self.fig3 = plt.figure()
        self.ax3 = self.fig2.add_subplot(223)
        self.ax4 = self.fig2.add_subplot(224)
        if pipe is None:
            plt.ion()
        else:
            self.timer = self.fig.canvas.new_timer(interval=100)
            self.timer.add_callback(self.update, pipe)
            self.timer.start()
        plt.show()

    def plot(self, plotting=Plotting()):
        p = plotting
        data = plotting.data
        model = plotting.model
        labels = p.model_labels 
        assert data.shape[1] == model.shape[1]
        assert self.dimension == data.shape[1]    
        if self.dimension == 2:
            self.plot2d(data, model, labels)
        elif self.dimension == 3:
            self.plot3d(data, model, labels)
        else:
            assert False
    
        self.fig.suptitle(f'score: {p.scores[-1]:.4f}; episode: {p.episodes[-1]}, rollout: {p.rollout_id}, runner: {p.runner_id}')
        self.fig.canvas.draw()   
        self.fig.savefig(self.log_dir+'result.pdf')             

        if p.model_image is not None:
            self.ax2.cla()
            self.ax2.imshow(p.model_image)

        self.ax3.cla()
        self.ax3.set_xlabel("time (s)")
        self.ax3.set_ylabel("score")
        self.ax3.plot(p.times, p.scores, 'r', label=p.optimizer_name)
        if self.compared is not None:
            self.ax3.plot(self.compared['evolved_time'], self.compared['evolved_scores'], 'g',
                          label=self.compared['cfg'].modeler_name)
        self.ax3.legend()
        # self.fig2.canvas.draw()

        self.ax4.cla()
        self.ax4.set_xlabel("episode")
        self.ax4.set_ylabel("score")
        self.ax4.plot(p.episodes, p.scores, 'r', label=p.optimizer_name)
        if self.compared is not None:
            self.ax4.plot(self.compared['evolved_iterations'], self.compared['evolved_scores'], 'g',
                          label=self.compared['cfg'].modeler_name)
        self.ax4.legend()
        # self.fig3.canvas.draw()

        self.fig2.canvas.draw()

    def plot3d(self, data, model, color):  
        self.ax.cla()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        # set_axes_equal(self.ax)             

        if data.size > 0:
            self.ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='black', marker='.', s=10.)

        if model.size > 0:      
            if color is None:
                self.ax.scatter(model[:, 0], model[:, 1], model[:, 2], c='g', marker='.', s=50.)
            else:
                assert color.shape[0] == model.shape[0]
                self.ax.scatter(model[:, 0], model[:, 1], model[:, 2], c=color, cmap='rainbow', marker='.', s=40.)


    def plot2d(self, data, model, labels):
        self.ax.cla()
        # self.ax.set_xlabel("x")
        # self.ax.set_ylabel("y")
        # self.ax.set_aspect('equal')

        if data.size > 0:
            (x, y) = (data[:, 1], -data[:, 0]) if self.rotate else (data[:, 0], data[:, 1])
            self.ax.scatter(x, y, c='black', marker='.', s=20.) 
        if model.size > 0:
            (x, y) = (model[:, 1], -model[:, 0]) if self.rotate else (model[:, 0], model[:, 1])            
            if labels is None:
                # (x, y) = (model[:, 1], -model[:, 0]) if self.rotate else (model[:, 0], model[:, 1])                
                self.ax.scatter(x, y, c='g', marker='.', s=40.)              
            else:
                assert labels.shape[0] == model.shape[0]
                # self.ax.scatter(model[:, 0], model[:, 1], c=labels, cmap='rainbow', marker='.', s=100.)
                self.ax.scatter(x, y, c=labels, cmap='rainbow', marker='.', s=40.)                
        self.ax.set_axis_off()
        


