class PlotManager:
    def __init__(self, visualization=None, compared_file=None, dimension=3, log_dir='', rotate=False):
        self.visualization = visualization
        self.compared_file = compared_file
        self.plotter = None
        self.plot_pipe = None
        self.plot_process = None
        self.plot_initialized = False
        self.dimension = dimension
        self.log_dir = log_dir
        self.rotate = rotate
        self.initialize_plot()

    def initialize_plot(self):
        if self.plot_initialized:
            return
        if self.visualization is not None:
            from .plotter import Plotter
            import matplotlib
            matplotlib.use('TkAgg')   
            self.plotter = Plotter(compared_file=self.compared_file, dimension=self.dimension, log_dir=self.log_dir, rotate=self.rotate)                
            if self.visualization == 'parallel':
                import multiprocessing as mp
                mp.freeze_support()
                ctx = mp.get_context('spawn')                                                 
                self.plot_pipe, plotter_pipe = ctx.Pipe()
                self.plot_process = ctx.Process(
                    target=self.plotter, args=(plotter_pipe,), daemon=True)
                self.plot_process.start()
            elif self.visualization == 'non-parallel':      
                self.plotter()
        self.plot_initialized = True

    def plot(self, **kwargs):
        if self.visualization is not None:
            from .plotter import Plotting
            p = Plotting()
            p.data = kwargs.get('data', None)
            p.model = kwargs.get('model', None)
            p.scores = kwargs.get('scores', None)
            assert p.scores is not None            
            p.times = kwargs.get('times', None)
            p.episodes = kwargs.get('episodes', None)
            p.optimizer_name = self.__class__.__name__
            p.runner_id = kwargs.get('runner_id', 0)
            p.rollout_id = kwargs.get('rollout_id', 0)
            p.model_labels = kwargs.get('model_labels', None)
            p.model_image = kwargs.get('model_image', None)
            if self.visualization == 'parallel':
                self.plot_pipe.send(p)
            elif self.visualization == 'non-parallel':
                self.plotter.plot(p)
            else:
                assert False

    def plot_model(self, model, data):
        if model is None:
            return
        if self.visualization is not None:
            from .plotter import Plotting
            plotting = Plotting()
            plotting.data = data
            plotting.model = model
            if self.visualization == 'parallel':
                self.plot_pipe.send(plotting)
            elif self.visualization == 'non-parallel':
                self.plotter.plot(plotting)
            else:
                assert False

    def close(self):
        if self.visualization != 'parallel':
            return
        self.plot_pipe.send('close')
        if self.plot_pipe.poll(200):
            try:
                self.plot_pipe.recv()
            except EOFError:
                pass
            except:
                assert False
        self.plot_process.terminate()
        self.plot_pipe.close()

