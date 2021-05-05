from utils.dynamic_plot import DynamicUpdate


class AccuracyPlot( DynamicUpdate ):    
    def __init__(self):
        super(AccuracyPlot, self).__init__()
        # Launch the graph
        self.on_launch()
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Test Accuracy")

    # The epoch function definition called in the trainer class
    def EpochCallback(self, epoch=None, accuracies=None, net=None, opt=None, cost=None):
        self.on_running(range(epoch+1), accuracies[:epoch+1])
