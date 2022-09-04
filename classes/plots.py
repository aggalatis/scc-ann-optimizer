import os
from matplotlib import pyplot as plt

class Plot:
    def __init__(self):
        self.plots_folder = os.getenv('PLOTS_FOLDER')

    def save_plot(self, train_data, val_data, model_id, type):
        try:
            plot_range = list(range(1, 101))
            plt.figure()
            plt.plot(plot_range, train_data, '-r', label='training data')
            plt.plot(plot_range, val_data, '-g', label='validation data')
            plt.xlabel('epochs')
            plt.ylabel(type)
            plt.legend(['Training Data', 'Validation Data'])
            plt.savefig(f"{self.plots_folder}\\{model_id}_{type}.png")
        except Exception as e:
            print(f"Saving exception: {e}")