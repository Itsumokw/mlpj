import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ForecastVisualizer:
    @staticmethod
    def plot_result(true: list, pred: list, save_path: str = None):
        plt.figure(figsize=(10,4))
        plt.plot(true, label='True Values', color='blue')
        plt.plot(pred, linestyle='--', label='Predictions', color='red')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    @classmethod
    def create_animation(cls, history: dict, save_path: str = 'forecast.gif'):
        fig, ax = plt.subplots()
        line_true, = ax.plot([], [], 'b-', label='True')
        line_pred, = ax.plot([], [], 'r--', label='Predicted')
        
        def init():
            ax.set_xlim(0, history['total_steps'])
            ax.set_ylim(min(history['true'])*0.9, max(history['true'])*1.1)
            return line_true, line_pred

        def update(frame):
            line_true.set_data(range(frame), history['true'][:frame])
            line_pred.set_data(range(frame), history['pred'][:frame])
            return line_true, line_pred

        ani = FuncAnimation(fig, update, frames=range(1, len(history['true'])+1),
                            init_func=init, blit=True)
        ani.save(save_path, writer='pillow', fps=5)