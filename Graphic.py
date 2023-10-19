import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150

class graphics:
    def print(*args, xlabel: str = None, ylabel: str = None, title: str = None, 
              log = False, xline = None, yline_min = None, yline_max = None):
        not_none_params = [item for item in args if item is not None]
        plt.plot(*not_none_params)

        plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
            plt.legend(labels=(ylabel), loc='upper left')
            plt.title(f"Зависимость {ylabel} от {xlabel}")
        if title is not None: plt.title(title)

        if (log): plt.yscale('log')
        plt.grid(True)
        plt.vlines(xline, yline_min, yline_max, colors = "black")
        plt.show()