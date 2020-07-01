
import matplotlib.pyplot as plt


def plot_precision_recall_curve(metrics, at=20, name=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    line_kwargs = {"drawstyle": "steps-post"}
    if name is not None:
        line_kwargs['label'] = name
    line_kwargs = dict(line_kwargs, **kwargs)

    line = ax.plot(metrics[f'r@{at}'], metrics[f'p@{at}'], **line_kwargs)
    ax.set(xlabel="Recall", ylabel="Precision")

    if "label" in line_kwargs:
        ax.legend(loc='upper right')

    return ax, line
