import matplotlib.pyplot as plt
import numpy as np


def plot_r2_lines():
    xmin, xmax, ymin, ymax = -3, 3, -3, 3
    ticks_frequency = 1

    # vectors
    xs, ys = [-3, -2, -1, 0, 1, 2, 3], [-2, -1, 0.5, 0.6, 1, 1.5, 2]
    # Plot points
    fig, axs = plt.subplots(1, 2, figsize=(8, 8))
    # ax.scatter(xs, ys, c=colors)
    ax = axs[0]
    ax.set_title("SSE")
    ax.scatter(xs, ys, c="blue")

    ax.plot([-5, 5], [-5, 5], c="black", ls='-', lw=1.5, alpha=0.5)
    ax.plot([-5, 5], [np.mean(ys), np.mean(ys)], c="orange", ls='-', lw=1.5, alpha=0.5)
    # Draw lines connecting points to axes
    for x, y in zip(xs, ys):
        ax.plot([x, x], [x, y], c="black", ls='--', lw=1.5, alpha=0.5)
        # ax.annotate(f"({x},{y})", (
        #     x + 0.03 if np.sign(x) == 1 else x - 0.13,
        #     y + 0.03 if np.sign(y) == 1 else y - 0.13
        # ), color="black", size=16)

    # Set identical scales for both axes
    ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1), aspect='equal')

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes
    # ax.set_xlabel('x', size=16, labelpad=-24, x=1.03)
    # ax.set_ylabel('y', size=16, labelpad=-21, y=1.02, rotation=0)

    # Create custom major ticks to determine position of tick labels
    x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])

    # Create minor ticks placed at each integer to enable drawing of minor grid
    # lines: note that this has no effect in this example with ticks_frequency=1
    ax.set_xticks(np.arange(xmin, xmax + 1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax + 1), minor=True)

    # Draw major and minor grid lines
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

    # Draw arrows
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)

    # handles, labels = ax.get_legend_handles_labels()
    # new_handles, new_labels, label_set = [], [], set()
    # for h, l in zip(handles, labels):
    #     if l in label_set:
    #         continue
    #     new_handles.append(h)
    #     new_labels.append(l)
    #     label_set.add(l)

    # ax.legend(handles=new_handles, labels= new_labels, loc="lower left", prop={'size': 16})
    ax = axs[1]

    ax.set_title("SST")
    ax.scatter(xs, ys, c="blue")

    ax.plot([-5, 5], [-5, 5], c="black", ls='-', lw=1.5, alpha=0.5)
    ax.plot([-5, 5], [np.mean(ys), np.mean(ys)], c="orange", ls='-', lw=1.5, alpha=0.5)
    # Draw lines connecting points to axes
    for x, y in zip(xs, ys):
        ax.plot([x, x], [np.mean(ys), y], c="orange", ls='--', lw=1.5, alpha=0.5)
        # ax.annotate(f"({x},{y})", (
        #     x + 0.03 if np.sign(x) == 1 else x - 0.13,
        #     y + 0.03 if np.sign(y) == 1 else y - 0.13
        # ), color="black", size=16)

    # Set identical scales for both axes
    ax.set(xlim=(xmin - 1, xmax + 1), ylim=(ymin - 1, ymax + 1), aspect='equal')

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes
    # ax.set_xlabel('x', size=16, labelpad=-24, x=1.03)
    # ax.set_ylabel('y', size=16, labelpad=-21, y=1.02, rotation=0)

    # Create custom major ticks to determine position of tick labels
    x_ticks = np.arange(xmin, xmax + 1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax + 1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])

    # Create minor ticks placed at each integer to enable drawing of minor grid
    # lines: note that this has no effect in this example with ticks_frequency=1
    ax.set_xticks(np.arange(xmin, xmax + 1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax + 1), minor=True)

    # Draw major and minor grid lines
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

    # Draw arrows
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)