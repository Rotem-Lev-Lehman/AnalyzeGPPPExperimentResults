import matplotlib.pyplot as plt
from domain_analyzer import get_regular_plot_styles_dict

styles_dict = get_regular_plot_styles_dict()
f = lambda style: plt.plot([], [],
                           ls=style['style'],
                           lw=style['width'],
                           color=style['color'])[0]

handles = [f(styles_dict[col]) for col in styles_dict.keys()]
labels = styles_dict.keys()
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False)


def export_legend(legend, filename=r'results\coverage_legend.png'):
    """ Exports the legend of the graphs to a new image file.

    :param legend: the legend we want to export
    :param filename: the path where we want to save the file to. Defaults to 'results\\coverage_legend.png'
    :type filename: str
    """
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


export_legend(legend)
