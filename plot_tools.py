#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com
# A set of convenience functions used for producing plots in `dabest`.


from .misc_tools import merge_two_dicts
import math
import warnings
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines
import matplotlib.axes as axes
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable, Union
from pandas.api.types import CategoricalDtype
from matplotlib.colors import ListedColormap

def halfviolin(v, half='right', fill_color='k', alpha=1,
                line_color='k', line_width=0):
    import numpy as np

    for b in v['bodies']:
        V = b.get_paths()[0].vertices

        mean_vertical = np.mean(V[:, 0])
        mean_horizontal = np.mean(V[:, 1])

        if half == 'right':
            V[:, 0] = np.clip(V[:, 0], mean_vertical, np.inf)
        elif half == 'left':
            V[:, 0] = np.clip(V[:, 0], -np.inf, mean_vertical)
        elif half == 'bottom':
            V[:, 1] = np.clip(V[:, 1], -np.inf, mean_horizontal)
        elif half == 'top':
            V[:, 1] = np.clip(V[:, 1], mean_horizontal, np.inf)

        b.set_color(fill_color)
        b.set_alpha(alpha)
        b.set_edgecolor(line_color)
        b.set_linewidth(line_width)



# def align_yaxis(ax1, v1, ax2, v2):
#     """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
#     # Taken from
#     # http://stackoverflow.com/questions/7630778/
#     # matplotlib-align-origin-of-right-axis-with-specific-left-axis-value
#     _, y1 = ax1.transData.transform((0, v1))
#     _, y2 = ax2.transData.transform((0, v2))
#     inv = ax2.transData.inverted()
#     _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
#     miny, maxy = ax2.get_ylim()
#     ax2.set_ylim(miny+dy, maxy+dy)
#
#
#
# def rotate_ticks(axes, angle=45, alignment='right'):
#     for tick in axes.get_xticklabels():
#         tick.set_rotation(angle)
#         tick.set_horizontalalignment(alignment)



def get_swarm_spans(coll):
    """
    Given a matplotlib Collection, will obtain the x and y spans
    for the collection. Will return None if this fails.
    """
    import numpy as np
    x, y = np.array(coll.get_offsets()).T
    try:
        return x.min(), x.max(), y.min(), y.max()
    except ValueError:
        return None



def error_bar(data, x, y, type='mean_sd', offset=0.2, ax=None,
              line_color="black", gap_width_percent=1, pos=[0, 1],
              method='gapped_lines', **kwargs):
    '''
    Function to plot the standard deviations as vertical errorbars.
    The mean is a gap defined by negative space.

    This function combines the functionality of gapped_lines(),
    proportional_error_bar(), and sankey_error_bar().

    Keywords
    --------
    data: pandas DataFrame.
        This DataFrame should be in 'long' format.

    x, y: string.
        x and y columns to be plotted.

    type: ['mean_sd', 'median_quartiles'], default 'mean_sd'
        Plots the summary statistics for each group. If 'mean_sd', then the
        mean and standard deviation of each group is plotted as a gapped line.
        If 'median_quantiles', then the median and 25th and 75th percentiles of
        each group is plotted instead.

    offset: float (default 0.3) or iterable.
        Give a single float (that will be used as the x-offset of all
        gapped lines), or an iterable containing the list of x-offsets.

    line_color: string (matplotlib color, default "black") or iterable of
        matplotlib colors.

        The color of the vertical line indicating the standard deviations.

    gap_width_percent: float, default 5
        The width of the gap in the line (indicating the central measure),
        expressed as a percentage of the y-span of the axes.

    ax: matplotlib Axes object, default None
        If a matplotlib Axes object is specified, the gapped lines will be
        plotted in order on this axes. If None, the current axes (plt.gca())
        is used.

    pos: list, default [0, 1]
        The positions of the error bars for the sankey_error_bar method.

    method: string, default 'gapped_lines'
        The method to use for drawing the error bars. Options are:
        'gapped_lines', 'proportional_error_bar', and 'sankey_error_bar'.

    kwargs: dict, default None
        Dictionary with kwargs passed to matplotlib.lines.Line2D
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    if gap_width_percent < 0 or gap_width_percent > 100:
        raise ValueError("`gap_width_percent` must be between 0 and 100.")
    if method not in ['gapped_lines', 'proportional_error_bar', 'sankey_error_bar']:
        raise ValueError("Invalid `method`. Must be one of 'gapped_lines', 'proportional_error_bar', or 'sankey_error_bar'.")

    if ax is None:
        ax = plt.gca()
    ax_ylims = ax.get_ylim()
    ax_yspan = np.abs(ax_ylims[1] - ax_ylims[0])
    gap_width = ax_yspan * gap_width_percent / 100

    keys = kwargs.keys()
    if 'clip_on' not in keys:
        kwargs['clip_on'] = False

    if 'zorder' not in keys:
        kwargs['zorder'] = 5

    if 'lw' not in keys:
        kwargs['lw'] = 2.

    if isinstance(data[x].dtype, pd.CategoricalDtype):
        group_order = pd.unique(data[x]).categories
    else:
        group_order = pd.unique(data[x])

    means = data.groupby(x)[y].mean().reindex(index=group_order)

    if method in ['proportional_error_bar', 'sankey_error_bar']:
        g = lambda x: np.sqrt((np.sum(x) * (len(x) - np.sum(x))) / (len(x) * len(x) * len(x)))
        sd = data.groupby(x)[y].apply(g)
    else:
        sd = data.groupby(x)[y].std().reindex(index=group_order)

    lower_sd = means - sd
    upper_sd = means + sd

    if (lower_sd < ax_ylims[0]).any() or (upper_sd > ax_ylims[1]).any():
        kwargs['clip_on'] = True

    medians = data.groupby(x)[y].median().reindex(index=group_order)
    quantiles = data.groupby(x)[y].quantile([0.25, 0.75]) \
        .unstack() \
        .reindex(index=group_order)
    lower_quartiles = quantiles[0.25]
    upper_quartiles = quantiles[0.75]

    if type == 'mean_sd':
        central_measures = means
        lows = lower_sd
        highs = upper_sd
    elif type == 'median_quartiles':
        central_measures = medians
        lows = lower_quartiles
        highs = upper_quartiles

    n_groups = len(central_measures)

    if isinstance(line_color, str):
        custom_palette = np.repeat(line_color, n_groups)
    else:
        if len(line_color) != n_groups:
            err1 = "{} groups are being plotted, but ".format(n_groups)
            err2 = "{} colors(s) were supplied in `line_color`.".format(len(line_color))
            raise ValueError(err1 + err2)
        custom_palette = line_color

    try:
        len_offset = len(offset)
    except TypeError:
        offset = np.repeat(offset, n_groups)
        len_offset = len(offset)

    if len_offset != n_groups:
        err1 = "{} groups are being plotted, but ".format(n_groups)
        err2 = "{} offset(s) were supplied in `offset`.".format(len_offset)
        raise ValueError(err1 + err2)

    kwargs['zorder'] = kwargs['zorder']

    for xpos, central_measure in enumerate(central_measures):
        kwargs['color'] = custom_palette[xpos]

        if method == 'sankey_error_bar':
            _xpos = pos[xpos] + offset[xpos]
        else:
            _xpos = xpos + offset[xpos]

        low = lows[xpos]
        low_to_mean = mlines.Line2D([_xpos, _xpos],
                                    [low, central_measure - gap_width],
                                    **kwargs)
        ax.add_line(low_to_mean)

        high = highs[xpos]
        mean_to_high = mlines.Line2D([_xpos, _xpos],
                                     [central_measure + gap_width, high],
                                     **kwargs)
        ax.add_line(mean_to_high)

def check_data_matches_labels(labels, data, side):
    '''
    Function to check that the labels and data match in the sankey diagram. 
    And enforce labels and data to be lists.
    Raises an exception if the labels and data do not match.

    Keywords
    --------
    labels: list of input labels
    data: Pandas Series of input data
    side: string, 'left' or 'right' on the sankey diagram
    '''
    if len(labels > 0):
        if isinstance(data, list):
            data = set(data)
        if isinstance(data, pd.Series):
            data = set(data.unique())
        if isinstance(labels, list):
            labels = set(labels)
        if labels != data:
            msg = "\n"
            if len(labels) <= 20:
                msg = "Labels: " + ",".join(labels) + "\n"
            if len(data) < 20:
                msg += "Data: " + ",".join(data)
            raise Exception('{0} labels and data do not match.{1}'.format(side, msg))
        
def normalize_dict(nested_dict, target):
    val = {}
    for key in nested_dict.keys():
        val[key] = np.sum([nested_dict[sub_key][key] for sub_key in nested_dict.keys()])
    
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            for subkey in value.keys():
                value[subkey] = value[subkey] * target[subkey]['right']/val[subkey]
    return nested_dict

def single_sankey(left, right, xpos=0, leftWeight=None, rightWeight=None, 
            colorDict=None, leftLabels=None, rightLabels=None, ax=None, 
            width=0.5, alpha=0.65, bar_width=0.2, rightColor=False, align='center'):

    '''
    Make a single Sankey diagram showing proportion flow from left to right
    Original code from: https://github.com/anazalea/pySankey
    Changes are added to normalize each diagram's height to be 1

    Keywords
    --------
    left: NumPy array 
        data on the left of the diagram
    right: NumPy array 
        data on the right of the diagram
        len(left) == len(right)
    xpos: float
        the starting point on the x-axis
    leftWeight: NumPy array
        weights for the left labels, if None, all weights are 1
    rightWeight: NumPy array
         weights for the right labels, if None, all weights are corresponding leftWeight
    colorDict: dictionary of colors for each label
        input format: {'label': 'color'}
    leftLabels: list
        labels for the left side of the diagram. The diagram will be sorted by these labels.
    rightLabels: list
        labels for the right side of the diagram. The diagram will be sorted by these labels.
    ax: matplotlib axes to be drawn on
    aspect: float
        vertical extent of the diagram in units of horizontal extent
    rightColor: bool
        if True, each strip of the diagram will be colored according to the corresponding left labels
    align: bool
        if 'center', the diagram will be centered on each xtick, 
        if 'edge', the diagram will be aligned with the left edge of each xtick
    '''

    # Initiating values
    if ax is None:
        ax = plt.gca()

    if leftWeight is None:
        leftWeight = []
    if rightWeight is None:
        rightWeight = []
    if leftLabels is None:
        leftLabels = []
    if rightLabels is None:
        rightLabels = []
    # Check weights
    if len(leftWeight) == 0:
        leftWeight = np.ones(len(left))
    if len(rightWeight) == 0:
        rightWeight = leftWeight

    # Create Dataframe
    if isinstance(left, pd.Series):
        left.reset_index(drop=True, inplace=True)
    if isinstance(right, pd.Series):
        right.reset_index(drop=True, inplace=True)
    dataFrame = pd.DataFrame({'left': left, 'right': right, 'leftWeight': leftWeight,
                              'rightWeight': rightWeight}, index=range(len(left)))
    
    if dataFrame[['left', 'right']].isnull().any(axis=None):
        raise Exception('Sankey graph does not support null values.')

    # Identify all labels that appear 'left' or 'right'
    allLabels = pd.Series(np.sort(np.r_[dataFrame.left.unique(), dataFrame.right.unique()])[::-1]).unique()

    # Identify left labels
    if len(leftLabels) == 0:
        leftLabels = pd.Series(np.sort(dataFrame.left.unique())[::-1]).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame['left'], 'left')

    # Identify right labels
    if len(rightLabels) == 0:
        rightLabels = pd.Series(np.sort(dataFrame.right.unique())[::-1]).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame['right'], 'right')

    # If no colorDict given, make one
    if colorDict is None:
        colorDict = {}
        palette = "hls"
        colorPalette = sns.color_palette(palette, len(allLabels))
        for i, label in enumerate(allLabels):
            colorDict[label] = colorPalette[i]
        fail_color = {0:"grey"}
        colorDict.update(fail_color)
    else:
        missing = [label for label in allLabels if label not in colorDict.keys()]
        if missing:
            msg = "The palette parameter is missing values for the following labels : "
            msg += '{}'.format(', '.join(missing))
            raise ValueError(msg)

    if align not in ("center", "edge"):
        err = '{} assigned for `align` is not valid.'.format(align)
        raise ValueError(err)
    if align == "center":
        try:
            leftpos = xpos - width / 2
        except TypeError as e:
            raise TypeError(f'the dtypes of parameters x ({xpos.dtype}) '
                            f'and width ({width.dtype}) '
                            f'are incompatible') from e
    else: 
        leftpos = xpos

    # Combine left and right arrays to have a pandas.DataFrame in the 'long' format
    left_series = pd.Series(left, name='values').to_frame().assign(groups='left')
    right_series = pd.Series(right, name='values').to_frame().assign(groups='right')
    concatenated_df = pd.concat([left_series, right_series], ignore_index=True)

    # Determine positions of left label patches and total widths
    # We also want the height of the graph to be 1
    leftWidths_norm = defaultdict()
    for i, leftLabel in enumerate(leftLabels):
        myD = {}
        myD['left'] = (dataFrame[dataFrame.left == leftLabel].leftWeight.sum()/ \
            dataFrame.leftWeight.sum())*(1-(len(leftLabels)-1)*0.02)
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['left']
        else:
            myD['bottom'] = leftWidths_norm[leftLabels[i - 1]]['top'] + 0.02
            myD['top'] = myD['bottom'] + myD['left']
            topEdge = myD['top']
        leftWidths_norm[leftLabel] = myD

    # Determine positions of right label patches and total widths
    rightWidths_norm = defaultdict()
    for i, rightLabel in enumerate(rightLabels):
        myD = {}
        myD['right'] = (dataFrame[dataFrame.right == rightLabel].rightWeight.sum()/ \
            dataFrame.rightWeight.sum())*(1-(len(leftLabels)-1)*0.02)
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['right']
        else:
            myD['bottom'] = rightWidths_norm[rightLabels[i - 1]]['top'] + 0.02
            myD['top'] = myD['bottom'] + myD['right']
            topEdge = myD['top']
        rightWidths_norm[rightLabel] = myD    

    # Total width of the graph
    xMax = width

    # Determine widths of individual strips, all widths are normalized to 1
    ns_l = defaultdict()
    ns_r = defaultdict()
    ns_l_norm = defaultdict()
    ns_r_norm = defaultdict()
    for leftLabel in leftLabels:
        leftDict = {}
        rightDict = {}
        for rightLabel in rightLabels:
            leftDict[rightLabel] = dataFrame[
                (dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)
                ].leftWeight.sum()
                
            rightDict[rightLabel] = dataFrame[
                (dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)
                ].rightWeight.sum()
        factorleft = leftWidths_norm[leftLabel]['left']/sum(leftDict.values())
        leftDict_norm = {k: v*factorleft for k, v in leftDict.items()}
        ns_l_norm[leftLabel] = leftDict_norm
        ns_r[leftLabel] = rightDict
    
    # ns_r should be using a different way of normalization to fit the right side
    # It is normalized using the value with the same key in each sub-dictionary

    ns_r_norm = normalize_dict(ns_r, rightWidths_norm)

    # Plot vertical bars for each label
    for leftLabel in leftLabels:
        ax.fill_between(
            [leftpos + (-(bar_width) * xMax), leftpos],
            2 * [leftWidths_norm[leftLabel]["bottom"]],
            2 * [leftWidths_norm[leftLabel]["bottom"] + leftWidths_norm[leftLabel]["left"]],
            color=colorDict[leftLabel],
            alpha=0.99,
        )
    for rightLabel in rightLabels:
        ax.fill_between(
            [xMax + leftpos, leftpos + ((1 + bar_width) * xMax)], 
            2 * [rightWidths_norm[rightLabel]['bottom']],
            2 * [rightWidths_norm[rightLabel]['bottom'] + rightWidths_norm[rightLabel]['right']],
            color=colorDict[rightLabel],
            alpha=0.99
        )

    # Plot error bars
    error_bar(concatenated_df, x='groups', y='values', ax=ax, offset=0, gap_width_percent=2,
              method="sankey_error_bar",
              pos=[(leftpos + (-(bar_width) * xMax) + leftpos)/2, \
                   (xMax + leftpos + leftpos + ((1 + bar_width) * xMax))/2])
    
    # Plot strips
    for leftLabel, rightLabel in itertools.product(leftLabels, rightLabels):
        labelColor = leftLabel
        if rightColor:
            labelColor = rightLabel
        if len(dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)]) > 0:
            # Create array of y values for each strip, half at left value,
            # half at right, convolve
            ys_d = np.array(50 * [leftWidths_norm[leftLabel]['bottom']] + \
                50 * [rightWidths_norm[rightLabel]['bottom']])
            ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
            ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
            ys_u = np.array(50 * [leftWidths_norm[leftLabel]['bottom'] + ns_l_norm[leftLabel][rightLabel]] + \
                50 * [rightWidths_norm[rightLabel]['bottom'] + ns_r_norm[leftLabel][rightLabel]])
            ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
            ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')

            # Update bottom edges at each label so next strip starts at the right place
            leftWidths_norm[leftLabel]['bottom'] += ns_l_norm[leftLabel][rightLabel]
            rightWidths_norm[rightLabel]['bottom'] += ns_r_norm[leftLabel][rightLabel]
            ax.fill_between(
                np.linspace(leftpos, leftpos + xMax, len(ys_d)), ys_d, ys_u, alpha=alpha,
                color=colorDict[labelColor], edgecolor='none'
            )
                
def sankeydiag(data, xvar, yvar, left_idx, right_idx, 
                leftLabels=None, rightLabels=None,  
                palette=None, ax=None, 
                one_sankey=False,
                width=0.4, rightColor=False,
                align='center', alpha=0.65, **kwargs):
    '''
    Read in melted pd.DataFrame, and draw multiple sankey diagram on a single axes
    using the value in column yvar according to the value in column xvar
    left_idx in the column xvar is on the left side of each sankey diagram
    right_idx in the column xvar is on the right side of each sankey diagram

    Keywords
    --------
    data: pd.DataFrame
        input data, melted dataframe created by dabest.load()
    xvar, yvar: string.
        x and y columns to be plotted.
    left_idx: str
        the value in column xvar that is on the left side of each sankey diagram
    right_idx: str
        the value in column xvar that is on the right side of each sankey diagram
        if len(left_idx) == 1, it will be broadcasted to the same length as right_idx
        otherwise it should have the same length as right_idx
    leftLabels: list
        labels for the left side of the diagram. The diagram will be sorted by these labels.
    rightLabels: list
        labels for the right side of the diagram. The diagram will be sorted by these labels.
    palette: str or dict
    ax: matplotlib axes to be drawn on
    one_sankey: bool 
        determined by the driver function on plotter.py. 
        if True, draw the sankey diagram across the whole raw data axes
    width: float
        the width of each sankey diagram
    align: str
        the alignment of each sankey diagram, can be 'center' or 'left'
    alpha: float
        the transparency of each strip
    rightColor: bool
        if True, each strip of the diagram will be colored according to the corresponding left labels
    colorDict: dictionary of colors for each label
        input format: {'label': 'color'}
    '''

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    if "width" in kwargs:
        width = kwargs["width"]

    if "align" in kwargs:
        align = kwargs["align"]
    
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
    
    if "rightColor" in kwargs:
        rightColor = kwargs["rightColor"]
    
    if "bar_width" in kwargs:
        bar_width = kwargs["bar_width"]

    if ax is None:
        ax = plt.gca()

    allLabels = pd.Series(np.sort(data[yvar].unique())[::-1]).unique()
        
    # Check if all the elements in left_idx and right_idx are in xvar column
    unique_xvar = data[xvar].unique()
    if not all(elem in unique_xvar for elem in left_idx):
        raise ValueError(f"{left_idx} not found in {xvar} column")
    if not all(elem in unique_xvar for elem in right_idx):
        raise ValueError(f"{right_idx} not found in {xvar} column")

    xpos = 0

    # For baseline comparison, broadcast left_idx to the same length as right_idx
    # so that the left of sankey diagram will be the same
    # For sequential comparison, left_idx and right_idx can have anything different 
    # but should have the same length
    if len(left_idx) == 1:
        broadcasted_left = np.broadcast_to(left_idx, len(right_idx))
    elif len(left_idx) != len(right_idx):
        raise ValueError(f"left_idx and right_idx should have the same length")
    else:
        broadcasted_left = left_idx

    if isinstance(palette, dict):
        if not all(key in allLabels for key in palette.keys()):
            raise ValueError(f"keys in palette should be in {yvar} column")
        else: 
            plot_palette = palette
    elif isinstance(palette, str):
        plot_palette = {}
        colorPalette = sns.color_palette(palette, len(allLabels))
        for i, label in enumerate(allLabels):
            plot_palette[label] = colorPalette[i]
    else:
        plot_palette = None

    for left, right in zip(broadcasted_left, right_idx):
        if one_sankey == False:
            single_sankey(data[data[xvar]==left][yvar], data[data[xvar]==right][yvar], 
                            xpos=xpos, ax=ax, colorDict=plot_palette, width=width, 
                            leftLabels=leftLabels, rightLabels=rightLabels, 
                            rightColor=rightColor, bar_width=bar_width,
                            align=align, alpha=alpha)
            xpos += 1
        else:
            xpos = 0 + bar_width/2
            width = 1 - bar_width
            single_sankey(data[data[xvar]==left][yvar], data[data[xvar]==right][yvar], 
                            xpos=xpos, ax=ax, colorDict=plot_palette, width=width, 
                            leftLabels=leftLabels, rightLabels=rightLabels, 
                            rightColor=rightColor, bar_width=bar_width,
                            align='edge', alpha=alpha)

    if one_sankey == False:
        sankey_ticks = [f"{left}\n v.s.\n{right}" for left, right in zip(broadcasted_left, right_idx)]
        ax.get_xaxis().set_ticks(np.arange(len(right_idx)))
        ax.get_xaxis().set_ticklabels(sankey_ticks)
    else:
        sankey_ticks = [broadcasted_left[0], right_idx[0]]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(sankey_ticks)


def horizontal_colormaker(number:int,custom_pal=None,desat_level:float=0.5):
    import seaborn as sns
    import matplotlib.pyplot as plt 

    # If no custom palette is provided, use the default seaborn palette
    if custom_pal is None:
        colors = sns.color_palette(n_colors=number)

    # If a tuple is provided, check it is the right length
    elif isinstance(custom_pal, tuple) or isinstance(custom_pal, list):
        if len(custom_pal) != number:
            raise ValueError('Number of colors inputted does not equal number of samples')
        else:
            colors = custom_pal

    # If a string is provided, check it is a matplotlib palette
    elif isinstance(custom_pal, str):
        # check it is in the list of matplotlib palettes.
        if custom_pal in plt.colormaps():
            colors = sns.color_palette(custom_pal, number)
        else:
            raise ValueError('The specified `custom_palette` {} is not a matplotlib palette. Please check.'.format(custom_pal))
    else:
        raise TypeError('Incorrect color input format')

    # Desaturate the colors
    desat_colors = [sns.desaturate(c, desat_level) for c in colors] 
    return desat_colors

def swarmplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    ax: axes.Subplot,
    order: List = None,
    hue: str = None,
    horizontal: bool = False,
    palette: Union[Iterable, str] = "black",
    zorder: float = 1,
    size: float = 5,
    side: str = "center",
    jitter: float = 1,
    filled: Union[bool, List, Tuple] = True,
    is_drop_gutter: bool = True,
    gutter_limit: float = 0.5,
    **kwargs,
):
    """
    API to plot a swarm plot.

    Parameters
    ----------
    data : pd.DataFrame
        The input data as a pandas DataFrame.
    x : str
        The column in the DataFrame to be used as the x-axis.
    y : str
        The column in the DataFrame to be used as the y-axis.
    ax : axes._subplots.Subplot | axes._axes.Axes
        Matplotlib AxesSubplot object for which the plot would be drawn on. Default is None.
    order : List
        The order in which x-axis categories should be displayed. Default is None.
    hue : str
        The column in the DataFrame that determines the grouping for color.
        If None (by default), it assumes that it is being grouped by x.
    palette : Union[Iterable, str]
        The color palette to be used for plotting. Default is "black".
    zorder : int | float
        The z-order for drawing the swarm plot wrt other matplotlib drawings. Default is 1.
    dot_size : int | float
        The size of the markers in the swarm plot. Default is 20.
    side : str
        The side on which points are swarmed ("center", "left", or "right"). Default is "center".
    jitter : int | float
        Determines the distance between points. Default is 1.
    filled : bool | List | Tuple
        Determines whether the dots in the swarmplot are filled or not. If set to False,
        dots are not filled. If provided as a List or Tuple, it should contain boolean values,
        each corresponding to a swarm group in order, indicating whether the dot should be
        filled or not.
    is_drop_gutter : bool
        If True, drop points that hit the gutters; otherwise, readjust them.
    gutter_limit : int | float
        The limit for points hitting the gutters.
    **kwargs:
        Additional keyword arguments to be passed to the swarm plot.

    Returns
    -------
    axes._subplots.Subplot | axes._axes.Axes
        Matplotlib AxesSubplot object for which the swarm plot has been drawn on.
    """
    s = SwarmPlot(data, x, y, ax, order, hue, palette, zorder, size, side, jitter)
    if horizontal == True:
        ax = s.plot_horizontal(is_drop_gutter, gutter_limit, ax, filled, **kwargs)
    else:
        ax = s.plot(is_drop_gutter, gutter_limit, ax, filled, **kwargs)
    return ax

class SwarmPlot:
    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        ax: axes.Subplot,
        order: List = None,
        hue: str = None,
        palette: Union[Iterable, str] = "black",
        zorder: float = 1,
        size: float = 5,
        side: str = "center",
        jitter: float = 1,
    ):
        """
        Initialize a SwarmPlot instance.

        Parameters
        ----------
        data : pd.DataFrame
            The input data as a pandas DataFrame.
        x : str
            The column in the DataFrame to be used as the x-axis.
        y : str
            The column in the DataFrame to be used as the y-axis.
        ax : axes.Subplot
            Matplotlib AxesSubplot object for which the plot would be drawn on.
        order : List
            The order in which x-axis categories should be displayed. Default is None.
        hue : str
            The column in the DataFrame that determines the grouping for color.
            If None (by default), it assumes that it is being grouped by x.
        palette : Union[Iterable, str]
            The color palette to be used for plotting. Default is "black".
        zorder : int | float
            The z-order for drawing the swarm plot wrt other matplotlib drawings. Default is 1.
        dot_size : int | float
            The size of the markers in the swarm plot. Default is 20.
        side : str
            The side on which points are swarmed ("center", "left", or "right"). Default is "center".
        jitter : int | float
            Determines the distance between points. Default is 1.

        Returns
        -------
        None
        """
        self.__x = x
        self.__y = y
        self.__order = order
        self.__hue = hue
        self.__zorder = zorder
        self.__palette = palette
        self.__jitter = jitter

        # Input validation
        self._check_errors(data, ax, size, side)

        self.__size = size * 4
        self.__side = side.lower()
        self.__data = data
        self.__color_col = self.__x if self.__hue is None else self.__hue

        # Generate default values
        if order is None:
            self.__order = self._generate_order()

        # Reformatting
        if not isinstance(self.__palette, dict):
            self.__palette = self._format_palette(self.__palette)
        data_copy = data.copy(deep=True)
        if not isinstance(self.__data[self.__x].dtype, pd.CategoricalDtype):
            # make x column into CategoricalDType to sort by
            data_copy[self.__x] = data_copy[self.__x].astype(
                CategoricalDtype(categories=self.__order, ordered=True)
            )
        data_copy.sort_values(by=[self.__x, self.__y], inplace=True)
        self.__data_copy = data_copy

        x_vals = range(len(self.__order))
        y_vals = self.__data_copy[self.__y]

        x_min = min(x_vals)
        x_max = max(x_vals)
        ax.set_xlim(left=x_min - 0.5, right=x_max + 0.5)

        y_range = max(y_vals) - min(y_vals)
        y_min = min(y_vals) - 0.05 * y_range
        y_max = max(y_vals) + 0.05 * y_range

        # ylim is set manually to override Axes.autoscale if it hasn't already been scaled at least once
        if ax.get_autoscaley_on():
            ax.set_ylim(bottom=y_min, top=y_max)

        figw, figh = ax.get_figure().get_size_inches()
        w = (ax.get_position().xmax - ax.get_position().xmin) * figw
        h = (ax.get_position().ymax - ax.get_position().ymin) * figh
        ax_xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
        ax_yspan = ax.get_ylim()[1] - ax.get_ylim()[0]

        # increases jitter distance based on number of swarms that is going to be drawn
        jitter = jitter * (1 + 0.05 * (math.log(ax_xspan)))

        gsize = (
            math.sqrt(self.__size) * 1.0 / (70 / jitter) * ax_xspan * 1.0 / (w * 0.8)
        )
        dsize = (
            math.sqrt(self.__size) * 1.0 / (70 / jitter) * ax_yspan * 1.0 / (h * 0.8)
        )
        self.__gsize = gsize
        self.__dsize = dsize

    def _check_errors(
        self, data: pd.DataFrame, ax: axes.Subplot, size: float, side: str
    ) -> None:
        """
        Check the validity of input parameters. Raises exceptions if detected.

        Parameters
        ----------
        data : pd.Dataframe
            Input data used for generation of the swarmplot.
        ax : axes.Subplot
            Matplotlib AxesSubplot object for which the plot would be drawn on.
        size : int | float
            scalar value determining size of dots of the swarmplot.
        side: str
            The side on which points are swarmed ("center", "left", or "right"). Default is "center".

        Returns
        -------
        None
        """
        # Type enforcement
        if not isinstance(data, pd.DataFrame):
            raise ValueError("`data` must be a Pandas Dataframe.")
        if not isinstance(ax, (axes._subplots.Subplot, axes._axes.Axes)):
            raise ValueError(
                f"`ax` must be a Matplotlib AxesSubplot. The current `ax` is a {type(ax)}"
            )
        if not isinstance(size, (int, float)):
            raise ValueError("`size` must be a scalar or float.")
        if not isinstance(side, str):
            raise ValueError(
                "Invalid `side`. Must be one of 'center', 'right', or 'left'."
            )
        if not isinstance(self.__x, str):
            raise ValueError("`x` must be a string.")
        if not isinstance(self.__y, str):
            raise ValueError("`y` must be a string.")
        if not isinstance(self.__zorder, (int, float)):
            raise ValueError("`zorder` must be a scalar or float.")
        if not isinstance(self.__jitter, (int, float)):
            raise ValueError("`jitter` must be a scalar or float.")
        if not isinstance(self.__palette, (str, Iterable)):
            raise ValueError(
                "`palette` must be either a string indicating a color name or an Iterable."
            )
        if self.__hue is not None and not isinstance(self.__hue, str):
            raise ValueError("`hue` must be either a string or None.")
        if self.__order is not None and not isinstance(self.__order, Iterable):
            raise ValueError("`order` must be either an Iterable or None.")

        # More thorough input validation checks
        if self.__x not in data.columns:
            err = "{0} is not a column in `data`.".format(self.__x)
            raise IndexError(err)
        if self.__y not in data.columns:
            err = "{0} is not a column in `data`.".format(self.__y)
            raise IndexError(err)
        if self.__hue is not None and self.__hue not in data.columns:
            err = "{0} is not a column in `data`.".format(self.__hue)
            raise IndexError(err)

        color_col = self.__x if self.__hue is None else self.__hue
        if self.__order is not None:
            for group_i in self.__order:
                if group_i not in pd.unique(data[self.__x]):
                    err = "{0} in `order` is not in the '{1}' column of `data`.".format(
                        group_i, self.__x
                    )
                    raise IndexError(err)

        if isinstance(self.__palette, str) and self.__palette.strip() == "":
            err = "`palette` cannot be an empty string. It must be either a string indicating a color name or an Iterable."
            raise ValueError(err)
        if isinstance(self.__palette, dict):
            for group_i, color_i in self.__palette.items():
                if group_i not in pd.unique(data[color_col]):
                    err = (
                        "{0} in `palette` is not in the '{1}' column of `data`.".format(
                            group_i, color_col
                        )
                    )
                    raise IndexError(err)
                if isinstance(color_i, str) and color_i.strip() == "":
                    err = "The color mapping for {0} in `palette` is an empty string. It must contain a color name.".format(
                        group_i
                    )
                    raise ValueError(err)

        if side.lower() not in ["center", "right", "left"]:
            raise ValueError(
                "Invalid `side`. Must be one of 'center', 'right', or 'left'."
            )

        return None

    def _generate_order(self) -> List:
        """
        Generates order value that determines the order in which x-axis categories should be displayed.

        Parameters
        ----------
        None

        Returns
        -------
        List:
            contains the order in which the x-axis categories should be displayed.
        """
        if isinstance(self.__data[self.__x].dtype, pd.CategoricalDtype):
            order = pd.unique(self.__data[self.__x]).categories.tolist()
        else:
            order = pd.unique(self.__data[self.__x]).tolist()

        return order

    def _format_palette(self, palette: Union[str, List, Tuple]) -> Dict:
        """
        Reformats palette into appropriate Dictionary form for swarm plot

        Parameters
        ----------
        palette: str | List | Tuple
            The color palette used for the swarm plot. Conventions are based on Matplotlib color
            specifications.

            Could be a singular string value - in which case, would be a singular color name.
            In the case of a List or Tuple - it could be a Sequence of color names or RGB(A) values.

        Returns
        -------
        Dict:
            Dictionary mapping unique groupings in the color column (of the data used for the swarm plot)
            to a color name (str) or a RGB(A) value (Tuple[float, float, float] | List[float, float, float]).
        """
        reformatted_palette = dict()
        groups = pd.unique(self.__data[self.__color_col]).tolist()

        if isinstance(palette, str):
            for group_i in groups:
                reformatted_palette[group_i] = palette
        if isinstance(palette, (list, tuple)):
            if len(groups) != len(palette):
                err = (
                    "unique values in '{0}' column in `data` "
                    "and `palette` do not have the same length. Number of unique values is {1} "
                    "while length of palette is {2}. The assignment of the colors in the "
                    "palette will be cycled."
                ).format(self.__color_col, len(groups), len(palette))
                warnings.warn(err)
            for i, group_i in enumerate(groups):
                reformatted_palette[group_i] = palette[i % len(palette)]

        return reformatted_palette

    def _swarm(
        self, values: Iterable[float], gsize: float, dsize: float, side: str
    ) -> pd.Series:
        """
        Perform the swarm algorithm to position points without overlap.

        Parameters
        ----------
        values : Iterable[int | float]
            The values to be plotted.
        gsize : int | float
            The size of the gap between points.
        dsize : int | float
            The size of the markers.
        side : str
            The side on which points are swarmed ("center", "left", or "right").

        Returns
        -------
        pd.Series:
            The x-offset values for the swarm plot.
        """
        # Input validation
        if not isinstance(values, Iterable):
            raise ValueError("`values` must be an Iterable")
        if not isinstance(gsize, (int, float)):
            raise ValueError("`gsize` must be a scalar or float.")
        if not isinstance(dsize, (int, float)):
            raise ValueError("`dsize` must be a scalar or float.")

        # Sorting algorithm based off of: https://github.com/mgymrek/pybeeswarm
        points_data = pd.DataFrame(
            {"y": [yval * 1.0 / dsize for yval in values], "x": [0] * len(values)}
        )
        for i in range(1, points_data.shape[0]):
            y_i = points_data["y"].values[i]
            points_placed = points_data[0:i]
            is_points_overlap = (
                abs(y_i - points_placed["y"]) < 1
            )  # Checks if y_i is overlapping with any points already placed
            if any(is_points_overlap):
                points_placed = points_placed[is_points_overlap]
                x_offsets = points_placed["y"].apply(
                    lambda y_j: math.sqrt(1 - (y_i - y_j) ** 2)
                )
                if side == "center":
                    potential_x_offsets = pd.Series(
                        [0]
                        + (points_placed["x"] + x_offsets).tolist()
                        + (points_placed["x"] - x_offsets).tolist()
                    )
                if side == "right":
                    potential_x_offsets = pd.Series(
                        [0] + (points_placed["x"] + x_offsets).tolist()
                    )
                if side == "left":
                    potential_x_offsets = pd.Series(
                        [0] + (points_placed["x"] - x_offsets).tolist()
                    )
                bad_x_offsets = []
                for x_i in potential_x_offsets:
                    dists = (y_i - points_placed["y"]) ** 2 + (
                        x_i - points_placed["x"]
                    ) ** 2
                    if any([item < 0.999 for item in dists]):
                        bad_x_offsets.append(True)
                    else:
                        bad_x_offsets.append(False)
                potential_x_offsets[bad_x_offsets] = np.infty
                abs_potential_x_offsets = [abs(_) for _ in potential_x_offsets]
                valid_x_offset = potential_x_offsets[
                    abs_potential_x_offsets.index(min(abs_potential_x_offsets))
                ]
                points_data.loc[i, "x"] = valid_x_offset
            else:
                points_data.loc[i, "x"] = 0

        points_data.loc[np.isnan(points_data["y"]), "x"] = np.nan

        return points_data["x"] * gsize

    def _adjust_gutter_points(
        self,
        points_data: pd.DataFrame,
        x_position: float,
        is_drop_gutter: bool,
        gutter_limit: float,
        value_column: str,
    ) -> pd.DataFrame:
        """
        Adjust points that hit the gutters or drop them based on the provided conditions.

        Parameters
        ----------
        points_data: pd.DataFrame
            Data containing coordinates of points for the swarm plot.
        x_position: int | float
            X-coordinate of the center of a singular swarm group of the swarm plot
        is_drop_gutter : bool
            If True, drop points that hit the gutters; otherwise, readjust them.
        gutter_limit : int | float
            The limit for points hitting the gutters.
        value_column : str
            column in points_data that contains the coordinates for the points in the axis against the gutter

        Returns
        -------
        pd.DataFrame:
            DataFrame with adjusted points based on the gutter limit.
        """
        if self.__side == "center":
            gutter_limit = gutter_limit / 2

        hit_gutter = abs(points_data[value_column] - x_position) >= gutter_limit
        total_num_of_points = points_data.shape[0]
        num_of_points_hit_gutter = points_data[hit_gutter].shape[0]
        if any(hit_gutter):
            if is_drop_gutter:
                # Drop points that hit gutter
                points_data.drop(points_data[hit_gutter].index.to_list(), inplace=True)
                err = (
                    "{0:.1%} of the points cannot be placed. "
                    "You might want to decrease the size of the markers."
                ).format(num_of_points_hit_gutter / total_num_of_points)
                warnings.warn(err)
            else:
                for i in points_data[hit_gutter].index:
                    points_data.loc[i, value_column] = np.sign(
                        points_data.loc[i, value_column]
                    ) * (x_position + gutter_limit)

        return points_data

    def plot(
        self,
        is_drop_gutter: bool,
        gutter_limit: float,
        ax: axes.Subplot,
        filled: Union[bool, List, Tuple],
        **kwargs,
    ) -> axes.Subplot:
        """
        Generate a swarm plot.

        Parameters
        ----------
        is_drop_gutter : bool
            If True, drop points that hit the gutters; otherwise, readjust them.
        gutter_limit : int | float
            The limit for points hitting the gutters.
        ax : axes.Subplot
            The matplotlib figure object to which the swarm plot will be added.
        filled : bool | List | Tuple
            Determines whether the dots in the swarmplot are filled or not. If set to False,
            dots are not filled. If provided as a List or Tuple, it should contain boolean values,
            each corresponding to a swarm group in order, indicating whether the dot should be
            filled or not.
        **kwargs:
            Additional keyword arguments to be passed to the scatter plot.

        Returns
        -------
        axes.Subplot:
            The matplotlib figure containing the swarm plot.
        """
        # Input validation
        if not isinstance(is_drop_gutter, bool):
            raise ValueError("`is_drop_gutter` must be a boolean.")
        if not isinstance(gutter_limit, (int, float)):
            raise ValueError("`gutter_limit` must be a scalar or float.")
        if not isinstance(filled, (bool, list, tuple)):
            raise ValueError("`filled` must be a boolean, list or tuple.")

        # More thorough input validation checks
        if isinstance(filled, (list, tuple)):
            if len(filled) != len(self.__order):
                err = (
                    "There are {0} unique values in `x` column in `data` "
                    "but `filled` has a length of {1}. If `filled` is a list "
                    "or a tuple, it must have the same length as the number of "
                    "unique values/groups in the `x` column of data."
                ).format(len(self.__order), len(filled))
                raise ValueError(err)
            if not all(isinstance(_, bool) for _ in filled):
                raise ValueError("All values in `filled` must be a boolean.")

        # Assumptions are that self.__data_copy is already sorted according to self.__order
        x_position = (
            0  # x-coordinate of center of each individual swarm of the swarm plot
        )
        x_tick_tabels = []
        for group_i, values_i in self.__data_copy.groupby(self.__x):
            x_new = []
            values_i_y = values_i[self.__y]
            x_offset = self._swarm(
                values=values_i_y,
                gsize=self.__gsize,
                dsize=self.__dsize,
                side=self.__side,
            )
            x_new = [
                x_position + offset for offset in x_offset
            ]  # apply x-offsets based on _swarm algo
            values_i["x_new"] = x_new
            values_i = self._adjust_gutter_points(
                values_i, x_position, is_drop_gutter, gutter_limit, "x_new"
            )
            x_tick_tabels.extend([group_i])
            x_position = x_position + 1

            if values_i.empty:
                ax.scatter(
                    values_i["x_new"],
                    values_i[self.__y],
                    s=self.__size,
                    zorder=self.__zorder,
                    **kwargs,
                )
                continue

            if self.__hue is not None:
                # color swarms based on `hue` column
                cmap_values, index = np.unique(
                    values_i[self.__hue], return_inverse=True
                )
                cmap = []
                for cmap_group_i in cmap_values:
                    cmap.append(self.__palette[cmap_group_i])
                cmap = ListedColormap(cmap)
                ax.scatter(
                    values_i["x_new"],
                    values_i[self.__y],
                    s=self.__size,
                    c=index,
                    cmap=cmap,
                    zorder=self.__zorder,
                    edgecolor="face",
                    **kwargs,
                )
            else:
                # color swarms based on `x` column
                if not isinstance(filled, bool):
                    facecolor = (
                        "none"
                        if not filled[x_position - 1]
                        else self.__palette[group_i]
                    )
                else:
                    facecolor = "none" if not filled else self.__palette[group_i]
                ax.scatter(
                    values_i["x_new"],
                    values_i[self.__y],
                    s=self.__size,
                    zorder=self.__zorder,
                    facecolor=facecolor,
                    edgecolor=self.__palette[group_i],
                    **kwargs,
                )

        ax.get_xaxis().set_ticks(np.arange(x_position))
        ax.get_xaxis().set_ticklabels(x_tick_tabels)

        return ax

    def plot_horizontal(
        self,
        is_drop_gutter: bool,
        gutter_limit: float,
        ax: axes.Subplot,
        filled: Union[bool, List, Tuple],
        **kwargs,
    ) -> axes.Subplot:
        """
        Generate a swarm plot.

        Parameters
        ----------
        is_drop_gutter : bool
            If True, drop points that hit the gutters; otherwise, readjust them.
        gutter_limit : int | float
            The limit for points hitting the gutters.
        ax : axes.Subplot
            The matplotlib figure object to which the swarm plot will be added.
        filled : bool | List | Tuple
            Determines whether the dots in the swarmplot are filled or not. If set to False,
            dots are not filled. If provided as a List or Tuple, it should contain boolean values,
            each corresponding to a swarm group in order, indicating whether the dot should be
            filled or not.
        **kwargs:
            Additional keyword arguments to be passed to the scatter plot.

        Returns
        -------
        axes.Subplot:
            The matplotlib figure containing the swarm plot.
        """
        # Input validation
        if not isinstance(is_drop_gutter, bool):
            raise ValueError("`is_drop_gutter` must be a boolean.")
        if not isinstance(gutter_limit, (int, float)):
            raise ValueError("`gutter_limit` must be a scalar or float.")
        if not isinstance(filled, (bool, list, tuple)):
            raise ValueError("`filled` must be a boolean, list or tuple.")

        # More thorough input validation checks
        if isinstance(filled, (list, tuple)):
            if len(filled) != len(self.__order):
                err = (
                    "There are {0} unique values in `x` column in `data` "
                    "but `filled` has a length of {1}. If `filled` is a list "
                    "or a tuple, it must have the same length as the number of "
                    "unique values/groups in the `x` column of data."
                ).format(len(self.__order), len(filled))
                raise ValueError(err)
            if not all(isinstance(_, bool) for _ in filled):
                raise ValueError("All values in `filled` must be a boolean.")

        # Assumptions are that self.__data_copy is already sorted according to self.__order
        y_position = (
            0  # x-coordinate of center of each individual swarm of the swarm plot
        )
        y_tick_tabels = []
        for group_i, values_i in self.__data_copy.groupby(self.__x):
            y_new = []
            values_i_y = values_i[self.__y]
            y_offset = self._swarm(
                values=values_i_y,
                gsize=self.__gsize,
                dsize=self.__dsize,
                side=self.__side,
            )
            y_new = [
                y_position + offset for offset in y_offset
            ]  # apply x-offsets based on _swarm algo
            values_i["y_new"] = y_new
            values_i = self._adjust_gutter_points(
                values_i, y_position, is_drop_gutter, gutter_limit, "y_new"
            )
            y_tick_tabels.extend([group_i])
            y_position = y_position + 1

            if values_i.empty:
                ax.scatter(
                    values_i[self.__y],
                    values_i["y_new"],
                    s=self.__size,
                    zorder=self.__zorder,
                    **kwargs,
                )
                continue

            if self.__hue is not None:
                # color swarms based on `hue` column
                cmap_values, index = np.unique(
                    values_i[self.__hue], return_inverse=True
                )
                cmap = []
                for cmap_group_i in cmap_values:
                    cmap.append(self.__palette[cmap_group_i])
                cmap = ListedColormap(cmap)
                ax.scatter(
                    values_i[self.__y],
                    values_i["y_new"],
                    s=self.__size,
                    c=index,
                    cmap=cmap,
                    zorder=self.__zorder,
                    edgecolor="face",
                    **kwargs,
                )
            else:
                # color swarms based on `y` column
                if not isinstance(filled, bool):
                    facecolor = (
                        "none"
                        if not filled[y_position - 1]
                        else self.__palette[group_i]
                    )
                else:
                    facecolor = "none" if not filled else self.__palette[group_i]
                ax.scatter(
                    values_i[self.__y],
                    values_i["y_new"],
                    s=self.__size,
                    zorder=self.__zorder,
                    facecolor=facecolor,
                    edgecolor=self.__palette[group_i],
                    **kwargs,
                )

        ax.get_yaxis().set_ticks(np.arange(y_position))
        ax.get_yaxis().set_ticklabels(y_tick_tabels)

        return ax