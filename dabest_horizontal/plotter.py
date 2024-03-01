#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com


def EffectSizeDataFramePlotter(EffectSizeDataFrame, **plot_kwargs):
    """
    Custom function that creates an estimation plot from an EffectSizeDataFrame.
    Keywords
    --------
    EffectSizeDataFrame: A `dabest` EffectSizeDataFrame object.
    **plot_kwargs:
        color_col=None
        raw_marker_size=6, es_marker_size=9,
        swarm_label=None, contrast_label=None, delta2_label=None,
        swarm_ylim=None, contrast_ylim=None, delta2_ylim=None,
        custom_palette=None, swarm_desat=0.5, halfviolin_desat=1,
        halfviolin_alpha=0.8,
        face_color = None,
        bar_label=None, bar_desat=0.8, bar_width = 0.5,bar_ylim = None,
        ci=None, ci_type='bca', err_color=None,
        float_contrast=True,
        show_pairs=True,
        show_delta2=True,
        group_summaries=None,
        group_summaries_offset=0.1,
        fig_size=None,
        dpi=100,
        ax=None,
        swarmplot_kwargs=None,
        violinplot_kwargs=None,
        slopegraph_kwargs=None,
        sankey_kwargs=None,
        reflines_kwargs=None,
        group_summary_kwargs=None,
        legend_kwargs=None,
    """

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore', 'This figure includes Axes that are not compatible with tight_layout')

    from .misc_tools import merge_two_dicts
    from .plot_tools import halfviolin, get_swarm_spans, error_bar, sankeydiag
    from ._stats_tools.effsize import _compute_standardizers, _compute_hedges_correction_factor

    import logging
    # Have to disable logging of warning when get_legend_handles_labels()
    # tries to get from slopegraph.
    logging.disable(logging.WARNING)

    # Save rcParams that I will alter, so I can reset back.
    original_rcParams = {}
    _changed_rcParams = ['axes.grid']
    for parameter in _changed_rcParams:
        original_rcParams[parameter] = plt.rcParams[parameter]

    plt.rcParams['axes.grid'] = False

    ytick_color = plt.rcParams["ytick.color"]
    face_color = plot_kwargs["face_color"]
    if plot_kwargs["face_color"] is None:
        face_color = "white"

    dabest_obj  = EffectSizeDataFrame.dabest_obj
    plot_data   = EffectSizeDataFrame._plot_data
    xvar        = EffectSizeDataFrame.xvar
    yvar        = EffectSizeDataFrame.yvar
    is_paired   = EffectSizeDataFrame.is_paired
    delta2      = EffectSizeDataFrame.delta2
    mini_meta   = EffectSizeDataFrame.mini_meta
    effect_size = EffectSizeDataFrame.effect_size
    proportional = EffectSizeDataFrame.proportional

    all_plot_groups = dabest_obj._all_plot_groups
    idx             = dabest_obj.idx

    if effect_size != "mean_diff" or not delta2:
        show_delta2 = False
    else:
        show_delta2 = plot_kwargs["show_delta2"]

    if effect_size != "mean_diff" or not mini_meta:
        show_mini_meta = False
    else:
        show_mini_meta = plot_kwargs["show_mini_meta"]

    if show_delta2 and show_mini_meta:
        err0 = "`show_delta2` and `show_mini_meta` cannot be True at the same time."
        raise ValueError(err0)

    # Disable Gardner-Altman plotting if any of the idxs comprise of more than
    # two groups or if it is a delta-delta plot.
    float_contrast   = plot_kwargs["float_contrast"]
    effect_size_type = EffectSizeDataFrame.effect_size
    if len(idx) > 1 or len(idx[0]) > 2:
        float_contrast = False

    if effect_size_type in ['cliffs_delta']:
        float_contrast = False

    if show_delta2 or show_mini_meta:
        float_contrast = False  

    if not is_paired:
        show_pairs = False
    else:
        show_pairs = plot_kwargs["show_pairs"]

    # Set default kwargs first, then merge with user-dictated ones.
    default_swarmplot_kwargs = {'size': plot_kwargs["raw_marker_size"]}
    if plot_kwargs["swarmplot_kwargs"] is None:
        swarmplot_kwargs = default_swarmplot_kwargs
    else:
        swarmplot_kwargs = merge_two_dicts(default_swarmplot_kwargs,
                                           plot_kwargs["swarmplot_kwargs"])

    # Barplot kwargs
    default_barplot_kwargs = {"estimator": np.mean, "ci": plot_kwargs["ci"]}

    if plot_kwargs["barplot_kwargs"] is None:
        barplot_kwargs = default_barplot_kwargs
    else:
        barplot_kwargs = merge_two_dicts(default_barplot_kwargs,
                                         plot_kwargs["barplot_kwargs"])

    # Sankey Diagram kwargs
    default_sankey_kwargs = {"width": 0.4, "align": "center",
                            "alpha": 0.4, "rightColor": False,
                            "bar_width":0.2}
    if plot_kwargs["sankey_kwargs"] is None:
        sankey_kwargs = default_sankey_kwargs
    else:
        sankey_kwargs = merge_two_dicts(default_sankey_kwargs,
                                        plot_kwargs["sankey_kwargs"])
                

    # Violinplot kwargs.
    default_violinplot_kwargs = {'widths':0.5, 'vert':True,
                               'showextrema':False, 'showmedians':False}
    if plot_kwargs["violinplot_kwargs"] is None:
        violinplot_kwargs = default_violinplot_kwargs
    else:
        violinplot_kwargs = merge_two_dicts(default_violinplot_kwargs,
                                            plot_kwargs["violinplot_kwargs"])

    # slopegraph kwargs.
    default_slopegraph_kwargs = {'lw':1, 'alpha':0.5}
    if plot_kwargs["slopegraph_kwargs"] is None:
        slopegraph_kwargs = default_slopegraph_kwargs
    else:
        slopegraph_kwargs = merge_two_dicts(default_slopegraph_kwargs,
                                            plot_kwargs["slopegraph_kwargs"])

    # Zero reference-line kwargs.
    default_reflines_kwargs = {'linestyle':'solid', 'linewidth':0.75,
                                'zorder': 2,
                                'color': ytick_color}
    if plot_kwargs["reflines_kwargs"] is None:
        reflines_kwargs = default_reflines_kwargs
    else:
        reflines_kwargs = merge_two_dicts(default_reflines_kwargs,
                                          plot_kwargs["reflines_kwargs"])

    # Legend kwargs.
    default_legend_kwargs = {'loc': 'upper left', 'frameon': False}
    if plot_kwargs["legend_kwargs"] is None:
        legend_kwargs = default_legend_kwargs
    else:
        legend_kwargs = merge_two_dicts(default_legend_kwargs,
                                        plot_kwargs["legend_kwargs"])

    # Group summaries kwargs.
    gs_default = {'mean_sd', 'median_quartiles', None}
    if plot_kwargs["group_summaries"] not in gs_default:
        raise ValueError('group_summaries must be one of'
        ' these: {}.'.format(gs_default) )

    default_group_summary_kwargs = {'zorder': 3, 'lw': 2,
                                    'alpha': 1}
    if plot_kwargs["group_summary_kwargs"] is None:
        group_summary_kwargs = default_group_summary_kwargs
    else:
        group_summary_kwargs = merge_two_dicts(default_group_summary_kwargs,
                                               plot_kwargs["group_summary_kwargs"])

    # Create color palette that will be shared across subplots.
    color_col = plot_kwargs["color_col"]
    if color_col is None:
        color_groups = pd.unique(plot_data[xvar])
        bootstraps_color_by_group = True
    else:
        if color_col not in plot_data.columns:
            raise KeyError("``{}`` is not a column in the data.".format(color_col))
        color_groups = pd.unique(plot_data[color_col])
        bootstraps_color_by_group = False
    if show_pairs:
        bootstraps_color_by_group = False

    # Handle the color palette.
    names = color_groups
    n_groups = len(color_groups)
    custom_pal = plot_kwargs["custom_palette"]
    swarm_desat = plot_kwargs["swarm_desat"]
    bar_desat = plot_kwargs["bar_desat"]
    contrast_desat = plot_kwargs["halfviolin_desat"]

    if custom_pal is None:
        unsat_colors = sns.color_palette(n_colors=n_groups)
    else:

        if isinstance(custom_pal, dict):
            groups_in_palette = {k: v for k,v in custom_pal.items()
                                 if k in color_groups}

            # # check that all the keys in custom_pal are found in the
            # # color column.
            # col_grps = {k for k in color_groups}
            # pal_grps = {k for k in custom_pal.keys()}
            # not_in_pal = pal_grps.difference(col_grps)
            # if len(not_in_pal) > 0:
            #     err1 = 'The custom palette keys {} '.format(not_in_pal)
            #     err2 = 'are not found in `{}`. Please check.'.format(color_col)
            #     errstring = (err1 + err2)
            #     raise IndexError(errstring)

            names = groups_in_palette.keys()
            unsat_colors = groups_in_palette.values()

        elif isinstance(custom_pal, list):
            unsat_colors = custom_pal[0: n_groups]

        elif isinstance(custom_pal, str):
            # check it is in the list of matplotlib palettes.
            if custom_pal in plt.colormaps():
                unsat_colors = sns.color_palette(custom_pal, n_groups)
            else:
                err1 = 'The specified `custom_palette` {}'.format(custom_pal)
                err2 = ' is not a matplotlib palette. Please check.'
                raise ValueError(err1 + err2)

    if custom_pal is None and color_col is None:
        swarm_colors = [sns.desaturate(c, swarm_desat) for c in unsat_colors]
        plot_palette_raw = dict(zip(names.categories, swarm_colors))

        bar_color = [sns.desaturate(c, bar_desat) for c in unsat_colors]
        plot_palette_bar = dict(zip(names.categories, bar_color))

        contrast_colors = [sns.desaturate(c, contrast_desat) for c in unsat_colors]
        plot_palette_contrast = dict(zip(names.categories, contrast_colors))

        # For Sankey Diagram plot, no need to worry about the color, each bar will have the same two colors
        # default color palette will be set to "hls"
        plot_palette_sankey = None

    else:
        swarm_colors = [sns.desaturate(c, swarm_desat) for c in unsat_colors]
        plot_palette_raw = dict(zip(names, swarm_colors))

        bar_color = [sns.desaturate(c, bar_desat) for c in unsat_colors]
        plot_palette_bar = dict(zip(names, bar_color))

        contrast_colors = [sns.desaturate(c, contrast_desat) for c in unsat_colors]
        plot_palette_contrast = dict(zip(names, contrast_colors))

        plot_palette_sankey = custom_pal

    # Infer the figsize.
    fig_size   = plot_kwargs["fig_size"]
    if fig_size is None:
        all_groups_count = np.sum([len(i) for i in dabest_obj.idx])
        # Increase the width for delta-delta graph
        if show_delta2 or show_mini_meta:
            all_groups_count += 2
        if is_paired and show_pairs is True and proportional is False:
            frac = 0.75
        else:
            frac = 1
        if float_contrast is True:
            height_inches = 4
            each_group_width_inches = 2.5 * frac
        else:
            height_inches = 6
            each_group_width_inches = 1.5 * frac

        width_inches = (each_group_width_inches * all_groups_count)
        fig_size = (width_inches, height_inches)

    # Initialise the figure.
    # sns.set(context="talk", style='ticks')
    init_fig_kwargs = dict(figsize=fig_size, dpi=plot_kwargs["dpi"]
                            ,tight_layout=True)

    width_ratios_ga = [2.5, 1]
    h_space_cummings = 0.3
    if plot_kwargs["ax"] is not None:
        # New in v0.2.6.
        # Use inset axes to create the estimation plot inside a single axes.
        # Author: Adam L Nekimken. (PR #73)
        inset_contrast = True
        rawdata_axes = plot_kwargs["ax"]
        ax_position = rawdata_axes.get_position()  # [[x0, y0], [x1, y1]]
        
        fig = rawdata_axes.get_figure()
        fig.patch.set_facecolor(face_color)
        
        if float_contrast is True:
            axins = rawdata_axes.inset_axes(
                    [1, 0,
                     width_ratios_ga[1]/width_ratios_ga[0], 1])
            rawdata_axes.set_position(  # [l, b, w, h]
                    [ax_position.x0,
                     ax_position.y0,
                     (ax_position.x1 - ax_position.x0) * (width_ratios_ga[0] /
                                                         sum(width_ratios_ga)),
                     (ax_position.y1 - ax_position.y0)])

            contrast_axes = axins

        else:
            axins = rawdata_axes.inset_axes([0, -1 - h_space_cummings, 1, 1])
            plot_height = ((ax_position.y1 - ax_position.y0) /
                           (2 + h_space_cummings))
            rawdata_axes.set_position(
                    [ax_position.x0,
                     ax_position.y0 + (1 + h_space_cummings) * plot_height,
                     (ax_position.x1 - ax_position.x0),
                     plot_height])

            # If the contrast axes are NOT floating, create lists to store
            # raw ylims and raw tick intervals, so that I can normalize
            # their ylims later.
            contrast_ax_ylim_low = list()
            contrast_ax_ylim_high = list()
            contrast_ax_ylim_tickintervals = list()
        contrast_axes = axins
        rawdata_axes.contrast_axes = axins

    else:
        inset_contrast = False
        # Here, we hardcode some figure parameters.
        if float_contrast is True:
            fig, axx = plt.subplots(
                    ncols=2,
                    gridspec_kw={"width_ratios": width_ratios_ga,
                                 "wspace": 0},
                                 **init_fig_kwargs)
            fig.patch.set_facecolor(face_color)

        else:
            fig, axx = plt.subplots(nrows=2,
                                    gridspec_kw={"hspace": 0.3},
                                    **init_fig_kwargs)
            fig.patch.set_facecolor(face_color)
            # If the contrast axes are NOT floating, create lists to store
            # raw ylims and raw tick intervals, so that I can normalize
            # their ylims later.
            contrast_ax_ylim_low = list()
            contrast_ax_ylim_high = list()
            contrast_ax_ylim_tickintervals = list()

        rawdata_axes  = axx[0]
        contrast_axes = axx[1]
    rawdata_axes.set_frame_on(False)
    contrast_axes.set_frame_on(False)

    redraw_axes_kwargs = {'colors'     : ytick_color,
                          'facecolors' : ytick_color,
                          'lw'      : 1,
                          'zorder'  : 10,
                          'clip_on' : False}

    swarm_ylim = plot_kwargs["swarm_ylim"]

    if swarm_ylim is not None:
        rawdata_axes.set_ylim(swarm_ylim)

    one_sankey = None
    if is_paired is not None:
        one_sankey = False # Flag to indicate if only one sankey is plotted.

    if show_pairs is True:
        # Determine temp_idx based on is_paired and proportional conditions
        if is_paired == "baseline":
            idx_pairs = [(control, test) for i in idx for control, test in zip([i[0]] * (len(i) - 1), i[1:])]
            temp_idx = idx if not proportional else idx_pairs
        else:
            idx_pairs = [(control, test) for i in idx for control, test in zip(i[:-1], i[1:])]
            temp_idx = idx if not proportional else idx_pairs

        # Determine temp_all_plot_groups based on proportional condition
        plot_groups = [item for i in temp_idx for item in i]
        temp_all_plot_groups = all_plot_groups if not proportional else plot_groups
        
        if proportional==False:
        # Plot the raw data as a slopegraph.
        # Pivot the long (melted) data.
            if color_col is None:
                pivot_values = yvar
            else:
                pivot_values = [yvar, color_col]
            pivoted_plot_data = pd.pivot(data=plot_data, index=dabest_obj.id_col,
                                         columns=xvar, values=pivot_values)
            x_start = 0
            for ii, current_tuple in enumerate(temp_idx):
                if len(temp_idx) > 1:
                    # Select only the data for the current tuple.
                    if color_col is None:
                        current_pair = pivoted_plot_data.reindex(columns=current_tuple)
                    else:
                        current_pair = pivoted_plot_data[yvar].reindex(columns=current_tuple)
                else:
                    if color_col is None:
                        current_pair = pivoted_plot_data
                    else:
                        current_pair = pivoted_plot_data[yvar]
                grp_count = len(current_tuple)
                # Iterate through the data for the current tuple.
                for ID, observation in current_pair.iterrows():
                    x_points = [t for t in range(x_start, x_start + grp_count)]
                    y_points = observation.tolist()

                    if color_col is None:
                        slopegraph_kwargs['color'] = ytick_color
                    else:
                        color_key = pivoted_plot_data[color_col,
                                                      current_tuple[0]].loc[ID]
                        if isinstance(color_key, str) == True:
                            slopegraph_kwargs['color'] = plot_palette_raw[color_key]
                            slopegraph_kwargs['label'] = color_key

                    rawdata_axes.plot(x_points, y_points, **slopegraph_kwargs)
                x_start = x_start + grp_count
            # Set the tick labels, because the slopegraph plotting doesn't.
            rawdata_axes.set_xticks(np.arange(0, len(temp_all_plot_groups)))
            rawdata_axes.set_xticklabels(temp_all_plot_groups)
            
        else:
            # Plot the raw data as a set of Sankey Diagrams aligned like barplot.
            group_summaries = plot_kwargs["group_summaries"]
            if group_summaries is None:
                group_summaries = "mean_sd"
            err_color = plot_kwargs["err_color"]
            if err_color == None:
                err_color = "black"

            if show_pairs is True:
                sankey_control_group = []
                sankey_test_group = []
                for i in temp_idx:
                    sankey_control_group.append(i[0])
                    sankey_test_group.append(i[1])                   

            if len(temp_all_plot_groups) == 2:
                one_sankey = True   
            
            # Replace the paired proportional plot with sankey diagram
            sankey = sankeydiag(plot_data, xvar=xvar, yvar=yvar, 
                            left_idx=sankey_control_group, 
                            right_idx=sankey_test_group,
                            palette=plot_palette_sankey,
                            ax=rawdata_axes, 
                            one_sankey=one_sankey,
                            **sankey_kwargs)
                            
    else:
        if proportional==False:
            # Plot the raw data as a swarmplot.
            rawdata_plot = sns.swarmplot(data=plot_data, x=xvar, y=yvar,
                                         ax=rawdata_axes,
                                         order=all_plot_groups, hue=color_col,
                                         palette=plot_palette_raw, zorder=1,
                                         **swarmplot_kwargs)
        else:
            # Plot the raw data as a barplot.
            bar1_df = pd.DataFrame({xvar: all_plot_groups, 'proportion': np.ones(len(all_plot_groups))})
            bar1 = sns.barplot(data=bar1_df, x=xvar, y="proportion",
                               ax=rawdata_axes,
                               order=all_plot_groups,
                               linewidth=2, facecolor=(1, 1, 1, 0), edgecolor=bar_color,
                               zorder=1)
            bar2 = sns.barplot(data=plot_data, x=xvar, y=yvar,
                               ax=rawdata_axes,
                               order=all_plot_groups,
                               palette=plot_palette_bar,
                               zorder=1,
                               **barplot_kwargs)
            # adjust the width of bars
            bar_width = plot_kwargs["bar_width"]
            for bar in bar1.patches:
                x = bar.get_x()
                width = bar.get_width()
                centre = x + width / 2.
                bar.set_x(centre - bar_width / 2.)
                bar.set_width(bar_width)

        # Plot the gapped line summaries, if this is not a Cumming plot.
        # Also, we will not plot gapped lines for paired plots. For now.
        group_summaries = plot_kwargs["group_summaries"]
        if group_summaries is None:
            group_summaries = "mean_sd"

        if group_summaries is not None and proportional==False:
            # Create list to gather xspans.
            xspans = []
            line_colors = []
            for jj, c in enumerate(rawdata_axes.collections):
                try:
                    _, x_max, _, _ = get_swarm_spans(c)
                    x_max_span = x_max - jj
                    xspans.append(x_max_span)
                except TypeError:
                    # we have got a None, so skip and move on.
                    pass

                if bootstraps_color_by_group is True:
                    line_colors.append(plot_palette_raw[all_plot_groups[jj]])

            if len(line_colors) != len(all_plot_groups):
                line_colors = ytick_color

            error_bar(plot_data, x=xvar, y=yvar,
                         # Hardcoded offset...
                         offset=xspans + np.array(plot_kwargs["group_summaries_offset"]),
                         line_color=line_colors,
                         gap_width_percent=1.5,
                         type=group_summaries, ax=rawdata_axes,
                         method="gapped_lines",
                         **group_summary_kwargs)

        if group_summaries is not None and proportional == True:

            err_color = plot_kwargs["err_color"]
            if err_color == None:
                err_color = "black"
            error_bar(plot_data, x=xvar, y=yvar,
                     offset=0,
                     line_color=err_color,
                     gap_width_percent=1.5,
                     type=group_summaries, ax=rawdata_axes,
                     method="proportional_error_bar",
                     **group_summary_kwargs)

    # Add the counts to the rawdata axes xticks.
    counts = plot_data.groupby(xvar).count()[yvar]
    ticks_with_counts = []
    for xticklab in rawdata_axes.xaxis.get_ticklabels():
        t = xticklab.get_text()
        if t.rfind("\n") != -1:
            te = t[t.rfind("\n") + len("\n"):]
            N = str(counts.loc[te])
            te = t
        else:
            te = t
            N = str(counts.loc[te])

        ticks_with_counts.append("{}\nN = {}".format(te, N))

    rawdata_axes.set_xticklabels(ticks_with_counts)

    # Save the handles and labels for the legend.
    handles, labels = rawdata_axes.get_legend_handles_labels()
    legend_labels  = [l for l in labels]
    legend_handles = [h for h in handles]
    if bootstraps_color_by_group is False:
        rawdata_axes.legend().set_visible(False)

    # Enforce the xtick of rawdata_axes to be 0 and 1 after drawing only one sankey
    if one_sankey:
        rawdata_axes.set_xticks([0, 1])

    # Plot effect sizes and bootstraps.
    # Take note of where the `control` groups are.
    if is_paired == "baseline" and show_pairs == True:
        if proportional == True and one_sankey == False:
            ticks_to_skip = []
            ticks_to_plot = np.arange(0, len(temp_all_plot_groups)/2).tolist()
            ticks_to_start_sankey = np.cumsum([len(i)-1 for i in idx]).tolist()
            ticks_to_start_sankey.pop()
            ticks_to_start_sankey.insert(0, 0)
        else:
            # ticks_to_skip = np.arange(0, len(temp_all_plot_groups), 2).tolist()
            # ticks_to_plot = np.arange(1, len(temp_all_plot_groups), 2).tolist()
            ticks_to_skip = np.cumsum([len(t) for t in idx])[:-1].tolist()
            ticks_to_skip.insert(0, 0)
            # Then obtain the ticks where we have to plot the effect sizes.
            ticks_to_plot = [t for t in range(0, len(all_plot_groups))
                        if t not in ticks_to_skip]
            ticks_to_skip_contrast = np.cumsum([(len(t)) for t in idx])[:-1].tolist()
            ticks_to_skip_contrast.insert(0, 0)
    else:
        if proportional == True and one_sankey == False:
            ticks_to_skip = [len(sankey_control_group)]
            # Then obtain the ticks where we have to plot the effect sizes.
            ticks_to_plot = [t for t in range(0, len(temp_idx))
                        if t not in ticks_to_skip]
            ticks_to_skip = []
            ticks_to_start_sankey = np.cumsum([len(i)-1 for i in idx]).tolist()
            ticks_to_start_sankey.pop()
            ticks_to_start_sankey.insert(0, 0)
        else:
            ticks_to_skip = np.cumsum([len(t) for t in idx])[:-1].tolist()
            ticks_to_skip.insert(0, 0)
            # Then obtain the ticks where we have to plot the effect sizes.
            ticks_to_plot = [t for t in range(0, len(all_plot_groups))
                        if t not in ticks_to_skip]

    # Plot the bootstraps, then the effect sizes and CIs.
    es_marker_size   = plot_kwargs["es_marker_size"]
    halfviolin_alpha = plot_kwargs["halfviolin_alpha"]

    ci_type = plot_kwargs["ci_type"]

    results      = EffectSizeDataFrame.results
    contrast_xtick_labels = []


    for j, tick in enumerate(ticks_to_plot):
        current_group     = results.test[j]
        current_control   = results.control[j]
        current_bootstrap = results.bootstraps[j]
        current_effsize   = results.difference[j]
        if ci_type == "bca":
            current_ci_low    = results.bca_low[j]
            current_ci_high   = results.bca_high[j]
        else:
            current_ci_low    = results.pct_low[j]
            current_ci_high   = results.pct_high[j]


        # Create the violinplot.
        # New in v0.2.6: drop negative infinities before plotting.
        v = contrast_axes.violinplot(current_bootstrap[~np.isinf(current_bootstrap)],
                                     positions=[tick],
                                     **violinplot_kwargs)
        # Turn the violinplot into half, and color it the same as the swarmplot.
        # Do this only if the color column is not specified.
        # Ideally, the alpha (transparency) fo the violin plot should be
        # less than one so the effect size and CIs are visible.
        if bootstraps_color_by_group is True:
            fc = plot_palette_contrast[current_group]
        else:
            fc = "grey"

        halfviolin(v, fill_color=fc, alpha=halfviolin_alpha)

        # Plot the effect size.
        contrast_axes.plot([tick], current_effsize, marker='o',
                           color=ytick_color,
                           markersize=es_marker_size)
        # Plot the confidence interval.
        contrast_axes.plot([tick, tick],
                           [current_ci_low, current_ci_high],
                           linestyle="-",
                           color=ytick_color,
                           linewidth=group_summary_kwargs['lw'])

        contrast_xtick_labels.append("{}\nminus\n{}".format(current_group,
                                                   current_control))

    # Plot mini-meta violin
    if show_mini_meta or show_delta2:
        if show_mini_meta:
            mini_meta_delta = EffectSizeDataFrame.mini_meta_delta
            data            = mini_meta_delta.bootstraps_weighted_delta
            difference      = mini_meta_delta.difference
            if ci_type == "bca":
                ci_low          = mini_meta_delta.bca_low
                ci_high         = mini_meta_delta.bca_high
            else:
                ci_low          = mini_meta_delta.pct_low
                ci_high         = mini_meta_delta.pct_high
        else: 
            delta_delta     = EffectSizeDataFrame.delta_delta
            data            = delta_delta.bootstraps_delta_delta
            difference      = delta_delta.difference
            if ci_type == "bca":
                ci_low          = delta_delta.bca_low
                ci_high         = delta_delta.bca_high
            else:
                ci_low          = delta_delta.pct_low
                ci_high         = delta_delta.pct_high
        #Create the violinplot.
        #New in v0.2.6: drop negative infinities before plotting.
        position = max(rawdata_axes.get_xticks())+2
        v = contrast_axes.violinplot(data[~np.isinf(data)],
                                     positions=[position],
                                     **violinplot_kwargs)

        fc = "grey"

        halfviolin(v, fill_color=fc, alpha=halfviolin_alpha)

        # Plot the effect size.
        contrast_axes.plot([position], difference, marker='o',
                           color=ytick_color,
                           markersize=es_marker_size)
        # Plot the confidence interval.
        contrast_axes.plot([position, position],
                           [ci_low, ci_high],
                           linestyle="-",
                           color=ytick_color,
                           linewidth=group_summary_kwargs['lw'])
        if show_mini_meta:
            contrast_xtick_labels.extend(["","Weighted delta"])
        else:
            contrast_xtick_labels.extend(["","delta-delta"])

    # Make sure the contrast_axes x-lims match the rawdata_axes xlims,
    # and add an extra violinplot tick for delta-delta plot.
    if show_delta2 is False and show_mini_meta is False:
        contrast_axes.set_xticks(rawdata_axes.get_xticks())
    else:
        temp = rawdata_axes.get_xticks()
        temp = np.append(temp, [max(temp)+1, max(temp)+2])
        contrast_axes.set_xticks(temp)

    if show_pairs is True:
        max_x = contrast_axes.get_xlim()[1]
        rawdata_axes.set_xlim(-0.375, max_x)

    if float_contrast is True:
        contrast_axes.set_xlim(0.5, 1.5)
    elif show_delta2 or show_mini_meta:
        # Increase the xlim of raw data by 2
        temp = rawdata_axes.get_xlim()
        if show_pairs:
            rawdata_axes.set_xlim(temp[0], temp[1]+0.25)
        else:
            rawdata_axes.set_xlim(temp[0], temp[1]+2)
        contrast_axes.set_xlim(rawdata_axes.get_xlim())
    else:
        contrast_axes.set_xlim(rawdata_axes.get_xlim())

    # Properly label the contrast ticks.
    for t in ticks_to_skip:
        contrast_xtick_labels.insert(t, "")
    
    contrast_axes.set_xticklabels(contrast_xtick_labels)

    if bootstraps_color_by_group is False:
        legend_labels_unique = np.unique(legend_labels)
        unique_idx = np.unique(legend_labels, return_index=True)[1]
        legend_handles_unique = (pd.Series(legend_handles, dtype="object").loc[unique_idx]).tolist()

        if len(legend_handles_unique) > 0:
            if float_contrast is True:
                axes_with_legend = contrast_axes
                if show_pairs is True:
                    bta = (1.75, 1.02)
                else:
                    bta = (1.5, 1.02)
            else:
                axes_with_legend = rawdata_axes
                if show_pairs is True:
                    bta = (1.02, 1.)
                else:
                    bta = (1.,1.)
            leg = axes_with_legend.legend(legend_handles_unique,
                                          legend_labels_unique,
                                          bbox_to_anchor=bta,
                                          **legend_kwargs)
            if show_pairs is True:
                for line in leg.get_lines():
                    line.set_linewidth(3.0)

    og_ylim_raw = rawdata_axes.get_ylim()
    og_xlim_raw = rawdata_axes.get_xlim()

    if float_contrast is True:
        # For Gardner-Altman plots only.

        # Normalize ylims and despine the floating contrast axes.
        # Check that the effect size is within the swarm ylims.
        if effect_size_type in ["mean_diff", "cohens_d", "hedges_g","cohens_h"]:
            control_group_summary = plot_data.groupby(xvar)\
                                             .mean(numeric_only=True).loc[current_control, yvar]
            test_group_summary = plot_data.groupby(xvar)\
                                          .mean(numeric_only=True).loc[current_group, yvar]
        elif effect_size_type == "median_diff":
            control_group_summary = plot_data.groupby(xvar)\
                                             .median().loc[current_control, yvar]
            test_group_summary = plot_data.groupby(xvar)\
                                          .median().loc[current_group, yvar]

        if swarm_ylim is None:
            swarm_ylim = rawdata_axes.get_ylim()

        _, contrast_xlim_max = contrast_axes.get_xlim()

        difference = float(results.difference[0])
        
        if effect_size_type in ["mean_diff", "median_diff"]:
            # Align 0 of contrast_axes to reference group mean of rawdata_axes.
            # If the effect size is positive, shift the contrast axis up.
            rawdata_ylims = np.array(rawdata_axes.get_ylim())
            if current_effsize > 0:
                rightmin, rightmax = rawdata_ylims - current_effsize
            # If the effect size is negative, shift the contrast axis down.
            elif current_effsize < 0:
                rightmin, rightmax = rawdata_ylims + current_effsize
            else:
                rightmin, rightmax = rawdata_ylims

            contrast_axes.set_ylim(rightmin, rightmax)

            og_ylim_contrast = rawdata_axes.get_ylim() - np.array(control_group_summary)

            contrast_axes.set_ylim(og_ylim_contrast)
            contrast_axes.set_xlim(contrast_xlim_max-1, contrast_xlim_max)

        elif effect_size_type in ["cohens_d", "hedges_g","cohens_h"]:
            if is_paired:
                which_std = 1
            else:
                which_std = 0
            temp_control = plot_data[plot_data[xvar] == current_control][yvar]
            temp_test    = plot_data[plot_data[xvar] == current_group][yvar]
            
            stds = _compute_standardizers(temp_control, temp_test)
            if is_paired:
                pooled_sd = stds[1]
            else:
                pooled_sd = stds[0]
            
            if effect_size_type == 'hedges_g':
                gby_count   = plot_data.groupby(xvar).count()
                len_control = gby_count.loc[current_control, yvar]
                len_test    = gby_count.loc[current_group, yvar]
                            
                hg_correction_factor = _compute_hedges_correction_factor(len_control, len_test)
                            
                ylim_scale_factor = pooled_sd / hg_correction_factor

            elif effect_size_type == "cohens_h":
                ylim_scale_factor = (np.mean(temp_test)-np.mean(temp_control)) / difference

            else:
                ylim_scale_factor = pooled_sd
                
            scaled_ylim = ((rawdata_axes.get_ylim() - control_group_summary) / ylim_scale_factor).tolist()

            contrast_axes.set_ylim(scaled_ylim)
            og_ylim_contrast = scaled_ylim

            contrast_axes.set_xlim(contrast_xlim_max-1, contrast_xlim_max)

        if one_sankey is None:
            # Draw summary lines for control and test groups..
            for jj, axx in enumerate([rawdata_axes, contrast_axes]):

                # Draw effect size line.
                if jj == 0:
                    ref = control_group_summary
                    diff = test_group_summary
                    effsize_line_start = 1

                elif jj == 1:
                    ref = 0
                    diff = ref + difference
                    effsize_line_start = contrast_xlim_max-1.1

                xlimlow, xlimhigh = axx.get_xlim()

                # Draw reference line.
                axx.hlines(ref,            # y-coordinates
                        0, xlimhigh,  # x-coordinates, start and end.
                        **reflines_kwargs)
                            
                # Draw effect size line.
                axx.hlines(diff,
                        effsize_line_start, xlimhigh,
                        **reflines_kwargs)
        else: 
            ref = 0
            diff = ref + difference
            effsize_line_start = contrast_xlim_max - 0.9
            xlimlow, xlimhigh = contrast_axes.get_xlim()
            # Draw reference line.
            contrast_axes.hlines(ref,            # y-coordinates
                    effsize_line_start, xlimhigh,  # x-coordinates, start and end.
                    **reflines_kwargs)
                        
            # Draw effect size line.
            contrast_axes.hlines(diff,
                    effsize_line_start, xlimhigh,
                    **reflines_kwargs)    
        rawdata_axes.set_xlim(og_xlim_raw) # to align the axis
        # Despine appropriately.
        sns.despine(ax=rawdata_axes,  bottom=True)
        sns.despine(ax=contrast_axes, left=True, right=False)

        # Insert break between the rawdata axes and the contrast axes
        # by re-drawing the x-spine.
        rawdata_axes.hlines(og_ylim_raw[0],                  # yindex
                            rawdata_axes.get_xlim()[0], 1.3, # xmin, xmax
                            **redraw_axes_kwargs)
        rawdata_axes.set_ylim(og_ylim_raw)

        contrast_axes.hlines(contrast_axes.get_ylim()[0],
                             contrast_xlim_max-0.8, contrast_xlim_max,
                             **redraw_axes_kwargs)


    else:
        # For Cumming Plots only.

        # Set custom contrast_ylim, if it was specified.
        if plot_kwargs['contrast_ylim'] is not None or (plot_kwargs['delta2_ylim'] is not None and show_delta2):

            if plot_kwargs['contrast_ylim'] is not None:
                custom_contrast_ylim = plot_kwargs['contrast_ylim']
                if plot_kwargs['delta2_ylim'] is not None and show_delta2:
                    custom_delta2_ylim = plot_kwargs['delta2_ylim']
                    if custom_contrast_ylim!=custom_delta2_ylim:
                        err1 = "Please check if `contrast_ylim` and `delta2_ylim` are assigned"
                        err2 = "with same values."
                        raise ValueError(err1 + err2)
            else:
                custom_delta2_ylim = plot_kwargs['delta2_ylim']
                custom_contrast_ylim = custom_delta2_ylim

            if len(custom_contrast_ylim) != 2:
                err1 = "Please check `contrast_ylim` consists of "
                err2 = "exactly two numbers."
                raise ValueError(err1 + err2)

            if effect_size_type == "cliffs_delta":
                # Ensure the ylims for a cliffs_delta plot never exceed [-1, 1].
                l = plot_kwargs['contrast_ylim'][0]
                h = plot_kwargs['contrast_ylim'][1]
                low = -1 if l < -1 else l
                high = 1 if h > 1 else h
                contrast_axes.set_ylim(low, high)
            else:
                contrast_axes.set_ylim(custom_contrast_ylim)

        # If 0 lies within the ylim of the contrast axes,
        # draw a zero reference line.
        contrast_axes_ylim = contrast_axes.get_ylim()
        if contrast_axes_ylim[0] < contrast_axes_ylim[1]:
            contrast_ylim_low, contrast_ylim_high = contrast_axes_ylim
        else:
            contrast_ylim_high, contrast_ylim_low = contrast_axes_ylim
        if contrast_ylim_low < 0 < contrast_ylim_high:
            contrast_axes.axhline(y=0, **reflines_kwargs)

        if is_paired == "baseline" and show_pairs == True:
            if proportional == True and one_sankey == False:
                rightend_ticks_raw = np.array([len(i)-2 for i in idx]) + np.array(ticks_to_start_sankey)
            else:    
                rightend_ticks_raw = np.array([len(i)-1 for i in temp_idx]) + np.array(ticks_to_skip)
            for ax in [rawdata_axes]:
                sns.despine(ax=ax, bottom=True)
        
                ylim = ax.get_ylim()
                xlim = ax.get_xlim()
                redraw_axes_kwargs['y'] = ylim[0]
        
                if proportional == True and one_sankey == False:
                    for k, start_tick in enumerate(ticks_to_start_sankey):
                        end_tick = rightend_ticks_raw[k]
                        ax.hlines(xmin=start_tick, xmax=end_tick,
                              **redraw_axes_kwargs)
                else:   
                    for k, start_tick in enumerate(ticks_to_skip):
                        end_tick = rightend_ticks_raw[k]
                        ax.hlines(xmin=start_tick, xmax=end_tick,
                              **redraw_axes_kwargs)
                ax.set_ylim(ylim)
                del redraw_axes_kwargs['y']
            
            if proportional == False:
                temp_length = [(len(i)-1) for i in idx]
            else:
                temp_length = [(len(i)-1)*2-1 for i in idx]
            if proportional == True and one_sankey == False:
                rightend_ticks_contrast = np.array([len(i)-2 for i in idx]) + np.array(ticks_to_start_sankey)
            else:   
                rightend_ticks_contrast = np.array(temp_length) + np.array(ticks_to_skip_contrast)
            for ax in [contrast_axes]:
                sns.despine(ax=ax, bottom=True)
        
                ylim = ax.get_ylim()
                xlim = ax.get_xlim()
                redraw_axes_kwargs['y'] = ylim[0]
        
                if proportional == True and one_sankey == False:
                    for k, start_tick in enumerate(ticks_to_start_sankey):
                        end_tick = rightend_ticks_contrast[k]
                        ax.hlines(xmin=start_tick, xmax=end_tick,
                                **redraw_axes_kwargs)
                else:
                    for k, start_tick in enumerate(ticks_to_skip_contrast):
                        end_tick = rightend_ticks_contrast[k]
                        ax.hlines(xmin=start_tick, xmax=end_tick,
                                **redraw_axes_kwargs)                
        
                ax.set_ylim(ylim)
                del redraw_axes_kwargs['y']
        else:
            # Compute the end of each x-axes line.
            if proportional == True and one_sankey == False:
                rightend_ticks = np.array([len(i)-2 for i in idx]) + np.array(ticks_to_start_sankey)
            else:
                rightend_ticks = np.array([len(i)-1 for i in idx]) + np.array(ticks_to_skip)
        
            for ax in [rawdata_axes, contrast_axes]:
                sns.despine(ax=ax, bottom=True)
            
                ylim = ax.get_ylim()
                xlim = ax.get_xlim()
                redraw_axes_kwargs['y'] = ylim[0]
            
                if proportional == True and one_sankey == False:
                    for k, start_tick in enumerate(ticks_to_start_sankey):
                        end_tick = rightend_ticks[k]
                        ax.hlines(xmin=start_tick, xmax=end_tick,
                                **redraw_axes_kwargs)
                else:
                    for k, start_tick in enumerate(ticks_to_skip):
                        end_tick = rightend_ticks[k]
                        ax.hlines(xmin=start_tick, xmax=end_tick,
                                **redraw_axes_kwargs)
            
                ax.set_ylim(ylim)
                del redraw_axes_kwargs['y']

    if show_delta2 is True or show_mini_meta is True:
        ylim = contrast_axes.get_ylim()
        redraw_axes_kwargs['y'] = ylim[0]
        x_ticks = contrast_axes.get_xticks()
        contrast_axes.hlines(xmin=x_ticks[-2], xmax=x_ticks[-1],
                              **redraw_axes_kwargs)
        del redraw_axes_kwargs['y']

    # Set raw axes y-label.
    swarm_label = plot_kwargs['swarm_label']
    if swarm_label is None and yvar is None:
        swarm_label = "value"
    elif swarm_label is None and yvar is not None:
        swarm_label = yvar

    bar_label = plot_kwargs['bar_label']
    if bar_label is None and effect_size_type != "cohens_h":
        bar_label = "proportion of success"
    elif bar_label is None and effect_size_type == "cohens_h":
        bar_label = "value"

    # Place contrast axes y-label.
    contrast_label_dict = {'mean_diff': "mean difference",
                           'median_diff': "median difference",
                           'cohens_d': "Cohen's d",
                           'hedges_g': "Hedges' g",
                           'cliffs_delta': "Cliff's delta",
                           'cohens_h': "Cohen's h"}

    if proportional == True and effect_size_type != "cohens_h":
        default_contrast_label = "proportion difference"
    else:
        default_contrast_label = contrast_label_dict[EffectSizeDataFrame.effect_size]


    if plot_kwargs['contrast_label'] is None:
        if is_paired:
            contrast_label = "paired\n{}".format(default_contrast_label)
        else:
            contrast_label = default_contrast_label
        contrast_label = contrast_label.capitalize()
    else:
        contrast_label = plot_kwargs['contrast_label']

    contrast_axes.set_ylabel(contrast_label)
    if float_contrast is True:
        contrast_axes.yaxis.set_label_position("right")

    # Set the rawdata axes labels appropriately
    if proportional == False:
        rawdata_axes.set_ylabel(swarm_label)
    else:
        rawdata_axes.set_ylabel(bar_label)
    rawdata_axes.set_xlabel("")

    # Because we turned the axes frame off, we also need to draw back
    # the y-spine for both axes.
    if float_contrast==False:
        rawdata_axes.set_xlim(contrast_axes.get_xlim())
    og_xlim_raw = rawdata_axes.get_xlim()
    rawdata_axes.vlines(og_xlim_raw[0],
                         og_ylim_raw[0], og_ylim_raw[1],
                         **redraw_axes_kwargs)

    og_xlim_contrast = contrast_axes.get_xlim()

    if float_contrast is True:
        xpos = og_xlim_contrast[1]
    else:
        xpos = og_xlim_contrast[0]

    og_ylim_contrast = contrast_axes.get_ylim()
    contrast_axes.vlines(xpos,
                         og_ylim_contrast[0], og_ylim_contrast[1],
                         **redraw_axes_kwargs)


    if show_delta2 is True:
        if plot_kwargs['delta2_label'] is None:
            delta2_label = "delta - delta"
        else: 
            delta2_label = plot_kwargs['delta2_label']
        delta2_axes = contrast_axes.twinx()
        delta2_axes.set_frame_on(False)
        delta2_axes.set_ylabel(delta2_label)
        og_xlim_delta = contrast_axes.get_xlim()
        og_ylim_delta = contrast_axes.get_ylim()
        delta2_axes.set_ylim(og_ylim_delta)
        delta2_axes.vlines(og_xlim_delta[1],
                         og_ylim_delta[0], og_ylim_delta[1],
                         **redraw_axes_kwargs)

    # Make sure no stray ticks appear!
    rawdata_axes.xaxis.set_ticks_position('bottom')
    rawdata_axes.yaxis.set_ticks_position('left')
    contrast_axes.xaxis.set_ticks_position('bottom')
    if float_contrast is False:
        contrast_axes.yaxis.set_ticks_position('left')

    # Reset rcParams.
    for parameter in _changed_rcParams:
        plt.rcParams[parameter] = original_rcParams[parameter]

    # Return the figure.
    return fig


def EffectSizeDataFramePlotterHorizontal(EffectSizeDataFrame, **kwargs):
    """
    Custom function that creates a Horizontal estimation plot from an EffectSizeDataFrame.
    Keywords
    --------
    EffectSizeDataFrame: A `dabest` EffectSizeDataFrame object.
    **plot_kwargs:
        raw_marker_size=6, es_marker_size=9,
        swarm_label=None, contrast_label=None, color_col=None,
        custom_palette=None, swarm_desat=0.5, halfviolin_desat=1,
        halfviolin_alpha=0.8,
        face_color = None,
        fig_size=None,
        dpi=100,
        ax=None,
        legend_kwargs=None,

        mean_gap_width_percent = 2,
        title = None,
        title_fontsize = 14,
        horizontal_plot_kwargs = None,
        horizontal_swarmplot_kwargs = None,
        horizontal_violinplot_kwargs = None,
        horizontal_table_kwargs = None,
        contrast_bar = False,
        contrast_bar_kwargs = None,
        contrast_dots = False,
        contrast_dots_kwargs = None,
    """
    ## Import Modules
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import warnings
    from .misc_tools import merge_two_dicts
    from .plot_tools import horizontal_colormaker,halfviolin

    ## Variables
    ### General Variables
    ax = kwargs["ax"]
    fig_size = kwargs["fig_size"] 
    show_mini_meta = kwargs["show_mini_meta"]
    dpi = kwargs["dpi"]
    ### SwarmPlot Variables
    mean_gap_width_percent = kwargs["mean_gap_width_percent"]
    raw_marker_size = kwargs["raw_marker_size"]    
    swarm_label = kwargs["swarm_label"]    
    ### ViolinPlot Variables
    es_marker_size = kwargs["es_marker_size"]
    halfviolin_alpha = kwargs["halfviolin_alpha"]
    contrast_label = kwargs["contrast_label"]
    contrast_bars = kwargs["contrast_bars"]
    contrast_dots = kwargs["contrast_dots"]
    contrast_dots_kwargs = kwargs["contrast_dots_kwargs"]
    
    if kwargs["face_color"] is None:
        face_color = "white"

    ### Dabest Variables
    dabest_obj = EffectSizeDataFrame.dabest_obj
    plot_data = EffectSizeDataFrame._plot_data
    xvar = EffectSizeDataFrame.xvar
    yvar = EffectSizeDataFrame.yvar
    is_paired = EffectSizeDataFrame.is_paired
    delta2 = EffectSizeDataFrame.delta2
    mini_meta = EffectSizeDataFrame.mini_meta
    effect_size = EffectSizeDataFrame.effect_size
    # proportional = EffectSizeDataFrame.proportional

    all_plot_groups = dabest_obj._all_plot_groups
    idx = dabest_obj.idx
    id_col = dabest_obj.id_col
    unpacked_idx = [item for sublist in idx for item in sublist]
    multi_unpaired_control = True if not is_paired and len(idx)>1 else False
    multi_paired_control = True if is_paired and any(len(x)>2 for x in idx) else False

    if not is_paired:
        show_pairs = False
    else:
        show_pairs = kwargs["show_pairs"]

    ### Num Exps
    number_of_groups = len(idx)
    Num_of_Exps_in_each_group = []
    for n in range(number_of_groups):
        Num_of_Exps_in_each_group.append(len(idx[n]))
    if is_paired:
        Num_Exps = len(unpacked_idx)
        if multi_paired_control:
            Num_of_contrasts = len(unpacked_idx) - number_of_groups
        else:
            Num_of_contrasts = number_of_groups
    else:
        Num_Exps = len(unpacked_idx)
        if mini_meta:
            Num_of_contrasts = number_of_groups
        else:
            Num_of_contrasts = len(unpacked_idx) - number_of_groups
    MM_Adj_Num_Exps = Num_Exps + 1 if mini_meta or delta2 else Num_Exps  # Mini-Meta Adjustment
    MM_Adj_Num_of_contrasts = Num_of_contrasts + 1 if mini_meta or delta2 else Num_of_contrasts  # Mini-Meta Adjustment

    color_col = kwargs["color_col"]
    ## Convert wide format to long format.
    # wide_format = True if xvar == None else False
    # if color_col == None:
    #     id_vars = [id_col] if is_paired else None
    # else:
    #     id_vars = [id_col,color_col] if is_paired else [color_col]
    #
    # if wide_format:
    #     plot_data = pd.melt(plot_data,value_vars=unpacked_idx, id_vars=id_vars)
    #     yvar='value'
    #     xvar='variable'

    ## Set default kwargs first, then merge with user-dictated ones

    ## Plot kwargs
    default_plot_kwargs = {'plot_width_ratios' : [1,0.7,0.3] , 'contrast_wspace' : 0.05}
    if kwargs["horizontal_plot_kwargs"] is None:
        plot_kwargs = default_plot_kwargs
    else:
        plot_kwargs = merge_two_dicts(default_plot_kwargs, kwargs["horizontal_plot_kwargs"])
    plot_width_ratios = plot_kwargs['plot_width_ratios']
    contrast_wspace=plot_kwargs['contrast_wspace']

    ## Swarmplot kwargs
    default_swarm_kwargs = {'paired_line_alpha': 0.3, 'paired_means_offset': 0.25, 'dot_alpha': 0.8, 'xlim': None,
                            'xlabel_fontsize': 11, 'ylabel_fontsize': 11, 'ylabel_show_samplesize': False,
                            'size': kwargs["raw_marker_size"]}

    if kwargs["horizontal_swarmplot_kwargs"] is None:
        swarm_kwargs = default_swarm_kwargs
    else:
        swarm_kwargs = merge_two_dicts(default_swarm_kwargs, kwargs["horizontal_swarmplot_kwargs"])

    #### Barplot kwargs
    default_barplot_kwargs = {'color':'grey','alpha':0.1,'zorder':0}
    if kwargs["barplot_kwargs"] is None:
        barplot_kwargs = default_barplot_kwargs
    else:
        barplot_kwargs = merge_two_dicts(default_barplot_kwargs, barplot_kwargs)

    # Violinplot kwargs
    default_violin_kwargs = {'contrast_xlim': None, 'contrast_xlabel_fontsize': 12}
    if kwargs["horizontal_violinplot_kwargs"] is None:
        violin_kwargs = default_violin_kwargs
    else:
        violin_kwargs = merge_two_dicts(default_violin_kwargs, kwargs["horizontal_violinplot_kwargs"])
    contrast_xlim = violin_kwargs['contrast_xlim']
    contrast_xlabel_fontsize = violin_kwargs['contrast_xlabel_fontsize']

    ## Legend kwargs.
    default_legend_kwargs = {'loc': 'upper left', 'frameon': False,}
    if kwargs["legend_kwargs"] is None:
        legend_kwargs = default_legend_kwargs
    else:
        legend_kwargs = merge_two_dicts(default_legend_kwargs,kwargs["legend_kwargs"])

    #### Contrast dots kwargs
    default_contrast_dots_kwargs = {'color': None, 'alpha': 0.5, 'size': 3}
    if contrast_dots_kwargs is None:
        contrast_dots_kwargs = default_contrast_dots_kwargs
    else:
        contrast_dots_kwargs = merge_two_dicts(default_contrast_dots_kwargs, contrast_dots_kwargs)
    single_color_contrast_dots = contrast_dots_kwargs['color']
    del contrast_dots_kwargs['color']

    ## Colors
    # if color_col is None:
    #     color_groups = pd.unique(plot_data[xvar])
    #     bootstraps_color_by_group = True
    #     # swarm_colors = horizontal_colormaker(number=number_of_groups if is_paired else Num_Exps,custom_pal=custom_palette,desat_level=swarm_desat)
    #     # halfviolin_colors = horizontal_colormaker(number=number_of_groups if is_paired else Num_Exps,custom_pal=custom_palette,desat_level=halfviolin_desat)
    # else:
    #     if color_col not in plot_data.columns:
    #         raise KeyError("`{}` is not a column in the data.".format(color_col))
    #     color_groups = pd.unique(plot_data[color_col])
    #     bootstraps_color_by_group = False
    # if show_pairs:
    #     bootstraps_color_by_group = False
    #     # color_col_nums = len(plot_data[color_col].unique())
    #     # if color_col_nums == 0:
    #     #     raise ValueError('Color column is empty or does not exist.')
    #     # else:
    #     #     swarm_colors = horizontal_colormaker(number=color_col_nums,custom_pal=custom_palette,desat_level=swarm_desat)
    #     #     halfviolin_colors = horizontal_colormaker(number=color_col_nums,custom_pal=custom_palette,desat_level=halfviolin_desat)

    custom_palette = kwargs["custom_palette"]
    swarm_desat = kwargs["swarm_desat"]
    bar_desat = kwargs["bar_desat"]
    contrast_desat = kwargs["halfviolin_desat"]

    if color_col == None:
        swarm_colors = horizontal_colormaker(number=number_of_groups if is_paired else Num_Exps,custom_pal=custom_palette,desat_level=swarm_desat)
        contrast_colors = horizontal_colormaker(number=number_of_groups if is_paired else Num_Exps,custom_pal=custom_palette,desat_level=contrast_desat)
    else:
        if color_col not in plot_data.columns:
            raise KeyError("`{}` is not a column in the data.".format(color_col))
        else:
            color_col_nums = len(plot_data[color_col].unique())
            if color_col_nums == 0:
                raise ValueError('Color column is empty or does not exist.')
            else:
                swarm_colors = horizontal_colormaker(number=color_col_nums,custom_pal=custom_palette,desat_level=swarm_desat)
                contrast_colors = horizontal_colormaker(number=color_col_nums,custom_pal=custom_palette,desat_level=contrast_desat)

    # Handling of color palette


    # if custom_palette is None:
    #     unsat_colors = sns.color_palette(n_colors=n_groups)
    # else:
    #     if isinstance(custom_palette, dict):
    #         groups_in_palette = {k: v for k,v in custom_palette.items()
    #                              if k in color_groups}
    #
    #         # # check that all the keys in custom_pal are found in the
    #         # # color column.
    #         # col_grps = {k for k in color_groups}
    #         # pal_grps = {k for k in custom_pal.keys()}
    #         # not_in_pal = pal_grps.difference(col_grps)
    #         # if len(not_in_pal) > 0:
    #         #     err1 = 'The custom palette keys {} '.format(not_in_pal)
    #         #     err2 = 'are not found in `{}`. Please check.'.format(color_col)
    #         #     errstring = (err1 + err2)
    #         #     raise IndexError(errstring)
    #
    #         names = groups_in_palette.keys()
    #         unsat_colors = groups_in_palette.values()
    #
    #     elif isinstance(custom_palette, list):
    #         unsat_colors = custom_palette[0: n_groups]
    #
    #     elif isinstance(custom_palette, str):
    #         # check it is in the list of matplotlib palettes.
    #         if custom_palette in plt.colormaps():
    #             unsat_colors = sns.color_palette(custom_palette, n_groups)
    #         else:
    #             err1 = 'The specified `custom_palette` {}'.format(custom_palette)
    #             err2 = ' is not a matplotlib palette. Please check.'
    #             raise ValueError(err1 + err2)
    #
    # if custom_palette is None and color_col is None:
    #     swarm_colors = [sns.desaturate(c, swarm_desat) for c in unsat_colors]
    #     plot_palette_raw = dict(zip(names.categories, swarm_colors))
    #
    #     bar_color = [sns.desaturate(c, bar_desat) for c in unsat_colors]
    #     plot_palette_bar = dict(zip(names.categories, bar_color))
    #
    #     contrast_colors = [sns.desaturate(c, contrast_desat) for c in unsat_colors]
    #     plot_palette_contrast = dict(zip(names.categories, contrast_colors))
    #
    #     # For Sankey Diagram plot, no need to worry about the color, each bar will have the same two colors
    #     # default color palette will be set to "hls"
    #     plot_palette_sankey = None
    #
    # else:
    #     swarm_colors = [sns.desaturate(c, swarm_desat) for c in unsat_colors]
    #     plot_palette_raw = dict(zip(names, swarm_colors))
    #
    #     bar_color = [sns.desaturate(c, bar_desat) for c in unsat_colors]
    #     plot_palette_bar = dict(zip(names, bar_color))
    #
    #     contrast_colors = [sns.desaturate(c, contrast_desat) for c in unsat_colors]
    #     plot_palette_contrast = dict(zip(names, contrast_colors))
    #
    #     plot_palette_sankey = custom_palette
    #
    # print(plot_palette_raw)
    # print(all_plot_groups)

    ## Checks
    # assert (not EffectSizeDataFrame.delta2), 'Horizontal Plot is currently unavailable for delta-delta experiments.'  ### Check if delta-delta experiment (not available)

    if EffectSizeDataFrame.proportional:   ### Check if proportional plot desired (not available yet)
        raise NotImplementedError('Horizontal Plot is currently unavailable for proportional data.')
    
    if effect_size != 'mean_diff' and mini_meta:   ### Check if mini-meta analysis is available
        raise ValueError('Mini-meta analysis is only available for Mean Diff analysis.')
    
    # if mini_meta and not is_paired:   ### Check if mini-meta and unpaired (not available)
    #     raise ValueError('Mini-meta analysis is only available for paired data.')
    
    if is_paired and mini_meta and any(len(x)>2 for x in idx):   ### Check if mini-meta and repeated measures (not available)
        raise ValueError('Mini-meta is unavailable for repeated measures.')

    ## Create Figure if no axes are specified
    if kwargs["ax"] == None:
        frac = 0.3 if is_paired or mini_meta else 0.5
        fig, ax = plt.subplots(1,1,figsize=(7,1+(frac*MM_Adj_Num_Exps)) if fig_size==None else fig_size,dpi=dpi)
        title = kwargs["title"]
        title_fontsize = kwargs["title_fontsize"]
        if title is not None:
            fig.suptitle(title,fontsize=title_fontsize)

        # all_groups_count = np.sum([len(i) for i in idx])
        # # Increase the width for delta-delta graph
        # if show_delta2 or show_mini_meta:
        #     all_groups_count += 2

    ## Inset Axes
    ax_position = ax.get_position()
    contrast_axes = ax.inset_axes([1+contrast_wspace, 0, (plot_width_ratios[1]/plot_width_ratios[0]), 1])
    table_axes = ax.inset_axes([1+contrast_wspace+(plot_width_ratios[1]/plot_width_ratios[0]), 0, (plot_width_ratios[2]/plot_width_ratios[0]), 1])

    ax.set_position([ax_position.x0,ax_position.y0,(ax_position.x1 - ax_position.x0) * (plot_width_ratios[0] / sum(plot_width_ratios)),(ax_position.y1 - ax_position.y0)])
    rawdata_axes = ax
    rawdata_axes.contrast_axes = contrast_axes
    rawdata_axes.table_axes = table_axes
    fig = rawdata_axes.get_figure()
    fig.patch.set_facecolor(face_color)

    swarm_paired_line_alpha = swarm_kwargs['paired_line_alpha']
    swarm_ylabel_show_samplesize = swarm_kwargs['ylabel_show_samplesize']
    swarm_paired_means_offset = swarm_kwargs['paired_means_offset']
    swarm_ylabel_fontsize = swarm_kwargs['ylabel_fontsize']
    dot_alpha = swarm_kwargs['dot_alpha']
    swarm_xlim = swarm_kwargs['xlim']
    swarm_xlabel_fontsize = swarm_kwargs['xlabel_fontsize']
    print(swarm_colors)

    ### Unpaired
    if not is_paired:
        if mini_meta or delta2:
            ordered_labels = unpacked_idx
            df_list = []
            for i, ypos in zip(ordered_labels, np.arange(2, MM_Adj_Num_Exps + 1, 1)[::-1]):
                _df = plot_data[plot_data[xvar] == i].copy()
                _df['ypos'] = ypos
                df_list.append(_df)
            ordered_df = pd.concat(df_list)
            sns.swarmplot(ax=rawdata_axes,data=ordered_df, x=yvar,y='ypos', orient="h",palette=swarm_colors[::-1] if color_col == None else swarm_colors,
                        alpha=dot_alpha,size=raw_marker_size-2,hue=color_col,order = np.arange(MM_Adj_Num_Exps+1))
            rawdata_axes.set_ylabel('')

            # plot_data.sort_values(by=[id_col], inplace=True)
            # #### Deal with color_col
            # if color_col != None:
            #     color_col_ind = plot_data[color_col].unique()
            #     color_col_dict = {}
            #     for n, c in zip(color_col_ind, swarm_colors):
            #         color_col_dict.update({n: c})
            #
            # #### Create the data tuples & Mean + SD tuples
            # output_x, output_y = [], []
            # color_col_names = []
            #
            # startpos = MM_Adj_Num_Exps
            # for n in np.arange(0, number_of_groups, 1):
            #     samplesize = len(plot_data[plot_data[xvar].str.contains(idx[n][0])])
            #     y_list, x_list = [], []
            #     for num, i in enumerate(idx[n]):
            #         y_list.append(samplesize * [startpos - num])
            #         x_list.append(plot_data[plot_data[xvar].str.contains(i)][yvar])
            #
            #     startpos = startpos - Num_of_Exps_in_each_group[n]
            #
            #     output_y.append(np.array(y_list))
            #     output_x.append(np.array(x_list))
            #     if color_col != None:
            #         color_col_names.append(np.array(plot_data[plot_data[xvar].str.contains(idx[n][0])][color_col]))


        else:
            ordered_labels = unpacked_idx
            df_list = []
            for i,ypos in zip(ordered_labels,np.arange(1,MM_Adj_Num_Exps+1,1)[::-1]):
                _df = plot_data[plot_data[xvar]==i].copy()
                _df['ypos'] = ypos
                df_list.append(_df)
            ordered_df = pd.concat(df_list)
            sns.swarmplot(ax=rawdata_axes,data=ordered_df, x=yvar,y='ypos', orient="h",palette=swarm_colors[::-1] if color_col == None else swarm_colors,
                        alpha=dot_alpha,size=raw_marker_size,hue=color_col,order = np.arange(MM_Adj_Num_Exps+1))
            rawdata_axes.set_ylabel('')

    ### Paired
    else:
        plot_data.sort_values(by=[id_col], inplace=True)
        #### Deal with color_col
        if color_col != None:
            color_col_ind = plot_data[color_col].unique()
            color_col_dict = {}
            for n,c in zip(color_col_ind,swarm_colors):
                color_col_dict.update({n: c})

        #### Create the data tuples & Mean + SD tuples
        output_x, output_y=[],[]
        means,sd, color_col_names=[],[],[]

        startpos = MM_Adj_Num_Exps
        for n in np.arange(0,number_of_groups,1):
            samplesize = len(plot_data[plot_data[xvar].str.contains(idx[n][0])])
            y_list,x_list=[],[]
            mean_list,sd_list=[],[]
            for num, i in enumerate(idx[n]):
                y_list.append(samplesize*[startpos - num])
                x_list.append(plot_data[plot_data[xvar].str.contains(i)][yvar])
                mean_list.append(plot_data[plot_data[xvar].str.contains(i)][yvar].mean())
                sd_list.append(plot_data[plot_data[xvar].str.contains(i)][yvar].std())
            startpos = startpos - Num_of_Exps_in_each_group[n]

            output_y.append(np.array(y_list))
            output_x.append(np.array(x_list))
            means.append(np.array(mean_list))
            sd.append(np.array(sd_list))
            
            if color_col != None:
                color_col_names.append(np.array(plot_data[plot_data[xvar].str.contains(idx[n][0])][color_col]))

        #### Plot the pairs of data
        if color_col != None:
            for x, y, cs in zip(output_x,output_y,color_col_names):  
                color_cols = [color_col_dict[i] for i in cs]
                for n,c in zip(range(0,len(x[0])),color_cols):
                    for length in range(0,(len(x)-1)):
                        rawdata_axes.plot([x[length][n],x[length+1][n]],[y[length][n],y[length+1][n]],color=c, alpha=swarm_paired_line_alpha)
        else:
            for x, y, c in zip(output_x,output_y,swarm_colors):  
                rawdata_axes.plot(x, y,color=c, alpha=swarm_paired_line_alpha)

        #### Plot Mean & SD tuples
        if not multi_paired_control:
            import matplotlib.lines as mlines
            ax_ylims = rawdata_axes.get_ylim()
            ax_yspan = np.abs(ax_ylims[1] - ax_ylims[0])
            gap_width = ax_yspan * mean_gap_width_percent/100

            mean_colors = swarm_colors if color_col == None else number_of_groups*['black']
            for m,n,c in zip(np.arange(0,number_of_groups,1),np.arange(0,MM_Adj_Num_Exps,2)[::-1],mean_colors):
                for a,b,d in zip([0,1],[2,1],[swarm_paired_means_offset,-swarm_paired_means_offset]):
                    b=b-1 if mini_meta or delta2 else b
                    mean_to_high = mlines.Line2D([means[m][a]+gap_width, means[m][a]+sd[m][a]],[n+b+d, n+b+d],color=c)
                    rawdata_axes.add_line(mean_to_high) 

                    low_to_mean = mlines.Line2D([means[m][a]-sd[m][a], means[m][a]-gap_width],[n+b+d, n+b+d],color=c)
                    rawdata_axes.add_line(low_to_mean)

    ### Select the bootstraps to plot
    bootstraps = [EffectSizeDataFrame.results.bootstraps[n] for n in range(Num_of_contrasts)]
    mean_diff  = [EffectSizeDataFrame.results.difference[n] for n in range(Num_of_contrasts)]
    bca_low    = [EffectSizeDataFrame.results.bca_low[n] for n in range(Num_of_contrasts)]
    bca_high   = [EffectSizeDataFrame.results.bca_high[n] for n in range(Num_of_contrasts)]

    if is_paired:
        if multi_paired_control:
            ypos,contrast_locs=[],[]
            start=1
            for n in Num_of_Exps_in_each_group:
                for i in range(n-1):
                    end = start + i
                    ypos.append(np.arange(1,MM_Adj_Num_Exps+1,1)[::-1][end])
                    contrast_locs.append(end)
                start =+ end + 2
        else:
            if mini_meta or delta2:
                ypos = np.insert(np.arange(2,MM_Adj_Num_Exps,2,dtype=float),0,0.25)[::-1]
            else:   
                ypos = np.arange(1,MM_Adj_Num_Exps+1,2)[::-1]
    else:
        if multi_unpaired_control:
            ypos,contrast_locs=[],[]
            start=1
            start_color = 0
            for n in Num_of_Exps_in_each_group:
                for i in range(n-1):
                    end = start + i
                    end_color = start_color + i
                    ypos.append(np.arange(1,MM_Adj_Num_Exps+1,1)[::-1][end])
                    contrast_locs.append(end_color)
                start =+ end + 2
                start_color =+ end_color + 2
        else:
            ypos = np.arange(1, MM_Adj_Num_of_contrasts + 1, 1)[::-1]

        if mini_meta or delta2:
            ypos = np.insert(np.arange(2, MM_Adj_Num_Exps, 2, dtype=float), 0, 0.25)[::-1]


    if mini_meta:
        bootstraps.append(EffectSizeDataFrame.mini_meta_delta.bootstraps_weighted_delta)
        mean_diff.append(EffectSizeDataFrame.mini_meta_delta.difference)
        bca_low.append(EffectSizeDataFrame.mini_meta_delta.bca_low)
        bca_high.append(EffectSizeDataFrame.mini_meta_delta.bca_high)

    if delta2:
        bootstraps.append(EffectSizeDataFrame.delta_delta.bootstraps_delta_delta)
        mean_diff.append(EffectSizeDataFrame.delta_delta.difference)
        bca_low.append(EffectSizeDataFrame.delta_delta.bca_low)
        bca_high.append(EffectSizeDataFrame.delta_delta.bca_high)

    default_violinplot_kwargs = {'widths': 1 if multi_paired_control or not is_paired else 2, 'vert':False,'showextrema':False, 'showmedians':False, 'positions': ypos}
    v = rawdata_axes.contrast_axes.violinplot(bootstraps, **default_violinplot_kwargs,)
    halfviolin(v,  half='top', alpha = halfviolin_alpha)

    ### Plot mean diff and bca_low and bca_high
    rawdata_axes.contrast_axes.plot(mean_diff,ypos, 'k.', markersize = es_marker_size)
    rawdata_axes.contrast_axes.plot([bca_low, bca_high], [ypos, ypos],'k', linewidth = 2.5)

    ### Add Grey bar
    if contrast_bars:
        for n,y in zip(np.arange(0,MM_Adj_Num_of_contrasts,1),ypos):
            rawdata_axes.contrast_axes.add_patch(mpatches.Rectangle((0,y), mean_diff[n], 0.5 if multi_paired_control or not is_paired else 1, **barplot_kwargs))

    print(contrast_colors)
    ### Violin colors
    if color_col == None:
        if multi_unpaired_control:
            for n,loc in zip(np.arange(0,MM_Adj_Num_of_contrasts,1), contrast_locs):
                rawdata_axes.contrast_axes.collections[n].set_fc(contrast_colors[loc])
        elif multi_paired_control:
            start = 0
            for n,c in zip(np.arange(0,number_of_groups,1),contrast_colors):
                for num, i in enumerate(idx[n][:-1]):
                    rawdata_axes.contrast_axes.collections[start+num].set_fc(c)
                start += (len(idx[n])-1)
        else:
            for n,c in zip(np.arange(0,MM_Adj_Num_of_contrasts,1),contrast_colors if is_paired else contrast_colors):
                rawdata_axes.contrast_axes.collections[n].set_fc(c)
    else:
        for n in np.arange(0,MM_Adj_Num_of_contrasts,1):
            rawdata_axes.contrast_axes.collections[n].set_fc('grey')

    # ### Delta dots?
    # if contrast_dots:
    #     if not is_paired or multi_paired_control:
    #         warnings.warn('Contrast dots are not supported for unpaired data or paired repeated measures. Plotting without...', UserWarning)
    #     else:
    #         df_list = []
    #         for n,ypos_dots in zip(range(len(idx)), ypos):
    #             _df = plot_data[plot_data[xvar]==idx[n][0]].copy()
    #             _df['ypos_dots'] = ypos_dots-0.5
    #             _df['value_exp'] = plot_data[plot_data[xvar]==idx[n][1]][yvar].values
    #             _df['Diff'] = _df['value_exp'] - _df[yvar]
    #             df_list.append(_df)
    #         delta_dot_df = pd.concat(df_list)
    #
    #         if single_color_contrast_dots == None:
    #             sns.stripplot(ax=rawdata_axes.contrast_axes,data=delta_dot_df, x='Diff',y='ypos_dots',native_scale=True, orient="h",
    #                         palette=halfviolin_colors[::-1] if color_col == None else halfviolin_colors,hue=color_col,legend=False,**contrast_dots_kwargs)
    #         else:
    #             sns.stripplot(ax=rawdata_axes.contrast_axes,data=delta_dot_df, x='Diff',y='ypos_dots',native_scale=True, orient="h",color=single_color_contrast_dots,
    #                           legend=False,**contrast_dots_kwargs)
    #         rawdata_axes.contrast_axes.set_ylabel('')

    ## Table axes
    ### Kwargs
    default_table_kwargs = {'color' : 'yellow','alpha' :0.2,'fontsize' : 12,'text_color' : 'black', 'text_units' : None,'paired_gap_dashes' : False}
    if kwargs["horizontal_table_kwargs"] is None:
        table_kwargs = default_table_kwargs
    else:
        table_kwargs = merge_two_dicts(default_table_kwargs, kwargs["horizontal_table_kwargs"])
    table_color = table_kwargs['color']
    table_alpha = table_kwargs['alpha']
    table_font_size = table_kwargs['fontsize'] if table_kwargs['text_units'] == None else table_kwargs['fontsize']-2
    table_text_color = table_kwargs['text_color']
    text_units = '' if table_kwargs['text_units'] == None else table_kwargs['text_units']
    table_gap_dashes = table_kwargs['paired_gap_dashes']
    
    ### Create a table of deltas
    cols=['Δ','N']
    lst = []
    for n in np.arange(0,Num_of_contrasts,1):
        lst.append([EffectSizeDataFrame.results.difference[n],0])
    if mini_meta:
        lst.append([EffectSizeDataFrame.mini_meta_delta.difference,0])
    elif delta2:
        lst.append([EffectSizeDataFrame.delta_delta.difference,0])
    tab = pd.DataFrame(lst, columns=cols)

    ### Plot the text
    for i,loc in zip(tab.index,ypos):
        if mini_meta or delta2:
            loc_new = loc if loc != 0.25 else loc+0.25
            rawdata_axes.table_axes.text(0.5, loc_new, "{:+.2f}".format(tab.iloc[i,0])+text_units,ha="center", va="center", color=table_text_color,size=table_font_size)
        else:
            rawdata_axes.table_axes.text(0.5, loc, "{:+.2f}".format(tab.iloc[i,0])+text_units,ha="center", va="center", color=table_text_color,size=table_font_size)

    ### Plot the dashes
    if mini_meta or delta2:
        no_contrast_positions = list(set([int(x-0.5) for x in ypos[:-1]]) ^ set(np.arange(2,Num_Exps+2,1)))
    else:
        no_contrast_positions = list(set([int(x-0.5) for x in ypos]) ^ set(np.arange(0,Num_Exps,1)))

    if table_gap_dashes or not is_paired or multi_paired_control:
        if not (mini_meta or delta2):
            for i in no_contrast_positions:
                rawdata_axes.table_axes.text(0.5, i+1, "—",ha="center", va="center", color=table_text_color,size=table_font_size)

    ### Parameters for table
    rawdata_axes.table_axes.axvspan(0, 1, facecolor=table_color, alpha=table_alpha)  #### Plot the background color
    rawdata_axes.table_axes.set_xticks([0.5])
    rawdata_axes.table_axes.set_xticklabels([])
    
    ## Final parameters
    ### Tick-params
    rawdata_axes.tick_params(left=True)
    rawdata_axes.contrast_axes.tick_params(left=False)
    rawdata_axes.table_axes.tick_params(left=False, bottom=False)

    ### X-label
    rawdata_axes.set_xlabel('Metric' if swarm_label == None else swarm_label,fontsize=swarm_xlabel_fontsize)
    rawdata_axes.contrast_axes.set_xlabel(contrast_label if contrast_label != None else 'Mean difference' if effect_size == 'mean_diff' else effect_size,fontsize=contrast_xlabel_fontsize)
    rawdata_axes.table_axes.set_xlabel('Δ')

    ### X-lim
    if type(swarm_xlim)==tuple or type(swarm_xlim)==list and len(swarm_xlim)==2:
        rawdata_axes.set_xlim(swarm_xlim[0],swarm_xlim[1])  
    elif swarm_xlim != None:
        raise ValueError('swarm_xlim should be a tuple or list of length 2. Defaulting to automatic scaling.')  

    if type(contrast_xlim)==tuple or type(contrast_xlim)==list and len(contrast_xlim)==2:
        rawdata_axes.contrast_axes.set_xlim(contrast_xlim[0],contrast_xlim[1])  
    elif contrast_xlim != None:
        raise ValueError('contrast_xlim should be a tuple or list and of length 2. Defaulting to automatic scaling.')  
    
    ### Y-ticks 
    yticklabels=[]
    if is_paired:
        rawdata_axes.set_yticks(np.arange(1,MM_Adj_Num_Exps+1,1))
        for n in np.arange(0,number_of_groups,1):
            for num, i in enumerate(idx[n]):
                if swarm_ylabel_show_samplesize:
                    ss = len(plot_data[plot_data[xvar].str.contains(i)][yvar])
                    yticklabels.append(i + ' (n='+str(ss)+')')
                else:
                    yticklabels.append(i)
        if mini_meta:
            rawdata_axes.set_yticks(np.insert(np.arange(2,MM_Adj_Num_Exps+1,1,dtype=float),0,0.5))
            yticklabels.append('Weighted Mean')
        elif delta2:
            rawdata_axes.set_yticks(np.insert(np.arange(2, MM_Adj_Num_Exps + 1, 1, dtype=float), 0, 0.5))
            yticklabels.append('delta-delta')
    else:
        if mini_meta:
            rawdata_axes.set_yticks(np.arange(1, MM_Adj_Num_Exps + 1, 1))
            for n in np.arange(0, number_of_groups, 1):
                for num, i in enumerate(idx[n]):
                    if swarm_ylabel_show_samplesize:
                        ss = len(plot_data[plot_data[xvar].str.contains(i)][yvar])
                        yticklabels.append(i + ' (n=' + str(ss) + ')')
                    else:
                        yticklabels.append(i)
            rawdata_axes.set_yticks(np.insert(np.arange(2, MM_Adj_Num_Exps + 1, 1, dtype=float), 0, 0.5))
            yticklabels.append('Weighted Mean')
        elif delta2:
            rawdata_axes.set_yticks(np.arange(1, MM_Adj_Num_Exps + 1, 1))
            for n in np.arange(0, number_of_groups, 1):
                for num, i in enumerate(idx[n]):
                    if swarm_ylabel_show_samplesize:
                        ss = len(plot_data[plot_data[xvar].str.contains(i)][yvar])
                        yticklabels.append(i + ' (n=' + str(ss) + ')')
                    else:
                        yticklabels.append(i)
            rawdata_axes.set_yticks(np.insert(np.arange(2, MM_Adj_Num_Exps + 1, 1, dtype=float), 0, 0.5))
            yticklabels.append('delta-delta')
        else:
            rawdata_axes.set_yticks(np.arange(1,MM_Adj_Num_Exps+1,1))
            for n in np.arange(0,MM_Adj_Num_Exps,1):
                if swarm_ylabel_show_samplesize:
                    ss = len(plot_data[plot_data[xvar].str.contains(unpacked_idx[n])][yvar])
                    yticklabels.append(unpacked_idx[n] + '\n' + ' (n='+str(ss)+')')
                else:
                    yticklabels.append(unpacked_idx[n])

    rawdata_axes.set_yticklabels(yticklabels[::-1],ma='center',fontsize = swarm_ylabel_fontsize)
    rawdata_axes.contrast_axes.set_yticks([])
    rawdata_axes.table_axes.set_yticks([])

    ### Y-lim
    if is_paired:
        for ax in [rawdata_axes,rawdata_axes.contrast_axes,rawdata_axes.table_axes]:
            if mini_meta or delta2:
                ax.set_ylim(-0.25,Num_Exps+2)
            else:
                ax.set_ylim(0 if contrast_dots else 0.5,Num_Exps+0.5)
    else:
        for ax in [rawdata_axes,rawdata_axes.contrast_axes,rawdata_axes.table_axes]:
            if mini_meta or delta2:
                ax.set_ylim(-0.25,Num_Exps+2)
            else:
                ax.set_ylim(0.5,Num_Exps+0.5)
    rawdata_axes.contrast_axes.plot([0, 0], rawdata_axes.contrast_axes.get_ylim(), 'k', linewidth = 1)

    ### y-spines
    sns.despine(ax=rawdata_axes)
    sns.despine(ax=rawdata_axes.contrast_axes, left=True)
    sns.despine(ax=rawdata_axes.table_axes, left=True, bottom=True)
    spine_xpos = rawdata_axes.get_xlim()[0]
    rawdata_axes.set_xlim(spine_xpos,rawdata_axes.get_xlim()[1])
    rawdata_axes.spines[['left']].set_visible(False) 

    if is_paired:
        line_start=2 if mini_meta or delta2 else 1
        end=0
        for n in Num_of_Exps_in_each_group[::-1]:
            line_end=line_start+(n-1)
            rawdata_axes.vlines(spine_xpos, line_start, line_end, colors='black', linestyles='solid', label='',linewidths=1)
            line_start=line_end+1
    else:
        if multi_unpaired_control:
            # line_start=1
            # end=0
            # for n in Num_of_Exps_in_each_group[::-1]:
            #     line_end=line_start+(n-1)
            #     rawdata_axes.vlines(spine_xpos, line_start, line_end, colors='black', linestyles='solid', label='',linewidths=1)
            #     line_start=line_end+1

            line_start = 2 if mini_meta or delta2 else 1
            for n in Num_of_Exps_in_each_group[::-1]:
                line_end = line_start + (n - 1)
                rawdata_axes.vlines(spine_xpos, line_start, line_end, colors='black', linestyles='solid', label='',
                                    linewidths=1)
                line_start = line_end + 1

        else:
            rawdata_axes.vlines(spine_xpos, 1, MM_Adj_Num_Exps, colors='black', linestyles='solid', label='',linewidths=1)
            
    ### Legend
    if color_col != None:
        if not is_paired:
            h1, l1 = rawdata_axes.get_legend_handles_labels()
            rawdata_axes.table_axes.legend(h1, l1,bbox_to_anchor=(0.8, 1.0), title=color_col,**legend_kwargs)
            rawdata_axes.legend().remove()
        else:
            color_col_ind = plot_data[color_col].unique()
            from matplotlib.lines import Line2D
            handles=[]
            for n,c in zip(color_col_ind,swarm_colors):
                handles.append(Line2D([0], [0], label=n, color=c))
            rawdata_axes.table_axes.legend(handles=handles,bbox_to_anchor=(0.85, 1.0), handlelength=1, title=color_col,**legend_kwargs)
    
    ## Return fig
    return fig
