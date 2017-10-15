"""Provide data exploration functions."""
# pylama: ignore=D103

import os

import matplotlib.pyplot as plt
import seaborn as sns

import constants


def time_series_violinplot_of_dataframe(dataframe,
                                        grouper,
                                        output_path,
                                        **violionplot_kwargs):
    for variable in dataframe.columns:
        fig, ax = plt.subplots()
        sns.violinplot(data=dataframe,
                       x=grouper,
                       y=variable,
                       ax=ax,
                       **violionplot_kwargs)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_path, variable + '.png'),
            dpi=500)


def main(dataframe):
    # Plot a histogram of every series in this dataframe.
    dataframe_w_days = dataframe.copy()
    dataframe_w_days.date.fillna(-1, inplace=True)
    time_series_violinplot_of_dataframe(dataframe_w_days,
                                        'date',
                                        constants.DE_VIOLIN,
                                        hue='occupancy',
                                        scale='width',
                                        scale_hue=True,
                                        split=True,
                                        inner=None)
    gb = dataframe_w_days.groupby(['date', 'occupancy'])  # noqa
    # print(gb.count())
    # print(dataframe_w_days.days.value_counts().sort_index())
    # import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
