'''
calculate and generate images of different stats from calendar_/all_events.csv
'''


import gen_image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorutils import hex_to_rgb, rgb_to_hex
from pandas import DataFrame
from sklearn.cluster import DBSCAN


def flatten(lst: list) -> list:
    '''
    flattens list (as np.flatten())
    '''
    return [item for sublist in lst for item in sublist]


def get_tracked_time_info(df: DataFrame, timerange: str,
                          postfix: str = '') -> pd.Timedelta:
    '''
    calculate total tracked time and save image
    '''
    df = df.copy()
    df['overlap'] = df.apply(
        lambda row: tuple(df[(df['start'] < row['end']) &
                             (df['end'] > row['start'])]
                          .index.tolist()), axis=1)
    noverlap_tracked = (df[df['overlap']
                           .apply(lambda x: len(x) == 1)].duration.sum())

    ov = df[df['overlap'].apply(lambda x: len(x) > 1)]
    overlap_tracked = (ov.groupby('overlap')
                       .apply(lambda x: x.end.max() - x.start.min())
                       .sum())
    total_tracked = noverlap_tracked + overlap_tracked
    total_tracked_m = f'{round(total_tracked / pd.Timedelta(minutes=1)):,}'
    minute_s = 'minute' if str(total_tracked_m).endswith('1') else 'minutes'
    h1 = f'You tracked {total_tracked_m}\n'\
        f'{minute_s} this {timerange}'
    h2 = f'It is {total_tracked.components[0]} days, '\
        f'{total_tracked.components[1]} hours, and '\
        f'{total_tracked.components[2]} minutes'

    gen_image.generate_spotify_style_image(
        h1, h2, f'pictures/00_tracked_time_{timerange}{postfix}.png')

    return total_tracked


def get_not_tracked_time_info(df: DataFrame,
                              total_tracked: pd.Timedelta,
                              timerange: str,
                              postfix: str = '') -> None:
    '''
    calculate total not tracked time and save image
    '''
    all_time = df.end.max() - df.start.min()
    not_tracked = all_time - total_tracked
    perc = round(not_tracked / all_time * 100, 2)
    not_tracked_m = f'{round(not_tracked / pd.Timedelta(minutes=1)):,}'
    minute_s = 'minute' if str(not_tracked_m).endswith('1') else 'minutes'
    missed_time_h1 = f'You missed only\n{not_tracked_m} '\
        f'{minute_s} this {timerange}'
    missed_time_h2 = f'which is {perc}% of all tracked time'

    gen_image.generate_spotify_style_image(
        missed_time_h1, missed_time_h2,
        f'pictures/01_missed_time_{timerange}{postfix}.png')


def get_multitasking_time_info(df: DataFrame, timerange: str,
                               postfix: str = '') -> None:
    '''
    calculate total multitasking time, top mt categories and save image
    create lollipop chart of multitasking events
    '''
    df = df.copy()
    df['overlap'] = df.apply(lambda row: df[(df['start'] < row['end']) &
                             (df['end'] > row['start'])].index.tolist(), axis=1)
    df_mt = df.explode('overlap')
    df_mt = df_mt[df_mt.index != df_mt['overlap']]
    df_mt.drop(columns=['start_local', 'description',
                        'end_local', 'duration'], inplace=True)
    df_mt = df_mt.loc[df_mt.overlap > df_mt.index]
    df_mt[['event_name_2', 'category_2', 'start_2', 'end_2']] = (
        df_mt.apply(
            lambda row: df.loc[
                row['overlap'], ['event_name', 'category', 'start', 'end']]
            .values, axis=1, result_type='expand'))
    df_mt = df_mt.loc[df_mt.event_name != df_mt.event_name_2]
    df_mt['intersecting_time'] = df_mt.apply(
        lambda row: min(row['end'], row['end_2'])
        - max(row['start'], row['start_2']), axis=1)

    pair_dfs = {}

    for key in ('category', 'event_name'):
        pair_dfs[key] = (
            df_mt
            .groupby([key, f'{key}_2'], as_index=False)
            .agg({'intersecting_time': 'sum'})
            .sort_values('intersecting_time', ascending=False))
        pair_dfs[key] = (
            pair_dfs[key]
            .groupby(pair_dfs[key][[key, f'{key}_2']]
                     .apply(sorted, axis=1)
                     .agg(tuple))
            .agg({'intersecting_time': 'sum'})
            .reset_index().sort_values('intersecting_time', ascending=False))

    fav_mt_categories = pair_dfs['category'].iloc[0, 0]
    fav_mt_time = pair_dfs['category'].iloc[0, 1] / pd.Timedelta(minutes=1)
    fav_mt_time = f'{round(fav_mt_time):,}'
    total_mt_time = df_mt['intersecting_time'].sum() / pd.Timedelta(minutes=1)
    total_mt_time = f'{round(total_mt_time):,}'
    h1 = f"You were multitasking\n{total_mt_time} minutes this {timerange}"
    h2 = 'Your favorite categories to mix were\n'\
        f'{fav_mt_categories[0]} and {fav_mt_categories[1]}\n'\
        f'You spent {fav_mt_time} minutes doing it\nat the same time'

    fav_mt_categories = flatten([i.split() for i in fav_mt_categories])

    gen_image.generate_spotify_style_image(
        h1, h2, f'pictures/05_mt_time_{timerange}{postfix}.png',
        fav_mt_categories)

    events = [' +\n'.join(i) for i in
              pair_dfs['event_name'].head()['index'].to_list()]
    minutes = (pair_dfs['event_name'].head().intersecting_time
               / pd.Timedelta(minutes=1)
               ).apply(round).to_list()
    colors = [[df.loc[df.event_name == i].iloc[0, 1] for i in tup]
              for tup in pair_dfs['event_name'].head()['index'].to_list()]
    merged_colors = [
        rgb_to_hex(
            np.array([hex_to_rgb(i) for i in tup]).mean(axis=0))
        for tup in colors]

    gen_image.groovy_lollipop_plot(
        events, minutes, merged_colors,
        'Your favorite activities\nto combine\n'
        f'this {timerange} were',
        f'pictures/06_mt_lollipop_{timerange}{postfix}.png')


def get_top_categories_events_info(df: DataFrame, timerange: str,
                                   postfix: str = '') -> None:
    '''
    calculate top categories, their time spent and save images
    '''
    df = df.loc[df.event_name != 'sleep'].copy()

    top_categories = (
        df
        .groupby(['category', 'category_color'], as_index=False)
        .agg({'duration': 'sum'})
        .sort_values('duration', ascending=False)
        .head(5)
    )

    top_categories.duration = round(top_categories.duration
                                    / pd.Timedelta(minutes=1))

    gen_image.groovy_barplot(
        top_categories.category,
        top_categories.duration,
        top_categories.category_color,
        f'Your top categories\nthis {timerange} were',
        f'pictures/03_top_categories_{timerange}{postfix}.png')

    top_events = (
        df
        .groupby(['event_name', 'category_color'], as_index=False)
        .agg({'duration': 'sum'})
        .sort_values('duration', ascending=False)
        .head(5)
    )

    top_events.duration = round(top_events.duration
                                / pd.Timedelta(minutes=1))

    gen_image.groovy_barplot(
        top_events.event_name.to_list(),
        top_events.duration.to_list(),
        top_events.category_color.to_list(),
        f'Your top activities\nthis {timerange} were',
        f'pictures/04_top_events_{timerange}{postfix}.png',
        mirrored=True)


def text_cloud_of_all_events(df: DataFrame, timerange: str,
                             postfix: str = '') -> None:
    '''
    calculate durations of all events and save wordcloud of them
    '''
    event_durations = (df
                       .groupby('event_name')
                       .agg({'duration': 'sum'})
                       / pd.Timedelta(minutes=1)
                       ).to_dict()['duration']
    event_colors = (
        df
        .groupby('event_name').category_color
        .agg(lambda x: x.value_counts().idxmax())
    ).to_dict()

    # ((df
    #  .groupby('event_name')
    #  .agg({'duration': 'sum'})
    #  / pd.Timedelta(minutes=1)
    #  ).sort_values('duration', ascending=False)
    #  .reset_index()
    #  .to_csv(f'to_wc_{timerange}{postfix}.csv', index=False))

    gen_image.groovy_textcloud(
        event_durations,
        event_colors,
        f'You had a\nwonderful {timerange}',
        f'pictures/07_events_cloud_{timerange}{postfix}.png')


def get_longest_event_per_day(df: DataFrame, timerange: str,
                              postfix: str = '') -> None:
    '''
    calculate longest event per day and save image
    with event, date, and duration
    '''
    df = df.loc[df.event_name != 'sleep'].copy()
    df['date'] = df.start_local.dt.date
    lepd = (  # longest event per day
        df.groupby(['date', 'event_name'], as_index=False).duration
        .sum().sort_values('duration', ascending=False)
    ).head(1).copy()
    duration = (
        f'{round(lepd.duration.iloc[0] / pd.Timedelta(minutes=1)):,}')
    minute_s = 'minute' if duration.endswith('1') else 'minutes'
    date = lepd.date.iloc[0].strftime('%B %-d')
    gen_image.generate_spotify_style_image(
        f'{date}\nwas special',
        f'You spent {duration} {minute_s}\n'
        f'{lepd.event_name.iloc[0]} this day\n'
        f'It was your longest activity\nin a single day this {timerange}',
        f'pictures/02_longest_event_per_day_{timerange}{postfix}.png',
        lepd.event_name.iloc[0].split()
    )


def get_average_categories_per_day(df: DataFrame, timerange: str,
                                   postfix: str = '') -> None:
    '''
    calculate average summed duration of events per category
    and save donutplot
    '''
    df = df.copy()
    df.loc[df.event_name == 'sleep', 'category'] = 'sleep'
    df['date'] = df.start_local.dt.date

    df_grouped = (
        df
        .groupby(['category', 'date'], as_index=False)
        .agg({'duration': 'sum'}))

    all_categories = df['category'].unique()
    all_dates = df['date'].unique()

    # Create a DataFrame with all combinations of date and category
    all_combinations = pd.MultiIndex.from_product(
        [all_dates, all_categories], names=['date', 'category'])
    all_df = pd.DataFrame(index=all_combinations).reset_index()

    # Merge the original df with the new one to fill in missing values
    result_df = (pd.merge(
        all_df, df_grouped,
        how='left', on=['date', 'category'])
        .fillna({'duration': pd.Timedelta(minutes=0)}))

    mean_res = (
        result_df
        .groupby('category')
        .duration
        .mean()
        .reset_index()
    )

    labels = mean_res.category.values
    durations = ((mean_res.duration / pd.Timedelta(minutes=1))
                 .apply(round).values)
    colors = np.array([df.loc[df.category == i].category_color.iloc[0]
                       for i in labels])

    sleep_idx = np.where(labels == 'sleep')[0]
    # darken category color for sleep
    colors[sleep_idx] = rgb_to_hex(
        tuple(i - 40 for i in hex_to_rgb(colors[sleep_idx][0])))

    gen_image.groovy_donutplot(
        labels,
        durations,
        colors,
        f'Your average day\nthis {timerange}',
        f'pictures/08_average_day_donut_{timerange}{postfix}.png',
    )


def cluster_by_time_duration(
        df: DataFrame,
        time_col: str = 'start_local',
        dur_col: str = 'duration',
        plot_clusters=False) -> DataFrame:
    '''
    use DBSCAN to clusterize events by starting time and duration
    '''
    # pylint: disable=invalid-name

    df['date'] = df[time_col].dt.date
    df['start_time'] = df[time_col].dt.time
    df['start_time_seconds'] = df['start_time'].apply(
        lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    df['duration_seconds'] = df[dur_col].dt.total_seconds()

    clustered_dfs = []

    for category, category_df in df.groupby(['category']):
        X = category_df[['start_time_seconds', 'duration_seconds']].values
        average_duration = category_df['duration_seconds'].mean()
        average_local_time = category_df['start_time_seconds'].mean()

        if category[0] in ['basic needs', 'chill', 'home']:
            duration_factor = 0.05
            local_time_factor = 0.05
        elif category[0] == 'sleep':
            duration_factor = 1
            local_time_factor = 1
        else:
            duration_factor = 0.1
            local_time_factor = 0.1
        eps = np.sqrt((average_duration * duration_factor)**2
                      + (average_local_time * local_time_factor)**2)

        dbscan = DBSCAN(
            eps=eps,
            min_samples=round(2 / 7 * df.date.nunique()))
        category_df['cluster'] = dbscan.fit_predict(X)

        if plot_clusters:
            unique_labels = set(category_df['cluster'])
            core_samples_mask = np.zeros_like(category_df['cluster'],
                                              dtype=bool)
            core_samples_mask[dbscan.core_sample_indices_] = True

            colors = [matplotlib.colormaps['Spectral'](each)
                      for each in np.linspace(0, 1, len(unique_labels))]
            plt.figure()

            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = category_df['cluster'] == k

                xy = X[class_member_mask & core_samples_mask]
                plt.plot(
                    xy[:, 0] / 3600,
                    xy[:, 1] / 3600,
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=14,
                    label=f'Cluster {k} (Core)'
                )

                xy = X[class_member_mask & ~core_samples_mask]
                plt.plot(
                    xy[:, 0] / 3600,
                    xy[:, 1] / 3600,
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=6,
                    label=f'Cluster {k}'
                )
            plt.title(f'Clusters for Category: {category}')
            plt.xlabel('Start Time (hours)')
            plt.ylabel('Duration (hours)')
            # plt.legend()
            plt.show()

        clustered_dfs.append(category_df)

    return pd.concat(clustered_dfs, ignore_index=True)


def get_median_day(df: DataFrame, timerange: str,
                   postfix: str = '') -> None:
    '''
    clusterize and calculate median categories start times and durations
    and save image with it
    '''
    df = df.copy()
    df.loc[df.event_name == 'sleep', 'category'] = 'sleep'
    clustered_df = cluster_by_time_duration(df)
    median_day = (
        clustered_df
        .loc[clustered_df.cluster != -1]  # no noise
        .groupby(['category', 'cluster'])
        .agg({'start_time_seconds': 'median', 'duration_seconds': 'median'})
        .reset_index()
    )

    cat_to_color = (
        df[['category', 'category_color']]
        .groupby('category')['category_color']
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict())

    cat_to_color['sleep'] = rgb_to_hex(
        tuple(i - 40 for i in hex_to_rgb(cat_to_color['sleep'])))

    gen_image.groovy_day_circle(
        median_day, cat_to_color,
        f'This is how you typically\nspent your days\nthis {timerange}',
        f'pictures/09_typical_day_circle_{timerange}{postfix}.png'
    )


def get_categories_dynamics(df: DataFrame, timerange: str,
                            postfix: str = '') -> None:
    '''
    Calculate and save image of duration dynamics per date per category
    '''
    df = df.copy()
    df['date'] = df.start_local.dt.date
    ridge_df = df.groupby(['date', 'category', 'category_color'],
                          as_index=False).duration.sum()
    ridge_df.duration = ridge_df.duration / pd.Timedelta(minutes=1)
    title = f'Dynamics of time\nspent per category\nthis {timerange}'
    output_path = f'pictures/10_categories_dynamics_{timerange}{postfix}.png'
    gen_image.groovy_ridgeplot(ridge_df, title, output_path)


def main() -> None:
    '''
    calculate and generate images of different stats
    from calendar_/all_events.csv to pictures/
    '''
    df_nsfw = pd.read_csv('calendar_/all_events.csv')
    df_sfw = df_nsfw.loc[
        ~df_nsfw.event_name.isin(['sex', 'masturbation', 'pooping'])].copy()

    print('calculating stats...')
    for df, postfix in zip((df_nsfw, df_sfw), ('_nsfw', '_sfw')):
        cols = ['start', 'end', 'start_local', 'end_local']
        df[cols] = df[cols].apply(pd.to_datetime)
        df.duration = pd.to_timedelta(df.duration)

        tr_to_days = {'week': 7, 'month': 30, 'year': 365, 'half-year': 183}
        for timerange in list(tr_to_days):
            df_part = df.loc[
                df['start'] >= (
                    pd.to_datetime('today', utc=True)
                    - pd.Timedelta(days=tr_to_days[timerange]))
            ]

            total_tracked = get_tracked_time_info(df_part, timerange, postfix)
            get_not_tracked_time_info(df_part, total_tracked,
                                      timerange, postfix)
            get_multitasking_time_info(df_part, timerange, postfix)
            get_top_categories_events_info(df_part, timerange, postfix)
            text_cloud_of_all_events(df_part, timerange, postfix)
            get_longest_event_per_day(df_part, timerange, postfix)
            get_average_categories_per_day(df_part, timerange, postfix)
            get_median_day(df_part, timerange, postfix)
            get_categories_dynamics(df_part, timerange, postfix)
    # print('done.')

# %%
if __name__ == '__main__':
    main()


# %%
# import sys
# sys.path.append('/home/klim/.virtualenvs/MyLifeWrapped/lib/python3.10/site-packages/')


# %%
# timerange = 'month'
# df = pd.read_csv('calendar_/all_events.csv')
# df = df.loc[
#     ~df.event_name.isin(['sex', 'masturbation', 'pooping'])].copy()
#
# cols = ['start', 'end', 'start_local', 'end_local']
# df[cols] = df[cols].apply(pd.to_datetime)
# df.duration = pd.to_timedelta(df.duration)
#
# tr_to_days = {'week': 7, 'month': 30, 'year': 365, 'half-year': 183}
#
# df = df.loc[
#     df['start'] >= (
#         pd.to_datetime('today', utc=True)
#         - pd.Timedelta(days=tr_to_days[timerange]))
# ]

# %%
