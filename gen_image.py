'''
this library contains functions to create beautiful plots
for MyLifeWrapped app
'''

import os
import random

import joypy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects, rcParams, transforms
from matplotlib.patches import Wedge
from pandas import DataFrame
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud

os.environ["XDG_SESSION_TYPE"] = "xcb"


def textsize(text, font):
    '''
    get width and height of text
    '''
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height


def crop_img(
    img: Image,
    output_path: str,
    new_height: int = 1024,
    ratio: float = .5625  # 9/16
) -> None:
    '''
    crop image
    '''
    width, height = img.size
    new_width = ratio * new_height
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))
    img.save(output_path)


def add_bg_and_h1(
    fg_image_path,
    title: str = None,
    y_title_mod: float = .125,
    x_mod: float = 0,
    y1_mod: float = 1,
    y2_mod: float = 1,
    bg_folder: str = '/home/klim/Documents/work/scripts/'
    'MyLifeWrapped/backgrounds'
) -> Image:
    '''
    add background to image and (optionally) title
    y_title_mod modifies where title is (by default at 1/8 of height from top)
    x_mod -- fg_image.width modifier in calculating x_position
    y1_mod -- background.height modifier in calculating y_position
    y2_mod -- fg_image.height modifier in calculating y_position
    '''
    random_background = random.choice(os.listdir(bg_folder))
    background_path = os.path.join(bg_folder, random_background)
    background = Image.open(background_path)

    fg_image = Image.open(fg_image_path)

    # Calculate the position to place the fg_image image at the center
    x_position = round(
        (background.width - fg_image.width + fg_image.width * x_mod) // 2)
    y_position = round(
        (background.height * y1_mod - fg_image.height * y2_mod) // 2)

    # Paste the fg_image onto the background
    background.paste(fg_image, (x_position, y_position), fg_image)

    if title is not None:
        h1_font = ImageFont.truetype('Ubuntu-M.ttf', 40)
        width, height = background.size
        lines_h1 = title.split('\n')
        draw = ImageDraw.Draw(background)
        total_height = sum(textsize(line, h1_font)[1]
                           for line in lines_h1)

        y_group_h1 = (height - total_height) * y_title_mod

        for line in lines_h1:
            textwidth, textheight = textsize(line, h1_font)
            x = (width - textwidth) / 2
            draw.text((x, y_group_h1), line, font=h1_font, fill='#1e1e1e')
            y_group_h1 += textheight + 5

    return background


def generate_spotify_style_image(h1: str,
                                 h2: str,
                                 output_path: str,
                                 bold_words: list = None) -> None:
    '''
    generate image with random background from bg_folder
    with text -- heading and regular
    it writes text in the middle
    handles line breaks
    and makes digits bold in regular text and words in list bold_words
    '''
    if bold_words is None:
        bold_words = []

    bg_folder = '/home/klim/Documents/work/scripts/MyLifeWrapped/backgrounds'
    random_background = random.choice(os.listdir(bg_folder))
    background_path = os.path.join(bg_folder, random_background)

    img = Image.open(background_path)
    draw = ImageDraw.Draw(img)

    h1_font = ImageFont.truetype("Ubuntu-M.ttf", 40)
    h2_font = ImageFont.truetype("Ubuntu-R.ttf", 25)
    h2_bold_font = ImageFont.truetype("Ubuntu-M.ttf", 25)

    width, height = img.size
    lines_h1 = h1.split('\n')
    lines_h2 = h2.split('\n')

    all_lines = lines_h1 + lines_h2
    total_height = sum(textsize(line, h1_font
                                if line in lines_h1 else h2_font)[1]
                       for line in all_lines)

    y_group_h1 = (height - total_height) / 2

    for line in lines_h1:
        textwidth, textheight = textsize(line, h1_font)
        x = (width - textwidth) / 2
        draw.text((x, y_group_h1), line, font=h1_font, fill='#1e1e1e')
        y_group_h1 += textheight + 5

    y_group_h2 = y_group_h1 + 30

    for line in lines_h2:
        textwidth, textheight = textsize(line, h2_font)
        x = (width - textwidth) / 2

        words = line.split()

        for word in words:
            if (any(char.isdigit() or char == '%' for char in word)
                    or word in bold_words):
                word_width, _ = textsize(word, h2_bold_font)
                draw.text((x, y_group_h2), word, font=h2_bold_font,
                          fill='#1e1e1e')
                x += word_width + textsize(' ', h2_font)[0]
            else:
                word_width, _ = textsize(word, h2_font)
                draw.text((x, y_group_h2), word, font=h2_font, fill='#1e1e1e')
                x += word_width + textsize(' ', h2_font)[0]

        y_group_h2 += textheight  # Move to the next line

    crop_img(img, output_path)


def groovy_barplot(labels: list,
                   values: list,
                   colors: list,
                   title: str,
                   output_path: str,
                   mirrored: bool = False) -> None:
    '''
    create beautiful barplot with rounded edges and custom colors
    '''
    sorted_indices = np.argsort(values)
    labels = np.array(labels)[sorted_indices]
    values = np.array(values)[sorted_indices]
    colors = np.array(colors)[sorted_indices]

    # Create a horizontal bar plot
    fig, ax = plt.subplots(figsize=(576 / 100, 576 / 100), dpi=100)

    # Customize the appearance
    bars = ax.barh(labels, values, color=colors, height=0.6, edgecolor='none')

    fig.patch.set_alpha(0)
    plt.axis('off')

    # Show only label names in the middle of each bar
    for bar_, label, color in zip(bars, labels, colors):
        fs_1 = bar_.get_height() * 70 * (bar_.get_width() / values.max())
        fontsize = min(fs_1, fs_1 / len(label) * 12)

        minute_s = (' minute' if str(bar_.get_width()).endswith('1')
                    else ' minutes')
        texxt = f'{int(bar_.get_width()):,} {minute_s}'
        ax.text(
            bar_.get_width() * .7 / 2 + 70,
            bar_.get_y() + bar_.get_height() * 1.4 / 2,
            label,
            ha='center', va='center', color='#1e1e1e', fontweight='bold',
            fontsize=fontsize,
            fontfamily='Ubuntu')
        ax.text(
            bar_.get_width() * .7 + values.max() * .1,
            bar_.get_y() + bar_.get_height() * 1.4 / 2,
            texxt,
            ha='right' if mirrored else 'left', va='center', color='#1e1e1e',
            fontfamily='Ubuntu',
            fontsize=bar_.get_height() * 25)

        ax.plot(
            [bar_.get_x(), (bar_.get_x() + bar_.get_width()) * .7],
            [bar_.get_y() + bar_.get_height() * 1.4 / 2,
             bar_.get_y() + bar_.get_height() * 1.4 / 2],
            linewidth=70,
            solid_capstyle="round",
            color=color)

    for patch in reversed(ax.patches):
        patch.remove()

    ax.margins(x=0)
    ax.set_xlim(0)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplots_adjust(left=0, right=.9)

    if mirrored:
        ax.invert_xaxis()

    # ax.set_title(title,
    #              fontdict={'fontsize': 30, 'fontweight': 'medium',
    #                        'fontname': "Ubuntu"})
    plt.tight_layout(pad=0)

    plt.savefig('transparent_chart.png',
                transparent=True)
    plt.close()

    img = add_bg_and_h1('transparent_chart.png', title, y1_mod=1.2)
    crop_img(img, output_path)


def groovy_lollipop_plot(labels: list,
                         values: list,
                         colors: list,
                         title: str,
                         output_path: str) -> None:
    '''
    create beautiful lollipop plot going in half new_circle
    '''
    # Sort that data, my dude
    sorted_indices = np.argsort(values)[::-1]
    labels = np.array(labels)[sorted_indices]
    values = np.array(values)[sorted_indices]
    colors = np.array(colors)[sorted_indices]

    fig, ax = plt.subplots(figsize=(576 * 1.4 / 100, 576 * 1.4 / 100), dpi=100)
    fig.patch.set_alpha(0)  # Transparent background, y'know

    plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    big_dark_circle = Wedge(center=(0.5, 0.5), r=0.5, theta1=90,
                            theta2=270, facecolor='#1E1E1E')
    ax.add_patch(big_dark_circle)

    gap = 0.088
    for i, color in enumerate(colors):
        radius = 0.5 - (i + 1) * gap
        light_circle = Wedge(center=(0.5, 0.5), r=radius - 0.0035, theta1=90,
                             theta2=270,
                             # color=color[1],
                             color=color,
                             width=.002, alpha=.5)
        ax.add_patch(light_circle)

        new_circle = Wedge(
            center=(0.5, 0.5), r=radius,
            # theta1=120 + (1 - values[i] / values.max()) * 150,
            theta1=90,
            # theta2=270,
            theta2=240 - (1 - values[i] / values.max()) * 150,
            # color=color[0],
            color=color,
            width=.005, alpha=1)
        ax.add_patch(new_circle)

        marker_radius = radius - 0.0035
        # marker_theta = np.radians(120 + (1 - values[i] / values.max()) * 150)
        marker_theta = np.radians(240 - (1 - values[i] / values.max()) * 150)
        marker_x = 0.5 + marker_radius * np.cos(marker_theta)
        marker_y = 0.5 + marker_radius * np.sin(marker_theta)
        ax.scatter(marker_x,
                   marker_y,
                   # color=color[0],
                   color=color,
                   s=50,
                   linewidths=1)

        text_theta1 = np.radians(270)
        text_x1 = 0.5 + marker_radius * np.cos(text_theta1) + .03
        text_y1 = 0.5 + marker_radius * np.sin(text_theta1)
        text_theta2 = np.radians(90)
        text_x2 = 0.5 + marker_radius * np.cos(text_theta2) + .03
        text_y2 = 0.5 + marker_radius * np.sin(text_theta2)

        ax.text(text_x1, text_y1,
                f'{values[i]:,} minutes',
                ha='left', va='center',
                color='#1e1e1e', fontsize=20, fontfamily='Ubuntu')
        ax.text(text_x2, text_y2,
                str(labels[i]),
                ha='left', va='center',
                color='#1e1e1e', fontsize=14, fontfamily='Ubuntu')

    ax.set_xlim(0, 1)
    plt.savefig('transparent_chart.png', bbox_inches='tight',
                transparent=True, pad_inches=0)
    plt.close()

    img = add_bg_and_h1(
        'transparent_chart.png',
        x_mod=1 / 5.6,
        y1_mod=1.2,
        title=title,
    )
    crop_img(img, output_path)


def groovy_textcloud(
    event_durations: dict,
    event_colors: dict,
    title: str,
    output_path: str,
) -> None:
    '''
    create textcloud with outlines from dicts
    evet_to_durations (in minutes) and event_to_color
    '''
    fig, ax = plt.subplots(
        figsize=(576 / 1.2 / 100, 576 / 1.2 * 1.5 / 100), dpi=100)
    fig.patch.set_alpha(0)

    def color_func(word,
                   # pylint: disable=unused-argument
                   font_size,
                   position,
                   orientation,
                   random_state=None,
                   **kwargs):
        return event_colors.get(word)

    wc = WordCloud(width=round(576 / 1.2),
                   height=round(576 / 1.2 * 1.5),
                   background_color=None,
                   mode='RGBA',
                   max_words=2000,
                   repeat=True,
                   font_path='/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf',
                   color_func=color_func
                   )
    # generate word cloud
    wc.generate_from_frequencies(event_durations)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    ax.margins(x=0)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0, top=1)
    plt.tight_layout(pad=0)

    plt.savefig('transparent_chart.png', bbox_inches='tight',
                transparent=True, pad_inches=0)
    plt.close()

    img = add_bg_and_h1(
        'transparent_chart.png',
        y1_mod=1.2,
        title=title,
        y_title_mod=.1
    )
    crop_img(img, output_path)


def groovy_donutplot(labels: list,
                     values: list,
                     colors: list,
                     title: str,
                     output_path: str,
                     units: str = ' min',
                     donut_width: float = .25) -> None:
    '''
    create beautiful wobbly donut plot with expanded sections
    '''
    percents = values / values.sum() * 100
    other_idx = np.where(percents < 4)
    if other_idx[0].size > 0:
        other_value = np.sum(values[other_idx])
        labels = np.delete(labels, other_idx)
        values = np.delete(values, other_idx)
        colors = np.delete(colors, other_idx)

        labels = np.append(labels, 'other')
        values = np.append(values, other_value)
        colors = np.append(colors, 'grey')

    sorted_indices = np.argsort(values)[::-1]
    labels = np.array(labels)[sorted_indices]
    values = np.array(values)[sorted_indices]
    colors = np.array(colors)[sorted_indices]
    percents = values / values.sum() * 100

    plt.xkcd()
    rcParams['path.effects'] = [patheffects.withStroke(linewidth=0)]
    fig, ax = plt.subplots(figsize=(576 / 100, 576 / 100), dpi=100)
    fig.patch.set_alpha(0)

    patches, _, _ = plt.pie(values,
                            colors=colors,
                            explode=[.1 for i in values],
                            autopct=lambda pct: f'{pct:.0f}%',
                            startangle=120,
                            counterclock=False,
                            shadow=True,
                            wedgeprops={'linewidth': 3,
                                        'edgecolor': (1, 1, 1, 0),
                                        'width': donut_width},
                            pctdistance=(1 + (1 - donut_width)) / 2,
                            textprops={'color': 'white',
                                       'fontfamily': 'Ubuntu',
                                       'weight': 'bold',
                                       'ha': 'center',
                                       'fontsize': 15}
                            # 'fontsize': 60 * donut_width}
                            )

    for i, patch in enumerate(patches):
        # Adjust coordinates for label placement
        angle = (patch.theta1 + patch.theta2) / 2
        x = (1 - donut_width - .05) * np.cos(np.radians(angle))
        y = (1 - donut_width - .05) * np.sin(np.radians(angle))
        ha = 'right' if angle > -90 else 'left'
        ta = 0.1 if 0 > angle > -180 else 0
        ax.text(x, y + ta, labels[i],
                color=colors[i], fontfamily='Ubuntu',
                weight='bold',
                ha=ha, va='center', fontsize=18)

        ax.text(x, y + ta - 0.12, f'{round(values[i]):,}{units}',
                color=colors[i], fontfamily='Ubuntu',
                weight='regular', ha=ha, fontsize=11)

    # Equal aspect ratio ensures that the donut is circular
    plt.axis('equal')
    ax.margins(x=0)
    plt.tight_layout(pad=0)

    plt.subplots_adjust(left=0.05, right=.95)

    plt.savefig('transparent_chart.png',
                transparent=True)
    plt.close()
    plt. rcdefaults()

    img = add_bg_and_h1(
        'transparent_chart.png',
        y1_mod=1.2,
        title=title,
        y_title_mod=1 / 6
    )
    crop_img(img, output_path)


def groovy_day_circle(
        median_day: DataFrame,
        cat_to_color: dict,
        title: str,
        output_path: str,
        gap: float = .03,
        width: float = .02) -> None:
    '''
    create beautiful circle timeplot with categories durations
    '''
    # sort
    category_max_duration = (
        median_day
        .groupby('category')['duration_seconds']
        .sum())
    sorting_key = (
        median_day['category']
        .map(category_max_duration)
        .sort_values(ascending=False)
        .index)
    median_day = median_day.loc[sorting_key]

    # plot
    fig, ax = plt.subplots(figsize=(576 / 100, 576 / 100,), dpi=100)
    fig.patch.set_alpha(0)
    plt.axis('off')

    radius = 0.49
    ax.add_patch(plt.Circle((0.5, 0.5), radius, color='#1E1E1E'))

    for hour in range(24):
        hour_theta = np.radians(90 - hour * 15)
        hour_x = 0.5 + (radius + .05) * np.cos(hour_theta)
        hour_y = 0.5 + (radius + .05) * np.sin(hour_theta)
        if hour not in [0, 6, 12, 18]:
            fontsize = 12
            fontweight = 'regular'
        else:
            fontsize = 16
            fontweight = 'bold'
        ax.text(hour_x, hour_y, f'{hour :02d}',
                color='#1E1E1E', ha='center', va='center', fontsize=fontsize,
                fontfamily='Ubuntu', fontweight=fontweight,
                rotation=-hour * 15)

    # def events_overlap(event1, event2):
    #     start1 = event1['start_time_seconds']
    #     end1 = start1 + event1['duration_seconds']
    #     start2 = event2['start_time_seconds']
    #     end2 = start2 + event2['duration_seconds']
    #
    #     return start1 < end2 and start2 < end1

    c = -1
    # prev_entry = {'category': None,
    #               'start_time_seconds': None,
    #               'duration_seconds': None}
    for _, row in median_day.iterrows():
        # if ((row.category != prev_entry['category'])
        #         or (row.category == prev_entry['category']
        #             and events_overlap(row, prev_entry))):
        c += 1
        r = radius - c * gap
        # print(row.category, r)
        theta2 = 90 - row.start_time_seconds / 3600 * 15
        theta1 = theta2 - row.duration_seconds / 3600 * 15
        wedge = Wedge(
            center=(0.5, 0.5),
            r=r,
            theta1=theta1,
            theta2=theta2,
            color=cat_to_color[row.category],
            width=width)
        ax.add_patch(plt.Circle((0.5, 0.5), r - width / 2,
                                ec=cat_to_color[row.category],
                                lw=1, alpha=.2, fill=False))
        ax.add_patch(wedge)

        # prev_entry['category'] = row.category
        # prev_entry['start_time_seconds'] = row.start_time_seconds
        # prev_entry['duration_seconds'] = row.duration_seconds

    ax.margins(x=0)
    plt.axis('equal')
    plt.tight_layout(pad=0)

    plt.subplots_adjust(left=0.05, right=.95)

    plt.savefig('transparent_chart.png',
                transparent=True)
    plt.close()

    # create wordcloud with cirle mask
    mask = np.array(Image.open('mask.png'))

    def color_func(word,
                   # pylint: disable=unused-argument
                   font_size,
                   position,
                   orientation,
                   random_state=None,
                   **kwargs):
        return cat_to_color.get(word)
    wc = WordCloud(
        background_color=None,
        mode='RGBA',
        repeat=False,
        font_path='/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf',
        color_func=color_func,
        mask=mask
    )
    wc.generate_from_frequencies(
        median_day
        .groupby('category')
        .duration_seconds
        .agg('sum').to_dict())
    wc.to_file('wc_masked.png')

    # put wc inside circleplot
    circleplot_img = Image.open('transparent_chart.png')
    wc_img = Image.open('wc_masked.png')
    circleplot_img_width, circleplot_img_height = circleplot_img.size
    n_cat = median_day.category.nunique()
    wc_img = wc_img.resize((round(circleplot_img_width / (n_cat * gap) / 9),
                            round(circleplot_img_height / (n_cat * gap) / 9)))
    wc_img_width, wc_img_height = wc_img.size

    # Calculate the center coordinates
    x = (circleplot_img_width - wc_img_width) // 2
    y = (circleplot_img_height - wc_img_height) // 2
    circleplot_img.paste(wc_img, (x, y), wc_img.split()[3])
    circleplot_img.save('transparent_chart_2.png')

    # add bg and crop
    img = add_bg_and_h1(
        'transparent_chart_2.png',
        y1_mod=1.2,
        title=title,
    )
    crop_img(img, output_path)


def groovy_ridgeplot(
        ridge_df: DataFrame,
        title: str,
        output_path: str) -> None:
    '''
    create ridgeplot by category with durations as densities and date as x
    ridge_df should have date, category, category_color, and duration (in mins)
    '''
    category_order = list(ridge_df['category'].unique())
    colors = list(ridge_df['category_color'].unique())
    df_expanded = ridge_df.loc[ridge_df.index.repeat(
        ridge_df['duration'].astype(int))]
    df_expanded = df_expanded.drop('duration', axis=1)
    df_expanded['date_ordinal'] = pd.to_datetime(
        df_expanded['date']).apply(lambda x: x.toordinal())
    df_expanded['category'] = pd.Categorical(
        df_expanded['category'],
        categories=category_order, ordered=True)

    # plt.figure(figsize=(576 / 100, 576 * 10 / 8 / 100), dpi=100)
    fig, axes = joypy.joyplot(
        data=df_expanded,
        by='category',
        column='date_ordinal',
        overlap=2,
        lw=0,
        color=colors,
        figsize=(576 / 100, 576 * (10 / 8) / 100),
    )

    min_date = df_expanded['date_ordinal'].min()
    max_date = df_expanded['date_ordinal'].max()
    # mid_date = (max_date - min_date) / 2 + min_date

    min_date, max_date = [pd.Timestamp.fromordinal(int(i)).strftime('%B %-d')
                          for i in [min_date, max_date]]

    for ax, category in zip(axes, category_order):
        ax.set_yticklabels([])  # Hide yticklabels
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        line = ax.get_lines()[1]
        x_values = line.get_xdata()
        densities = line.get_ydata()
        if np.sum(densities) == 0:
            continue
        cumulative_density = np.cumsum(densities) / np.sum(densities)
        median_value_index = np.argmin(np.abs(cumulative_density - 0.5))
        median_x_value = x_values[median_value_index]

        ax.text(median_x_value, .25, category, transform=trans,
                va='center', ha='center', color='#1E1E1E',
                fontname='Ubuntu', fontsize=18,
                zorder=10)
    plt.tick_params(width=0)
    plt.xticks([])
    n_spaces_min = 7 if len(max_date) > len(min_date) else 5
    n_spaces_max = 7 if len(min_date) > len(max_date) else 5
    plt.text(0.025, .5, ((min_date + ' ' * n_spaces_min) * 4).strip(),
             transform=fig.transFigure,
             va='center', ha='center', color='#1E1E1E',
             fontname='Ubuntu', fontsize=14, rotation=90,
             zorder=2.5)
    plt.text(.975, .5, ((max_date + ' ' * n_spaces_max) * 4).strip(),
             transform=fig.transFigure,
             va='center', ha='center', color='#1E1E1E',
             fontname='Ubuntu', fontsize=14, rotation=-90,
             zorder=2.5)

    plt.margins(0, 0)
    # plt.tight_layout(pad=0)

    plt.subplots_adjust(left=.05, right=.95)

    plt.savefig('transparent_chart.png', bbox_inches='tight',
                transparent=True, pad_inches=0)
    plt.close()

    img = add_bg_and_h1(
        'transparent_chart.png',
        y1_mod=1.15,
        title=title,
        y_title_mod=.125
    )
    crop_img(img, output_path)


def groovy_3d_barplot(
        df: DataFrame,
        title: str,
        output_path: str) -> None:
    '''
    create 3d barplot with categories, weekdays and values
    '''
    category_colors = dict(zip(df['category'].unique(),
                               df['category_color'].unique()))

    df['weekday'] = pd.to_datetime(df['date']).dt.strftime('%a')
    median_duration_by_category = df.groupby(
        'category')['duration'].mean().sort_values()
    sorted_categories = median_duration_by_category.index.tolist()

    fig = plt.figure(
        figsize=(576 * 1.3 / 100, 576 * 1.3 * 1.5 / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1, df['category'].nunique() / 7, 1.5))

    df = df.groupby(['category', 'weekday'], as_index=False).agg(
        {'duration': 'median'})

    # Convert categorical variables to numeric codes
    weekday_order = ['Mon', 'Tue', 'Wed',
                     'Thu', 'Fri', 'Sat', 'Sun']
    df['weekday'] = pd.Categorical(df['weekday'], categories=weekday_order,
                                   ordered=True)
    df['weekday_code'] = df['weekday'].cat.codes

    df['category'] = pd.Categorical(df['category'],
                                    categories=sorted_categories,
                                    ordered=True)

    df['category_code'] = df['category'].cat.codes

    x = df['weekday_code']
    y = df['category_code']
    z = df['duration']

    # Create 3D bars
    ax.bar3d(x, y, 0, 1, 1, z, shade=True,
             color=[category_colors[cat] for cat in df['category']],
             ec='#1e1e1e', lw=1)

    # Set tick labels
    ax.set_xticks(df['weekday_code'].unique())
    ax.set_xticklabels(df['weekday'].unique())
    ax.set_yticks(df['category_code'].unique())
    ax.set_yticklabels(df['category'].unique())
    ax.set_zlim(bottom=-100)

    ax.view_init(elev=10, azim=-60)

    ax.set_axis_off()

    for i, label in zip(df['weekday_code'].unique(), df['weekday'].unique()):
        ax.text(i + 1, -1, 0, label,
                color='#1e1e1e', ha='center', va='center',
                zdir='x', fontname='Ubuntu', fontsize=14)

    for i, (label, color) in enumerate(zip(
            sorted_categories,
            [category_colors[cat] for cat in sorted_categories])):
        ax.text(7.25, i, 0, label, color=color,
                ha='center', va='top', zdir='z',
                fontname='Ubuntu', fontweight='bold', fontsize=18)

    for i, row in df[df['weekday'] == 'Sun'].iterrows():
        if row['duration'] < df['duration'].max() * .1:
            continue
        x_coord = 6
        y_coord = row['category_code'] + 2.5
        z_coord = row['duration'] / 2 - df.duration.max() * .05
        annotation_text = f"{round(row['duration'])} min"
        ax.text(x_coord, y_coord, z_coord, annotation_text,
                color='#bbb', ha='center', va='center',
                zdir='z', fontname='Ubuntu', fontsize=8)

    ax.margins(0, 0, 0)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0, right=1)

    plt.savefig('transparent_chart.png', bbox_inches='tight',
                transparent=True, pad_inches=0)
    plt.close()

    img = add_bg_and_h1(
        'transparent_chart.png',
        y1_mod=1.15,
        # x_mod=.05,
        title=title,
        y_title_mod=.1
    )
    crop_img(img, output_path)
