#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

def get_coordinates(r, c):
    """
    r: number of rows; int
    c: number of columns; int
    
    return: list with coordinates of each element from matrix (r x c)
    """
    arr = np.zeros((r, c))
    coords = []
    for index, value in np.ndenumerate(arr):
        coords.append(index)
    return coords
            

def plot_hists(df, features, color, r=None, c=None):
    """
    df: dataframe which contains all necessary features; pandas DataFrame
    features: features explored by histograms; list
    color: the color of the charts; any format available for matplotlib
    r: number of rows in the figure; int or None, default None
    c: number of columns in the figure; int or None, default None
    
    return: plots histograms for each feature
    """
    n = len(features)
    
    if n == 1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(df[f'{features[0]}'], bins=60, color=color)
        plt.show()
        
    else:
        if r != 1:
            fig, axs = plt.subplots(r, c, figsize=(c*5, r*4))
            coordinates = get_coordinates(r, c)
            coords = coordinates[:n]

            for f in features:
                i = features.index(f)
                axs[coords[i]].hist(df[f'{f}'], bins=60, color=color)
                axs[coords[i]].set_xlabel(f)

            if len(features) < r*c:
                erased = coordinates[n:]
                for e in erased:
                    axs[e].axis('off')
            plt.show()

        else:
            fig, ax = plt.subplots(1, c, figsize=(n*6, 4))
            for f in features:
                i = features.index(f)

                ax[i].hist(df[f'{f}'], bins=60, color=color)
                ax[i].set_xlabel(f)
            plt.show()
            
            
def plot_log_hists(df, features, color, r=None, c=None):
    """
    df: dataframe which contains all necessary features; pandas DataFrame
    features: features explored by histograms; list
    color: the color of the charts; any format available for matplotlib
    r: number of rows in the figure; int or None, default None
    c: number of columns in the figure; int or None, default None
    
    return: plots histograms of ln(X + 1) for each feature (X)
    """
    n = len(features)
    
    if n == 1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(np.log1p(df[f'{features[0]}']), bins=60, color=color)
        plt.show()
        
    else:
        if r != 1:
            fig, axs = plt.subplots(r, c, figsize=(c*5, r*4))
            coordinates = get_coordinates(r, c)
            coords = coordinates[:n]

            for f in features:
                i = features.index(f)
                axs[coords[i]].hist(np.log1p(df[f'{f}']), bins=60, color=color)
                axs[coords[i]].set_xlabel(f)

            if len(features) < r*c:
                erased = coordinates[n:]
                for e in erased:
                    axs[e].axis('off')
            plt.show()

        else:
            fig, ax = plt.subplots(1, c, figsize=(n*6, 4))
            for f in features:
                i = features.index(f)

                ax[i].hist(np.log1p(df[f'{f}']), bins=60, color=color)
                ax[i].set_xlabel(f)
            plt.show()
            

def plot_boxplots(df, features, color, r=None, c=None):
    """
    df: dataframe which contains all necessary features; pandas DataFrame
    features: features explored by boxplots; list
    color: the color of the charts; any format available for matplotlib
    r: number of rows in the figure; int or None, default None
    c: number of columns in the figure; int or None, default None
    
    return: plots simple boxplots for each feature
    """
    n = len(features)
    flierprops = dict(marker='s', markerfacecolor=color, markeredgecolor='none')
    medianprops = dict(color='black', linewidth=3)
    
    if n == 1:
        fig, ax = plt.subplots(figsize=(3.7, 6))
        ax.boxplot(df[f'{features[0]}'],
                   flierprops=flierprops,
                   medianprops=medianprops)
        ax.grid(visible=False, axis='x', which='both')
        ax.tick_params(axis='x',
                       which='both',
                       bottom=False,
                       top=False,
                       labelbottom=False)
        plt.show()
        
    else:
        if r != 1:
            fig, axs = plt.subplots(r, c, figsize=(c*3.7, r*6))
            coordinates = get_coordinates(r, c)
            coords = coordinates[:n]

            for f in features:
                i = features.index(f)
                axs[coords[i]].boxplot(df[f'{f}'],
                                       flierprops=flierprops,
                                       medianprops=medianprops)
                axs[coords[i]].grid(visible=False, axis='x', which='both')
                axs[coords[i]].set_xlabel(f)
                axs[coords[i]].tick_params(axis='x',
                                           which='both',
                                           bottom=False,
                                           top=False,
                                           labelbottom=False)

            if len(features) < r*c:
                erased = coordinates[n:]
                for e in erased:
                    axs[e].axis('off')
            plt.show()

        else:
            fig, ax = plt.subplots(1, c, figsize=(n*3.7, 6))
            for f in features:
                i = features.index(f)

                ax[i].boxplot(df[f'{f}'],
                              flierprops=flierprops,
                              medianprops=medianprops)
                ax[i].grid(visible=False, axis='x', which='both')
                ax[i].set_xlabel(f)
                ax[i].tick_params(axis='x',
                                  which='both',
                                  bottom=False,
                                  top=False,
                                  labelbottom=False)
            plt.show()
            
            
def plot_corr(df, features, save=False, title=None, l=10, w=8, tri=True, main=False, cmap='coolwarm'):
    """
    df: dataframe which contains all necessary features; pandas DataFrame
    features: features explored by the correlation map; list
    digits: number of digits after the decimal point for correlations; int, default 2
    l: length of the chart; int or float, default 9
    w: width of the chart; int or float, default 7
    tri: whether the corrplot is triangle; bool, default True (elements above the main diagonal are hidden)
    main: whether to plot the main diagonal; bool, default False
    cmap: colormap of the chart; any format available for seaborn heatmap, default 'coolwarm'
    
    return: plots a correlation map
    """
    corr = df[features].corr()
    k = 0 if main==True else -1
    antimask = np.tri(len(corr), k=k)
    n = corr.shape[0]
    mask = np.absolute(antimask - np.ones((n, n)))
    mask = mask if tri==True else None
    
    xtl = features if main==True else features[:-1] + [' ']
    ytl = features if main==True else [' '] + features[1:]
    
    plt.subplots(figsize = (l, w))
    sns.heatmap(corr,
                mask=mask,
                annot=True,
                linewidths=.5,
                fmt= '.2f',
                vmin=-1,
                vmax=1,
                cmap=cmap,
                square=True,
                xticklabels=xtl,
                yticklabels=ytl)
    
    if save == True:
        plt.savefig(f'figures/{title}.png', bbox_inches='tight', dpi=100)
    
    plt.show()   


def get_other_color(data, df, ax, highlt_color):
    """
    data: the original dataframe
    df: the dataframe grouped by a feature
    ax: matplotlib ax
    highlt_color: the color to highlight bars; str
    
    return: figure where minor categories are highlited
    """
    for i in df.index:
        if df.loc[i] / len(data) < 0.02:
            ax.patches[df.index.get_indexer([f'{i}'])[0]].set_facecolor(highlt_color)
    return ax


def plot_barcharts(df, features, group, indx, basic_color, highlt_color, r=None, c=None):
    """
    documentation
    """
    
    n = len(features)
    
    fig, axs = plt.subplots(r, c, figsize=(c*7, r*6))
    
    coordinates = get_coordinates(r, c)
    coords = coordinates[:n]
    
    for (i, feature) in list(zip(coords, features)):
        df_group = df.groupby(feature).count()[group]

        opts = list(map(int, list(df_group.index))) if indx == 'float' else list(df_group.index)

        freq = df_group / sum(df_group) * 100
        y_pos = np.arange(len(opts))
        
        axs[i].barh(y_pos, freq, color=basic_color, align='center')
        get_other_color(df, df_group, axs[i], highlt_color)
        
        for u, v in enumerate(freq):
            clr = highlt_color if v < 2 else basic_color
            axs[i].text(v + 2, u + 0.15, str(round(v, 2))+'%', color=clr, fontweight='bold')
        
        axs[i].set_yticks(y_pos)
        axs[i].set_yticklabels(opts)
        axs[i].invert_yaxis()
        axs[i].set_xlim(0, 100)
        axs[i].set_xlabel('% of values', color='grey')
        axs[i].set_ylabel(feature, color='grey')
        axs[i].grid(color='whitesmoke')
            
    if len(features) < r*c:
        erased = coordinates[n:]
        for e in erased:
            axs[e].axis('off')

    plt.show()
    
    
def plot_mean_target_map(df, features, group, order, title, l=12, w=8, palette='coolwarm'):
    """
    documentation
    """
    subtotal = df.groupby(features[1]).mean()[group]
    mean_target = pd.pivot_table(data=df,
                                 index=features[1],
                                 columns=features[0],
                                 values=group,
                                 aggfunc=np.mean,
                                 fill_value=0)
    mean_target = mean_target[order]
    mean_target['Subtotal'] = subtotal
    
    plt.figure(figsize = (l, w))
    hm = sns.heatmap(mean_target,
                     cmap=palette,
                     linewidth=.5,
                     annot=True)
    hm.set(title = title,
           xlabel = features[0],
           ylabel = features[1])

    plt.show()    
    
def plot_count_target_map(df, features, group, order, title, l=12, w=8, palette='coolwarm'):
    """
    documentation
    """
    subtotal = df.groupby(features[1]).count()[group]
    count_target = pd.pivot_table(data=df,
                                  index=features[1],
                                  columns=features[0],
                                  values=group,
                                  aggfunc=np.ma.count,
                                  fill_value=0)
    count_target = count_target[order]
    count_target['Subtotal'] = subtotal
    
    plt.figure(figsize = (l, w))
    hm = sns.heatmap(count_target,
                     cmap=palette,
                     linewidth=.5,
                     annot=True)
    hm.set(title = title,
           xlabel = features[0],
           ylabel = features[1])

    plt.show()
    
    
def plot_sum_target_map(df, features, group, order, title, l=12, w=8, palette='coolwarm'):
    """
    documentation
    """
    subtotal = df.groupby(features[1]).sum()[group]
    sum_target = pd.pivot_table(data=df,
                                index=features[1],
                                columns=features[0],
                                values=group,
                                aggfunc=np.sum,
                                fill_value=0)
    sum_target = sum_target[order]
    sum_target['Subtotal'] = subtotal
    
    plt.figure(figsize = (l, w))
    hm = sns.heatmap(sum_target,
                     cmap=palette,
                     linewidth=.5,
                     annot=True)
    hm.set(title = title,
           xlabel = features[0],
           ylabel = features[1])

    plt.show()