import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes

import pandas as pd
import seaborn as sns
import ptitprince as pt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import arviz as az

from rvm_analysis.save_tools import save_in_folder
from rvm_analysis.colours import get_cell_colour

from typing import Optional

class BrokenAxes:
    """
    A class for splitting the xaxis into different scales, and plotting this as one function.
    """
    def __init__(self, figsize: tuple=(5,3), width_ratios:tuple =(1, 4), d=0.015, xlims=((-4, 0), (0, 100)),fontsize=7):
        self.fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios, wspace=0)
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["axes.spines.right"] = False
        # Create two real axes
        self.ax_left = self.fig.add_subplot(gs[0])
        self.ax_right = self.fig.add_subplot(gs[1], sharey=self.ax_left)

        # Set xlimits
        self.ax_left.set_xlim(*xlims[0])
        self.ax_right.set_xlim(*xlims[1])

        # We change the fontsize of minor ticks label 
        self.ax_left.tick_params(axis='both', which='major', labelsize=fontsize)
        self.ax_right.tick_params(axis='both', which='minor', labelsize=fontsize)
        self.ax_right.tick_params(axis='both', which='major', labelsize=fontsize)
        self.ax_left.tick_params(axis='both', which='minor', labelsize=fontsize)

        # Hide spines and ticks
        self.ax_left.spines['right'].set_visible(False)
        self.ax_right.spines['left'].set_visible(False)
        self.ax_right.yaxis.set_visible(False)

        # Store xlims for dispatching plotting
        self.left_xlim = xlims[0]
        self.right_xlim = xlims[1]

        # # Draw break marks
        kwargs = dict(color='k', clip_on=False)
        # for ax, sign in zip((self.ax_left, self.ax_right), (-1, 1)):
        #     kwargs.update(transform=ax.transAxes)
        #     ax.plot([sign * d, sign * -d], [-d, +d], **kwargs)
        #     ax.plot([sign * d, sign * -d], [1 - d, 1 + d], **kwargs)

        # Right edge of left axis
        # self.ax_left.plot([1 - d, 1 + d], [0.5 - d, 0.5 + d], transform=self.ax_left.transAxes, **kwargs)
        # self.ax_left.plot([1 - d, 1 + d], [0.5 + d, 0.5 - d], transform=self.ax_left.transAxes, **kwargs)

        #! Just two break mark between the two axes
        self.ax_right.plot([-d, +d], [- d, + d], transform=self.ax_right.transAxes, **kwargs)
        self.ax_right.plot([-d, +d], [1 - d, 1 + d], transform=self.ax_right.transAxes, **kwargs)

    def plot(self, x, y, **kwargs):
        x = np.asarray(x)
        y = np.asarray(y)

        mask_left = (x >= self.left_xlim[0]) & (x <= self.left_xlim[1])
        mask_right = (x >= self.right_xlim[0]) & (x <= self.right_xlim[1])

        if np.any(mask_left):
            self.ax_left.plot(x[mask_left], y[mask_left], **kwargs)
        if np.any(mask_right):
            self.ax_right.plot(x[mask_right], y[mask_right], **kwargs)

        # Optionally hide the 0 label to avoid overlap
        xticks = self.ax_left.get_xticks()
        xticklabels = [str(int(t)) if t != self.left_xlim[1] else '' for t in xticks]
        self.ax_left.set_xticklabels(xticklabels)

    def scatter(self,x,y,**kwargs):
        x = np.asarray(x)
        y = np.asarray(y)

        mask_left = (x >= self.left_xlim[0]) & (x <= self.left_xlim[1])
        mask_right = (x >= self.right_xlim[0]) & (x <= self.right_xlim[1])

        if np.any(mask_left):
            self.ax_left.scatter(x[mask_left], y[mask_left], **kwargs)
        if np.any(mask_right):
            self.ax_right.scatter(x[mask_right], y[mask_right], **kwargs)  

    def fill_between(self,x,lower, upper, **kwargs):
        x = np.asarray(x)
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        mask_left = (x >= self.left_xlim[0]) & (x <= self.left_xlim[1])
        mask_right = (x >= self.right_xlim[0]) & (x <= self.right_xlim[1])

        if np.any(mask_left):
            self.ax_left.fill_between(x[mask_left], lower[mask_left],upper[mask_left], **kwargs)
        if np.any(mask_right):
            self.ax_right.fill_between(x[mask_right], lower[mask_right], upper[mask_right],**kwargs)

    def set_xlabel(self, label,**kwargs):
        self.ax_right.set_xlabel(label,**kwargs)

    def set_ylabel(self, label,**kwargs):
        self.ax_left.set_ylabel(label,**kwargs)

    def set_title(self, label):
        self.fig.suptitle(label)

    def set_ylim(self, bottom, top):
        self.ax_left.set_ylim(bottom, top)
        self.ax_right.set_ylim(bottom, top)

    def set_yscale(self,*args):
        self.ax_left.set_yscale(*args)
        self.ax_right.set_yscale(*args)

    def show(self):
        plt.tight_layout()
        plt.show()


def plot_rainclouds(data_csv_str=None,y='pseudo-r2',df=None,xlim: Optional[list]=None,ax: Optional[Axes]=None,
                    offset=0.0,move=0.0,custom_palette=None,xlabel=None,
                    cloud_linewidths=None):
    """
    Plots rainclouds for gpytorch metric data. See gpytorch_models for the
    code which generates the data.
    """
    # Load the data
    if df is None:
        df = pd.read_csv(data_csv_str)

    # Set the visual style
    sns.set_theme(style="white", font_scale=1.0)

    # Create the figure
    if ax is None:
        f, ax = plt.subplots(figsize=(10, 6))

    # # Define the order for consistent plotting
    # order = sorted(df['cell_name'].unique())

    # Raincloud plot
    plt.rcParams['font.size'] = 7
    pt.RainCloud(x='cell_name', y=y, data=df, palette=custom_palette, bw=.2, width_viol=.7, ax=ax, orient='h',
                 move=move,box_showfliers=False,point_size=3,box_linewidth=0.5,linewidth=0.5,box_whiskerprops={"linewidth": 0.5},
                 )#offset=offset + max(0.15/1.8, .15) + .05,move=move

    # Define custom line widths
    if cloud_linewidths is not None:
        custom_cloud_linewidths = cloud_linewidths#

        # Manually update the cloud outlines (violin patches)
        from matplotlib.collections import PolyCollection

        cloud_index = 0
        for collection in ax.collections:
            if isinstance(collection, PolyCollection):
                if cloud_index < len(custom_cloud_linewidths):
                    collection.set_linewidth(custom_cloud_linewidths[cloud_index])
                    cloud_index += 1

        # Optionally force re-render
        plt.draw()
    
    # ax.set_title(f"Raincloud Plot of {y} by Cell Name")
    # ax.set_ylabel("")
    ax.set_xlabel("Pseudo-R² Value ($>0$ improves on mean)" if xlabel is None else xlabel,fontsize=7)
    ax.set_ylabel("Cell Type",fontsize=7)
    if xlim is not None:
        ax.set_xlim(*xlim)
    plt.rcParams['axes.linewidth'] = 0.5
    ax.spines[['top','right']].set_visible(False)
    ax.vlines(0,*ax.get_ylim(),linestyle="--",linewidth=0.5,color='black')
    ax.tick_params(which='both',labelsize=7,length=1,width=1)
    # plt.subplots_adjust(hspace=0.1)  # Increase as needed (default is ~0.125)

    if ax is None:
        plt.tight_layout()
        plt.show()


def plot_interactive_raincloud(data_csv_str, y='pseudo-r2'):
    """
    Creates an interactive raincloud-like plot using Plotly where hovering over points shows model_id.
    """
    # Load the data
    df = pd.read_csv(data_csv_str)

    # Define consistent order
    order = sorted(df['cell_name'].unique())

    # Plotly violin + strip for raincloud effect
    fig = px.violin(
        df,
        y='cell_name',
        x=y,
        color='cell_name',
        orientation='h',
        box=True,         # box plot inside violin
        points='all',     # show all individual points
        hover_data=['model_id'],  # show model_id on hover
        category_orders={'cell_name': order},
        template="simple_white"
    )

    fig.update_layout(
        title=f"Raincloud-like Plot of {y} by Cell Name",
        xaxis_title="R² Value",
        yaxis_title="Cell Type",
        showlegend=False
    )

    fig.show()


def plot_predictions(counts,X_test,count_pred_mean,lambda_preds,lambda_mean,preds_numpy):
    bax = BrokenAxes()
    # bax.scatter(df['time'], df['count'], alpha=0.3, label='Raw data (stacked trials)')
    bax.plot(counts['time'],counts['count'],label="true",linewidth=0.5,color='black',alpha=0.5)
    bax.plot(X_test,count_pred_mean,label="predicted_counts")
    # bax.plot(X_test,count_pred_upper,label="predicted_counts")
    # bax.plot(X_test,count_pred_lower,label="predicted_counts")
    # lambda_hpd = az.hdi(lambda_preds, hdi_prob=0.94)
    bax.plot(X_test, lambda_mean, color='red', label='Mean predicted count')
    az.plot_hdi(x=X_test,y = lambda_preds,ax=bax,smooth=False)
    az.plot_hdi(x = X_test,y=preds_numpy.T,ax=bax,smooth=False,color='pink')
    # az.plot_hdi(x=X_test,y = count_preds,color='green',ax=bax,smooth=False)
    # az.plot_posterior(poisson_counts, hdi_prob=0.95,ax=bax)

    # plt.plot(times,counts)
    # plt.plot(X_test,df.groupby('time', as_index=False)['count'][::10].mean())
    # plt.fill_between(unique_times, lambda_hpd[:, 0], lambda_hpd[:, 1], color='red', alpha=0.3, label='94% HDI')
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.title("Posterior Predictive Mean per Time Point")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predictions_paper(counts,X_test,count_pred_mean,lambda_preds,lambda_mean,preds_numpy,
                           cell_name='OFF',figsize=(3.5,1.5), log_y = False,
                           **save_kwargs):
    """Plots predicted counts and confidence intervals."""

    bax = BrokenAxes(figsize=figsize,xlims=((-10,0),(0,100)))

    bax.plot(counts['time'],counts['count'],label="Observed count",linewidth=0.2,
             color=get_cell_colour(names=cell_name),alpha=1.0)
    bax.plot(X_test, lambda_mean, label='Mean predicted count',linewidth=0.5,color='black')
    az.plot_hdi(x=X_test,y = lambda_preds,ax=bax,smooth=False,color='black',
                fill_kwargs={"alpha":0.3,"zorder": 1000,"linewidth":0.0})
    az.plot_hdi(x = X_test,y=preds_numpy.T,ax=bax,smooth=False,color='grey',fill_kwargs={"alpha": 0.4,"linewidth":0.0})

    bax.set_ylabel("Total spikes (5ms bin width)",fontsize=7)
    bax.set_xlabel("Time (s)",fontsize=7)
    plt.legend(fontsize=7)
    plt.grid(False)
    ylims = plt.ylim()
    bax.set_ylim(*ylims)
    if log_y:
        bax.set_yscale("log")
    plt.vlines(0,*ylims,color='black',linestyle="--")
    save_in_folder(**save_kwargs,pad_inches=0.1)

def plot_loo_comparison(traces,cell_name,figsize=(3.5,2),**save_kwargs):
    """Makes a clean and tidy plot of the LOO comparison between models."""
    df_comp_loo = az.compare(traces,ic='loo')
    fig, ax = plt.subplots(figsize=figsize)
    ax = az.plot_compare(df_comp_loo,ax=ax,title=None,textsize=7,
                    plot_kwargs={"color_ic":get_cell_colour(names=cell_name)},show=False)
    leg = ax.get_legend()
    leg.set_loc("best")
    ax.legend(fontsize=7)
    ax.set_ylabel("Ranked Models",fontsize=7)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel("Log ELPD (LOO)")
    save_in_folder(**save_kwargs)