# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

# font size
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12

# typeface
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True

def CreatePlotGrid(title, xlabel, ylabel, zlabel, aspectmode):
    
    # plot
    fig, ((axXZ, axYZ), (axXY, ax3D)) = plt.subplots(2, 2)
    fig.suptitle(title, fontsize=14)
    fig.set_size_inches(10, 10)
    fig.subplots_adjust(hspace=0.2)
    fig.subplots_adjust(wspace=0.5)

    # XZ Plane
    axXZ.set_title(xlabel + zlabel + ' Plane')
    axXZ.xaxis.set_label_text(xlabel + ' axis (km)')
    axXZ.yaxis.set_label_text(zlabel + ' axis (km)')
    axXZ.set_aspect(aspectmode)

    # YZ Plane
    axYZ.set_title(ylabel + zlabel + ' Plane')
    axYZ.xaxis.set_label_text(ylabel + ' axis (km)')
    axYZ.yaxis.set_label_text(zlabel + ' axis (km)')
    axYZ.set_aspect(aspectmode)

    # XY Plane
    axXY.set_title(xlabel + ylabel + ' Plane')
    axXY.xaxis.set_label_text(xlabel + ' axis (km)')
    axXY.yaxis.set_label_text(ylabel + ' axis (km)')
    axXY.set_aspect(aspectmode)

    # plot in 3D
    ax3D.axis('off')
    ax3D = fig.add_subplot(224, projection='3d') # "224" means "2x2 grid, 4th subplot"
    ax3D.set_title('3D View in ' + xlabel + ylabel + zlabel + ' Frame')
    ax3D.xaxis.set_label_text(xlabel + ' axis (km)')
    ax3D.yaxis.set_label_text(ylabel + ' axis (km)')
    ax3D.zaxis.set_label_text(zlabel + ' axis (km)')
    ax3D.set_aspect(aspectmode)
    
    # for publishing in paper
    
    # this script for setting width & height is from http://damon-is-a-geek.com/publication-ready-the-first-time-beautiful-reproducible-plots-with-matplotlib.html
    WIDTH = 345.0  # the LaTeX result from \the\textwidth
    FACTOR = 0.90  # the fraction of the width you'd like the figure to occupy
    fig_width_pt  = WIDTH * FACTOR
    inches_per_pt = 1.0 / 72.27
    golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
    fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio   # figure height in inches
    fig_dims      = [fig_width_in, fig_height_in] # fig dims as a list

    # Y vs X in RLP, R vs I in RIC
    fig1, (ax1) = plt.subplots(1,1);
    #fig1.figsize = fig_dims    
    fig1.set_size_inches(fig_dims)
    ax1.set_title(xlabel + ylabel + ' Plane')
    ax1.xaxis.set_label_text(ylabel + ' axis (km)')
    ax1.yaxis.set_label_text(xlabel + ' axis (km)')
    ax1.set_aspect(aspectmode)

    # B vs V in VNB
    fig2, (ax2) = plt.subplots(1,1);
    #fig2.figsize = fig_dims
    fig2.set_size_inches(fig_dims)
    ax2.set_title(xlabel + zlabel + ' Plane')
    ax2.xaxis.set_label_text(xlabel + ' axis (km)')
    ax2.yaxis.set_label_text(zlabel + ' axis (km)')
    ax2.set_aspect(aspectmode)
    
    axis_array = ((axXZ, axYZ), (axXY, ax3D), (ax1, ax2))
    
    return axis_array

    
def SetPlotGridData(axis_array, data, style, color, label):
    
    # Allowed colors:
    # b: blue
    # g: green
    # r: red
    # c: cyan
    # m: magenta
    # y: yellow
    # k: black
    # w: white
    
    ((axXZ, axYZ), (axXY, ax3D), (ax1, ax2)) = axis_array

    if style == 'points':
        markersize = 5
        markertype = 'o'
    elif style == 'line':
        markersize = 1
        markertype = '-'
    elif style == 'dotted':
        markersize = 1
        markertype = ':'
    elif style == 'star':
        markersize = 5
        markertype = '*'
        
    # add data to plots 
    ax1.plot(data.y, data.x, markertype, markersize=markersize, color=color, label=label)
    ax2.plot(data.x, data.z, markertype, markersize=markersize, color=color, label=label)
    
    axXZ.plot(data.x, data.z, markertype, markersize=markersize, color=color)
    axYZ.plot(data.y, data.z, markertype, markersize=markersize, color=color)
    axXY.plot(data.x, data.y, markertype, markersize=markersize, color=color)
    #ax3D.plot(data.x.values, data.y.values, data.z.values, markertype, markersize=markersize, color=color)
        
def ConfigurePlotLegend(axis_array):
    
    ((axXZ, axYZ), (axXY, ax3D), (ax1, ax2)) = axis_array
    
    handles, labels = ax1.get_legend_handles_labels()
    
    ax1.legend([handles[1], handles[2], handles[3]], [labels[1], labels[2], labels[3]], loc='lower right')
    ax2.legend([handles[1], handles[2], handles[3]], [labels[1], labels[2], labels[3]], loc='lower right')
    #'Linear Propagation, Linear dV', 'Nonlinear Propagation, Linear dV', 'Nonlinear Propagation, Targeted dV''])
    
    #plt.savefig('..\LaTeX\Images\RIC.pdf', format='pdf')
    

