# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def CreatePlotGrid(title, xlabel, ylabel, zlabel, aspectmode):
    
    # plot
    fig, ((axXZ, axYZ), (axXY, ax3D)) = plt.subplots(2, 2)
    fig.suptitle(title, fontsize=14)
    fig.set_size_inches(10, 10)
    fig.subplots_adjust(hspace=0.2)
    fig.subplots_adjust(wspace=0.5)

    # XZ Plane
    axXZ.set_title(xlabel + zlabel + ' Plane')
    axXZ.xaxis.set_label_text(xlabel + ' axis')
    axXZ.yaxis.set_label_text(zlabel + ' axis')
    axXZ.set_aspect(aspectmode)

    # YZ Plane
    axYZ.set_title(ylabel + zlabel + ' Plane')
    axYZ.xaxis.set_label_text(ylabel + ' axis')
    axYZ.yaxis.set_label_text(zlabel + ' axis')
    axYZ.set_aspect(aspectmode)

    # XY Plane
    axXY.set_title(xlabel + ylabel + ' Plane')
    axXY.xaxis.set_label_text(xlabel + ' axis')
    axXY.yaxis.set_label_text(ylabel + ' axis')
    axXY.set_aspect(aspectmode)

    # plot in 3D
    ax3D.axis('off')
    ax3D = fig.add_subplot(224, projection='3d') # "224" means "2x2 grid, 4th subplot"
    ax3D.set_title('3D View in ' + xlabel + ylabel + zlabel + ' Frame')
    ax3D.xaxis.set_label_text(xlabel + ' axis')
    ax3D.yaxis.set_label_text(ylabel + ' axis')
    ax3D.zaxis.set_label_text(zlabel + ' axis')
    ax3D.set_aspect(aspectmode)
    
    return axXZ, axYZ, axXY, ax3D

    
def SetPlotGridData(axXZ, axYZ, axXY, ax3D, data, style, color):
    
    # Allowed colors:
    # b: blue
    # g: green
    # r: red
    # c: cyan
    # m: magenta
    # y: yellow
    # k: black
    # w: white

    if style == 'points':
        markersize = 5
        markertype = 'o'
    elif style == 'line':
        markersize = 1
        markertype = '-'
        
    # add data to plots 
    axXZ.plot(data.x, data.z, markertype, markersize=markersize, color=color)
    axYZ.plot(data.y, data.z, markertype, markersize=markersize, color=color)
    axXY.plot(data.x, data.y, markertype, markersize=markersize, color=color)
    ax3D.plot(data.x.values, data.y.values, data.z.values, markertype, markersize=markersize, color=color)
        
    #ax3D.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
        

