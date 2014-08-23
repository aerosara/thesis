# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def PlotGrid(title, xlabel, ylabel, zlabel, data, points, aspectmode):
    
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
    
    # add points to plots
    for key in points:
        axXZ.plot(points[key][0], points[key][2], 'o', markersize=5, label=key)
        axYZ.plot(points[key][1], points[key][2], 'o', markersize=5, label=key)
        axXY.plot(points[key][0], points[key][1], 'o', markersize=5, label=key)
        ax3D.plot([points[key][0]], [points[key][1]], [points[key][2]], 'o', markersize=5, label=key)
        
    # add data to plots
    for key in data:
        axXZ.plot(data[key]['x'], data[key]['z'], '-', label=key)
        axYZ.plot(data[key]['y'], data[key]['z'], '-', label=key)
        axXY.plot(data[key]['x'], data[key]['y'], '-', label=key)
        ax3D.plot(data[key]['x'], data[key]['y'], data[key]['z'], '-', label=key)
        
    ax3D.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
        

