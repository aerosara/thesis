# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%reset
%pylab
%pdb off

# Can do "%pylab" or "%pylab inline"

# Cheat sheet:
# np.array([v1, v2])
# np.array([rVec[0], iVec[0], cVec[0]])  # makes a 3x3 matrix
# np.linspace(v1, v2, numPoints)
# np.concatenate(( a1, a2 ))
# print shape(waypoints.RIC)

# <headingcell level=3>

# Import libraries

# <codecell>


import numpy as np
import pandas as pd
import scipy.integrate as integrate

from IPython.display import display
from IPython.core.display import HTML

import thesis_functions.utilities
from thesis_functions.initial_conditions import initial_condition_sets
from thesis_functions.visualization import CreatePlotGrid, SetPlotGridData, ConfigurePlotLegend
from thesis_functions.astro import FindOrbitCenter, ComputeLibrationPoints, stop_yEquals0, stop_zEquals0
from thesis_functions.astro import ComputeNonlinearDerivs, ComputeRelmoDynamicsMatrix
from thesis_functions.astro import odeintNonlinearDerivs, odeintNonlinearDerivsWithLinearRelmoSTM, odeintNonlinearDerivsWithLinearRelmo
from thesis_functions.astro import ComputeRequiredVelocity, PropagateSatelliteAndChaser, TargetRequiredVelocity 
from thesis_functions.astro import PropagateSatellite, ComputeOffsets, ConvertOffset, BuildRICFrame, BuildVNBFrame
from thesis_functions.astro import BuildRICFrames, BuildVNBFrames, ConvertOffsets

from thesis_functions.major_simulation_components import set_up_target, compute_RLP_properties, plot_full_orbit, define_waypoints_RIC
from thesis_functions.major_simulation_components import convert_waypoints_RLP_VNB, travel_waypoints, compute_waypoint_metrics

# <headingcell level=3>

# Set up plots

# <codecell>


# Create plots

# Allowed colors:
# b: blue
# g: green
# r: red
# c: cyan
# m: magenta
# y: yellow
# k: black
# w: white

# Allowed axis modes: 'auto' and 'equal'
axis_mode = 'auto'

# Plots of offset in RLP, RIC, VNB frames
axis_array_RLP = CreatePlotGrid('Offset between Satellites 1 and 2 in RLP Frame', 'X', 'Y', 'Z', axis_mode)
axis_array_RIC = CreatePlotGrid('Offset between Satellites 1 and 2 in RIC Frame', 'R', 'I', 'C', axis_mode)
axis_array_VNB = CreatePlotGrid('Offset between Satellites 1 and 2 in VNB Frame', 'V', 'N', 'B', axis_mode)

# <codecell>


def run_waypoint_traveler(halo, clock_angle, approach, timescale, spacing):
    
    target_initial_state, period, mu = set_up_target(halo, clock_angle, initial_condition_sets)
    
    RLP_properties = compute_RLP_properties(target_initial_state, mu)
    
    plot_full_orbit(target_initial_state, RLP_properties, period, mu)
    
    print RLP_properties
    
    waypoints = define_waypoints_RIC(approach, spacing, timescale, RLP_properties, axis_array_RIC)
    
    waypoints = convert_waypoints_RLP_VNB(target_initial_state, waypoints, RLP_properties, axis_array_RLP, axis_array_VNB)
    
    # set_up_plots()
    
    waypoints, waypoint_velocities = travel_waypoints(target_initial_state, waypoints, RLP_properties, axis_array_RLP, axis_array_RIC, axis_array_VNB)
    
    waypoint_metrics = compute_waypoint_metrics(halo, clock_angle, approach, timescale, spacing, waypoints, waypoint_velocities, RLP_properties)

    # Period of libration point orbit (in nondimensional time units)
    #print 'Period of libration point orbit in seconds', period*RLP_properties.time_const

    #print 'waypoints.RLP_achieved_analytic_nonlin', display(HTML(waypoints.RLP_achieved_analytic_nonlin.to_html()))
    #print 'waypoints.RLP_achieved_targeted_nonlin', display(HTML(waypoints.RLP_achieved_targeted_nonlin.to_html()))

    return waypoint_metrics

# <headingcell level=3>

# Test Case Inputs

# <codecell>


#halo_cases = ['small', 'medium', 'large', 'greater']
halo_cases = ['EM']
#halo_cases = ['small']
#clock_angles = np.arange(0.0, 360.0, 10.0)
#clock_angles = np.arange(0.0, 360.0, 30.0)
clock_angles = np.array([0.0])

# not used yet:
approach_cases = ['+R', '-R', '+I', '-I', '+C', '-C']
timescales = ['fast', 'medium', 'slow']
spacings = ['close', 'medium', 'far']

halo = halo_cases[0]
clock_angle = clock_angles[0]
approach = approach_cases[0]
timescale = timescales[0]
spacing = spacings[0]

print halo, clock_angle, approach, timescale, spacing

# <codecell>


#run_waypoint_traveler(halo, clock_angle, approach, timescale, spacing)


#results = pd.DataFrame(...)
#for halo, clock_angle, approach, timescale, spacing in configuration:

for halo in halo_cases:
    for clock_angle in clock_angles:

        current_results = run_waypoint_traveler(halo, clock_angle, approach, timescale, spacing)    
#    #results = results.append(current_results)

# <codecell>


# <rawcell>

# 
# results.loc[(halo, clock_angle, approach, timescale, spacing)]
# 
# waypoint_metrics.groupby(["approach", "halo"]).apply(lambda x: pd.DataFrame({"foo": [x.DV_analytic.sum()]}))
# 
# df = pd.DataFrame({"bar": [2,4,6,8]})
# 
# foo = conditions.groupby("halo").apply(run_simulation)
# results = conditions.apply(run_simulation)

