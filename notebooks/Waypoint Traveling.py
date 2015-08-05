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
import matplotlib.pyplot as plt
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

from thesis_functions.major_simulation_components import set_up_target, compute_RLP_properties, set_active_point, plot_full_orbit, plot_initial_condition, define_waypoints_RIC
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

# create plots showing the target satellite in one full orbit
axis_array_RLP_absolute = CreatePlotGrid('Satellite 1 Orbit in RLP Frame', 'X', 'Y', 'Z', 'equal')
    

# <codecell>


first = True 

def run_waypoint_traveler(halo, clock_angle, approach, timescale, spacing):
    
    target_initial_state, period, mu = set_up_target(halo, clock_angle, initial_condition_sets, axis_array_RLP, axis_array_RIC, axis_array_VNB)
    
    RLP_properties = compute_RLP_properties(mu)
    
    set_active_point(target_initial_state, RLP_properties)
    
    if (first == True):
        plot_full_orbit(target_initial_state, RLP_properties, period, mu, axis_array_RLP_absolute)
    
    plot_initial_condition(target_initial_state, RLP_properties, axis_array_RLP_absolute)
    
    #print RLP_properties
    
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
approach_cases = ['+R', '-R', '+I', '-I', '+C', '-C']

# not used yet:
timescales = ['fast', 'medium', 'slow']
spacings = ['close', 'medium', 'far']

# Used for first set of results in paper:
#halo_cases = ['EM']
#clock_angles = np.array([0.0])

# Used for second set of results in paper:
halo_cases = ['EM']
clock_angles = np.arange(0.0, 360.0, 1.0)

halo = halo_cases[0]
clock_angle = clock_angles[0]
approach = '+I'
timescale = timescales[0]
spacing = spacings[0]

print halo, clock_angle, approach, timescale, spacing

# <codecell>


#run_waypoint_traveler(halo, clock_angle, approach, timescale, spacing)


#results = pd.DataFrame(...)
#for halo, clock_angle, approach, timescale, spacing in configuration:


summary_metrics = pd.DataFrame({'halo':           len(clock_angles),
                             'clock_angle':    len(clock_angles),
                             'approach':       len(clock_angles),
                             'timescale':      len(clock_angles),
                             'spacing':        len(clock_angles),
                             'sum_DV_targeted': len(clock_angles),
                             'sum_DV_analytic': len(clock_angles),
                             'sum_DV_magnitude_difference': len(clock_angles),
                             'sum_DV_angle_difference': len(clock_angles),
                             'sum_achieved_position_error_analytic': len(clock_angles),
                             'sum_achieved_position_error_targeted': len(clock_angles)},
                            index=[clock_angles])

for halo in halo_cases:
    for clock_angle in clock_angles:

        current_results = run_waypoint_traveler(halo, clock_angle, approach, timescale, spacing)   
        
        # compute and record summary metrics
        summary_metrics.halo.loc[clock_angle] = halo
        summary_metrics.clock_angle.loc[clock_angle] = clock_angle
        summary_metrics.approach.loc[clock_angle] = approach
        summary_metrics.timescale.loc[clock_angle] = timescale
        summary_metrics.spacing.loc[clock_angle] = spacing
        summary_metrics.sum_DV_targeted.loc[clock_angle] = current_results.DV_targeted.sum()
        summary_metrics.sum_DV_analytic.loc[clock_angle] = current_results.DV_analytic.sum()
        summary_metrics.sum_DV_magnitude_difference.loc[clock_angle] = current_results.DV_magnitude_difference.abs().sum()
        summary_metrics.sum_DV_angle_difference.loc[clock_angle] = current_results.DV_angle_difference.sum()
        summary_metrics.sum_achieved_position_error_analytic.loc[clock_angle] = current_results.achieved_position_error_analytic.sum()
        summary_metrics.sum_achieved_position_error_targeted.loc[clock_angle] = current_results.achieved_position_error_targeted.sum()
        
        first = False
    
print 'summary_metrics', display(HTML(summary_metrics.to_html(float_format=lambda x: '{0:.3f}'.format(x))))

#    #results = results.append(current_results)

# <codecell>


fig3, (ax3) = plt.subplots(1,1);
ax3.plot(summary_metrics.clock_angle, summary_metrics.sum_DV_analytic, label='Sum of Linear \(\Delta V\)')
ax3.plot(summary_metrics.clock_angle, summary_metrics.sum_DV_targeted, label='Sum of Targeted \(\Delta V\)')
#lims = ylim()
#ylim([0, lims[1]]) 
ax3.set_title('Total Rendezvous \(\Delta V\) vs. Clock Angle')
ax3.xaxis.set_label_text('Clock Angle (degrees)')
ax3.yaxis.set_label_text('Sum of \(\Delta V\) (m/s)')
ax3.legend(loc='upper right')

fig4, (ax4) = plt.subplots(1,1);
ax4.plot(summary_metrics.clock_angle, summary_metrics.sum_DV_magnitude_difference)
ax4.set_title('Linear-Targeted \(\Delta V\) Difference vs. Clock Angle')
ax4.xaxis.set_label_text('Clock Angle (degrees)')
ax4.yaxis.set_label_text('\(\Delta V\) Difference (m/s)')

fig5, (ax5) = plt.subplots(1,1);
ax5.plot(summary_metrics.clock_angle, summary_metrics.sum_achieved_position_error_analytic, label='Sum of Linear Position Error')
ax5.plot(summary_metrics.clock_angle, summary_metrics.sum_achieved_position_error_targeted, label='Sum of Targeted Position Error')
ax5.semilogy()
ax5.set_title('Total Rendezvous Position Error vs. Clock Angle')
ax5.xaxis.set_label_text('Clock Angle (degrees)')
ax5.yaxis.set_label_text('Sum of Position Error (log(m))')
ax5.legend(loc='upper right')

fig6, (ax6) = plt.subplots(1,1);
ax6.plot(summary_metrics.clock_angle, summary_metrics.sum_DV_angle_difference)
ax6.set_title('Linear-Targeted \(\Delta V\) Angle Difference vs. Clock Angle')
ax6.xaxis.set_label_text('Clock Angle (degrees)')
ax6.yaxis.set_label_text('\(\Delta V\) Angle Difference (deg)')

fig7, (ax7) = plt.subplots(1,1);
ax7.plot(summary_metrics.clock_angle, (summary_metrics.sum_DV_analytic-summary_metrics.sum_DV_targeted).abs()/summary_metrics.sum_DV_analytic*100)
ax7.set_title('Linear-Targeted \(\Delta V\) Difference vs. Clock Angle')
ax7.xaxis.set_label_text('Clock Angle (degrees)')
ax7.yaxis.set_label_text('\(\Delta V\) Difference (\%)')

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

