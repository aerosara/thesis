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
#from pycse import odelay
#from IPython.html.widgets import interact, interactive
#from IPython.display import clear_output, display, HTML

import thesis_functions.utilities
from thesis_functions.initial_conditions import initial_condition_sets
from thesis_functions.visualization import CreatePlotGrid, SetPlotGridData
from thesis_functions.astro import FindOrbitCenter, ComputeLibrationPoints, stop_yEquals0, stop_zEquals0
from thesis_functions.astro import ComputeNonlinearDerivs, ComputeRelmoDynamicsMatrix
from thesis_functions.astro import odeintNonlinearDerivs, odeintNonlinearDerivsWithLinearRelmoSTM, odeintNonlinearDerivsWithLinearRelmo
from thesis_functions.astro import ComputeRequiredVelocity, PropagateSatelliteAndChaser
from thesis_functions.astro import PropagateSatellite, ComputeOffsets, ConvertOffset, BuildRICFrame, BuildVNBFrame

# <headingcell level=3>

# Initial Conditions

# <codecell>


# Set initial conditions for the target satellite 

# The initial condition DataFrame contains initial conditions from Barbee, Howell, and Sharp

# Barbee's initial conditions are a planar (Lyapunov) orbit at Earth/Moon L1

# Each initial_condition_set has attributes: author, test_case, mu, x, z, y_dot, t
initial_condition_set = initial_condition_sets.loc["Barbee", 1]

mu = initial_condition_set.mu

# target_initial_state is passed to the odeint functions, so it needs to have exactly these 6 elements in the correct order
target_initial_state = pd.Series({
    'x':     initial_condition_set.x,
    'y':     0.0,
    'z':     initial_condition_set.z,
    'x_dot': 0.0,
    'y_dot': initial_condition_set.y_dot,
    'z_dot': 0.0})

# reassign so that the series maintains the required order for its values
target_initial_state = target_initial_state[['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']]

# TODO: start target satellite from different points along its orbit.  
#       Look at how delta-V changes; also maybe linear relmo will be a better approximation along other parts of the orbit.

initial_condition_sets
# initial_condition_sets.loc['Barbee']
# initial_condition_sets.loc['Barbee', 1].x
#print display(HTML(initial_condition_sets.to_html()))

# <codecell>


# RLP_properties will have attributes: X1, X2, L1, L2, L3, L4, L5
# X1 and X2 are positions of larger and smaller bodies along X axis
RLP_properties = ComputeLibrationPoints(mu)

# The FindOrbitCenter function doesn't work if you only propagate a partial orbit, so just treat L1 as the center
center = RLP_properties.L1

# In nondimensional units, r12 = 1, M = 1, timeConst = Period/(2pi) = 1, G = 1
RLP_properties['m1']  = 5.97219e24        # Earth (kg)
RLP_properties['m2']  = 7.34767309e22     # Moon (kg)
RLP_properties['G']   = 6.67384e-11/1e9   # m3/(kg*s^2) >> converted to km3
RLP_properties['r12'] = 384400.0          # km
RLP_properties['M']   = RLP_properties.m1 + RLP_properties.m2

# This is how you convert between dimensional time (seconds) and non-dimensional time 
RLP_properties['time_const'] = RLP_properties.r12**(1.5) / (RLP_properties.G * RLP_properties.M)**(0.5) # (units are seconds)

# Period in seconds of Moon around Earth
RLP_properties['T'] = 2.0 * np.pi * RLP_properties.time_const   

# Period of libration point orbit (in nondimensional time units)
period = initial_condition_set.t  
print 'Period of libration point orbit in seconds', period*RLP_properties.time_const

RLP_properties

# <headingcell level=3>

# Define Waypoints

# <codecell>


# TODO: input waypoints in any frame (RLP, RIC, or VNB)
# TODO: get decent test cases in the Sun-Earth-Moon frame
# TODO: report/plot position error at each waypoint
# TODO: report/plot delta-V at each waypoint

# TODO: look at waypoints with different spacing, different approach directions, different amount of time between points
# TODO: would be nice to have a function that generically converts waypoints between frames (e.g. arguments = WaypointDictionary, inputframe, outputframe)

# Create a collection of waypoints which we initially populate in RIC coordinates
waypoint_RIC_coordinates = np.array([[0.0, 1000.0/RLP_properties.r12, 0.0],
                                    [0.0,  275.0/RLP_properties.r12, 0.0],   # move 725 km  # 400% errors
                                    [0.0,  180.0/RLP_properties.r12, 0.0],   # move 95 km  # 400% errors
                                    [0.0,  100.0/RLP_properties.r12, 0.0],   # 40% errors
                                    [0.0,   15.0/RLP_properties.r12, 0.0],   # 8% errors
                                    [0.0,    5.0/RLP_properties.r12, 0.0],   # 10% errors
                                    [0.0,    1.0/RLP_properties.r12, 0.0],
                                    [0.0,   0.03/RLP_properties.r12, 0.0],
                                    [0.0,    0.0/RLP_properties.r12, 0.0]])

# Time points
waypoint_times = np.array([0.0, 2.88, 4.70, 5.31, 5.67, 6.03, 6.64, 7.0, 7.26])*86400.0/RLP_properties.time_const

# Create data panel which will hold the waypoints in RIC, RLP, and VNB frames, indexed by time
waypoints = pd.Panel(items = ['RIC', 'RLP', 'VNB', 
                              'RIC_achieved', 'RLP_achieved', 'VNB_achieved'],
                     major_axis = waypoint_times, # time points
                     minor_axis = list('xyz'))    # coordinate labels

# Copy the RIC waypoint data into the panel
waypoints['RIC'] = waypoint_RIC_coordinates

waypoints.RIC

# <headingcell level=3>

# Convert Waypoints from RIC to RLP and VNB

# <codecell>


## Create a set of waypoint intervals.
waypoint_time_intervals = zip(waypoints.major_axis[:-1], waypoints.major_axis[1:])
#waypoint_time_intervals = zip(waypoints.major_axis[3:4], waypoints.major_axis[4:5])  # Starting at 4th point for now


## Convert the first waypoint to RLP and VNB

# these matrices convert from RLP coordinates to the RIC  and VNB frames at the timestamp of the first point
RLPtoRIC = BuildRICFrame(target_initial_state, center)
RLPtoVNB = BuildVNBFrame(target_initial_state, center)

# this matrix converts from RIC to RLP at the timestamp of the first point
RICtoRLP = np.linalg.inv(RLPtoRIC)

# Calculate the waypoint in the RLP and VNB frames and store it
waypoints.RLP.iloc[0] = ConvertOffset(waypoints.RIC.iloc[0], RICtoRLP)
waypoints.VNB.iloc[0] = ConvertOffset(waypoints.RLP.iloc[0], RLPtoVNB)


## Convert the remaining waypoints to RLP and VNB

target_initial_state_for_segment = target_initial_state.copy()

for start, end in waypoint_time_intervals:

    print 'percentage of orbit covered getting to next point (by time):', (end - start)/period*100.0
    
    # array of time points
    timespan_for_segment = np.linspace(start, end, 500)

    # Build an ephem for the given timespan up to the next waypoint.
    # target_ephem_for_segment is a DataFrame with attributes x, y, z, x_dot, y_dot, z_dot and is indexed by timespan_for_segment
    # this is the target satellite position and velocity over time in RLP frame from integrating initial state with full nonlinear dynamics
    target_ephem_for_segment = PropagateSatellite(mu, timespan_for_segment, target_initial_state_for_segment)
    
    # Get the target satellite state at the end of the current segment
    target_state_at_endpoint = target_ephem_for_segment.iloc[-1]

    # Build RIC and VNB frames
    # these matrices convert from RLP coordinates to the RIC  and VNB frames at the timestamp of the next point
    RLPtoRIC = BuildRICFrame(target_state_at_endpoint, center)
    RLPtoVNB = BuildVNBFrame(target_state_at_endpoint, center)
    
    # this matrix converts from RIC to RLP at the timestamp of the next point
    RICtoRLP = np.linalg.inv(RLPtoRIC)
    
    # Calculate the waypoint in the RLP and VNB frames and store it
    waypoints.RLP.loc[end] = ConvertOffset(waypoints.RIC.loc[end], RICtoRLP)
    waypoints.VNB.loc[end] = ConvertOffset(waypoints.RLP.loc[end], RLPtoVNB)
    
    # Reset the state as the last entry in the ephem.
    target_initial_state_for_segment = target_ephem_for_segment.irow(-1)  # TODO: need to use .copy here?  or anywhere else?
    
print display(HTML(waypoints.RLP.to_html()))
print display(HTML(waypoints.VNB.to_html()))

# <headingcell level=3>

# Set up plots

# <rawcell>

# 
# # Create plots
# 
# # Allowed colors:
# # b: blue
# # g: green
# # r: red
# # c: cyan
# # m: magenta
# # y: yellow
# # k: black
# # w: white
# 
# #fig1 = plt.figure()
# #fig2 = plt.figure()
# #ax1 = fig1.add_subplot(111)
# #ax2 = fig2.add_subplot(111)
# #ax1.set_title('dx_LINEAR vs timespan')
# #ax2.set_title('Difference between LINEAR and NONLINEAR: dy vs dx')
# 
# # Plots of offset in RLP, RIC, VNB frames
# axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP = CreatePlotGrid('Offset between Satellites 1 and 2 in RLP Frame', 'X', 'Y', 'Z', 'auto')
# axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC = CreatePlotGrid('Offset between Satellites 1 and 2 in RIC Frame', 'R', 'I', 'C', 'auto')
# axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB = CreatePlotGrid('Offset between Satellites 1 and 2 in VNB Frame', 'V', 'N', 'B', 'auto')
# 
# # add zero point to plots (this is location of target satellite)
# points = {}
# data = {}
# points['zero'] = {'xyz':[0,0,0], 'color':'k'}
# SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, data, points)
# SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, data, points)
# SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, data, points)
# points = {}
# 
# # add all waypoints to RLP, RIC, and VNB plots
# for w in Waypoints:
#     points['w'] = {'xyz':np.array(Waypoints[w]['r_RLP'])*r12, 'color':'c'}
#     SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, data, points)
#     
#     points['w'] = {'xyz':np.array(Waypoints[w]['r_RIC'])*r12, 'color':'c'}
#     SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, data, points)
#     
#     points['w'] = {'xyz':np.array(Waypoints[w]['r_VNB'])*r12, 'color':'c'}
#     SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, data, points)    
# 
# points = {}

# <headingcell level=3>

# Travel between waypoints

# <codecell>

#print target_initial_state
#print waypoint_time_intervals[0][0]
#print waypoints.RLP.loc[waypoint_time_intervals[0][0]]
#print target_initial_state[['x_dot', 'y_dot', 'z_dot']]
print waypoints.RLP

# <codecell>


target_initial_state_for_segment = target_initial_state.copy()
chaser_initial_state_relative = pd.Series(index = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot'])

# create Panel for waypoint velocities
waypoint_velocities = pd.Panel(items = ['RLP_pre_maneuver_absolute', 'RLP_post_maneuver_absolute',
                                        'RLP_pre_maneuver_relative', 'RLP_post_maneuver_relative',
                                        'RLP_delta_v'],
                     major_axis = waypoint_times, # time points
                     minor_axis = ['x_dot', 'y_dot', 'z_dot'])    # coordinate labels

# assume starts exactly from first waypoint with same velocity as target satellite (for lack of any better velocity values at this point)
waypoints.RLP_achieved.iloc[0] = waypoints.RLP.iloc[0]
waypoint_velocities.RLP_pre_maneuver_absolute.iloc[0] = target_initial_state[['x_dot', 'y_dot', 'z_dot']]

# Travel between waypoints
for start, end in waypoint_time_intervals:
    
    # Pull out the RLP vector of the current and next waypoint
    current_waypoint = waypoints.RLP_achieved.loc[start]
    next_waypoint = waypoints.RLP.loc[end]
    
    ## Compute required velocity to travel between waypoints

    # Compute required velocity at point 1 to take us to point 2 within time (t2-t1)
    # This is from Lian et al.
    # Method signature:
    # initialRelativeVelocity = ComputeRequiredVelocity(initialState1ForSegment, initialRelativePosition, initialTime, targetRelativePosition, targetTime)
    chaser_initial_velocity_relative = ComputeRequiredVelocity(target_initial_state_for_segment, current_waypoint, start, next_waypoint, end, mu)

    waypoint_velocities.RLP_post_maneuver_relative.loc[start] = chaser_initial_velocity_relative
    
    #print 'initial chaser relative velocity', chaser_initial_velocity_relative

    chaser_initial_state_relative[['x', 'y', 'z']] = current_waypoint
    chaser_initial_state_relative[['x_dot', 'y_dot', 'z_dot']] = chaser_initial_velocity_relative

    ## Integrate first satellite with full nonlinear dynamics and second satellite with linear relmo dynamics

    # array of time points
    timespan_for_segment = np.linspace(start, end, 500)

    # compute target satellite position and velocity over time in RLP frame by integrating initial state with full nonlinear dynamics
    # compute offset between target and chaser satellite over time in RLP frame by integrating initial offset with linearized relmo dynamics
    target_ephem_for_segment, chaser_offset_linear_RLP = PropagateSatelliteAndChaser(mu, timespan_for_segment, target_initial_state_for_segment, chaser_initial_state_relative)

    ##  Integrate second satellite with full nonlinear dynamics

    # initial state of second satellite in absolute RLP coordinates (not relative to first satellite)
    #chaser_initial_state_absolute_for_segment = np.array(target_initial_state_for_segment) - np.array(chaser_initial_state_relative) # remove this if code runs ok with version on next line
    chaser_initial_state_absolute_for_segment = target_initial_state_for_segment - chaser_initial_state_relative

    # compute chaser satellite position and velocity over time in RLP frame by integrating initial state with full nonlinear dynamics
    chaser_ephem_for_segment = PropagateSatellite(mu, timespan_for_segment, chaser_initial_state_absolute_for_segment);
    
    # Compute offsets in RLP frame based on nonlinear motion
    chaser_offset_nonlin_RLP = ComputeOffsets(timespan_for_segment, target_ephem_for_segment, chaser_ephem_for_segment);
    
    offsets_linear_RIC = pd.DataFrame([['x', 'y', 'z']], index=timespan_for_segment)
    offsets_nonlin_RIC = pd.DataFrame([['x', 'y', 'z']], index=timespan_for_segment)
    offsets_linear_VNB = pd.DataFrame([['x', 'y', 'z']], index=timespan_for_segment)
    offsets_nonlin_VNB = pd.DataFrame([['x', 'y', 'z']], index=timespan_for_segment)
    
    ## Offsets in RIC and VNB
    for t in timespan_for_segment:
        
        # Build RIC and VNB frames
        RLPtoRIC = BuildRICFrame(target_ephem_for_segment.loc[t], center)
        RLPtoVNB = BuildVNBFrame(target_ephem_for_segment.loc[t], center)

        # Compute offsets in RIC frame
        offsets_linear_RIC.loc[t] = ConvertOffset(chaser_offset_linear_RLP.loc[t, ['x', 'y', 'z']], RLPtoRIC);
        offsets_nonlin_RIC.loc[t] = ConvertOffset(chaser_offset_nonlin_RLP.loc[t, ['x', 'y', 'z']], RLPtoRIC);

        # Compute offsets in VNB frame
        offsets_linear_VNB.loc[t] = ConvertOffset(chaser_offset_linear_RLP.loc[t, ['x', 'y', 'z']], RLPtoVNB);
        offsets_nonlin_VNB.loc[t] = ConvertOffset(chaser_offset_nonlin_RLP.loc[t, ['x', 'y', 'z']], RLPtoVNB);   
    
    # Build RIC and VNB frames
    RLPtoRIC = BuildRICFrame(target_ephem_for_segment.loc[end], center)
    RLPtoVNB = BuildVNBFrame(target_ephem_for_segment.loc[end], center)

    ## Compute delta-V
    
    # post-maneuver velocity at current waypoint
    waypoint_velocities.RLP_post_maneuver_absolute.loc[start] = chaser_ephem_for_segment.loc[start, ['x_dot', 'y_dot', 'z_dot']]
    
    # compute delta-V executed at current waypoint
    waypoint_velocities.RLP_delta_v.loc[start] = waypoint_velocities.RLP_post_maneuver_absolute.loc[start] - waypoint_velocities.RLP_pre_maneuver_absolute.loc[start]
    
    # pre-maneuver velocity for next waypoint (end of current propagation segment)
    waypoint_velocities.RLP_pre_maneuver_absolute.loc[end] = chaser_ephem_for_segment.loc[end, ['x_dot', 'y_dot', 'z_dot']]
    
    # TODO: also compute the delta-V based only on the linear relmo propagation and compare the delta-V to the nonlinear one currently being computed
    #      (this means we would need to propagate forward from the nominal waypoint instead of only propagating forward from the achieved waypoint)
    # pre-maneuver relative velocity when arriving at next waypoint, based on linear propagation
    #Waypoints[nextPoint]['v_RLP_pre_LINEAR'] = np.array([ dxdot_LINEAR[-1], dydot_LINEAR[-1], dzdot_LINEAR[-1] ])

    ## Output that gets fed into next iteration/segment
    
    # Record updated primary satellite initial state for next segment
    target_initial_state_for_segment = target_ephem_for_segment.loc[end, ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']]
    
    # Record updated/achieved chaser satellite waypoint for next segment
    waypoints.RLP_achieved.loc[end] = chaser_offset_nonlin_RLP.loc[end, ['x', 'y', 'z']]
    
    # compute updated/achieved waypoint location in RIC and VNB
    waypoints.RIC_achieved.loc[end] = ConvertOffset(waypoints.RLP_achieved.loc[end], RLPtoRIC)
    waypoints.VNB_achieved.loc[end] = ConvertOffset(waypoints.RLP_achieved.loc[end], RLPtoVNB)
    
    ## VISUALIZATIONS

    #ax1.plot(timespan, dx_LINEAR*r12)

    # Compare linear relmo propagation to nonlinear dynamics
    #ax2.plot((dx_NONLIN - dx_LINEAR)/np.amax(np.absolute(dx_LINEAR))*100.0, (dy_NONLIN - dy_LINEAR)/np.amax(np.absolute(dy_LINEAR))*100.0)
    
    # create data dictionaries
    #dataoffsetRLP = {};
    #dataoffsetRLP['linear'] = {'x':dx_LINEAR*r12, 'y':dy_LINEAR*r12, 'z':dz_LINEAR*r12, 'color':'g'}
    #dataoffsetRLP['nonlin'] = {'x':dx_NONLIN*r12, 'y':dy_NONLIN*r12, 'z':dz_NONLIN*r12, 'color':'r'}
    
    #dataoffsetRIC = {};
    #dataoffsetRIC['linear'] = {'x':dr_LINEAR*r12, 'y':di_LINEAR*r12, 'z':dc_LINEAR*r12, 'color':'g'}
    #ataoffsetRIC['nonlin'] = {'x':dr_NONLIN*r12, 'y':di_NONLIN*r12, 'z':dc_NONLIN*r12, 'color':'r'}
    
    #dataoffsetVNB = {};
    #dataoffsetVNB['linear'] = {'x':dv_LINEAR*r12, 'y':dn_LINEAR*r12, 'z':db_LINEAR*r12, 'color':'g'}
    #dataoffsetVNB['nonlin'] = {'x':dv_NONLIN*r12, 'y':dn_NONLIN*r12, 'z':db_NONLIN*r12, 'color':'r'}

    # Plot offset (relative motion) between satellites 1 and 2 in RLP frame and add achieved waypoint (end of current segment) to plot
    #points[nextPoint] = {'xyz':np.array(Waypoints[nextPoint]['r_RLP_achieved'])*r12, 'color':'m'}
    #SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, dataoffsetRLP, points)
    
    # Plot offset (relative motion) between satellites 1 and 2 in RIC frame and add achieved waypoint (start and end of current segment) to plot
    #points[nextPoint] = {'xyz':np.array(Waypoints[nextPoint]['r_RIC_achieved'])*r12, 'color':'m'}
    #SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, dataoffsetRIC, points)
    
    # Plot offset (relative motion) between satellites 1 and 2 in VNB frame and add achieved waypoint (start and end of current segment) to plot
    #points[nextPoint] = {'xyz':np.array(Waypoints[nextPoint]['r_VNB_achieved'])*r12, 'color':'m'}
    #SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, dataoffsetVNB, points)
    
    #points = {}
    
## final delta-V

# final post-maneuver velocity is same as the target satellite's velocity
waypoint_velocities.RLP_post_maneuver_absolute.loc[end] = target_ephem_for_segment.loc[end, ['x_dot', 'y_dot', 'z_dot']]

# compute final delta-V
waypoint_velocities.RLP_delta_v.loc[end] = waypoint_velocities.RLP_post_maneuver_absolute.loc[end] - waypoint_velocities.RLP_pre_maneuver_absolute.loc[end]

# <codecell>

waypoints.RIC_achieved
waypoints.RLP_achieved
waypoint_velocities.RLP_post_maneuver_absolute
chaser_ephem_for_segment.loc[start, ['x_dot', 'y_dot', 'z_dot']]
chaser_offset_linear_RLP
chaser_offset_nonlin_RLP
offsets_linear_RIC

# <codecell>


waypoint_delta_v = pd.Series(index=waypoint_times)

# compute delta-V magnitude and report to screen
for t in waypoint_times:
    waypoint_delta_v.loc[t] = np.linalg.norm(waypoint_velocities.RLP_delta_v.loc[t], 2)*RLP_properties.r12/RLP_properties.time_const*1000.0  # m/s
    
waypoint_delta_v

# <headingcell level=3>

# Unused version of waypoint traveler - stores cumulative ephems

# <rawcell>

# 
# # Create a panel that represents the ephem of each satellite.
# #t = 0.0
# #target_ephem_cumulative = pd.Panel(items=["RLP"], major_axis=[t], minor_axis=["x", "y", "z", "x_dot", "y_dot", "z_dot"])
# #chaser_ephem_cumulative = target_ephem_cumulative.copy()
# 
# # Configure the initial states of each ephem.
# #target_ephem_cumulative.loc["RLP", t] = target_initial_state.values
# 
# # assume the chaser starts exactly from first waypoint with same velocity as the target satellite (for lack of any better velocity values at this point)
# #Waypoints[0]['r_RLP_achieved'] = Waypoints[0]['r_RLP']
# #Waypoints[0]['v_RLP_abs_premaneuver'] = initialstate1[3:6]
# waypoints.RLP_achieved.iloc[0] = waypoints.RLP.iloc[0]
# waypoints.PreManeuverVelocity.iloc[0] = target_initial_state['x_dot', 'y_dot', 'z_dot']
# 
# #chaser_ephem_cumulative.loc["RLP_LINEAR", t, ["x", "y", "z"]] = waypoints.RLP.iloc[0]
# #chaser_ephem_cumulative.loc["RLP", t, ["x", "y", "z"]] = waypoints.RLP.iloc[0]
# #chaser_ephem_cumulative.loc["RLP_LINEAR", t, ["x", "y", "z"]] = waypoints.RLP.loc[waypoint_time_intervals[0][0]]  # use waypoint corresponding to first time in waypoint_time_intervals
# #chaser_ephem_cumulative.loc["RLP", t, ["x", "y", "z"]] = waypoints.RLP.loc[waypoint_time_intervals[0][0]]
# 
# # Next, simulate the two spacecraft for each waypoint interval.
# for start, end in waypoint_time_intervals:
# 
#     # array of time points
#     timespan_for_segment = np.linspace(start, end, 500)
#     
#     target_initial_state_for_segment = target_ephem_cumulative.loc["RLP", start, ["x", "y", "z"]]
# 
#     # Pull out the RLP vector of the current and next waypoint
#     current_waypoint = waypoints.RLP_achieved.loc[start]
#     next_waypoint = waypoints.RLP.loc[end]
# 
#     chaser_initial_state_for_segment = chaser_ephem_cumulative.loc["RLP", start, ["x", "y", "z"]]
# 
#     ## Compute required velocity to travel between waypoints
# 
#     # Compute required velocity at point 1 to take us to point 2 within time (t2-t1)
#     # This is from Lian et al.
#     # Method signature:
#     # initialRelativeVelocity = ComputeRequiredVelocity(initialState1ForSegment, initialRelativePosition, initialTime, targetRelativePosition, targetTime)
#     #Waypoints[currentPoint]['v_RLP'] = ComputeRequiredVelocity(initialState1ForSegment, Waypoints[currentPoint]['r_RLP_achieved'], Waypoints[currentPoint]['t'], Waypoints[nextPoint]['r_RLP'], Waypoints[nextPoint]['t'], mu)
#     required_relative_velocity = ComputeRequiredVelocity(target_initial_state_for_segment, chaser_initial_state_for_segment, start, next_waypoint, end, mu)
# 
#     waypoints.PostManeuverVelocity.loc[start] = required_relative_velocity
#     
#     #initialRelativeState = np.concatenate(( Waypoints[currentPoint]['r_RLP_achieved'], Waypoints[currentPoint]['v_RLP'] ))
# 
#     # Calculate required velocity.
#     #required_velocity = required_relative_velocity - target_ephem_cumulative.loc["rlp", start, ["x_dot", "y_dot", "z_dot"]]
# 
#     # Store the required velocity.
#     #chaser_ephem_cumulative.loc["RLP", start, ["x_dot", "y_dot", "z_dot"]] = required_velocity
# 
#     # Calculate the relative state between the two spacecraft.
#     initial_relative_state = target_ephem_cumulative.loc["RLP", start] - chaser_ephem_cumulative.loc["rlp", start]
# 
#     ## Integrate first satellite with full nonlinear dynamics and second satellite with linear relmo dynamics
#     
#     # Propagate the target spacecraft using nonlinear dynamics and generate linear offset.
#     # compute target satellite position and velocity over time in RLP frame by integrating initial state with full nonlinear dynamics
#     # compute offset between target and chaser satellite over time in RLP frame by integrating initial offset with linearized relmo dynamics
#     target_ephem_for_segment, linear_offset_for_segment = PropagateSatelliteAndChaser(mu, timespan_for_segment, target_ephem_cumulative.loc["RLP", start], initial_relative_state)
# 
#     # Propagate the chaser spacecraft using nonlinear dynamics.
#     chaser_ephem_for_segment = PropagateSatellite(mu, timespan_for_segment, chaser_ephem_cumulative.loc["RLP", start])
#     
#     waypoints.RLP_achieved.loc[end] = chaser_ephem_for_segment.loc['RLP', end, ['x', 'y', 'z']]
#     
#     waypoints.PreManeuverVelocity.loc[end] = chaser_ephem_for_segment.loc['RLP', end, ['x_dot', 'y_dot', 'z_dot']]
#     
#     
# 
#     # We need to re-index our ephems. Boo.
#     target_ephem_cumulative = target_ephem_cumulative.reindex(major_axis=np.unique(np.concatenate((target_ephem_cumulative.major_axis.values, time))))
#     chaser_ephem_cumulative = chaser_ephem_cumulative.reindex(major_axis=np.unique(np.concatenate((chaser_ephem_cumulative.major_axis.values, time))))
# 
#     # Store the ephems.
#     target_ephem_cumulative.loc["RLP", timespan_for_segment] = target_ephem_for_segment.values
#     chaser_ephem_cumulative.loc["RLP", timespan_for_segment] = chaser_ephem_for_segment.values
#     chaser_ephem_cumulative.loc["RLP_LINEAR", timespan_for_segment] = (target_ephem_for_segment + linear_offset_for_segment).values

# <codecell>


