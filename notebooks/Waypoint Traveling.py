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
from thesis_functions.astro import ComputeRequiredVelocity, PropagateSatelliteAndChaser, TargetRequiredVelocity 
from thesis_functions.astro import PropagateSatellite, ComputeOffsets, ConvertOffset, BuildRICFrame, BuildVNBFrame
from thesis_functions.astro import BuildRICFrames, BuildVNBFrames, ConvertOffsets

# <headingcell level=3>

# Initial Conditions

# <codecell>


# Set initial conditions for the target satellite 

# The initial condition DataFrame contains initial conditions from Barbee, Howell, and Sharp

# Barbee's initial conditions are a planar (Lyapunov) orbit at Earth/Moon L1

# Each initial_condition_set has attributes: author, test_case, mu, x, z, y_dot, t
initial_condition_set = initial_condition_sets.loc['Barbee', 1]

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
target_initial_state = target_initial_state.loc[['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']]

# TODO: start target satellite from different points along its orbit.  
#       Look at how delta-V changes; also maybe linear relmo will be a better approximation along other parts of the orbit.

initial_condition_sets
#initial_condition_sets.loc['Barbee']

# <codecell>


# RLP_properties will have attributes: X1, X2, L1, L2, L3, L4, L5
# X1 and X2 are positions of larger and smaller bodies along X axis
RLP_properties = ComputeLibrationPoints(mu)

# The FindOrbitCenter function doesn't work if you only propagate a partial orbit, so just treat L1/L2 as the center
center = RLP_properties.L1
#center = RLP_properties.L2

system = 'Earth-Moon'
#system = 'Sun-Earth'

if (system == 'Earth-Moon'):
    
    # In nondimensional units, r12 = 1, M = 1, timeConst = Period/(2pi) = 1, G = 1
    RLP_properties['m1']  = 5.97219e24        # Earth (kg)
    RLP_properties['m2']  = 7.34767309e22     # Moon (kg)
    RLP_properties['r12'] = 384400.0          # km
    RLP_properties['G']   = 6.67384e-11/1e9   # m3/(kg*s^2) >> converted to km3
    RLP_properties['M']   = RLP_properties.m1 + RLP_properties.m2
    
elif (system == 'Sun-Earth'):
    
    # In nondimensional units, r12 = 1, M = 1, timeConst = Period/(2pi) = 1, G = 1
    RLP_properties['m1']  = 1.988435e30  #1.989e30                      # Sun (kg)
    RLP_properties['m2']  = 5.9721986e24 + 7.34767309e22    # Earth + Moon (kg)
    RLP_properties['r12'] = 149597870.7                   # km
    RLP_properties['G']   = 6.67384e-11/1e9               # m3/(kg*s^2) >> converted to km3
    RLP_properties['M']   = RLP_properties.m1 + RLP_properties.m2

#TU_SE_days = 58.1313429643148; % days = 5022548.03211679872 seconds

# This is how you convert between dimensional time (seconds) and non-dimensional time 
RLP_properties['time_const'] = RLP_properties.r12**(1.5) / (RLP_properties.G * RLP_properties.M)**(0.5) # (units are seconds)

# Period in seconds of secondary around primary
RLP_properties['T'] = 2.0 * np.pi * RLP_properties.time_const   

# Period of libration point orbit (in nondimensional time units)
period = initial_condition_set.t  
print 'Period of libration point orbit in seconds', period*RLP_properties.time_const

RLP_properties

# <headingcell level=3>

# Define Waypoints

# <codecell>

# TODO: input waypoints in any frame (RLP, RIC, or VNB)
# TODO: report/plot position error at each waypoint
# TODO: report/plot delta-V at each waypoint

# TODO: look at waypoints with different spacing, different approach directions, different amount of time between points
# TODO: would be nice to have a function that generically converts waypoints between frames (e.g. arguments = WaypointDictionary, inputframe, outputframe)

# Create a collection of waypoints which we initially populate in RIC coordinates
waypoint_RIC_coordinates = np.array([ #[0.0, 1000.0, 0.0],
                                      #[0.0,  275.0, 0.0],   # move 725 km  # 400% errors
                                      #[0.0,  180.0, 0.0],   # move 95 km  # 400% errors
                                    [0.0,  100.0, 0.0],   # 40% errors
                                    [0.0,   15.0, 0.0],   # 8% errors
                                    [0.0,    5.0, 0.0],   # 10% errors
                                    [0.0,    1.0, 0.0],
                                    [0.0,   0.03, 0.0],
                                    [0.0,    0.0, 0.0]])/RLP_properties.r12

# Time points
waypoint_times = np.array([#0.0, 2.88, 4.70, 
                           5.31, 5.67, 6.03, 6.64, 7.0, 7.26])*86400.0/RLP_properties.time_const
# Create data panel which will hold the waypoints in RIC, RLP, and VNB frames, indexed by time
waypoints = pd.Panel(items = ['RIC', 'RLP', 'VNB', 
                              'RIC_achieved_targeted_nonlin', 'RLP_achieved_targeted_nonlin', 'VNB_achieved_targeted_nonlin',
                              'RIC_achieved_analytic_nonlin', 'RLP_achieved_analytic_nonlin', 'VNB_achieved_analytic_nonlin'],
                     major_axis = waypoint_times, # time points
                     minor_axis = list('xyz'))    # coordinate labels

# Create a set of waypoint intervals
waypoint_time_intervals = zip(waypoints.major_axis[:-1], waypoints.major_axis[1:])

# Copy the RIC waypoint data into the panel
waypoints['RIC'] = waypoint_RIC_coordinates

print 'waypoints.RIC'
waypoints.RIC

# <headingcell level=3>

# Convert Waypoints from RIC to RLP and VNB

# <codecell>


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
    target_initial_state_for_segment = target_ephem_for_segment.irow(-1)
    
print 'waypoints.RLP', display(HTML(waypoints.RLP.to_html()))
print 'waypoints.VNB', display(HTML(waypoints.VNB.to_html()))

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

#fig1 = plt.figure()
#fig2 = plt.figure()
#ax1 = fig1.add_subplot(111)
#ax2 = fig2.add_subplot(111)
#ax1.set_title('dx_LINEAR vs timespan')
#ax2.set_title('Difference between LINEAR and NONLINEAR: dy vs dx')

# allowed axis modes: 'auto' and 'equal'

# Plots of offset in RLP, RIC, VNB frames
axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP = CreatePlotGrid('Offset between Satellites 1 and 2 in RLP Frame', 'X', 'Y', 'Z', 'auto')
axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC = CreatePlotGrid('Offset between Satellites 1 and 2 in RIC Frame', 'R', 'I', 'C', 'auto')
axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB = CreatePlotGrid('Offset between Satellites 1 and 2 in VNB Frame', 'V', 'N', 'B', 'auto')

# add all waypoints to RLP, RIC, and VNB plots
SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, waypoints.RLP*RLP_properties.r12, 'points', 'c')
SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, waypoints.RIC*RLP_properties.r12, 'points', 'c')
SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, waypoints.VNB*RLP_properties.r12, 'points', 'c')

# <headingcell level=3>

# Travel between waypoints

# <codecell>


# set up initial state Series objects
target_initial_state_for_segment       = target_initial_state.copy()
chaser_initial_state_relative_analytic = pd.Series(index = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']) # relative means origin is target satellite
chaser_initial_state_absolute_analytic = pd.Series(index = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']) # absolute means origin is origin of RLP frame
chaser_initial_state_absolute_targeted = pd.Series(index = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']) 
chaser_initial_state_missed_maneuver   = target_initial_state.copy()

# create Panel for waypoint velocities
# these velocities are all absolute (they are wrt the origin of the RLP frame, not wrt the target satellite)
waypoint_velocities = pd.Panel(items = ['RLP_pre_maneuver_targeted_nonlin', 'RLP_post_maneuver_targeted_nonlin', 'RLP_delta_v_targeted_nonlin',
                                        'RLP_pre_maneuver_analytic_linear', 'RLP_post_maneuver_analytic_linear', 'RLP_delta_v_analytic_linear'],
                     major_axis = waypoint_times, # time points
                     minor_axis = ['x_dot', 'y_dot', 'z_dot'])    # coordinate labels

# assume chaser starts exactly from first waypoint
waypoints.RLP_achieved_targeted_nonlin.iloc[0] = waypoints.RLP.iloc[0]
waypoints.RLP_achieved_analytic_nonlin.iloc[0] = waypoints.RLP.iloc[0]

# assume chaser starts with same velocity as target satellite (for lack of any better velocity values at this point)
# TODO: maybe haloize the chaser initial state so that its starting pre-maneuver velocity at the first waypoint is the velocity it would have in a halo orbit
waypoint_velocities.RLP_pre_maneuver_targeted_nonlin.iloc[0] = target_initial_state.loc[['x_dot', 'y_dot', 'z_dot']]
waypoint_velocities.RLP_pre_maneuver_analytic_linear.iloc[0] = target_initial_state.loc[['x_dot', 'y_dot', 'z_dot']]

# create Panel for holding offsets between target and chaser over time for the current segment
offsets = pd.Panel(items = ['RLP_analytic_linear_nominal', 'RLP_analytic_linear_achieved', 'RLP_analytic_nonlin', 'RLP_targeted_nonlin', 'RLP_missed_maneuver',
                            'RIC_analytic_linear_nominal', 'RIC_analytic_linear_achieved', 'RIC_analytic_nonlin', 'RIC_targeted_nonlin', 'RIC_missed_maneuver',
                            'VNB_analytic_linear_nominal', 'VNB_analytic_linear_achieved', 'VNB_analytic_nonlin', 'VNB_targeted_nonlin', 'VNB_missed_maneuver'],
                     major_axis = timespan_for_segment, # time points
                     minor_axis = ['x', 'y', 'z'])    # coordinate labels

# offsets Panel Legend: 
# analytic_linear_nominal  - traveling between original, nominal waypoints
# analytic_linear_achieved - traveling between waypoints that were achieved by targeted nonlinear model
# analytic_nonlin          - propagating same initial state as analytic_linear_achieved, but now in nonlinear model
# targeted_nonlin
# targeted_missed_maneuver

# Travel between waypoints
for start, end in waypoint_time_intervals:

    # array of time points
    timespan_for_segment = np.linspace(start, end, 500)
    
    offsets.major_axis = timespan_for_segment

    #******************************************************#
    
    ## Travel between the *nominal* waypoints using *only* the analytic delta-V computation propagated in the linear model for chaser
    
    # Pull out the RLP vector of the current and next waypoint, using the originally designed waypoints
    current_waypoint = waypoints.RLP.loc[start]
    next_waypoint = waypoints.RLP.loc[end]

    # Compute required velocity at point 1 to take us to point 2 within time (t2-t1)
    chaser_initial_velocity_relative_analytic = ComputeRequiredVelocity(target_initial_state_for_segment, current_waypoint, start, next_waypoint, end, mu)
    
    # Set up starting position and velocity of chaser
    chaser_initial_state_relative_analytic.loc[['x', 'y', 'z']] = current_waypoint
    chaser_initial_state_relative_analytic.loc[['x_dot', 'y_dot', 'z_dot']] = chaser_initial_velocity_relative_analytic

    ## Integrate first satellite with full nonlinear dynamics and second satellite with linear relmo dynamics
    # compute target satellite position and velocity over time in RLP frame by integrating initial state with full nonlinear dynamics
    # and compute offset between target and chaser satellite over time in RLP frame by integrating initial offset with linearized relmo dynamics
    target_ephem_for_segment, offsets_RLP_analytic_linear_nominal = PropagateSatelliteAndChaser(mu, timespan_for_segment, target_initial_state_for_segment, chaser_initial_state_relative_analytic)
    
    # copy position offsets into Panel
    offsets.RLP_analytic_linear_nominal = offsets_RLP_analytic_linear_nominal
    
    ## Compute analytic delta-V (based on original/nominal waypoint positions)
    chaser_initial_state_absolute_analytic = target_initial_state_for_segment - chaser_initial_state_relative_analytic
    chaser_final_state_absolute_analytic = target_ephem_for_segment.loc[end] - offsets_RLP_analytic_linear_nominal.loc[end]
    
    # post-maneuver velocity at current waypoint
    waypoint_velocities.RLP_post_maneuver_analytic_linear.loc[start] = chaser_initial_state_absolute_analytic.loc[['x_dot', 'y_dot', 'z_dot']]
    
    # pre-maneuver velocity for next waypoint (end of current propagation segment)
    waypoint_velocities.RLP_pre_maneuver_analytic_linear.loc[end] = chaser_final_state_absolute_analytic.loc[['x_dot', 'y_dot', 'z_dot']]  
    
    #******************************************************#
    
    ## Travel between the waypoints that are achieved by the nonlinear/targeted model;
    ## Use the linear model come up with a delta-V guess for the nonlinear model
    
    # Pull out the RLP vector of the current waypoint, starting from where the targeted nonlinear model left off
    current_waypoint = waypoints.RLP_achieved_targeted_nonlin.loc[start]

    # Compute required velocity at point 1 to take us to point 2 within time (t2-t1)
    chaser_initial_velocity_relative_analytic = ComputeRequiredVelocity(target_initial_state_for_segment, current_waypoint, start, next_waypoint, end, mu)
    
    # Set up starting position and velocity of chaser
    chaser_initial_state_relative_analytic.loc[['x', 'y', 'z']] = current_waypoint
    chaser_initial_state_relative_analytic.loc[['x_dot', 'y_dot', 'z_dot']] = chaser_initial_velocity_relative_analytic

    ## Integrate first satellite with full nonlinear dynamics and second satellite with linear relmo dynamics
    # compute target satellite position and velocity over time in RLP frame by integrating initial state with full nonlinear dynamics
    # and compute offset between target and chaser satellite over time in RLP frame by integrating initial offset with linearized relmo dynamics
    target_ephem_for_segment, offsets.RLP_analytic_linear_achieved = PropagateSatelliteAndChaser(mu, timespan_for_segment, target_initial_state_for_segment, chaser_initial_state_relative_analytic)
    
    #******************************************************#
    
    ## Integrate second satellite with full nonlinear dynamics using analytic delta-V

    # initial state of second satellite in absolute RLP coordinates (not relative to first satellite)
    chaser_initial_state_absolute_analytic = target_initial_state_for_segment - chaser_initial_state_relative_analytic
    
    # compute chaser satellite position and velocity over time in RLP frame by integrating initial state with full nonlinear dynamics
    chaser_ephem_for_segment_analytic = PropagateSatellite(mu, timespan_for_segment, chaser_initial_state_absolute_analytic)
    
    #******************************************************#
    
    ## Integrate second satellite with full nonlinear dynamics using targeted delta-V
    
    # use analytic delta-V as initial guess for targeter
    chaser_velocity_initial_guess_absolute = chaser_initial_state_absolute_analytic.loc[['x_dot', 'y_dot', 'z_dot']]
    chaser_initial_state_absolute_targeted.loc[['x', 'y', 'z']] = chaser_initial_state_absolute_analytic.loc[['x', 'y', 'z']]
    chaser_initial_state_absolute_targeted.loc[['x_dot', 'y_dot', 'z_dot']] = TargetRequiredVelocity(target_initial_state_for_segment, chaser_velocity_initial_guess_absolute, current_waypoint, start, next_waypoint, end, mu)
    
    # compute chaser satellite position and velocity over time in RLP frame by integrating initial state with full nonlinear dynamics
    chaser_ephem_for_segment_targeted = PropagateSatellite(mu, timespan_for_segment, chaser_initial_state_absolute_targeted)
    
    ## Compute delta-V based on targeted path between actual/achieved waypoints
    
    # post-maneuver velocity at current waypoint
    waypoint_velocities.RLP_post_maneuver_targeted_nonlin.loc[start] = chaser_initial_state_absolute_targeted.loc[['x_dot', 'y_dot', 'z_dot']]
    
    # pre-maneuver velocity for next waypoint (end of current propagation segment)
    waypoint_velocities.RLP_pre_maneuver_targeted_nonlin.loc[end] = chaser_ephem_for_segment_targeted.loc[end, ['x_dot', 'y_dot', 'z_dot']]
    
    #******************************************************#
    
    ## Integrate second satellite with full nonlinear dynamics assuming no delta-V was applied ("missed maneuver")
    
    chaser_ephem_for_segment_missed_maneuver = PropagateSatellite(mu, timespan_for_segment, chaser_initial_state_missed_maneuver)
    
    #******************************************************#
    
    ## Compute offsets in RLP frame based on nonlinear motion
    offsets.RLP_analytic_nonlin = ComputeOffsets(timespan_for_segment, target_ephem_for_segment, chaser_ephem_for_segment_analytic)
    offsets.RLP_targeted_nonlin = ComputeOffsets(timespan_for_segment, target_ephem_for_segment, chaser_ephem_for_segment_targeted)
    offsets.RLP_missed_maneuver = ComputeOffsets(timespan_for_segment, target_ephem_for_segment, chaser_ephem_for_segment_missed_maneuver)
    
    ## Build RIC and VNB frames
    RLPtoRIC = BuildRICFrames(target_ephem_for_segment, center)
    RLPtoVNB = BuildVNBFrames(target_ephem_for_segment, center)

    # Compute offsets in RIC frame
    offsets.RIC_analytic_linear_nominal  = ConvertOffsets(offsets.RLP_analytic_linear_nominal, RLPtoRIC);
    offsets.RIC_analytic_linear_achieved = ConvertOffsets(offsets.RLP_analytic_linear_achieved, RLPtoRIC);
    offsets.RIC_analytic_nonlin = ConvertOffsets(offsets.RLP_analytic_nonlin, RLPtoRIC);
    offsets.RIC_targeted_nonlin = ConvertOffsets(offsets.RLP_targeted_nonlin, RLPtoRIC);
    offsets.RIC_missed_maneuver = ConvertOffsets(offsets.RLP_missed_maneuver, RLPtoRIC);

    # Compute offsets in VNB frame
    offsets.VNB_analytic_linear_nominal  = ConvertOffsets(offsets.RLP_analytic_linear_nominal, RLPtoVNB);
    offsets.VNB_analytic_linear_achieved = ConvertOffsets(offsets.RLP_analytic_linear_achieved, RLPtoVNB);
    offsets.VNB_analytic_nonlin = ConvertOffsets(offsets.RLP_analytic_nonlin, RLPtoVNB);
    offsets.VNB_targeted_nonlin = ConvertOffsets(offsets.RLP_targeted_nonlin, RLPtoVNB);
    offsets.VNB_missed_maneuver = ConvertOffsets(offsets.RLP_missed_maneuver, RLPtoVNB);
    
    #******************************************************#
    
    ## Output that gets fed into next iteration/segment
    
    # Record updated primary satellite initial state for next segment
    target_initial_state_for_segment = target_ephem_for_segment.loc[end, ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']]
    
    # use final state from previous segment as initial state for "missed maneuver" propagation for next segment
    chaser_initial_state_missed_maneuver = chaser_ephem_for_segment_targeted.iloc[-1] 
    
    # Record updated/achieved chaser satellite waypoint for next segment
    waypoints.RLP_achieved_targeted_nonlin.loc[end] = offsets.RLP_targeted_nonlin.loc[end]  
    waypoints.RLP_achieved_analytic_nonlin.loc[end] = offsets.RLP_analytic_nonlin.loc[end]   
    
    # Build RIC and VNB frames at next waypoint
    RLPtoRIC = BuildRICFrame(target_ephem_for_segment.loc[end], center)
    RLPtoVNB = BuildVNBFrame(target_ephem_for_segment.loc[end], center)
    
    # compute updated/achieved waypoint locations in RIC and VNB
    waypoints.RIC_achieved_targeted_nonlin.loc[end] = ConvertOffset(waypoints.RLP_achieved_targeted_nonlin.loc[end], RLPtoRIC)
    waypoints.VNB_achieved_targeted_nonlin.loc[end] = ConvertOffset(waypoints.RLP_achieved_targeted_nonlin.loc[end], RLPtoVNB)
    
    waypoints.RIC_achieved_analytic_nonlin.loc[end] = ConvertOffset(waypoints.RLP_achieved_analytic_nonlin.loc[end], RLPtoRIC)
    waypoints.VNB_achieved_analytic_nonlin.loc[end] = ConvertOffset(waypoints.RLP_achieved_analytic_nonlin.loc[end], RLPtoVNB)
    
    #******************************************************#
    
    ## VISUALIZATIONS
    
    # plot offsets (relative motion) between satellites 1 and 2 in RLP, RIC, and VNB frames
    SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, offsets.RLP_analytic_linear_nominal*RLP_properties.r12, 'line', 'g')
    SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, offsets.RLP_analytic_linear_achieved*RLP_properties.r12, 'line', 'c')
    SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, offsets.RLP_analytic_nonlin*RLP_properties.r12, 'line', 'r')
    SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, offsets.RLP_targeted_nonlin*RLP_properties.r12, 'line', 'b')
    SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, offsets.RLP_missed_maneuver*RLP_properties.r12, 'dotted', 'b')
    
    SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, offsets.RIC_analytic_linear_nominal*RLP_properties.r12, 'line', 'g')
    SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, offsets.RIC_analytic_linear_achieved*RLP_properties.r12, 'line', 'c')
    SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, offsets.RIC_analytic_nonlin*RLP_properties.r12, 'line', 'r')
    SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, offsets.RIC_targeted_nonlin*RLP_properties.r12, 'line', 'b')
    SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, offsets.RIC_missed_maneuver*RLP_properties.r12, 'dotted', 'b')
    
    SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, offsets.VNB_analytic_linear_nominal*RLP_properties.r12, 'line', 'g')
    SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, offsets.VNB_analytic_linear_achieved*RLP_properties.r12, 'line', 'c')
    SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, offsets.VNB_analytic_nonlin*RLP_properties.r12, 'line', 'r')
    SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, offsets.VNB_targeted_nonlin*RLP_properties.r12, 'line', 'b')
    SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, offsets.VNB_missed_maneuver*RLP_properties.r12, 'dotted', 'b')
    
    #******************************************************#
    
# add achieved waypoints to plots
SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, waypoints.RLP_achieved_targeted_nonlin*RLP_properties.r12, 'points', 'm')
SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, waypoints.RIC_achieved_targeted_nonlin*RLP_properties.r12, 'points', 'm')
SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, waypoints.VNB_achieved_targeted_nonlin*RLP_properties.r12, 'points', 'm')

# final post-maneuver velocity is same as the target satellite's velocity
waypoint_velocities.RLP_post_maneuver_targeted_nonlin.loc[end] = target_ephem_for_segment.loc[end, ['x_dot', 'y_dot', 'z_dot']]
waypoint_velocities.RLP_post_maneuver_analytic_linear.loc[end] = target_ephem_for_segment.loc[end, ['x_dot', 'y_dot', 'z_dot']]

# compute delta-V's
waypoint_velocities.RLP_delta_v_targeted_nonlin = waypoint_velocities.RLP_post_maneuver_targeted_nonlin - waypoint_velocities.RLP_pre_maneuver_targeted_nonlin
waypoint_velocities.RLP_delta_v_analytic_linear = waypoint_velocities.RLP_post_maneuver_analytic_linear - waypoint_velocities.RLP_pre_maneuver_analytic_linear

# <codecell>


print 'waypoints.RLP_achieved_analytic_nonlin', display(HTML(waypoints.RLP_achieved_analytic_nonlin.to_html()))
print 'waypoints.RLP_achieved_targeted_nonlin', display(HTML(waypoints.RLP_achieved_targeted_nonlin.to_html()))

# <codecell>

#waypoints.RIC_achieved
#waypoints.RLP_achieved
#waypoint_velocities.RLP_post_maneuver_absolute
#chaser_ephem_for_segment.loc[start, ['x_dot', 'y_dot', 'z_dot']]

#print type(offsets_linear_RLP.loc[t, ['x', 'y', 'z']])

#offsets_linear_RLP[['x', 'y', 'z']]*RLP_properties.r12
#offsets_linear_RLP[['x', 'y', 'z']]*RLP_properties.r12
#print start, end
#print waypoint_time_intervals
#print type(offsets_linear_RLP[['x', 'y', 'z']]*RLP_properties.r12)

#waypoints.RLP_achieved.loc[end]
#offsets_linear_RIC
#waypoint_velocities.RLP_delta_v
#waypoint_velocities.RLP_post_maneuver_absolute - waypoint_velocities.RLP_pre_maneuver_absolute

#offsets.RLP_targeted_nonlin[['x', 'y', 'z']]
#current_waypoint

#np.dot(waypoint_velocities.RLP_delta_v_targeted_nonlin.loc[t], waypoint_velocities.RLP_delta_v_analytic_linear.loc[t])
#np.linalg.norm(waypoint_velocities.RLP_delta_v_analytic_linear.loc[t])
#waypoint_velocities.RLP_delta_v_targeted_nonlin

#offsets.RLP_analytic_linear.loc[t].iloc[0:3] #, ['x', 'y', 'z']]
#offsets.RLP_analytic_linear.loc[:,['x', 'y', 'z']]

#test = pd.Series({
#    'x':     initial_condition_set.x,
#    'y':     0.0,
#    'z':     initial_condition_set.z,
#    'x_dot': 0.0,
#    'y_dot': initial_condition_set.y_dot,
#    'z_dot': 0.0})
#print test

#test.reindex_axis(index=['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot'], axis=0)
#test.reindex(index=['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot'])
#print test

# reassign so that the series maintains the required order for its values
#test = test[['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']]
#df.reindex(index=[date1, date2, date3], columns=['A', 'B', 'C'])

# <codecell>


halo = 'small'
clock_angle = 0.0
approach = '+R'
timescale = 'med'
spacing = 'med'

waypoint_metrics = pd.DataFrame({'halo':           halo,
                                 'clock_angle':    clock_angle,
                                 'approach':       approach,
                                 'timescale':      timescale,
                                 'spacing':        spacing,
                                 'waypoint_times': waypoint_times,
                                 'DV_targeted': np.zeros((len(waypoint_times))),
                                 'DV_analytic': np.zeros((len(waypoint_times))),
                                 'DV_magnitude_difference': np.zeros((len(waypoint_times))),
                                 'DV_angle_difference': np.zeros((len(waypoint_times))),
                                 'achieved_position_error_analytic': np.zeros((len(waypoint_times))),
                                 'achieved_position_error_targeted': np.zeros((len(waypoint_times)))},
                            index=np.arange(len(waypoint_times)))

for point in np.arange(len(waypoint_times)):
    #for t in waypoint_times:
    t = waypoint_times[point]
    
    # compute delta-V magnitude in m/s
    waypoint_metrics.DV_targeted.iloc[point] = np.linalg.norm(waypoint_velocities.RLP_delta_v_targeted_nonlin.loc[t], 2)*RLP_properties.r12/RLP_properties.time_const*1000.0  # m/s
    waypoint_metrics.DV_analytic.iloc[point] = np.linalg.norm(waypoint_velocities.RLP_delta_v_analytic_linear.loc[t], 2)*RLP_properties.r12/RLP_properties.time_const*1000.0  # m/s
    
    # compute angle between targeted delta-V vector and analytic delta-V vector in degrees
    waypoint_metrics.DV_angle_difference.iloc[point] = math.degrees(math.acos(
                                                                np.dot(waypoint_velocities.RLP_delta_v_targeted_nonlin.loc[t], waypoint_velocities.RLP_delta_v_analytic_linear.loc[t])/
                                                                (np.linalg.norm(waypoint_velocities.RLP_delta_v_targeted_nonlin.loc[t])*
                                                                 np.linalg.norm(waypoint_velocities.RLP_delta_v_analytic_linear.loc[t]))))
    
    # compute delta-V magnitude difference between targeted and analytic, in m/s
    waypoint_metrics.DV_magnitude_difference.iloc[point] = waypoint_metrics.DV_targeted.iloc[point] - waypoint_metrics.DV_analytic.iloc[point]
    
    # compute position error in achieved waypoint location when applying analytic delta-V in nonlinear propagation, in meters
    waypoint_metrics.achieved_position_error_analytic.iloc[point] = np.linalg.norm(waypoints.RLP.loc[t] - waypoints.RLP_achieved_analytic_nonlin.loc[t])*RLP_properties.r12*1000 # meters

    waypoint_metrics.achieved_position_error_targeted.iloc[point] = np.linalg.norm(waypoints.RLP.loc[t] - waypoints.RLP_achieved_targeted_nonlin.loc[t])*RLP_properties.r12*1000 # meters

print 'waypoint_metrics', display(HTML(waypoint_metrics.to_html()))

waypoint_metrics.to_csv('output/run_' + halo + '_' + str(clock_angle) + '_' + approach + '_' + timescale + '_' + spacing + '.csv')

# <codecell>


