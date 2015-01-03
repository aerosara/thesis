# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


import numpy as np
import pandas as pd
import scipy.integrate as integrate
import math

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

# <headingcell level=3>

# Initial Conditions

# <codecell>


def set_up_target(halo, clock_angle, initial_condition_sets, axis_array_RLP, axis_array_RIC, axis_array_VNB):

    # Set initial conditions for the target satellite 

    # The initial condition DataFrame contains initial conditions from Barbee, Howell, and Sharp

    # Barbee's first set of initial conditions are a planar (Lyapunov) orbit at Earth/Moon L1
    # Barbee's next 4 sets of initial conditions are halo orbits at Sun/Earth L2
    
    #halo_cases = ['small', 'medium', 'large', 'greater']
    #index = halo_cases.index(halo) + 2  # add two because the indexing starts from 1 and the halo orbits start from 2 
    
    halo_cases = ['EM', 'small', 'medium', 'large', 'greater']
    index = halo_cases.index(halo) + 1  # add one because the indexing starts from 1 and the orbits start from 1
        
    # Each initial_condition_set has attributes: author, test_case, mu, x, z, y_dot, t
    initial_condition_set = initial_condition_sets.loc['Barbee', index]

    mu = initial_condition_set.mu
    
    period = initial_condition_set.t

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

    #print 'initial_condition_sets: Barbee'
    #print display(HTML(initial_condition_sets.loc['Barbee'].to_html()))

    # propagate target initial state to the desired starting clock angle
    if (clock_angle != 0.0):

        # propagate target satellite to desired starting clock angle
        timespan_to_clock_angle = np.linspace(0.0, initial_condition_set.t*clock_angle/360, 500)

        # initial_state_to_clock_angle is a DataFrame with attributes x, y, z, x_dot, y_dot, z_dot and is indexed by timespan_to_clock_angle
        # this is the target satellite position and velocity over time in RLP frame from integrating initial state with full nonlinear dynamics
        initial_state_to_clock_angle = PropagateSatellite(mu, timespan_to_clock_angle, target_initial_state)

        target_initial_state = initial_state_to_clock_angle.iloc[-1]
    
    return target_initial_state, period, mu

# <headingcell level=3>

# Compute RLP properties

# <codecell>


def compute_RLP_properties(mu):
    
    # libration_points DataFrame will have attributes: X1, X2, L1, L2, L3, L4, L5
    # X1 and X2 are positions of larger and smaller bodies along X axis
    libration_points = ComputeLibrationPoints(mu)
    
    # RLP_properties will contain the libration_points information plus other constants
    RLP_properties = libration_points.copy()
    
    # gravitational constant
    RLP_properties['G']   = 6.67384e-11/1e9 # m3/(kg*s^2) >> converted to km3
    
    # store mu in RLP_properties as well
    RLP_properties['mu'] = mu

    # determine system from mass ratio
    if (mu == 0.012277471):
        system = 'Earth-Moon'
    elif (mu == 3.04009784138267e-06):
        system = 'Sun-Earth'
        
    # determine r12 and body masses for this system
    if (system == 'Earth-Moon'):

        # In nondimensional units, r12 = 1, M = 1, timeConst = Period/(2pi) = 1, G = 1
        RLP_properties['m1']  = 5.97219e24        # Earth (kg)
        RLP_properties['m2']  = 7.34767309e22     # Moon (kg)
        RLP_properties['r12'] = 384400.0          # km

    elif (system == 'Sun-Earth'):

        # In nondimensional units, r12 = 1, M = 1, timeConst = Period/(2pi) = 1, G = 1
        RLP_properties['m1']  = 1.988435e30  #1.989e30          # Sun (kg)
        RLP_properties['m2']  = 5.9721986e24 + 7.34767309e22    # Earth + Moon (kg)
        RLP_properties['r12'] = 149597870.7                     # km
        #TU_SE_days = 58.1313429643148; % days = 5022548.03211679872 seconds

    # compute total mass M for this system
    RLP_properties['M']   = RLP_properties.m1 + RLP_properties.m2
    
    # This is how you convert between dimensional time (seconds) and non-dimensional time 
    RLP_properties['time_const'] = RLP_properties.r12**(1.5) / (RLP_properties.G * RLP_properties.M)**(0.5) # (units are seconds)

    # Period in seconds of secondary around primary
    RLP_properties['T'] = 2.0 * np.pi * RLP_properties.time_const   
    
    #print 'RLP_properties'
    #print RLP_properties

    return RLP_properties


def set_active_point(target_initial_state, RLP_properties):
        
    # The FindOrbitCenter function doesn't work if you only propagate a partial orbit, so just treat L1/L2 as the center
    # determine which libration point we're orbiting by looking at magnitude/sign of the x component of the initial state
    if (0.0 < target_initial_state.x < 1.0):
        RLP_properties['center'] = RLP_properties.L1
    elif (target_initial_state.x > 1.0):
        RLP_properties['center'] = RLP_properties.L2
    

# <codecell>


def plot_full_orbit(target_initial_state, RLP_properties, period, mu, axis_array_RLP_absolute):
    
    timespan_to_full_orbit = np.linspace(0.0, period, 500)
    
    # propagate satellite one full orbit
    target_ephem_full_orbit = PropagateSatellite(mu, timespan_to_full_orbit, target_initial_state)
    
    # add data to plot
    SetPlotGridData(axis_array_RLP_absolute, target_ephem_full_orbit*RLP_properties.r12, 'line', 'b', 'Target')
    
    # plot L1 point
    L1_point = pd.Series({
        'x':     RLP_properties.L1[0],
        'y':     RLP_properties.L1[1],
        'z':     RLP_properties.L1[2]})
    
    SetPlotGridData(axis_array_RLP_absolute, L1_point*RLP_properties.r12, 'points', 'k', 'L1')
    
    return
    
def plot_initial_condition(target_initial_state, RLP_properties, axis_array_RLP_absolute):
    
    # plot initial condition
    SetPlotGridData(axis_array_RLP_absolute, target_initial_state*RLP_properties.r12, 'star', 'm', 'Initial Conditions')
    
    return

# <headingcell level=3>

# Define Waypoints

# <codecell>


def define_waypoints_RIC(approach, spacing, timescale, RLP_properties, axis_array_RIC):

    # Create a collection of waypoints which we initially populate in RIC coordinates
    waypoint_RIC_coordinates = np.array([#[0.0, 1000.0, 0.0],
                                         #[0.0,  275.0, 0.0],   # move 725 km
                                         #[0.0,  180.0, 0.0],   # move 95 km
                                        #[0.0,  100.0, 0.0],
                                        [0.0,   15.0, 0.0],
                                        [0.0,    5.0, 0.0],
                                        [0.0,    1.0, 0.0],
                                        #[0.0,   0.03, 0.0],
                                        [0.0,    0.0, 0.0]])/RLP_properties.r12

    # Time points
    waypoint_times = np.array([#0.0, 
                               #2.88, 4.70, 
                               #5.31, 
                               5.67, 6.03, 6.64, 
                               #7.0, 
                               7.26])*86400.0/RLP_properties.time_const
                               #0.  ,  0.36,  0.97,  1.59])*86400.0/RLP_properties.time_const
    
    # Create data panel which will hold the waypoints in RIC, RLP, and VNB frames, indexed by time
    waypoints = pd.Panel(items = ['RIC', 'RLP', 'VNB', 
                                  'RIC_achieved_targeted_nonlin', 'RLP_achieved_targeted_nonlin', 'VNB_achieved_targeted_nonlin',
                                  'RIC_achieved_analytic_nonlin', 'RLP_achieved_analytic_nonlin', 'VNB_achieved_analytic_nonlin'],
                         major_axis = waypoint_times, # time points
                         minor_axis = list('xyz'))    # coordinate labels

    # Copy the RIC waypoint data into the panel
    waypoints['RIC'] = waypoint_RIC_coordinates

    #print 'waypoints.RIC * RLP_properties.r12'
    #print display(HTML((waypoints.RIC*RLP_properties.r12).to_html()))
    
    # add all waypoints to RIC plots
    SetPlotGridData(axis_array_RIC, waypoints.RIC*RLP_properties.r12, 'points', 'c', 'Nominal Waypoints')
    
    return waypoints

# <headingcell level=3>

# Convert Waypoints from RIC to RLP and VNB

# <codecell>


def convert_waypoints_RLP_VNB(target_initial_state, waypoints, RLP_properties, axis_array_RLP, axis_array_VNB):
    
    ## Convert the first waypoint to RLP and VNB

    # these matrices convert from RLP coordinates to the RIC  and VNB frames at the timestamp of the first point
    RLPtoRIC = BuildRICFrame(target_initial_state, RLP_properties.center)
    RLPtoVNB = BuildVNBFrame(target_initial_state, RLP_properties.center)

    # this matrix converts from RIC to RLP at the timestamp of the first point
    RICtoRLP = np.linalg.inv(RLPtoRIC)

    # Calculate the waypoint in the RLP and VNB frames and store it
    waypoints.RLP.iloc[0] = ConvertOffset(waypoints.RIC.iloc[0], RICtoRLP)
    waypoints.VNB.iloc[0] = ConvertOffset(waypoints.RLP.iloc[0], RLPtoVNB)

    # make a temporary copy of the initial state which gets overwritten at the start of each segment
    target_initial_state_for_segment = target_initial_state.copy()

    # Create a set of waypoint intervals
    waypoint_times = waypoints.major_axis
    waypoint_time_intervals = zip(waypoint_times[:-1], waypoint_times[1:])

    ## Convert the remaining waypoints to RLP and VNB

    for start, end in waypoint_time_intervals:

        #print 'percentage of orbit covered getting to next point (by time):', (end - start)/period*100.0

        # array of time points
        timespan_for_segment = np.linspace(start, end, 500)

        # Build an ephem for the given timespan up to the next waypoint.
        # target_ephem_for_segment is a DataFrame with attributes x, y, z, x_dot, y_dot, z_dot and is indexed by timespan_for_segment
        # this is the target satellite position and velocity over time in RLP frame from integrating initial state with full nonlinear dynamics
        target_ephem_for_segment = PropagateSatellite(RLP_properties.mu, timespan_for_segment, target_initial_state_for_segment)

        # Get the target satellite state at the end of the current segment
        target_state_at_endpoint = target_ephem_for_segment.iloc[-1]

        # Build RIC and VNB frames
        # these matrices convert from RLP coordinates to the RIC  and VNB frames at the timestamp of the next point
        RLPtoRIC = BuildRICFrame(target_state_at_endpoint, RLP_properties.center)
        RLPtoVNB = BuildVNBFrame(target_state_at_endpoint, RLP_properties.center)

        # this matrix converts from RIC to RLP at the timestamp of the next point
        RICtoRLP = np.linalg.inv(RLPtoRIC)

        # Calculate the waypoint in the RLP and VNB frames and store it
        waypoints.RLP.loc[end] = ConvertOffset(waypoints.RIC.loc[end], RICtoRLP)
        waypoints.VNB.loc[end] = ConvertOffset(waypoints.RLP.loc[end], RLPtoVNB)

        # Reset the state as the last entry in the ephem.
        target_initial_state_for_segment = target_state_at_endpoint
    
    #print 'waypoints.RLP * RLP_properties.r12', display(HTML((waypoints.RLP*RLP_properties.r12).to_html()))
    #print 'waypoints.VNB * RLP_properties.r12', display(HTML((waypoints.VNB*RLP_properties.r12).to_html()))

    # add all waypoints to RLP and VNB plots
    SetPlotGridData(axis_array_RLP, waypoints.RLP*RLP_properties.r12, 'points', 'c', 'Nominal Waypoints')
    SetPlotGridData(axis_array_VNB, waypoints.VNB*RLP_properties.r12, 'points', 'c', 'Nominal Waypoints')
    
    return waypoints

# <headingcell level=3>

# Travel between waypoints

# <codecell>


def travel_waypoints(target_initial_state, waypoints, RLP_properties, axis_array_RLP, axis_array_RIC, axis_array_VNB):
    
    waypoint_times = waypoints.major_axis
    
    # set up initial state Series objects
    target_initial_state_for_segment       = target_initial_state.copy()  # make a temporary copy of the initial state which gets overwritten at the start of each segment
    chaser_initial_state_relative_analytic = pd.Series(index = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']) # relative means origin is target satellite
    chaser_initial_state_absolute_analytic = pd.Series(index = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']) # absolute means origin is origin of RLP frame
    chaser_initial_state_absolute_targeted = pd.Series(index = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']) 
    chaser_initial_state_missed_maneuver   = target_initial_state.copy() # this will get overwritten with the final targeted satellite state at the end of each segment

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
                         major_axis = np.arange(500),     # temporary time points - will get overwritten in each segment
                         minor_axis = ['x', 'y', 'z'])    # coordinate labels

    # offsets Panel Legend: 
    # analytic_linear_nominal  - traveling between original, nominal waypoints
    # analytic_linear_achieved - traveling between waypoints that were achieved by targeted/nonlinear model
    # analytic_nonlin          - propagating same initial state as analytic_linear_achieved, but now in nonlinear model
    # targeted_nonlin
    # targeted_missed_maneuver

    # Create a set of waypoint intervals
    waypoint_time_intervals = zip(waypoint_times[:-1], waypoint_times[1:])

    # Travel between waypoints
    for start, end in waypoint_time_intervals:

        # array of time points
        timespan_for_segment = np.linspace(start, end, 500)

        offsets.major_axis = timespan_for_segment

        #******************************************************#

        ## Travel between the *nominal* waypoints using only the analytic delta-V computation propagated in the linear model for chaser

        # Pull out the RLP vector of the current and next waypoint, using the originally designed waypoints
        current_waypoint = waypoints.RLP.loc[start]
        next_waypoint = waypoints.RLP.loc[end]

        # Compute required velocity at point 1 to take us to point 2 within time (t2-t1)
        chaser_initial_velocity_relative_analytic = ComputeRequiredVelocity(target_initial_state_for_segment, current_waypoint, start, next_waypoint, end, RLP_properties.mu)

        # Set up starting position and velocity of chaser
        chaser_initial_state_relative_analytic.loc[['x', 'y', 'z']] = current_waypoint
        chaser_initial_state_relative_analytic.loc[['x_dot', 'y_dot', 'z_dot']] = chaser_initial_velocity_relative_analytic

        ## Integrate first satellite with full nonlinear dynamics and second satellite with linear relmo dynamics
        # compute target satellite position and velocity over time in RLP frame by integrating initial state with full nonlinear dynamics
        # and compute offset between target and chaser satellite over time in RLP frame by integrating initial offset with linearized relmo dynamics
        target_ephem_for_segment, offsets_RLP_analytic_linear_nominal = PropagateSatelliteAndChaser(RLP_properties.mu, timespan_for_segment, target_initial_state_for_segment, chaser_initial_state_relative_analytic)

        # copy position offsets into Panel (velocities are not stored in the offsets Panel)
        offsets.RLP_analytic_linear_nominal = offsets_RLP_analytic_linear_nominal

        ## Compute analytic delta-V (based on original/nominal waypoint positions)
        chaser_initial_state_absolute_analytic = target_initial_state_for_segment - chaser_initial_state_relative_analytic
        chaser_final_state_absolute_analytic = target_ephem_for_segment.loc[end] - offsets_RLP_analytic_linear_nominal.loc[end]

        # post-maneuver velocity at current waypoint
        waypoint_velocities.RLP_post_maneuver_analytic_linear.loc[start] = chaser_initial_state_absolute_analytic.loc[['x_dot', 'y_dot', 'z_dot']]

        # pre-maneuver velocity for next waypoint (end of current propagation segment)
        waypoint_velocities.RLP_pre_maneuver_analytic_linear.loc[end] = chaser_final_state_absolute_analytic.loc[['x_dot', 'y_dot', 'z_dot']]  

        #******************************************************#

        ## Travel between the waypoints that are achieved by the nonlinear/targeted model
        # We are using the linear model to come up with a delta-V guess for the nonlinear model

        # Pull out the RLP vector of the current waypoint, starting from where the targeted nonlinear model left off
        current_waypoint = waypoints.RLP_achieved_targeted_nonlin.loc[start]

        # Compute required velocity at point 1 to take us to point 2 within time (t2-t1)
        chaser_initial_velocity_relative_analytic = ComputeRequiredVelocity(target_initial_state_for_segment, current_waypoint, start, next_waypoint, end, RLP_properties.mu)

        # Set up starting position and velocity of chaser
        chaser_initial_state_relative_analytic.loc[['x', 'y', 'z']] = current_waypoint
        chaser_initial_state_relative_analytic.loc[['x_dot', 'y_dot', 'z_dot']] = chaser_initial_velocity_relative_analytic

        ## Integrate first satellite with full nonlinear dynamics and second satellite with linear relmo dynamics
        # compute target satellite position and velocity over time in RLP frame by integrating initial state with full nonlinear dynamics
        # and compute offset between target and chaser satellite over time in RLP frame by integrating initial offset with linearized relmo dynamics
        target_ephem_for_segment, offsets.RLP_analytic_linear_achieved = PropagateSatelliteAndChaser(RLP_properties.mu, timespan_for_segment, target_initial_state_for_segment, chaser_initial_state_relative_analytic)

        #******************************************************#

        ## Integrate second satellite with full nonlinear dynamics using analytic delta-V

        # initial state of second satellite in absolute RLP coordinates (not relative to first satellite)
        chaser_initial_state_absolute_analytic = target_initial_state_for_segment - chaser_initial_state_relative_analytic

        # compute chaser satellite position and velocity over time in RLP frame by integrating initial state with full nonlinear dynamics
        chaser_ephem_for_segment_analytic_nonlin = PropagateSatellite(RLP_properties.mu, timespan_for_segment, chaser_initial_state_absolute_analytic)

        #******************************************************#

        ## Integrate second satellite with full nonlinear dynamics using targeted delta-V

        # use analytic delta-V as initial guess for targeter
        chaser_velocity_initial_guess_absolute = chaser_initial_state_absolute_analytic.loc[['x_dot', 'y_dot', 'z_dot']]
        chaser_initial_state_absolute_targeted.loc[['x', 'y', 'z']] = chaser_initial_state_absolute_analytic.loc[['x', 'y', 'z']]
        chaser_initial_state_absolute_targeted.loc[['x_dot', 'y_dot', 'z_dot']] = TargetRequiredVelocity(target_initial_state_for_segment, chaser_velocity_initial_guess_absolute, current_waypoint, start, next_waypoint, end, RLP_properties.mu)

        # compute chaser satellite position and velocity over time in RLP frame by integrating initial state with full nonlinear dynamics
        chaser_ephem_for_segment_targeted = PropagateSatellite(RLP_properties.mu, timespan_for_segment, chaser_initial_state_absolute_targeted)

        ## Compute delta-V based on targeted path between actual/achieved waypoints

        # post-maneuver velocity at current waypoint
        waypoint_velocities.RLP_post_maneuver_targeted_nonlin.loc[start] = chaser_initial_state_absolute_targeted.loc[['x_dot', 'y_dot', 'z_dot']]

        # pre-maneuver velocity for next waypoint (end of current propagation segment)
        waypoint_velocities.RLP_pre_maneuver_targeted_nonlin.loc[end] = chaser_ephem_for_segment_targeted.loc[end, ['x_dot', 'y_dot', 'z_dot']]

        #******************************************************#

        ## Integrate second satellite with full nonlinear dynamics assuming no delta-V was applied ("missed maneuver")

        chaser_ephem_for_segment_missed_maneuver = PropagateSatellite(RLP_properties.mu, timespan_for_segment, chaser_initial_state_missed_maneuver)

        #******************************************************#

        ## Compute offsets in RLP frame based on nonlinear motion
        offsets.RLP_analytic_nonlin = ComputeOffsets(timespan_for_segment, target_ephem_for_segment, chaser_ephem_for_segment_analytic_nonlin)
        offsets.RLP_targeted_nonlin = ComputeOffsets(timespan_for_segment, target_ephem_for_segment, chaser_ephem_for_segment_targeted)
        offsets.RLP_missed_maneuver = ComputeOffsets(timespan_for_segment, target_ephem_for_segment, chaser_ephem_for_segment_missed_maneuver)

        ## Build RIC and VNB frames
        RLPtoRIC = BuildRICFrames(target_ephem_for_segment, RLP_properties.center)
        RLPtoVNB = BuildVNBFrames(target_ephem_for_segment, RLP_properties.center)

        # Compute offsets in RIC frame
        offsets.RIC_analytic_linear_nominal  = ConvertOffsets(offsets.RLP_analytic_linear_nominal, RLPtoRIC);
        offsets.RIC_analytic_linear_achieved = ConvertOffsets(offsets.RLP_analytic_linear_achieved, RLPtoRIC);
        offsets.RIC_analytic_nonlin          = ConvertOffsets(offsets.RLP_analytic_nonlin, RLPtoRIC);
        offsets.RIC_targeted_nonlin          = ConvertOffsets(offsets.RLP_targeted_nonlin, RLPtoRIC);
        offsets.RIC_missed_maneuver          = ConvertOffsets(offsets.RLP_missed_maneuver, RLPtoRIC);

        # Compute offsets in VNB frame
        offsets.VNB_analytic_linear_nominal  = ConvertOffsets(offsets.RLP_analytic_linear_nominal, RLPtoVNB);
        offsets.VNB_analytic_linear_achieved = ConvertOffsets(offsets.RLP_analytic_linear_achieved, RLPtoVNB);
        offsets.VNB_analytic_nonlin          = ConvertOffsets(offsets.RLP_analytic_nonlin, RLPtoVNB);
        offsets.VNB_targeted_nonlin          = ConvertOffsets(offsets.RLP_targeted_nonlin, RLPtoVNB);
        offsets.VNB_missed_maneuver          = ConvertOffsets(offsets.RLP_missed_maneuver, RLPtoVNB);

        #******************************************************#

        ## Output that gets fed into next segment

        # Record updated primary satellite initial state for next segment
        target_initial_state_for_segment = target_ephem_for_segment.loc[end]

        # use final state from previous segment as initial state for "missed maneuver" propagation for next segment
        chaser_initial_state_missed_maneuver = chaser_ephem_for_segment_targeted.loc[end]

        # Record updated/achieved chaser satellite waypoint for next segment
        waypoints.RLP_achieved_targeted_nonlin.loc[end] = offsets.RLP_targeted_nonlin.loc[end]
        waypoints.RLP_achieved_analytic_nonlin.loc[end] = offsets.RLP_analytic_nonlin.loc[end]

        # Build RIC and VNB frames at next waypoint
        RLPtoRIC = BuildRICFrame(target_ephem_for_segment.loc[end], RLP_properties.center)
        RLPtoVNB = BuildVNBFrame(target_ephem_for_segment.loc[end], RLP_properties.center)

        # compute updated/achieved waypoint locations in RIC and VNB
        waypoints.RIC_achieved_targeted_nonlin.loc[end] = ConvertOffset(waypoints.RLP_achieved_targeted_nonlin.loc[end], RLPtoRIC)
        waypoints.VNB_achieved_targeted_nonlin.loc[end] = ConvertOffset(waypoints.RLP_achieved_targeted_nonlin.loc[end], RLPtoVNB)

        waypoints.RIC_achieved_analytic_nonlin.loc[end] = ConvertOffset(waypoints.RLP_achieved_analytic_nonlin.loc[end], RLPtoRIC)
        waypoints.VNB_achieved_analytic_nonlin.loc[end] = ConvertOffset(waypoints.RLP_achieved_analytic_nonlin.loc[end], RLPtoVNB)

        #******************************************************#

        ## VISUALIZATIONS

        # plot offsets (relative motion) between satellites 1 and 2 in RLP, RIC, and VNB frames
        SetPlotGridData(axis_array_RLP, offsets.RLP_analytic_linear_nominal*RLP_properties.r12, 'line', 'g', 'Linear Propagation, Linear \(\Delta V\)')
        #SetPlotGridData(axis_array_RLP, offsets.RLP_analytic_linear_achieved*RLP_properties.r12, 'dotted', 'r')
        SetPlotGridData(axis_array_RLP, offsets.RLP_analytic_nonlin*RLP_properties.r12, 'line', 'r', 'Nonlinear Propagation, Linear \(\Delta V\)')
        SetPlotGridData(axis_array_RLP, offsets.RLP_targeted_nonlin*RLP_properties.r12, 'line', 'b', 'Nonlinear Propagation, Targeted \(\Delta V\)')
        #SetPlotGridData(axis_array_RLP, offsets.RLP_missed_maneuver*RLP_properties.r12, 'dotted', 'b')

        SetPlotGridData(axis_array_RIC, offsets.RIC_analytic_linear_nominal*RLP_properties.r12, 'line', 'g', 'Linear Propagation, Linear \(\Delta V\)')
        #SetPlotGridData(axis_array_RIC, offsets.RIC_analytic_linear_achieved*RLP_properties.r12, 'dotted', 'r')
        SetPlotGridData(axis_array_RIC, offsets.RIC_analytic_nonlin*RLP_properties.r12, 'line', 'r', 'Nonlinear Propagation, Linear \(\Delta V\)')
        SetPlotGridData(axis_array_RIC, offsets.RIC_targeted_nonlin*RLP_properties.r12, 'line', 'b', 'Nonlinear Propagation, Targeted \(\Delta V\)')
        #SetPlotGridData(axis_array_RIC, offsets.RIC_missed_maneuver*RLP_properties.r12, 'dotted', 'b')

        SetPlotGridData(axis_array_VNB, offsets.VNB_analytic_linear_nominal*RLP_properties.r12, 'line', 'g', 'Linear Propagation, Linear \(\Delta V\)')
        #SetPlotGridData(axis_array_VNB, offsets.VNB_analytic_linear_achieved*RLP_properties.r12, 'dotted', 'r')
        SetPlotGridData(axis_array_VNB, offsets.VNB_analytic_nonlin*RLP_properties.r12, 'line', 'r', 'Nonlinear Propagation, Linear \(\Delta V\)')
        SetPlotGridData(axis_array_VNB, offsets.VNB_targeted_nonlin*RLP_properties.r12, 'line', 'b', 'Nonlinear Propagation, Targeted \(\Delta V\)')
        #SetPlotGridData(axis_array_VNB, offsets.VNB_missed_maneuver*RLP_properties.r12, 'dotted', 'b')

        #******************************************************#

    ConfigurePlotLegend(axis_array_RIC)
    ConfigurePlotLegend(axis_array_RLP)
    
    # add achieved waypoints to plots
    SetPlotGridData(axis_array_RLP, waypoints.RLP_achieved_targeted_nonlin*RLP_properties.r12, 'points', 'm', 'Achieved Waypoints')
    SetPlotGridData(axis_array_RIC, waypoints.RIC_achieved_targeted_nonlin*RLP_properties.r12, 'points', 'm', 'Achieved Waypoints')
    SetPlotGridData(axis_array_VNB, waypoints.VNB_achieved_targeted_nonlin*RLP_properties.r12, 'points', 'm', 'Achieved Waypoints')

    # final post-maneuver velocity is same as the target satellite's velocity
    waypoint_velocities.RLP_post_maneuver_targeted_nonlin.loc[end] = target_ephem_for_segment.loc[end, ['x_dot', 'y_dot', 'z_dot']]
    waypoint_velocities.RLP_post_maneuver_analytic_linear.loc[end] = target_ephem_for_segment.loc[end, ['x_dot', 'y_dot', 'z_dot']]

    # compute delta-V's
    waypoint_velocities.RLP_delta_v_targeted_nonlin = waypoint_velocities.RLP_post_maneuver_targeted_nonlin - waypoint_velocities.RLP_pre_maneuver_targeted_nonlin
    waypoint_velocities.RLP_delta_v_analytic_linear = waypoint_velocities.RLP_post_maneuver_analytic_linear - waypoint_velocities.RLP_pre_maneuver_analytic_linear

    return waypoints, waypoint_velocities

# <headingcell level=3>

# Compute waypoint metrics

# <codecell>


def compute_waypoint_metrics(halo, clock_angle, approach, timescale, spacing, waypoints, waypoint_velocities, RLP_properties):
    
    waypoint_times = waypoints.major_axis
    
    num_waypoints = len(waypoint_times)
    
    # create waypoint_metrics DataFrame for holding key output
    waypoint_metrics = pd.DataFrame({'halo':           halo,
                                     'clock_angle':    clock_angle,
                                     'approach':       approach,
                                     'timescale':      timescale,
                                     'spacing':        spacing,
                                     'DV_targeted': np.zeros(num_waypoints),
                                     'DV_analytic': np.zeros(num_waypoints),
                                     'DV_magnitude_difference': np.zeros(num_waypoints),
                                     'DV_angle_difference': np.zeros(num_waypoints),
                                     'achieved_position_error_analytic': np.zeros(num_waypoints),
                                     'achieved_position_error_targeted': np.zeros(num_waypoints)},
                                index=np.arange(num_waypoints))

    # loop over waypoint index values
    for point in np.arange(num_waypoints):
        
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

    #print 'waypoint_metrics', display(HTML(waypoint_metrics.to_html(float_format=lambda x: '{0:.3f}'.format(x))))

    waypoint_metrics.to_csv('output/run_' + halo + '_' + str(clock_angle) + '_' + approach + '_' + timescale + '_' + spacing + '.csv')
    
    return waypoint_metrics

