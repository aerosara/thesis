import pandas as pd
import numpy as np
import scipy.integrate as integrate

import thesis_functions.utilities

from thesis_functions.initial_conditions import initial_conditions

from thesis_functions.initialconditions import InputDataDictionary, SetInitialConditions
from thesis_functions.visualization import CreatePlotGrid, SetPlotGridData
from thesis_functions.astro import FindOrbitCenter, ComputeLibrationPoints, stop_yEquals0, stop_zEquals0
from thesis_functions.astro import ComputeNonlinearDerivs, ComputeRelmoDynamicsMatrix
from thesis_functions.astro import odeintNonlinearDerivs, odeintNonlinearDerivsWithLinearRelmoSTM, odeintNonlinearDerivsWithLinearRelmo
from thesis_functions.astro import ComputeRequiredVelocity, PropagateSatelliteAndChaser
from thesis_functions.astro import PropagateSatellite, ComputeOffsets, ConvertOffsets, ConvertOffset, BuildFrames

initial_condition = initial_conditions.loc["Barbee", 1]

mu = initial_condition.mu

initial_state = pd.Series({
    "x": initial_condition.x,
    "y": 0.0,
    "z": initial_condition.z,
    "x_dot": 0.0,
    "y_dot": initial_condition.y_dot,
    "z_dot": 0.0})

# X1 and X2 are positions of larger and smaller bodies along X axis
libration_system = ComputeLibrationPoints(mu)

# The FindOrbitCenter function doesn't work if you only propagate a partial orbit, so just treat L1 as the center
center = libration_system.L1

params = pd.Series({
    "m1": 5.97219e24,     # Earth (kg)
    "m2": 7.34767309e22,  # Moon (kg)
    "G": 6.67384e-11/1e9, # m3/(kg*s^2) >> converted to km3
    "r12": 384400.0})     # km

params.loc["M"] = params.m1 + params.m2

# This is how you convert between dimensional time (seconds) and non-dimensional time (units are seconds)
time_const = params.r12**(1.5) / (params.G * params.M)**(0.5)

period = 2 * np.pi * time_const   # Period in seconds of Moon around Earth

#  Period of libration point orbit (in nondimensional time units)
period = initial_condition.t

# Create a collection of waypoints where we initially population in RIC cooredinates.
waypoints = pd.Panel(np.array([np.vstack((np.zeros(6), np.array([100, 15, 5, 1, 0.03, 0.0]), np.zeros(6)))]).transpose((0, 2, 1)),
    items=["ric"],
    major_axis=np.array([5.31, 5.67, 6.03, 6.64, 7.0, 7.26]) * 86400 / time_const,
    minor_axis=list("xyz"))

# Append a waypoint at t=0 for initial state.
t = 0.0
frame = BuildFrames(initial_state.to_frame().transpose(), center).iloc[0]
ric_to_rlp = np.linalg.inv(frame.loc["ric"])
waypoints.loc["ric", t] = initial_state[list("xyz")].values
waypoints.loc["rlp", t] = np.dot(waypoints.loc["ric", t], ric_to_rlp)
waypoints.loc["vnb", t] = np.dot(waypoints.loc["rlp", t], frame.loc["vnb"])

# Finally, re-sort our waypoints.
waypoints = waypoints.sort_index(1)

# Create a Panel to store waypoint frames.
waypoint_frames = pd.Panel()

# Prepend 0 to the list of waypoint times and create a set of
# waypoint intervals.
waypoint_intervals = zip(waypoints.major_axis[:-1], waypoints.major_axis[1:])

state = initial_state

for start, end in waypoint_intervals:

    time = np.linspace(start, end, 500)

    # Build an ephem for the given timespan up to the waypoint.
    ephem = PropagateSatellite(mu, time, state)

    # Build the corresponding frames.
    frames = BuildFrames(ephem, center)

    # Select the last item in our frames collection as the waypoint frame.
    waypoint_frame = frames.iloc[-1]

    # Calculate the matrix to go from RIC to RLP
    ric_to_rlp = np.linalg.inv(waypoint_frame.loc["ric"])

    # Calculate the waypoint in the RLP frame and store it
    waypoints.loc["rlp", end] = np.dot(waypoints.loc["ric", end], ric_to_rlp)
    waypoints.loc["vnb", end] = np.dot(waypoints.loc["rlp", end], waypoint_frame.loc["vnb"])

    # Reset the state as the last entry in the ephem.
    state = ephem.irow(-1)

# Create a panel the represents the ephem of each satellite.
t = 0.0
target_satellite_ephem = pd.Panel(items=["rlp"], major_axis=[t], minor_axis=["x", "y", "z", "x_dot", "y_dot", "z_dot"])
chaser_satellite_ephem = target_satellite_ephem.copy()

# Configure the initial states of each ephem.
target_satellite_ephem.loc["rlp", t] = initial_state.values

# For the follower, use the position from the initial waypoint
chaser_satellite_ephem.loc["rlp_linear", t, ["x", "y", "z"]] = waypoints["rlp"].iloc[0]
chaser_satellite_ephem.loc["rlp", t, ["x", "y", "z"]] = waypoints["rlp"].iloc[0]

# Next, simulate the two spacecraft for each waypoint interval.

for start, end in waypoint_intervals:

    time = np.linspace(start, end, 500)

    # Select out the RLP vector of the next waypoint.
    next_waypoint = waypoints["rlp"][(waypoints.major_axis > start)].iloc[0]

    chaser_satellite_state = chaser_satellite_ephem.loc["rlp", start, ["x", "y", "z"]]

    # Compute the required velocity at the current waypoint.
    required_relative_velocity = ComputeRequiredVelocity(state, chaser_satellite_state, start, next_waypoint, end, mu)

    # Calculate required velocity.
    required_velocity = required_relative_velocity - target_satellite_ephem.loc["rlp", start, ["x_dot", "y_dot", "z_dot"]]

    # Store the required velocity.
    chaser_satellite_ephem.loc["rlp", start, ["x_dot", "y_dot", "z_dot"]] = required_velocity

    # Calculate the relative state between the two spacecraft.
    relative_state = target_satellite_ephem.loc["rlp", start] - chaser_satellite_ephem.loc["rlp", start]

    # Propagate the target spacecraft using nonlinear dynamics and generate linear offset.
    target_ephem, linear_offset = PropagateSatelliteAndChaser(mu, time, target_satellite_ephem.loc["rlp", start], relative_state)

    # Propagate the chaser spacecraft using nonlinear dynamics.
    chaser_ephem = PropagateSatellite(mu, time, chaser_satellite_ephem.loc["rlp", start])

    # We need to re-index our ephems. Boo.
    target_satellite_ephem = target_satellite_ephem.reindex(major_axis=np.unique(np.concatenate((target_satellite_ephem.major_axis.values, time))))
    chaser_satellite_ephem = chaser_satellite_ephem.reindex(major_axis=np.unique(np.concatenate((chaser_satellite_ephem.major_axis.values, time))))

    # Store the ephems.
    target_satellite_ephem.loc["rlp", time] = target_ephem.values
    chaser_satellite_ephem.loc["rlp", time] = chaser_ephem.values
    chaser_satellite_ephem.loc["rlp_linear", time] = (target_ephem + linear_offset).values
