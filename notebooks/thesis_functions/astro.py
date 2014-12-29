# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


import numpy as np
import pandas as pd
    
from scipy.optimize import fsolve
    
import scipy.integrate as integrate

# <codecell>

# Thesis astrodynamics functions

def FindOrbitCenter(x, y, z):

    # identify x-coordinate corresponding to maximum y-amplitude, 
    #          y-coordinate corresponding to maximum x-amplitude, 
    #          and z-coordinate corresponding to maximum y-amplitude  #y=0
    center = [x[y.argmax()], y[x.argmax()], z[y.argmax()]]  #np.abs(y).argmin()]]

    print 'center = ', center
    
    return center

# <codecell>


def ComputeLibrationPoints(mu):
    
    # Inputs: mu = m2/M = (mass of smaller body) / (total mass)
    
    # In nondimensional units, r12 = 1, M = 1, Period/(2pi) = 1, G = 1
    
    # Position of larger body along X axis:
    X1 = np.array([-mu, 0, 0]);
    
    # Position of smaller body along X axis:
    X2 = np.array([1.0-mu, 0, 0]);
    
    # Functions from notes from Brent Barbee's class ENAE601, 10/12/2011, and HW 4, 10/17/2011
    def f_L1(x, mu):   
        
        p = 1.0 - mu - x
        return (1.0 - mu)*(p**3.0)*(p**2.0 - 3.0*p + 3.0) - mu*(p**2.0 + p + 1.0)*(1.0 - p)**3.0

    def f_L2(x, mu):    
        
        p = mu - 1.0 + x
        return (1.0 - mu)*(p**3.0)*(p**2.0 + 3.0*p + 3.0) - mu*(p**2.0 + p + 1.0)*(1.0 - p)*(p + 1.0)**2.0
    
    def f_L3(x, mu):
        
        p = -x - mu
        return (1.0 - mu)*(p**2.0 + p + 1.0)*(p - 1.0)*(p + 1.0)**2.0 + mu*(p**3.0)*(p**2.0 + 3.0*p + 3.0)
        
        
    # Find roots of the functions with fsolve, providing an initial guess
    l1 = fsolve(f_L1, 0.7, args=(mu,));
    l2 = fsolve(f_L2, 1.2, args=(mu,));
    l3 = fsolve(f_L3, -1.1, args=(mu,));
    
    # L1
    L1 = np.array([l1[0], 0.0, 0.0]);
    
    # L2
    L2 = np.array([l2[0], 0.0, 0.0]);
    
    # L3
    L3 = np.array([l3[0], 0.0, 0.0]);
    
    # L4
    L4 = np.array([0.5 - mu, np.sqrt(3.0)/2.0, 0.0]);

    # L5
    L5 = np.array([0.5 - mu, -np.sqrt(3.0)/2.0, 0.0]);
    
    return pd.Series({
        "X1": X1,
        "X2": X2,
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "L4": L4,
        "L5": L5})

# <codecell>


# Define stopping conditions, which can be used with odelay (from pycse)
def stop_yEquals0(state, t):
    isterminal = True
    direction = 0
    value = state[1]  # y = 0
    return value, isterminal, direction

def stop_zEquals0(state, t):
    isterminal = True
    direction = 0
    value = state[2]  # z = 0
    return value, isterminal, direction

# <headingcell level=3>

# Dynamics and ODE functions to integrate

# <codecell>


def ComputeNonlinearDerivs(x, y, z, xdot, ydot, zdot, mu):
    
    # Position of larger body along X axis:
    X1 = np.array([-mu, 0, 0]);
    
    # Position of smaller body along X axis:
    X2 = np.array([1.0 - mu, 0, 0]);
    
    # distances from primary masses to target satellite
    r1 = np.sqrt((x-X1[0])**2.0 + y**2.0 + z**2.0);
    r2 = np.sqrt((x-X2[0])**2.0 + y**2.0 + z**2.0);

    # Compute nonlinear derivatives for target satellite in RLP frame
    targetStateDerivs = [xdot, 
                         ydot,
                         zdot, 
                         x + 2.0*ydot + (1 - mu)*(-mu - x)/(r1**3.0) + mu*(1 - mu - x)/(r2**3.0),
                         y - 2.0*xdot - (1 - mu)*y/(r1**3.0) - mu*y/(r2**3.0),
                         -(1 - mu)*z/(r1**3.0) - mu*z/(r2**3.0)]
    
    return targetStateDerivs
    

def ComputeRelmoDynamicsMatrix(x, y, z, mu):
    
    # set mu1, mu2 - the gravitational parameters of the larger and smaller bodies
    mu1 = 1.0 - mu
    mu2 = mu
    
    # Position of larger body along X axis:
    X1 = np.array([-mu, 0, 0]);
    
    # Position of smaller body along X axis:
    X2 = np.array([1.0 - mu, 0, 0]);
    
    # unit vectors from primary masses to target satellite
    e1 = np.array([x-X1[0], y, z])
    e2 = np.array([x-X2[0], y, z])
    
    # distances from primary masses to target satellite
    r1 = np.sqrt((x-X1[0])**2.0 + y**2.0 + z**2.0);
    r2 = np.sqrt((x-X2[0])**2.0 + y**2.0 + z**2.0);
    
    c1 = mu1/r1**3.0
    c2 = mu2/r2**3.0
    
    # set up 3x3 identity matrix and zeroes matrix
    I3 = np.eye(3)
    Z3 = np.zeros((3,3))
    
    # In non-dimensional units, omega = sqrt(GM/(r^3)) = 1
    w = 1.0;
    
    # Cross-product matrix
    wx = np.array([[0.0,  -w, 0.0], 
                   [  w, 0.0, 0.0],
                   [0.0, 0.0, 0.0]])
    
    X = -(c1 + c2)*I3 + 3.0*c1*np.outer(e1, e1) + 3.0*c2*np.outer(e2, e2) - np.dot(wx, wx)
    
    mwx2 = -2.0*wx  # paper says -2[wx]T (which equals 2[wx]), but Phd says -2[wx]
    
    # Linearized system dynamics matrix
    A = np.vstack([np.hstack([Z3, I3]),
                   np.hstack([X,  mwx2])])
    
    return A


def odeintNonlinearDerivs(inputstate, timespan, mu):
    
    x, y, z, x_dot, y_dot, z_dot = inputstate
    
    # Compute nonlinear derivatives for the satellite in RLP frame
    derivs = ComputeNonlinearDerivs(x, y, z, x_dot, y_dot, z_dot, mu)
    
    return derivs


# These derivs are from Luquette
def odeintNonlinearDerivsWithLinearRelmoSTM(inputstate, timespan, mu):
    
    # Position and velocity of target satellite in RLP frame
    x, y, z, xdot, ydot, zdot = inputstate[0:6]
    
    # This should always be the Identity matrix at t0
    Phi = inputstate[6:42].reshape(6,6)
    
    # Compute linearized system dynamics matrix for relmo
    A = ComputeRelmoDynamicsMatrix(x, y, z, mu);
    
    # Compute STM derivates using linearized relmo dynamics
    PhiDot = np.dot(A, Phi)
    
    # Compute nonlinear derivatives for target satellite in RLP frame
    targetStateDerivs = ComputeNonlinearDerivs(x, y, z, xdot, ydot, zdot, mu)
    
    # Concatenate derivatives
    derivs = np.concatenate((targetStateDerivs, PhiDot.reshape(1,36)[0]))
    
    return derivs


# This is from Luquette
def odeintNonlinearDerivsWithLinearRelmo(inputstate, timespan, mu):
    
    # position and velocity of target satellite in RLP frame
    x, y, z, xdot, ydot, zdot = inputstate[0:6]
    
    # offset position and velocity of chaser satellite wrt target satellite in RLP frame
    chaserInputState = inputstate[6:12]
    
    # Compute linearized system dynamics matrix for relmo
    A = ComputeRelmoDynamicsMatrix(x, y, z, mu);
    
    # Compute nonlinear derivatives for target satellite in RLP frame
    targetStateDerivs = ComputeNonlinearDerivs(x, y, z, xdot, ydot, zdot, mu)
    
    # Compute derivates for offset of chaser wrt target in RLP frame using linearized relmo dynamics
    chaserStateDerivs = np.dot(A, chaserInputState)
    
    # Concatenate derivatives
    derivs = np.concatenate((targetStateDerivs, chaserStateDerivs))
    
    return derivs


# <headingcell level=3>

# Waypoint Targeting

# <codecell>


# Compute required velocity at point 1 to take us to point 2 within time (t2-t1)
# This formula is from Lian et al.
def ComputeRequiredVelocity(initialstate1, initialRelativePosition, initialTime, targetRelativePosition, targetTime, mu):
        
    # initial state of the target SC and STM
    # initial state for the STM is just the identity matrix mapping from t1 to t1
    I6 = np.eye(6);
    initialstateForSTM = np.concatenate((initialstate1, I6.reshape(1,36)[0]))

    # array of time points to integrate over to compute the STM
    timespan = np.linspace(initialTime, targetTime, 500)

    # integrate first satellite and STM from t1 to t2
    statesOverTime1 = integrate.odeint(odeintNonlinearDerivsWithLinearRelmoSTM, initialstateForSTM, timespan, (mu,))  # "extra arguments must be given in a tuple"

    # transpose so that timepoints are columns and elements of the state are rows
    statesOverTime1 = statesOverTime1.T

    # select rows 7-42 (36 rows)
    Phi = statesOverTime1[6:42]

    # select the last column (last time point), and convert it into a 6x6 matrix
    Phi = Phi[:,-1].reshape(6,6)

    # pull out top left corner and top right corner
    # these are the state transition matrices of the (position at time 2) with 
    # respect to the (position at time 1) and (velocity at time 1)
    Phi11 = Phi[:3, :3]
    Phi12 = Phi[:3, 3:6]

    # Invert Phi12 to get the (velocity at time 1) with respect to the (positions at times 1 and 2)
    Phi12I = np.linalg.inv(Phi12)
    
    # Compute required velocity at point 1 to take us to point 2 within time (t2-t1)
    # This formula is from Lian et al.
    initialRelativeVelocity = np.dot(Phi12I, targetRelativePosition - np.dot(Phi11, initialRelativePosition))
    
    return initialRelativeVelocity


def PropagateSatelliteAndChaser(mu, timespan, initialstate1, initialRelativeState):
    
    ## FIRST SATELLITE NONLINEAR AND SECOND SATELLITE LINEAR RELMO
    
    # initial state of first satellite in absolute RLP coordinates and second satellite wrt first
    initialstateForRelmo = np.concatenate(( initialstate1, initialRelativeState ))

    # integrate first and second satellites and STM from t1 to t2
    statesOverTime1 = integrate.odeint(odeintNonlinearDerivsWithLinearRelmo, initialstateForRelmo, timespan, (mu,))  # "extra arguments must be given in a tuple"

    # transpose so that timepoints are columns and elements of the state are rows
    statesOverTime1 = statesOverTime1.T

    # target satellite position and velocity over time in RLP frame from integrating initial state with full nonlinear dynamics
    primaryStatesOverTime = statesOverTime1[0:6]  # rows 1-6
    x1, y1, z1, xdot1, ydot1, zdot1 = primaryStatesOverTime
    
    # offset between target and chaser satellite over time in RLP frame from integrating initial offset with linearized relmo dynamics
    relativeStatesFromLinearRelmoOverTime = statesOverTime1[6:12] # rows 7-12
    dx_LINEAR, dy_LINEAR, dz_LINEAR, dxdot_LINEAR, dydot_LINEAR, dzdot_LINEAR = relativeStatesFromLinearRelmoOverTime

    # set up target satellite "ephemeris" DataFrame
    target_satellite = pd.DataFrame({
        "x": x1,
        "y": y1,
        "z": z1,
        "x_dot": xdot1,
        "y_dot": ydot1,
        "z_dot": zdot1}, index=timespan)
    
    # reassign so that the data maintains the required order for its values
    target_satellite = target_satellite.loc[:, ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']]

    # set up chaser satellite offset "ephemeris" DataFrame
    offset_linear = pd.DataFrame({
        "x": dx_LINEAR,
        "y": dy_LINEAR,
        "z": dz_LINEAR,
        "x_dot": dxdot_LINEAR,
        "y_dot": dydot_LINEAR,
        "z_dot": dzdot_LINEAR}, index=timespan)
    
    # reassign so that the data maintains the required order for its values
    offset_linear = offset_linear.loc[:, ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']]
    
    return target_satellite, offset_linear

# <codecell>


import matplotlib.pyplot as plt

def TargetRequiredVelocity(target_initial_state, chaser_velocity_initial_guess, current_waypoint, start, next_waypoint, end, mu):
    
    # all vectors in RLP
    
    #print 'next_waypoint'
    #print next_waypoint
    
    perturbation = 0.00001
    tolerance = 0.000000001
    max_iterations = 10.0
    
    chaser_velocity_next_guess = chaser_velocity_initial_guess.copy()
    
    timespan = np.linspace(start, end, 500)
    
    iteration_count = 0.0
    errors = np.array([100, 100, 100])
    
    while (np.any(np.abs(errors) > tolerance)) and (iteration_count < max_iterations):
        
        #print 'chaser_velocity_next_guess'
        #print chaser_velocity_next_guess
        
        chaser_initial_state = pd.Series({
                                        'x':     target_initial_state.x - current_waypoint.x,
                                        'y':     target_initial_state.y - current_waypoint.y,
                                        'z':     target_initial_state.z - current_waypoint.z,
                                        'x_dot': chaser_velocity_next_guess.x_dot,
                                        'y_dot': chaser_velocity_next_guess.y_dot,
                                        'z_dot': chaser_velocity_next_guess.z_dot})

        # reassign so that the series maintains the required order for its values
        chaser_initial_state = chaser_initial_state.loc[['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']]

        # propagate target and chaser each using full nonlinear dynamics
        target_ephem = PropagateSatellite(mu, timespan, target_initial_state)
        chaser_ephem = PropagateSatellite(mu, timespan, chaser_initial_state)

        achieved_relative_position_nominal = target_ephem.loc[end, ['x', 'y', 'z']] - chaser_ephem.loc[end, ['x', 'y', 'z']]
        #print 'achieved_relative_position_nominal'
        #print achieved_relative_position_nominal
        
        partials_matrix = np.zeros((3,3))

        for index in [3, 4, 5]:

            # apply perturbation to current element of velocity vector
            chaser_initial_state.iloc[index] += perturbation
            
            #print 'index', index

            # propagate target and chaser each using full nonlinear dynamics
            target_ephem = PropagateSatellite(mu, timespan, target_initial_state)
            chaser_ephem = PropagateSatellite(mu, timespan, chaser_initial_state)

            achieved_relative_position_perturbed = target_ephem.loc[end, ['x', 'y', 'z']] - chaser_ephem.loc[end, ['x', 'y', 'z']]
            #print 'achieved_relative_position_perturbed'
            #print achieved_relative_position_perturbed

            partials_row = (achieved_relative_position_perturbed - achieved_relative_position_nominal)/perturbation
            #print 'partials row'
            #print partials_row

            partials_matrix[index-3,:] = partials_row

            chaser_initial_state.iloc[index] -= perturbation

        #print 'partials matrix'
        #print partials_matrix
        
        inverse_partials = np.linalg.inv(partials_matrix)

        #print 'inverse_partials'
        #print inverse_partials
        
        errors = next_waypoint - achieved_relative_position_nominal
        
        #print 'errors'
        #print errors
        
        correction = np.dot(inverse_partials, errors)
        
        #print 'correction'
        #print correction

        #chaser_velocity_next_guess[['x_dot', 'y_dot', 'z_dot']] = chaser_initial_state[['x_dot', 'y_dot', 'z_dot']] + np.dot(inverse_partials, errors)
        chaser_velocity_next_guess.loc[['x_dot', 'y_dot', 'z_dot']] += correction
        
        iteration_count += 1.0
        
        #plt.plot([iteration_count], [np.linalg.norm(errors)])
        #plt.show()
    
    chaser_initial_velocity_targeted = chaser_velocity_next_guess
    print 'start time', start,  'end time', end, 'iteration count', iteration_count
    
    return chaser_initial_velocity_targeted

# <codecell>


def PropagateSatellite(mu, timespan, initialstate):
    
    mu = mu;
    
    initialstate_internal = initialstate.copy() # so that initialstate is not modified outside this function # TODO: find other places where I need to be using copy()
    
    # integrate first satellite
    statesOverTime = integrate.odeint(odeintNonlinearDerivs, initialstate_internal, timespan, (mu,))  # "extra arguments must be given in a tuple"

    #timespan, statesOverTime, EventTime, EventState, EventIndex = odelay(nonlinearDerivativesFunction, initialstate, timespan, events=[stop_zEquals0])

    #print 'initialstate = ', initialstate
    #print 't = ', t # timestamps corresponding to the output states over time
    #print 'statesOverTime = ', statesOverTime
    #print 'EventTime = ', EventTime
    #print 'EventState = ', EventState
    #print 'EventIndex = ', EventIndex
    #print len(timespan)

    x, y, z, x_dot, y_dot, z_dot = statesOverTime.T
    
    ephem = pd.DataFrame({
        "x": x,
        "y": y,
        "z": z,
        "x_dot": x_dot,
        "y_dot": y_dot,
        "z_dot": z_dot}, index=timespan)
    
    # reassign so that the data maintains the required order for its values
    ephem = ephem.loc[:, ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']]
    
    return ephem
   

# <codecell>


def ComputeOffsets(timespan, ephem1, ephem2):

    # compute trajectory offset in RLP frame
    dx = ephem1.x - ephem2.x
    dy = ephem1.y - ephem2.y
    dz = ephem1.z - ephem2.z
    #dxdot = xdot1 - xdot2
    #dydot = ydot1 - ydot2
    #dzdot = zdot1 - zdot2
    
    offsets = pd.DataFrame({
        "x": dx,
        "y": dy,
        "z": dz}, index=timespan)
    
    return offsets

# <codecell>


def ConvertOffset(input_offset, rotation_matrix):
    
    # compute trajectory offset in new frame (e.g. RIC, VNB)
        
    # input_offset is the input offset vector (dx, dy, dz)
    # rotation_matrix is the matrix formed by the basis vectors (basis1,basis2,basis3) converting from the input frame to the output frame
    # output_offset is the output offset vector (db1, db2, db3)
    
    # compute dot products
    output_offset = np.dot(input_offset, rotation_matrix)
    
    return output_offset


def ConvertOffsets(input_offset, rotation_matrix):
    
    # compute trajectory offset in new frame (e.g. RIC, VNB)
        
    # input_offset is the input offset vector (dx, dy, dz)
    # rotation_matrix is the matrix formed by the basis vectors (basis1,basis2,basis3) converting from the input frame to the output frame
    # output_offset is the output offset vector (db1, db2, db3)
    
    # input_offset is Nx3
    # rotation_matrix is Nx3x3
    # want output_offset to be Nx3
    
    output_offset = np.einsum('ij,ijk->ik', input_offset, rotation_matrix)
    
    return output_offset

# <codecell>


def BuildRICFrame(state, center):
    
    # build RIC frame based on state
    rVec = state.loc[["x", "y", "z"]] - center  # this is a Series
    vVec = state.loc[["x_dot", "y_dot", "z_dot"]]
    
    cVec = np.cross(rVec, vVec)
    iVec = np.cross(cVec, rVec)
    
    # unitize RIC frame vectors
    rVec /= np.linalg.norm(rVec)
    iVec /= np.linalg.norm(iVec)
    cVec /= np.linalg.norm(cVec)
    
    RLPtoRIC = np.dstack((rVec, iVec, cVec)) # this is an ndarray
    
    return RLPtoRIC

def BuildVNBFrame(state, center):
    
    # build VNB frame based on state
    rVec = state.loc[["x", "y", "z"]] - center
    vVec = state.loc[["x_dot", "y_dot", "z_dot"]]

    nVec = np.cross(rVec, vVec)
    bVec = np.cross(vVec, nVec)

    # unitize VNB frame vectors
    vVec /= np.linalg.norm(vVec)
    nVec /= np.linalg.norm(nVec)
    bVec /= np.linalg.norm(bVec)
    
    RLPtoVNB = np.dstack((vVec, nVec, bVec))
        
    return RLPtoVNB


def BuildRICFrames(ephem, center):
    
    # build RIC frame based on state
    rVec = (ephem.loc[:, ["x", "y", "z"]] - center).values     # this is a 500x3 ndarray
    vVec = (ephem.loc[:, ["x_dot", "y_dot", "z_dot"]]).values
    
    cVec = np.cross(rVec, vVec)
    iVec = np.cross(cVec, rVec)
    
    # unitize RIC frame vectors
    rVec /= np.linalg.norm(rVec, axis=1)[:, np.newaxis]
    iVec /= np.linalg.norm(iVec, axis=1)[:, np.newaxis]
    cVec /= np.linalg.norm(cVec, axis=1)[:, np.newaxis]
    # need newaxis or else get 'ValueError: operands could not be broadcast together with shapes (500,3) (500) (500,3)'
    
    RLPtoRIC_matrices = np.dstack((rVec, iVec, cVec)) # this is an ndarray
    
    # TODO: do something like this so that the input can be a series or a dataframe
    #if type(ephem) == pd.Series:
    #    items = 1
    #elif type(ephem) == pd.DataFrame:
    #    items = ephem.index
    
    RLPtoRIC = pd.Panel(RLPtoRIC_matrices,
                        items=ephem.index, 
                        major_axis=list("RIC"),
                        minor_axis=list("xyz"))
    
    return RLPtoRIC


def BuildVNBFrames(ephem, center):
    
    # build VNB frame based on state
    rVec = (ephem.loc[:, ["x", "y", "z"]] - center).values
    vVec = (ephem.loc[:, ["x_dot", "y_dot", "z_dot"]]).values

    nVec = np.cross(rVec, vVec)
    bVec = np.cross(vVec, nVec)

    # unitize VNB frame vectors
    vVec /= np.linalg.norm(vVec, axis=1)[:, np.newaxis]
    nVec /= np.linalg.norm(nVec, axis=1)[:, np.newaxis]
    bVec /= np.linalg.norm(bVec, axis=1)[:, np.newaxis]
    
    RLPtoVNB_matrices = np.dstack((vVec, nVec, bVec))
        
    RLPtoVNB = pd.Panel(RLPtoVNB_matrices,
                        items=ephem.index, 
                        major_axis=list("VNB"),
                        minor_axis=list("xyz"))
    
    return RLPtoVNB

