# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


import numpy as np
    
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
    
    return X1, X2, L1, L2, L3, L4, L5;

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


# Not being used - don't have a decent test case with good values for the input state and BL1 (aka c2)
#def linearDerivativesFunction(inputstate, timespan):
#    x, y, z, xdot, ydot, zdot = inputstate
#    
#    #BL1 = 3.329168
#    #BL1 = 4.06107
#    BL1 = 0.012155092
#    
#    derivs = [xdot,
#              ydot,
#              zdot,
#              2.0*ydot + (2.0*BL1 + 1.0)*x,
#              -2.0*xdot - (BL1 - 1.0)*y,
#              -BL1*z]
#    
#    return derivs


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


# This full nonlinear version is what's being used
def odeintNonlinearDerivs(inputstate, timespan, mu):
    
    x, y, z, xdot, ydot, zdot = inputstate
    
    # Compute nonlinear derivatives for the satellite in RLP frame
    derivs = ComputeNonlinearDerivs(x, y, z, xdot, ydot, zdot, mu)
    
    return derivs


# This is from Luquette
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
# This is from Lian et al.
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
    # This is from Lian et al.
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

    return x1, y1, z1, xdot1, ydot1, zdot1, dx_LINEAR, dy_LINEAR, dz_LINEAR, dxdot_LINEAR, dydot_LINEAR, dzdot_LINEAR

# <codecell>


def PropagateSatellite(mu, timespan, initialstate1):
    
    mu = mu;
    
    # integrate first satellite
    statesOverTime1 = integrate.odeint(odeintNonlinearDerivs, initialstate1, timespan, (mu,))  # "extra arguments must be given in a tuple"

    #timespan, statesOverTime1, EventTime, EventState, EventIndex = odelay(nonlinearDerivativesFunction, initialstate1, timespan, events=[stop_zEquals0])

    #print 'initialstate1 = ', initialstate1
    #print 't = ', t # timestamps corresponding to the output states over time
    #print 'statesOverTime = ', statesOverTime
    #print 'EventTime = ', EventTime
    #print 'EventState = ', EventState
    #print 'EventIndex = ', EventIndex
    #print len(timespan)

    x1, y1, z1, xdot1, ydot1, zdot1 = statesOverTime1.T
    
    return x1, y1, z1, xdot1, ydot1, zdot1
   

# <codecell>


def ComputeOffsets(timespan, x1, y1, z1, xdot1, ydot1, zdot1, x2, y2, z2, xdot2, ydot2, zdot2):

    # compute trajectory offset in RLP frame
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    #dxdot = xdot1 - xdot2
    #dydot = ydot1 - ydot2
    #dzdot = zdot1 - zdot2

    # temporarily disabling plot
    if (0 == 1):
        
        # compute total distance offset
        distance = np.linalg.norm(np.array([dx, dy, dz]),2,0)

        # plot total distance offset over time
        figDeltaMag, axDeltaMag = plt.subplots()
        axDeltaMag.plot(timespan, distance, 'o-')
        figDeltaMag.suptitle('Total Distance Offset Over Time')

    #figXY, axXDotYDot = plt.subplots()
    #axXDotYDot.plot(timespan, xdot)
    #axXDotYDot.plot(timespan, ydot)
    
    return dx, dy, dz

# <codecell>


def ConvertOffsets(dx, dy, dz, basis1, basis2, basis3):

    # x,y,z are input offset vectors
    # basis1,basis2,basis3 are basis vectors converting from the input frame to the output frame
    # db1,db2,db3 are the output offset vectors
    
    # compute trajectory offset in new frame (e.g. RIC, VNB)

    ## This approach is more intuitive:
    # compute dot products
    db1 = np.zeros(len(dx))
    db2 = np.zeros(len(dx))
    db3 = np.zeros(len(dx))
    
    for ii in range(0, len(basis1)):
        db1[ii] = np.dot([dx[ii], dy[ii], dz[ii]], basis1[ii])
        db2[ii] = np.dot([dx[ii], dy[ii], dz[ii]], basis2[ii])
        db3[ii] = np.dot([dx[ii], dy[ii], dz[ii]], basis3[ii])
    
    ## This approach might be faster:
    # compute dot products
    #db1 = np.einsum('ij,ij->i', np.array([dx, dy, dz]).T, basis1)
    #db2 = np.einsum('ij,ij->i', np.array([dx, dy, dz]).T, basis2)
    #db3 = np.einsum('ij,ij->i', np.array([dx, dy, dz]).T, basis3)
    
    return db1, db2, db3

def ConvertOffset(dx, dy, dz, basis1, basis2, basis3):

    # x,y,z are input offset vectors
    # basis1,basis2,basis3 are basis vectors converting from the input frame to the output frame
    # db1,db2,db3 are the output offset vectors
    
    # compute trajectory offset in new frame (e.g. RIC, VNB)

    # compute dot products
    db1 = np.dot([dx, dy, dz], basis1)
    db2 = np.dot([dx, dy, dz], basis2)
    db3 = np.dot([dx, dy, dz], basis3)

    return db1, db2, db3

# <codecell>


def BuildRICFrame(x1, y1, z1, xdot1, ydot1, zdot1, center):
    
    # build RIC frame based on satellite 1

    rVec = np.array([x1-center[0], y1-center[1], z1-center[2]]).T

    vVec = np.array([xdot1, ydot1, zdot1]).T

    cVec = np.cross(rVec, vVec)

    iVec = np.cross(cVec, rVec)
    
    # unitize RIC frame vectors
    rVec = np.divide(rVec, np.linalg.norm(rVec,2,1)[:,None])
    cVec = np.divide(cVec, np.linalg.norm(cVec,2,1)[:,None])
    iVec = np.divide(iVec, np.linalg.norm(iVec,2,1)[:,None])

    return rVec, iVec, cVec


def BuildVNBFrame(x1, y1, z1, xdot1, ydot1, zdot1, center):
    
    # build VNB frame based on satellite 1

    rVec = np.array([x1-center[0], y1-center[1], z1-center[2]]).T

    vVec = np.array([xdot1, ydot1, zdot1]).T

    nVec = np.cross(rVec, vVec)

    bVec = np.cross(vVec, nVec)

    # unitize VNB frame vectors
    vVec = np.divide(vVec, np.linalg.norm(vVec,2,1)[:,None])
    nVec = np.divide(nVec, np.linalg.norm(nVec,2,1)[:,None])
    bVec = np.divide(bVec, np.linalg.norm(bVec,2,1)[:,None])
        
    return vVec, nVec, bVec

# <codecell>


