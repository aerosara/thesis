# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


import numpy as np
    

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
    
    # Compute the locations of the libration points

    # ported this from Matlab code found at http://www.math.rutgers.edu/~jmireles/matLabPage.html - need to QA against a textbook

    # In numpy.roots(p), p[0] corresponds to highest-order term in polynomial

    l = 1.0 - mu

    # L1
    p_L1 = [1, 2.0*(mu-l), l**2.0 - 4.0*l*mu + mu**2.0, 2.0*mu*l*(l-mu) + mu-l, mu**2.0*l**2.0 + 2.0*(l**2.0 + mu*2.0), mu**3.0 - l**3.0]

    L1 = np.real([i for i in np.roots(p_L1) if (i > -mu and i < l)])

    L1 = np.append(L1, [0.0, 0.0])

    # L2
    p_L2 = [1, 2.0*(mu-l), l**2.0 - 4.0*l*mu + mu**2.0, 2*mu*l*(l-mu) - (mu+l), mu**2.0*l**2.0 + 2.0*(l**2.0 - mu**2.0), -(mu**3.0 + l**3.0)]

    L2 = np.real([i for i in np.roots(p_L2) if (i > -mu and i > l)])

    L2 = np.append(L2, [0.0, 0.0])

    # L3
    p_L3 = [1.0, 2.0*(mu-l), l**2.0 - 4.0*mu*l + mu**2.0, 2.0*mu*l*(l-mu) + (l+mu), mu**2.0*l**2.0 + 2*(mu**2 - l**2.0), l**3.0 + mu**3.0]

    L3 = np.real([i for i in np.roots(p_L3) if i < -mu])

    L3 = np.append(L3, [0.0, 0.0])

    # L4
    L4 = [-mu + 0.5, np.sqrt(3.0)/2.0, 0.0]

    # L5
    L5 = [-mu + 0.5, -np.sqrt(3.0)/2.0, 0.0]
    
    return L1, L2, L3, L4, L5

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

# <codecell>


# Not being used - don't have a decent test case with good values for the input state and BL1 (aka c2)
def linearDerivativesFunction(inputstate, timespan):
    x, y, z, xdot, ydot, zdot = inputstate
    
    #BL1 = 3.329168
    #BL1 = 4.06107
    BL1 = 0.012155092
    
    derivs = [xdot,
              ydot,
              zdot,
              2.0*ydot + (2.0*BL1 + 1.0)*x,
              -2.0*xdot - (BL1 - 1)*y,
              -BL1*z]
    
    return derivs

# This full nonlinear version is what's being used
def nonlinearDerivativesFunction(inputstate, timespan, mu):
    
    x, y, z, xdot, ydot, zdot = inputstate
    
    # distances
    r1 = np.sqrt((mu+x)**2.0 + y**2.0 + z**2.0);
    r2 = np.sqrt((1-mu-x)**2.0 + y**2.0 + z**2.0);
    
    # masses
    #m1 = 1 - mu;
    #m2 = mu;
    #G = 1;

    derivs = [xdot, 
              ydot,
              zdot, 
              x + 2.0*ydot + (1 - mu)*(-mu - x)/(r1**3.0) + mu*(1 - mu - x)/(r2**3.0),
              y - 2.0*xdot - (1 - mu)*y/(r1**3.0) - mu*y/(r2**3.0),
              -(1 - mu)*z/(r1**3.0) - mu*z/(r2**3.0)]
    
    return derivs

# <codecell>


def PropagateSatellite(mu, timespan, initialstate1):
    
    import scipy.integrate as integrate
    
    mu = mu;
    
    # integrate first satellite
    statesOverTime1 = integrate.odeint(nonlinearDerivativesFunction, initialstate1, timespan, (mu,))  # "extra arguments must be given in a tuple"

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


