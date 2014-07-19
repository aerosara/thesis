# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=3>

# CRTBP Derivatives, Stopping Conditions, and Initial Conditions

# <codecell>


%reset
%pylab inline

import scipy.integrate as integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from pycse import odelay
from scipy.optimize import fsolve

from thesis_functions.astro import FindOrbitCenter, ComputeLibrationPoints, stop_yEquals0, stop_zEquals0, PropagateSatellite, BuildRICFrame, BuildVNBFrame  #nonlinearDerivativesFunction
from thesis_functions.visualization import PlotGrid

def stop_maxTime(state, t):
    isterminal = True
    direction = 0
    value = abs(t)-3  # stop if time is greater than 2 units
    return value, isterminal, direction

def nonlinearDerivativesFunction(inputstate, timespan):
    
    x, y, z, xdot, ydot, zdot = inputstate
    
    # distances
    r1 = np.sqrt((mu+x)**2.0 + y**2.0 + z**2.0);
    r2 = np.sqrt((1-mu-x)**2.0 + y**2.0 + z**2.0);
    
    derivs = [xdot, 
              ydot,
              zdot, 
              x + 2.0*ydot + (1 - mu)*(-mu - x)/(r1**3.0) + mu*(1 - mu - x)/(r2**3.0),
              y - 2.0*xdot - (1 - mu)*y/(r1**3.0) - mu*y/(r2**3.0),
              -(1 - mu)*z/(r1**3.0) - mu*z/(r2**3.0)]
    
    return derivs

#muArray_H = [0.04]
#testcases_H = np.array([[0.723268, 0.040000, 0.198019, 1.300177*4.0]]) # x, z, ydot, T

testcase = 0
#mu = muArray_H[testcase]
#timespan = np.linspace(0, testcases_H[testcase, 3], 2000) # Howell
#initialstate_H = [testcases_H[testcase, 0], 0, testcases_H[testcase,1], 0, testcases_H[testcase, 2], 0]  # Howell

# create a perturbed initial state
#initialstate2 = np.zeros(len(initialstate_H))
#initialstate2[0:3] = initialstate_H[0:3]
#initialstate2[3:6] = [x * 1.001 for x in initialstate_H[3:6]]

#print initialstate2

# <headingcell level=3>

# Define functions: ForceForwardToNextEvent, ForceForwardNEvents, FuncToSolve, Haloize

# <codecell>


''' from the pysce source code at https://github.com/jkitchin/pycse/blob/2c9ff7fe81b0dfb34b3edf87356817312910b799/pycse/PYCSE.py#L78
odelay(func, y0, xspan, events, fsolve_args=None, **kwargs):
    ode wrapper with events func is callable, with signature func(Y, x)
    y0 are the initial conditions xspan is what you want to integrate
    over

    events is a list of callable functions with signature event(Y, x).
    These functions return zero when an event has happened.
    
    [value, isterminal, direction] = event(Y, x)
    value is the value of the event function. When value = 0, an event is  triggered

    isterminal = True if the integration is to terminate at a zero of
    this event function, otherwise, False.

    direction = 0 if all zeros are to be located (the default), +1
    if only zeros where the event function is increasing, and -1 if
    only zeros where the event function is decreasing.

    fsolve_args is a dictionary of options for fsolve
    kwargs are any additional options you want to send to odeint.
    '''

def ForceForwardToNextEvent(initialstate, eventType):
    
    # take a small step forward (0.1 time units) so that the integration is sure to move forward from the current state
    shorttimespan = np.linspace(0.0, 0.1, 10)
    computeOneStep = integrate.odeint(nonlinearDerivativesFunction, initialstate, shorttimespan)
    
    # this is the new initial state (last vector from short propagation above)
    initialstate = computeOneStep[9]
    
    # step to the next event (given maximum time constraint)
    t, statesOverTime, EventTime, EventState, EventIndex = odelay(nonlinearDerivativesFunction, initialstate, timespan, events=[eventType, stop_maxTime])
    
    EventTime = EventTime + 0.1;
    
    return EventTime, EventState

def ForceForwardNEvents(inputstate, eventType, nEvents):
    
    # propagate forward the specified nEvents
    for x in xrange(0, nEvents):
        eTime, eState = ForceForwardToNextEvent(inputstate, eventType)
        inputstate = eState[0]
        # seems like you can't look ahead more than 2-3 plane crossings without doing deltaV's
        
    return eTime, eState
    

# step to abs(y) < 1e-11 (per Howell)

# adjust so xdot, zdot are zero (within 1e-8) when y is zero
# can adjust: initial x, z, or ydot

def FuncToSolve(initialguesses, inputstate, holdXorZ, numEventsToLookAhead):
    ''' initialguesses will contain [x and ydot] or [z and ydot] or just [ydot], depending on value of holdXorZ. 
        inputstate will contain the full position and velocity state.
        the values inside initialguesses will be adjusted to get perpendicular XZ plane crossings after numEventsToLookAhead.'''
    
    # set the initialguesses into the appropriate components of the inputstate
    if (holdXorZ == 'X'):
        inputstate[2] = initialguesses[0]  # Z
        inputstate[4] = initialguesses[1]  # Ydot
    elif (holdXorZ == 'Z'):
        inputstate[0] = initialguesses[0]  # X
        inputstate[4] = initialguesses[1]  # Ydot
    elif (holdXorZ == 'both'):
        # limit to 1% updates
        #if ((abs(initialguesses[0]) - abs(inputstate[4]))/abs(inputstate[4]) < 0.01):
        inputstate[4] = initialguesses[0]  # Ydot
            #inputstate[5] = initialguesses[1]
        
    # propagate forward the specified numEventsToLookAhead
    eTime, eState = ForceForwardNEvents(inputstate, stop_yEquals0, numEventsToLookAhead)
    
    result = [eState[0,3], eState[0,5]]  # Vx, Vz = 0 gives a perpendicular XZ plane crossing
    
    return result  # fsolve will look for a solution that makes the results = 0


def Haloize(inputstate, holdXorZ, numEventsToLookAhead):
    
    # the 'initialguesses' vector that gets passed to the FuncToSolve has to be the parameters 
    # that fsolve is allowed to vary in order to get the results to be zero
    if (holdXorZ == 'X'):
        initialguesses = [inputstate[2], inputstate[4]]  # Z and Ydot
    elif (holdXorZ == 'Z'):
        initialguesses = [inputstate[0], inputstate[4]]  # X and Ydot
    elif (holdXorZ == 'both'):
        #initialguesses = [inputstate[4], inputstate[5]]  # Ydot and Zdot
        initialguesses = inputstate[4]  # Ydot only
    
    # refine initial guesses
    ''' from fsolve documentation at http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html:
    scipy.optimize.fsolve(func, x0, args=(), ...)
    func: A function that takes at least one (possibly vector) argument.
    x0: The starting estimate for the roots of func(x) = 0.
    args: Any extra arguments to func. '''
    solution = fsolve(FuncToSolve, initialguesses, (inputstate, holdXorZ, numEventsToLookAhead))

    # print the update applied to X/Z and Vy
    print 'solution = ', solution, 'solution-initialguesses = ', solution-initialguesses  

    # TODO: make this work even if the input state isn't at a plane crossing... I guess you would need to let the solver adjust the input Vx and Vz (in addition to Vy) if you're not at the XZ plane crossing.......

    # copy solution from fsolve into haloState
    haloState = inputstate
    
    if (holdXorZ == 'X'):
        haloState[2] = solution[0]
        haloState[4] = solution[1]
    elif (holdXorZ == 'Z'):
        haloState[0] = solution[0]
        haloState[4] = solution[1]
    elif (holdXorZ == 'both'):
        haloState[4] = solution[0]
        #haloState[5] = solution[1]
    
    return haloState

# <headingcell level=3>

# Set input state and call Haloize

# <codecell>


# test case
#initialguesses_x_ydot = [0.723268, 0.198019] # x, ydot
#initz = 0.04000

#initialguesses_x_ydot = [0.723268, 0.198019] # x, ydot
#initz = 0.050

#inputstate = [initialguesses_x_ydot[0], 0.0, initz, 0.0, initialguesses_x_ydot[1], 0.0]

#inputstate = [0.72311425511821148, 0.0, 0.05, 0.0, 0.20832957653736492, 0.0]  # solution from Z = 0.05
mu = 0.04
timespan = np.linspace(0.0, 3.0, 200)
inputstate = [0.723, 0.0, 0.05, 0.0, 0.208, 0.0]

print 'inputstate = ', inputstate


# inputs to halo-ization
numEventsToLookAhead = 1
holdXorZ = 'both'  # 'X' 'Z' 'both'


# result from function using initial guesses (what are the Vx and Vz components at the 
# specified XZ plane crossing when just using the input state?  fsolve will adjust inputs so that this result gets close to zero)
eTime, initialGuessResult = ForceForwardNEvents(inputstate, stop_yEquals0, numEventsToLookAhead)  
print 'eTime = ', eTime, ', initialGuessResult = ', initialGuessResult

# we have the option to hold the X or Z component constant; the other component (X or Z)
# and the Ydot will be adjusted to get a halo orbit (perpendicular XZ plane crossings)
haloState = Haloize(inputstate, holdXorZ, numEventsToLookAhead)  
print 'haloState = ', haloState

# result from function using solution (Vx and Vz should be close to zero)
eTime, SolutionResult = ForceForwardNEvents(haloState, stop_yEquals0, numEventsToLookAhead)  
print 'eTime = ', eTime, ', SolutionResult = ', SolutionResult

# <codecell>


# determine orbit period
eTime, eState = ForceForwardToNextEvent(haloState, stop_yEquals0)

# propagate for X orbits (eTime[0]*4.0 = 2 orbits)
timespan = np.linspace(0.0, eTime[0]*2.0, 200)

statesOverTime = integrate.odeint(nonlinearDerivativesFunction, haloState, timespan)

print 'eTime[0] = ', eTime[0]

# <codecell>


x1, y1, z1, xdot1, ydot1, zdot1 = statesOverTime.T

center = FindOrbitCenter(x1, y1, z1);

data = {'sat1': {'x':x1, 'y':y1, 'z':z1}}
points = {'center': center}
PlotGrid('Satellite 1 in RLP Frame', 'X', 'Y', 'Z', data, points, 'equal')

