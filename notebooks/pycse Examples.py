# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=3>

# Example from pycse 1

# <codecell>


# copied from http://kitchingroup.cheme.cmu.edu/blog/tag/events/

from pycse import odelay
import matplotlib.pyplot as plt
import numpy as np

def ode(Y,x):
    y1, y2 = Y
    dy1dx = y2
    dy2dx = -y1
    return [dy1dx, dy2dx]

def event1(Y, x):
    y1, y2 = Y
    value = y2 - (-1.0)
    isterminal = True
    direction  = 0
    return value, isterminal, direction

def event2(Y, x):
    dy1dx, dy2dx = ode(Y,x)
    value = dy1dx - 0.0
    isterminal = False
    direction = -1  # derivative is decreasing towards a maximum
    return value, isterminal, direction

Y0 = [2.0, 1.0]
xspan = np.linspace(0, 5)
X, Y, XE, YE, IE = odelay(ode, Y0, xspan, events=[event1, event2])

plt.plot(X, Y)
for ie,xe,ye in zip(IE, XE, YE):
    if ie == 1: #this is the second event
        y1,y2 = ye
        plt.plot(xe, y1, 'ro') 
        
plt.legend(['$y_1$', '$y_2$'], loc='best')
#plt.savefig('images/odelay-mult-eq.png')
plt.show()

# <headingcell level=3>

# Example from pycse 2

# <codecell>


# copied from: http://kitchingroup.cheme.cmu.edu/pycse/pycse.html#sec-10-1-8

# 10.1.8 Stopping the integration of an ODE at some condition

from pycse import *
import numpy as np

k = 0.23
Ca0 = 2.3

def dCadt(Ca, t):
    return -k * Ca**2

def stop(Ca, t):
    isterminal = True
    direction = 0
    value = 1.0 - Ca
    return value, isterminal, direction

tspan = np.linspace(0.0, 10.0)

t, CA, TE, YE, IE = odelay(dCadt, Ca0, tspan, events=[stop])

print 'At t = {0:1.2f} seconds the concentration of A is {1:1.2f} mol/L.'.format(t[-1], float(CA[-1]))

# <headingcell level=3>

# fsolve example

# <codecell>

from math import cos

def func(x):
    return x + 2*cos(x)  # finds where this is zero

def func2(x):
    out = [x[0]*cos(x[1]) - 4]
    out.append(x[1]*x[0] - x[1] - 5)
    return out  # finds where both elements of this array are zero


from scipy.optimize import fsolve
x0 = fsolve(func, 0.3)  # initial guess
print x0
print func(x0)
#-1.02986652932

x02 = fsolve(func2, [1, 1]) # initial guesses
print x02
print func2(x02)
#[ 6.50409711  0.90841421]

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

from thesis_functions.initialconditions import InputDataDictionary, SetInitialConditions
from thesis_functions.astro import FindOrbitCenter, ComputeLibrationPoints, stop_yEquals0, stop_zEquals0, linearDerivativesFunction, nonlinearDerivativesFunction, PropagateSatellite, BuildRICFrame, BuildVNBFrame
from thesis_functions.visualization import PlotGrid


def stop_maxTime(state, t):
    isterminal = True
    direction = 0
    value = abs(t)-2  # stop if time is greater than 2 units
    return value, isterminal, direction

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

from thesis_functions.initialconditions import InputDataDictionary, SetInitialConditions
from thesis_functions.astro import FindOrbitCenter, ComputeLibrationPoints, stop_yEquals0, stop_zEquals0, linearDerivativesFunction, nonlinearDerivativesFunction, PropagateSatellite, BuildRICFrame, BuildVNBFrame
from thesis_functions.visualization import PlotGrid


def stop_maxTime(state, t):
    isterminal = True
    direction = 0
    value = abs(t)-2  # stop if time is greater than 2 units
    return value, isterminal, direction

ICs = InputDataDictionary()

mu, timespan, initialstate1 = SetInitialConditions(ICs, ICset = 'Howell', ICtestcase = 0, numPoints = 200)
    
L1, L2, L3, L4, L5 = ComputeLibrationPoints(mu)

x1, y1, z1, xdot1, ydot1, zdot1 = PropagateSatellite(mu, timespan, initialstate1)
    
center = FindOrbitCenter(x1, y1, z1);

# Plot satellite 1 in RLP frame
data = {'sat1': {'x':x1, 'y':y1, 'z':z1}}
points = {'L1': L1, 'center': center}
PlotGrid('Satellite 1 in RLP Frame', 'X', 'Y', 'Z', data, points, 'equal')
 


muArray_H = [0.04]
testcases_H = np.array([[0.723268, 0.040000, 0.198019, 1.300177*4.0]]) # x, z, ydot, T

testcase = 0
mu = muArray_H[testcase]
timespan = np.linspace(0, testcases_H[testcase, 3], 2000) # Howell
initialstate_H = [testcases_H[testcase, 0], 0, testcases_H[testcase,1], 0, testcases_H[testcase, 2], 0]  # Howell

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

def ForceForwardToNextEvent(inputstate, eventType):
    
    smallTimeStep = 1.0e-3
    TimeOfNextEvent = 0
    initialstate = inputstate
    
    #while (TimeOfNextEvent == 0):  # for testing (runs while loop once)
    while (TimeOfNextEvent < smallTimeStep):
        
        t, statesOverTime, EventTime, EventState, EventIndex = odelay(nonlinearDerivativesFunction, initialstate, timespan, events=[eventType, stop_maxTime])
        
        if (len(EventTime) > 0):
            TimeOfNextEvent = EventTime[0]
        
        # If odelay didn't actually step forward, force the state forward a small amount and then try again
        if (TimeOfNextEvent < smallTimeStep):
            
            # take one small step
            computeOneStep = integrate.odeint(nonlinearDerivativesFunction, initialstate, [0.0, smallTimeStep])
            
            initialstate = computeOneStep[1]
    
    return EventTime, EventState

def ForceForwardNEvents(inputstate, eventType, nEvents):
    
    # propagate forward the specified numEventsToLookAhead
    eState = np.zeros((1,6))
    eState[0] = inputstate
    for x in xrange(0, numEventsToLookAhead):
        eTime, eState = ForceForwardToNextEvent(inputstate, eventType)  
        # seems like you can't look ahead more than 2-3 plane crossings without doing deltaV's
        
    return eTime, eState
    

# step to abs(y) < 1e-11 (per Howell)

# adjust so xdot, zdot are zero (within 1e-8) when y is zero
# can adjust: initial x, z, or ydot

def FuncToSolve(initialguesses, inputstate, holdXorZ, numEventsToLookAhead):
    ''' initialguesses will contain [x and ydot] or [z and ydot], depending on value of holdXorZ. 
        inputstate will contain the full position and velocity state.
        the values inside initialguesses will be adjusted to get perpendicular XZ plane crossings after numEventsToLookAhead.'''
    
    # really this section of code shouldn't change anything because the initialguesses should already be equal to the appropriate components of the inputstate... but making sure that everything is consistent..
    if (holdXorZ == 'X'):
        inputstate[2] = initialguesses[0]
        inputstate[4] = initialguesses[1]
    elif (holdXorZ == 'Z'):
        inputstate[0] = initialguesses[0]
        inputstate[4] = initialguesses[1]
    elif (holdXorZ == 'both'):
        inputstate[4] = initialguesses
        
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
        initialguesses = inputstate[4]
    
    # refine initial guesses
    ''' from fsolve documentation at http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html:
    scipy.optimize.fsolve(func, x0, args=(), ...)
    func: A function that takes at least one (possibly vector) argument.
    x0: The starting estimate for the roots of func(x) = 0.
    args: Any extra arguments to func. '''
    solution = fsolve(FuncToSolve, initialguesses, (inputstate, holdXorZ, numEventsToLookAhead))

    # print the update applied to X/Z and Vy
    print 'solution = ', solution, 'solution-initialguesses = ', solution-initialguesses_x_ydot  

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
    
    return haloState

# <headingcell level=3>

# Set input state and call Haloize

# <codecell>

# test case
initialguesses_x_ydot = [0.723268, 0.198019] # x, ydot
initz = 0.04000

initialguesses_x_ydot = [0.723268, 0.198019] # x, ydot
initz = 0.050

inputstate = [initialguesses_x_ydot[0], 0.0, initz, 0.0, initialguesses_x_ydot[1], 0.0]

#inputstate = [0.72311425511821148, 0.0, 0.05, 0.0, 0.20832957653736492, 0.0]  # solution from Z = 0.05
inputstate = [0.723, 0.0, 0.05, 0.0, 0.208, 0.0]

print 'inputstate = ', inputstate


# inputs to halo-ization
numEventsToLookAhead = 1
holdXorZ = 'both'  # 'X' 'Z' 'both'


# result from function using initial guesses (what are the Vx and Vz components at the 
# specified XZ plane crossing when just using the input state?  fsolve will adjust inputs so that this result gets close to zero)
eTime, initialGuessResult = ForceForwardNEvents(inputstate, stop_yEquals0, numEventsToLookAhead)  
print 'initialGuessResult = ', initialGuessResult

# we have the option to hold the X or Z component constant; the other component (X or Z)
# and the Ydot will be adjusted to get a halo orbit (perpendicular XZ plane crossings)
haloState = Haloize(inputstate, holdXorZ, numEventsToLookAhead)  
print 'haloState = ', haloState

# result from function using solution (Vx and Vz should be close to zero)
eTime, SolutionResult = ForceForwardNEvents(haloState, stop_yEquals0, numEventsToLookAhead)  
print 'SolutionResult = ', SolutionResult

# <codecell>


# determine orbit period
eTime, eState = ForceForwardToNextEvent(haloState, stop_yEquals0)

# propagate for X orbits (eTime[0]*4.0 = 2 orbits)
timespan = np.linspace(0.0, eTime[0]*2.0, 2000)

statesOverTime = integrate.odeint(nonlinearDerivativesFunction, haloState, timespan)

print 'eTime[0] = ', eTime[0]

# <codecell>


x1, y1, z1, xdot1, ydot1, zdot1 = statesOverTime.T

# identify x-coordinate corresponding to maximum y-amplitude, 
#          y-coordinate corresponding to maximum x-amplitude, 
#          and z-coordinate corresponding to maximum y-amplitude  #y=0
center = [x1[y1.argmax()], y1[x1.argmax()], z1[y1.argmax()]]  #np.abs(y1).argmin()]]
          
print 'center = ', center

# plot in RLP
figRLP, ((axXZ, axYZ), (axXY, ax3D)) = plt.subplots(2, 2)
figRLP.suptitle('Satellite 1 in RLP Frame')
figRLP.set_size_inches(10, 10)
figRLP.subplots_adjust(hspace=0.2)
figRLP.subplots_adjust(wspace=0.5)

# XZ Plane
axXZ.plot(x1, z1, 'r-')
#axXZ.plot(L1[0], L1[2], 'g*', markersize=10)
axXZ.plot(center[0], center[2], 'w*', markersize=10)
axXZ.set_title('XZ Plane')
axXZ.xaxis.set_label_text('X axis')
axXZ.yaxis.set_label_text('Z axis')
axXZ.set_aspect('equal')

# YZ Plane
axYZ.plot(y1, z1, 'r-')
#axYZ.plot(L1[1], L1[2], 'g*', markersize=10)
axYZ.plot(center[1], center[2], 'w*', markersize=10)
axYZ.set_title('YZ Plane')
axYZ.xaxis.set_label_text('Y axis')
axYZ.yaxis.set_label_text('Z axis')
axYZ.set_aspect('equal')

# XY Plane
axXY.plot(x1, y1, 'r-')
#axXY.plot(L1[0], L1[1], 'g*', markersize=10)
axXY.plot(center[0], center[1], 'w*', markersize=10)
axXY.set_title('XY Plane')
axXY.xaxis.set_label_text('X axis')
axXY.yaxis.set_label_text('Y axis')
axXY.set_aspect('equal')

# 3D View
ax3D.axis('off')
ax3D = figRLP.add_subplot(224, projection='3d')
ax3D.plot(x1, y1, z1, 'r-')
#ax3D.plot([L1[0]], [L1[1]], [L1[2]], 'g*', markersize=10)
ax3D.plot([center[0]], [center[1]], [center[2]], 'w*', markersize=10)
ax3D.set_title('3D View in RLP Frame')
ax3D.xaxis.set_label_text('X axis')
ax3D.yaxis.set_label_text('Y axis')
ax3D.zaxis.set_label_text('Z axis')
#ax3D.set_aspect('equal')

# <codecell>


# <codecell>


# <codecell>


