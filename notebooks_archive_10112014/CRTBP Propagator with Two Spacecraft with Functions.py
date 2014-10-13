# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%reset
%pylab inline
%pdb off

# <headingcell level=3>

# Import libraries and define derivative function(s) for ODE's

# <codecell>


import numpy as np
from pycse import odelay
from IPython.html.widgets import interact, interactive
from IPython.display import clear_output, display, HTML

from thesis_functions.initialconditions import InputDataDictionary, SetInitialConditions
from thesis_functions.astro import FindOrbitCenter, ComputeLibrationPoints, stop_yEquals0, stop_zEquals0, linearDerivativesFunction, nonlinearDerivativesFunction, PropagateSatellite, ComputeOffsets, ConvertOffsets, BuildRICFrame, BuildVNBFrame
from thesis_functions.visualization import PlotGrid

# <codecell>


#def generateClosedInitialConditions(inputstate, timespan, componentToVary, outputstate):
    
    # propagate inputstate to plane crossing
    #timespan, statesOverTime1, EventTime, EventState, EventIndex = odelay(nonlinearDerivativesFunction, inputstate, timespan, events=[stop_yEquals0])

#print 'initialstate1 = ', initialstate1
#print 't = ', t # timestamps corresponding to the output states over time
#print 'statesOverTime = ', statesOverTime
#print 'EventTime = ', EventTime
#print 'EventState = ', EventState
#print 'EventIndex = ', EventIndex
#print len(timespan)
    
    #statesOverTime1 = integrate.odeint(nonlinearDerivativesFunction, initialstate1, timespan)
    
    # if goal parameter is larger than tolerance, then
    
    # adjust the componentToVary in the appropriate direction and repeat

# <headingcell level=3>

# Pull a state to use as the initial state for second satellite

# <codecell>


def InitializeSecondSatellite(thetaindex, x1, y1, z1, xdot1, ydot1, zdot1, center):
    
    ## Theta Angle Approach
    # compute angle theta (in XY plane for now) wrt center for each point along the orbit

    theta = np.zeros(len(x1))

    theta = np.degrees( np.arctan2( x1 - center[0], y1 - center[1] ))
        
    # user input:

    thetadirection = np.sign(theta[1] - theta[0])

    desiredtheta = theta[thetaindex]# + thetadirection*5 # theta[2] # degrees

    # temporarily disabling plot
    if (1 == 0):     
        figTheta, axTheta = plt.subplots()
        axTheta.plot(theta, 'o-')
        figTheta.suptitle('Theta over time of Satellite 1')

    # copy state from desired 'theta' value into second satellite initial state

    desiredthetaindex = (np.abs(theta-desiredtheta)).argmin()
    #print 'theta[0] = ', theta[0]
    #print 'index = ', desiredthetaindex
    #print 'theta[index] = ', theta[desiredthetaindex]

    initialstate2 = [x1[desiredthetaindex], y1[desiredthetaindex], z1[desiredthetaindex], 
                     xdot1[desiredthetaindex], ydot1[desiredthetaindex], zdot1[desiredthetaindex]]
    
    #return initialstate2


    ## VNB velocity offset approach

    # compute a small offset to velocity along the velocity direction

    # assign initial state to second satellite
    factor = thetaindex*1.001
    initialstate2 = [x1[0], y1[0], z1[0], 
                     xdot1[0]*factor, ydot1[0]*factor, zdot1[0]*factor]
    
    return initialstate2


    #generate closed halo...


    ## VNB position offset approach

    # compute a small offset to position along the velocity direction

    # assign initial state to second satellite
    #initialstate2 = [x1[0] + xdot1[0]*0.001, y1[0] + ydot1[0]*0.001, z1[0] + zdot1[0]*0.001, 
    #                 xdot1[0], ydot1[0], zdot1[0]]

# <headingcell level=3>

# Propagate second satellite and compare to first

# <codecell>


# First satellite 

ICs = InputDataDictionary()

mu, timespan, initialstate1 = SetInitialConditions(ICs, ICset = 'Barbee', ICtestcase = 0, numPoints = 200)
    
X1, X2, L1, L2, L3, L4, L5 = ComputeLibrationPoints(mu)

# In nondimensional units, r12 = 1, M = 1, timeConst = Period/(2pi) = 1, G = 1
r12 = 384400.0;
timeConst = r12**(1.5)/mu**(0.5)

# Re-dimensionalize
#X1 = X1*r12;
#X2 = X2*r12;
#L1 = L1*r12;
#L2 = L2*r12;
#L3 = L3*r12;
#L4 = L4*r12;
#L5 = L5*r12;

x1, y1, z1, xdot1, ydot1, zdot1 = PropagateSatellite(mu, timespan, initialstate1)

# Re-dimensionalize
#x1 = x1*r12;
#y1 = y1*r12;
#z1 = z1*r12;
#xdot1 = xdot1*r12/timeConst;
#ydot1 = ydot1*r12/timeConst;
#zdot1 = zdot1*r12/timeConst;

center = FindOrbitCenter(x1, y1, z1);

# Plot satellite 1 in RLP frame
data = {'sat1': {'x':x1, 'y':y1, 'z':z1}}
points = {'L1': L1, 'center': center, }
PlotGrid('Satellite 1 in RLP Frame', 'X', 'Y', 'Z', data, points, 'equal')
    
rVec, iVec, cVec = BuildRICFrame(x1, y1, z1, xdot1, ydot1, zdot1, center)
    
vVec, nVec, bVec = BuildVNBFrame(x1, y1, z1, xdot1, ydot1, zdot1, center)
    

# Second satellite (as function of theta index)

data.clear();

# create empty dictionaries
dataRLP = {};
dataoffsetRLP = {};
dataRIC = {};
dataVNB = {};

#def SecondSatellite(thetaindex):
#for thetaindex in np.arange(10, 50, 10): # thetaindex range
for thetaindex in np.arange(1, 5, 1): # VNB factor range
    
    initialstate2 = InitializeSecondSatellite(thetaindex, x1, y1, z1, xdot1, ydot1, zdot1, center);
    
    x2, y2, z2, xdot2, ydot2, zdot2 = PropagateSatellite(mu, timespan, initialstate2);

    # Compute offsets in RLP frame
    dx, dy, dz = ComputeOffsets(timespan, x1, y1, z1, xdot1, ydot1, zdot1, x2, y2, z2, xdot2, ydot2, zdot2);
    
    # Compute offsets in RIC frame
    dr, di, dc = ConvertOffsets(dx, dy, dz, rVec, iVec, cVec);

    # Compute offsets in VNB frame
    dv, dn, db = ConvertOffsets(dx, dy, dz, vVec, nVec, bVec);
    
    dataRLP['sat1' + str(thetaindex)] = {'x':x1, 'y':y1, 'z':z1}
    dataRLP['sat2' + str(thetaindex)] = {'x':x2, 'y':y2, 'z':z2}
    
    dataoffsetRLP['offsetRLP' + str(thetaindex)] = {'x':dx, 'y':dy, 'z':dz}
    dataRIC['offsetRIC' + str(thetaindex)] = {'x':dr, 'y':di, 'z':dc}
    dataVNB['offsetVNB' + str(thetaindex)] = {'x':dv, 'y':dn, 'z':db}

# Plot both satellites in RLP
#data = {'sat1': {'x':x1, 'y':y1, 'z':z1}, 'sat2': {'x':x2, 'y':y2, 'z':z2}}
points = {'L1': L1, 'L2': L2, 'center': center, 'M2': X2}
PlotGrid('Satellites 1 and 2 in RLP Frame', 'X', 'Y', 'Z', dataRLP, points, 'equal')

# Plot offset (relative motion) between satellites 1 and 2 in RLP
#data = {'offsetRLP': {'x':dx, 'y':dy, 'z':dz}}
points = {'zero': [0,0,0]}
PlotGrid('Offset between Satellites 1 and 2 in RLP Frame', 'X', 'Y', 'Z', dataoffsetRLP, points, 'auto')
    
# Plot relative motion in RIC frame
#data = {'offsetRIC': {'x':dr, 'y':di, 'z':dc}}
points = {'zero': [0,0,0]}
PlotGrid('Offset between Satellites 1 and 2 in RIC Frame', 'R', 'I', 'C', dataRIC, points, 'auto')
    
# Plot relative motion in VNB frame
#data = {'offsetVNB': {'x':dv, 'y':dn, 'z':db}}
points = {'zero': [0,0,0]}
PlotGrid('Offset between Satellites 1 and 2 in VNB Frame', 'V', 'N', 'B', dataVNB, points, 'auto')
    
    #return dx, dy, dz, x2, y2, z2
    
    
#def RLPBothSatellitesWidget(thetaindex):
    
#    SecondSatellite(thetaindex)
    
#    RLPBothSatellitesPlots(x1, y1, z1, x2, y2, z2, L1, center)
    
#def RLPOffsetWidget(thetaindex):
    
#    SecondSatellite(thetaindex)
    
#    RLPOffsetPlots(dx, dy, dz)
    
#def RICWidget(thetaindex):
    
#    SecondSatellite(thetaindex)
    
#    RICPlots(dx, dy, dz, rVec, iVec, cVec)
    
#def VNBWidget(thetaindex):
    
#    SecondSatellite(thetaindex)
    
#    VNBPlots(dx, dy, dz, vVec, nVec, bVec)
    

# <codecell>

#for theta in np.arange(10, 50, 10):
#    SecondSatellite(theta);

#w = interactive(SecondSatellite, thetaindex=(0,50))
#display(w)

#wRLPBothSatellites = interactive(RLPBothSatellitesWidget, thetaindex=(0,50))
#display(wRLPBothSatellites)

#wRLPOffset = interactive(RLPOffsetWidget, thetaindex=(0,50))
#display(wRLPOffset)

#wRIC = interactive(RICWidget, thetaindex=(0,50))
#display(wRIC)

#wVNB = interactive(VNBWidget, thetaindex=(0,50))
#display(wVNB)

# <codecell>

#%pdb

# <codecell>


