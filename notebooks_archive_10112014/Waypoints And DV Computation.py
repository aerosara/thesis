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

# <headingcell level=3>

# Import libraries

# <codecell>


import numpy as np
import scipy.integrate as integrate
#from pycse import odelay
#from IPython.html.widgets import interact, interactive
#from IPython.display import clear_output, display, HTML

from thesis_functions.initialconditions import InputDataDictionary, SetInitialConditions
from thesis_functions.visualization import CreatePlotGrid, SetPlotGridData
from thesis_functions.astro import FindOrbitCenter, ComputeLibrationPoints, stop_yEquals0, stop_zEquals0
from thesis_functions.astro import ComputeNonlinearDerivs, ComputeRelmoDynamicsMatrix
from thesis_functions.astro import odeintNonlinearDerivs, odeintNonlinearDerivsWithLinearRelmoSTM, odeintNonlinearDerivsWithLinearRelmo
from thesis_functions.astro import ComputeRequiredVelocity, PropagateSatelliteAndChaser
from thesis_functions.astro import PropagateSatellite, ComputeOffsets, ConvertOffsets, ConvertOffset, BuildRICFrame, BuildVNBFrame

# <headingcell level=3>

# Initial Conditions

# <codecell>

# First satellite 

# The initial condition dictionary contains initial conditions from Barbee, Howell, and Sharp
ICs = InputDataDictionary()

# Barbee's initial conditions are a planar (Lyapunov) orbit at Earth/Moon L1
mu, timespan, initialstate1 = SetInitialConditions(ICs, ICset = 'Barbee', ICtestcase = 0, numPoints = 200)

# X1 and X2 are positions of larger and smaller bodies along X axis
X1, X2, L1, L2, L3, L4, L5 = ComputeLibrationPoints(mu)

# The FindOrbitCenter function doesn't work if you only propagate a partial orbit, so just treat L1 as the center
center = L1

# Build instantaneous RIC and VNB frames
#x1, y1, z1, xdot1, ydot1, zdot1 = initialstate1
#x1 = np.array([initialstate1[0]])
#y1 = np.array([initialstate1[1]])
#z1 = np.array([initialstate1[2]])
#xdot1 = np.array([initialstate1[3]])
#ydot1 = np.array([initialstate1[4]])
#zdot1 = np.array([initialstate1[5]])
#rVec, iVec, cVec = BuildRICFrame(x1, y1, z1, xdot1, ydot1, zdot1, center)
#vVec, nVec, bVec = BuildVNBFrame(x1, y1, z1, xdot1, ydot1, zdot1, center)

# TODO: start target satellite from different points along its orbit.  
#       Look at how delta-V changes; also maybe linear relmo will be a better approximation along other parts of the orbit.

# <headingcell level=3>

# Define Waypoints

# <codecell>

# In nondimensional units, r12 = 1, M = 1, timeConst = Period/(2pi) = 1, G = 1
m1 = 5.97219e24;      # Earth  # kg
m2 = 7.34767309e22;   # Moon   # kg
M = m1 + m2;
G = 6.67384e-11/1e9;  # m3/(kg*s^2) >> converted to km3
r12 = 384400.0;       # km

timeConst = r12**(1.5)/(G*M)**(0.5)  # units are seconds  # this is how you convert between dimensional time (seconds) and non-dimensional time
print 'timeconst', timeConst

T = 2.0*np.pi*timeConst   # Period in seconds of Moon around Earth
print 'Period of Moon around Earth in seconds', T

period = np.max(timespan) # Period of libration point orbit (in nondimensional time units)
print 'Period of libration point orbit in seconds', period*timeConst

# TODO: input waypoints in any frame (RLP, RIC, or VNB)
# TODO: get decent test cases in the Sun-Earth-Moon frame
# TODO: report/plot position error at each waypoint
# TODO: report/plot delta-V at each waypoint

Waypoints = dict();
Waypoints[0] = {'t'    : 0.0,
                'r_RIC': [0.0, 1000.0/r12, 0.0]};

Waypoints[0] = {'t'    : 86400.0*2.88/timeConst,      # 2.88 days
                'r_RIC': [0.0, 275.0/r12, 0.0]};  # move 725 km  # 400% errors

Waypoints[0] = {'t'    : 86400.0*4.70/timeConst,      # 1.82 days
                'r_RIC': [0.0, 180.0/r12, 0.0]};  # move 95 km  # 400% errors

Waypoints[0] = {'t'    : 86400.0*5.31/timeConst,
                'r_RIC': [0.0, 100.0/r12, 0.0]};  # 40% errors

Waypoints[1] = {'t'    : 86400.0*5.67/timeConst,
                'r_RIC': [0.0, 15.0/r12, 0.0]};  # 8% errors

Waypoints[2] = {'t'    : 86400.0*6.03/timeConst,
                'r_RIC': [0.0, 5.0/r12, 0.0]};  # 10% errors

Waypoints[3] = {'t'    : 86400.0*6.64/timeConst,
                'r_RIC': [0.0, 1.0/r12, 0.0]};  # 

Waypoints[4] = {'t'    : 86400.0*7.0/timeConst,
                'r_RIC': [0.0, 0.030/r12, 0.0]};  # 

Waypoints[5] = {'t'    : 86400.0*7.26/timeConst,
                'r_RIC': [0.0, 0.0/r12, 0.0]};

# TODO: look at waypoints with different spacing, different approach directions, different amount of time between points

# <headingcell level=3>

# Convert Waypoints from RIC to RLP

# <codecell>


# Build RIC-to-RLP frame (inverse of RLP-to-RIC frame) at each waypoint time and convert waypoints from RIC to RLP

# TODO: would be nice to have a function that generically converts waypoints between frames (e.g. arguments = WaypointDictionary, inputframe, outputframe)

initialState1ForSegment = initialstate1

S = [0, 1, 2, 3, 4]
#S = [0]

for currentPoint in S:
    
    nextPoint = currentPoint + 1;    
    print currentPoint, Waypoints[currentPoint]
    print 'percentage of orbit covered getting to next point (by time):', (Waypoints[nextPoint]['t'] - Waypoints[currentPoint]['t'])/period*100.0
    
    # array of time points
    timespanForSegment = np.linspace(Waypoints[currentPoint]['t'], Waypoints[nextPoint]['t'], 500)

    # target satellite position and velocity over time in RLP frame from integrating initial state with full nonlinear dynamics
    x1, y1, z1, xdot1, ydot1, zdot1 = PropagateSatellite(mu, timespanForSegment, initialState1ForSegment);

    # Build RIC and VNB frames
    rVec, iVec, cVec = BuildRICFrame(x1, y1, z1, xdot1, ydot1, zdot1, center)
    vVec, nVec, bVec = BuildVNBFrame(x1, y1, z1, xdot1, ydot1, zdot1, center)

    ## for the first point only
    
    if (currentPoint == S[0]):
        
        # TODO: clean up a little because there's very similar repeated code here
        # START REPEAT 1
        # this matrix converts from RLP coordinates to the RIC frame at the timestamp of the current (first) point
        RLPtoRIC = np.array([rVec[0], iVec[0], cVec[0]])

        # this matrix converts from RIC to RLP at the timestamp of the current (first) point
        RICtoRLP = np.linalg.inv(RLPtoRIC)
        RLPxVec = RICtoRLP[:,0]
        RLPyVec = RICtoRLP[:,1]
        RLPzVec = RICtoRLP[:,2]
        
        # get the coordinates of the current waypoint in RIC
        [drW, diW, dcW] = Waypoints[currentPoint]['r_RIC']

        # Convert current waypoint from RIC frame to RLP frame at the timestamp of the current (first) point
        dxW, dyW, dzW = ConvertOffset(drW, diW, dcW, RLPxVec, RLPyVec, RLPzVec);
        Waypoints[currentPoint]['r_RLP'] = [dxW, dyW, dzW]
        
        # Convert current waypoint to VNB frame
        dvW, dnW, dbW = ConvertOffset(dxW, dyW, dzW, vVec[0], nVec[0], bVec[0]);
        Waypoints[currentPoint]['r_VNB'] = [dvW, dnW, dbW]
        # END REPEAT 1

    ## for all points
    
    # START REPEAT 2
    # this matrix converts from RLP coordinates to the RIC frame at the timestamp of the next point
    RLPtoRIC = np.array([rVec[-1], iVec[-1], cVec[-1]])
    
    # this matrix converts from RIC to RLP at the timestamp of the next point
    RICtoRLP = np.linalg.inv(RLPtoRIC)
    RLPxVec = RICtoRLP[:,0]
    RLPyVec = RICtoRLP[:,1]
    RLPzVec = RICtoRLP[:,2]
        
    # next point
    [drW, diW, dcW] = Waypoints[nextPoint]['r_RIC']
    
    # Convert next waypoint from RIC frame to RLP frame at the timestamp of the next point
    dxW, dyW, dzW = ConvertOffset(drW, diW, dcW, RLPxVec, RLPyVec, RLPzVec);
    Waypoints[nextPoint]['r_RLP'] = [dxW, dyW, dzW]
    
    # Convert waypoint to VNB frame
    dvW, dnW, dbW = ConvertOffset(dxW, dyW, dzW, vVec[-1], nVec[-1], bVec[-1]);
    Waypoints[nextPoint]['r_VNB'] = [dvW, dnW, dbW]
    # END REPEAT 2 
    
    # Record updated primary satellite initial state for next segment
    initialState1ForSegment = np.array([ x1[-1], y1[-1], z1[-1], xdot1[-1], ydot1[-1], zdot1[-1] ])
    

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

# Plots of offset in RLP, RIC, VNB frames
axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP = CreatePlotGrid('Offset between Satellites 1 and 2 in RLP Frame', 'X', 'Y', 'Z', 'auto')
axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC = CreatePlotGrid('Offset between Satellites 1 and 2 in RIC Frame', 'R', 'I', 'C', 'auto')
axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB = CreatePlotGrid('Offset between Satellites 1 and 2 in VNB Frame', 'V', 'N', 'B', 'auto')

# add zero point to plots (this is location of target satellite)
points = {}
data = {}
points['zero'] = {'xyz':[0,0,0], 'color':'k'}
SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, data, points)
SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, data, points)
SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, data, points)
points = {}

# add all waypoints to RLP, RIC, and VNB plots
for w in Waypoints:
    points['w'] = {'xyz':np.array(Waypoints[w]['r_RLP'])*r12, 'color':'c'}
    SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, data, points)
    
    points['w'] = {'xyz':np.array(Waypoints[w]['r_RIC'])*r12, 'color':'c'}
    SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, data, points)
    
    points['w'] = {'xyz':np.array(Waypoints[w]['r_VNB'])*r12, 'color':'c'}
    SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, data, points)    

points = {}

# <headingcell level=3>

# Travel between waypoints

# <codecell>


initialState1ForSegment = initialstate1

# assume starts exactly from first waypoint with same velocity as target satellite (for lack of any better velocity values at this point)
Waypoints[0]['r_RLP_achieved'] = Waypoints[0]['r_RLP']
Waypoints[0]['v_RLP_abs_premaneuver'] = initialstate1[3:6]

# Travel between waypoints
for currentPoint in S:

    nextPoint = currentPoint + 1;
    
    ## Compute required velocity to travel between waypoints

    # Compute required velocity at point 1 to take us to point 2 within time (t2-t1)
    # This is from Lian et al.
    # Method signature:
    # initialRelativeVelocity = ComputeRequiredVelocity(initialState1ForSegment, initialRelativePosition, initialTime, targetRelativePosition, targetTime)
    Waypoints[currentPoint]['v_RLP'] = ComputeRequiredVelocity(initialState1ForSegment, Waypoints[currentPoint]['r_RLP_achieved'], Waypoints[currentPoint]['t'], Waypoints[nextPoint]['r_RLP'], Waypoints[nextPoint]['t'], mu)

    #print 'initial chaser relative velocity', Waypoints[currentPoint]['v_RLP']

    initialRelativeState = np.concatenate(( Waypoints[currentPoint]['r_RLP_achieved'], Waypoints[currentPoint]['v_RLP'] ))

    ## Integrate first satellite with full nonlinear dynamics and second satellite with linear relmo dynamics

    # array of time points
    timespanForSegment = np.linspace(Waypoints[currentPoint]['t'], Waypoints[nextPoint]['t'], 500)

    # compute target satellite position and velocity over time in RLP frame by integrating initial state with full nonlinear dynamics
    # compute offset between target and chaser satellite over time in RLP frame by integrating initial offset with linearized relmo dynamics
    x1, y1, z1, xdot1, ydot1, zdot1, dx_LINEAR, dy_LINEAR, dz_LINEAR, dxdot_LINEAR, dydot_LINEAR, dzdot_LINEAR = PropagateSatelliteAndChaser(mu, timespanForSegment, initialState1ForSegment, initialRelativeState)

    ##  Integrate second satellite with full nonlinear dynamics

    # initial state of second satellite in absolute RLP coordinates (not relative to first satellite)
    initialstate2 = np.array(initialState1ForSegment) - np.array(initialRelativeState)

    # compute chaser satellite position and velocity over time in RLP frame by integrating initial state with full nonlinear dynamics
    x2, y2, z2, xdot2, ydot2, zdot2 = PropagateSatellite(mu, timespanForSegment, initialstate2);
    
    # Compute offsets in RLP frame based on nonlinear motion
    dx_NONLIN, dy_NONLIN, dz_NONLIN = ComputeOffsets(timespanForSegment, x1, y1, z1, xdot1, ydot1, zdot1, x2, y2, z2, xdot2, ydot2, zdot2);
    
    ## Offsets in RIC and VNB
    
    # Build RIC and VNB frames
    rVec, iVec, cVec = BuildRICFrame(x1, y1, z1, xdot1, ydot1, zdot1, center)
    vVec, nVec, bVec = BuildVNBFrame(x1, y1, z1, xdot1, ydot1, zdot1, center)

    # Compute offsets in RIC frame
    dr_LINEAR, di_LINEAR, dc_LINEAR = ConvertOffsets(dx_LINEAR, dy_LINEAR, dz_LINEAR, rVec, iVec, cVec);
    dr_NONLIN, di_NONLIN, dc_NONLIN = ConvertOffsets(dx_NONLIN, dy_NONLIN, dz_NONLIN, rVec, iVec, cVec);

    # Compute offsets in VNB frame
    dv_LINEAR, dn_LINEAR, db_LINEAR = ConvertOffsets(dx_LINEAR, dy_LINEAR, dz_LINEAR, vVec, nVec, bVec);
    dv_NONLIN, dn_NONLIN, db_NONLIN = ConvertOffsets(dx_NONLIN, dy_NONLIN, dz_NONLIN, vVec, nVec, bVec);   
    
    ## Compute delta-V
    
    # post-maneuver velocity at current waypoint
    Waypoints[currentPoint]['v_RLP_abs_postmaneuver'] = np.array([ xdot2[0], ydot2[0], zdot2[0] ])
    
    # compute delta-V executed at current waypoint
    Waypoints[currentPoint]['deltaV'] = Waypoints[currentPoint]['v_RLP_abs_postmaneuver'] - Waypoints[currentPoint]['v_RLP_abs_premaneuver']
    
    # pre-maneuver velocity for next waypoint (end of current propagation segment)
    Waypoints[nextPoint]['v_RLP_abs_premaneuver'] = np.array([ xdot2[-1], ydot2[-1], zdot2[-1] ])
    
    # TODO: also compute the delta-V based only on the linear relmo propagation and compare the delta-V to the nonlinear one currently being computed
    #      (this means we would need to propagate forward from the nominal waypoint instead of only propagating forward from the achieved waypoint)
    # pre-maneuver relative velocity when arriving at next waypoint, based on linear propagation
    #Waypoints[nextPoint]['v_RLP_pre_LINEAR'] = np.array([ dxdot_LINEAR[-1], dydot_LINEAR[-1], dzdot_LINEAR[-1] ])

    ## Output that gets fed into next iteration/segment
    
    # Record updated primary satellite initial state for next segment
    initialState1ForSegment = np.array([ x1[-1], y1[-1], z1[-1], xdot1[-1], ydot1[-1], zdot1[-1] ])
    
    # Record updated/achieved chaser satellite waypoint for next segment
    Waypoints[nextPoint]['r_RLP_achieved'] = np.array([ dx_NONLIN[-1], dy_NONLIN[-1], dz_NONLIN[-1] ])
    
    # compute updated/achieved waypoint location in RIC and VNB
    [dxW, dyW, dzW] = Waypoints[nextPoint]['r_RLP_achieved']
    
    drW, diW, dcW = ConvertOffset(dxW, dyW, dzW, rVec[-1], iVec[-1], cVec[-1]);
    dvW, dnW, dbW = ConvertOffset(dxW, dyW, dzW, vVec[-1], nVec[-1], bVec[-1]);
    
    Waypoints[nextPoint]['r_RIC_achieved'] = [drW, diW, dcW]
    Waypoints[nextPoint]['r_VNB_achieved'] = [dvW, dnW, dbW]
    
    
    ## VISUALIZATIONS

    #ax1.plot(timespan, dx_LINEAR*r12)

    # Compare linear relmo propagation to nonlinear dynamics
    #ax2.plot((dx_NONLIN - dx_LINEAR)/np.amax(np.absolute(dx_LINEAR))*100.0, (dy_NONLIN - dy_LINEAR)/np.amax(np.absolute(dy_LINEAR))*100.0)
    
    # create data dictionaries
    dataoffsetRLP = {};
    dataoffsetRLP['linear'] = {'x':dx_LINEAR*r12, 'y':dy_LINEAR*r12, 'z':dz_LINEAR*r12, 'color':'g'}
    dataoffsetRLP['nonlin'] = {'x':dx_NONLIN*r12, 'y':dy_NONLIN*r12, 'z':dz_NONLIN*r12, 'color':'r'}
    
    dataoffsetRIC = {};
    dataoffsetRIC['linear'] = {'x':dr_LINEAR*r12, 'y':di_LINEAR*r12, 'z':dc_LINEAR*r12, 'color':'g'}
    dataoffsetRIC['nonlin'] = {'x':dr_NONLIN*r12, 'y':di_NONLIN*r12, 'z':dc_NONLIN*r12, 'color':'r'}
    
    dataoffsetVNB = {};
    dataoffsetVNB['linear'] = {'x':dv_LINEAR*r12, 'y':dn_LINEAR*r12, 'z':db_LINEAR*r12, 'color':'g'}
    dataoffsetVNB['nonlin'] = {'x':dv_NONLIN*r12, 'y':dn_NONLIN*r12, 'z':db_NONLIN*r12, 'color':'r'}

    # Plot offset (relative motion) between satellites 1 and 2 in RLP frame and add achieved waypoint (end of current segment) to plot
    points[nextPoint] = {'xyz':np.array(Waypoints[nextPoint]['r_RLP_achieved'])*r12, 'color':'m'}
    SetPlotGridData(axXZ_RLP, axYZ_RLP, axXY_RLP, ax3D_RLP, dataoffsetRLP, points)
    
    # Plot offset (relative motion) between satellites 1 and 2 in RIC frame and add achieved waypoint (start and end of current segment) to plot
    points[nextPoint] = {'xyz':np.array(Waypoints[nextPoint]['r_RIC_achieved'])*r12, 'color':'m'}
    SetPlotGridData(axXZ_RIC, axYZ_RIC, axXY_RIC, ax3D_RIC, dataoffsetRIC, points)
    
    # Plot offset (relative motion) between satellites 1 and 2 in VNB frame and add achieved waypoint (start and end of current segment) to plot
    points[nextPoint] = {'xyz':np.array(Waypoints[nextPoint]['r_VNB_achieved'])*r12, 'color':'m'}
    SetPlotGridData(axXZ_VNB, axYZ_VNB, axXY_VNB, ax3D_VNB, dataoffsetVNB, points)
    
    points = {}


## final delta-V
currentPoint = nextPoint

# final post-maneuver velocity is same as the target satellite's velocity
Waypoints[currentPoint]['v_RLP_abs_postmaneuver'] = np.array([ xdot1[-1], ydot1[-1], zdot1[-1] ])

# compute final delta-V
Waypoints[currentPoint]['deltaV'] = Waypoints[currentPoint]['v_RLP_abs_postmaneuver'] - Waypoints[currentPoint]['v_RLP_abs_premaneuver']

# <codecell>


# compute delta-V magnitude and report to screen
for w in Waypoints:
    Waypoints[w]['deltaVmag'] = np.linalg.norm(Waypoints[w]['deltaV'],2)*r12/timeConst*1000  # m/s
    print Waypoints[w]['deltaVmag'], Waypoints[w]['deltaV']
    

# <codecell>


