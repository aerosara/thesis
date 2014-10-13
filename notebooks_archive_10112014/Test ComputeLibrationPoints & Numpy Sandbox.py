# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


import numpy as np

from thesis_functions.astro import ComputeLibrationPoints


# Inputs: m1 = mass of larger body, 
#         m2 = mass of smaller body, 
#         r12 = distance from m1 to m2
#         M = Total mass of the system
m1 = 1.989e30;        # Sun
m2 = 5.97219e24;      # Earth
#m2 = 7.34767309e22;  # Moon
r12 = 149600000.0;    # Sun-to-Earth
M = m1 + m2;

# Gravitational constant G = 1 in nondimensional units
G = 6.67384e-11;  # m3 kg-1 s-2

# mu = Gravitational parameter = GM
# For nondimensional computations, mu = M2/M
mu = m2/M;

# Angular rotation rate of system:
omega = np.sqrt(G*M/(r12**3.0));

print m1, m2, r12, M, mu;

# Compute Libration Points and positions of primary bodies
X1, X2, L1, L2, L3, L4, L5 = ComputeLibrationPoints(mu)

# To re-dimensionalize, multiply all distances by r12
print 'X1', X1;
print 'X1', X1*r12;
print 'X2', X2;
print 'X2', X2*r12;
print 'L1', L1;
print 'L1', L1*r12;
print 'L2', L2;
print 'L2', L2*r12;
print 'L3', L3;
print 'L3', L3*r12;
print 'L4', L4;
print 'L4', L4*r12;
print 'L5', L5;
print 'L5', L5*r12;

# <codecell>


# Earth/Moon Test

m1 = 5.97219e24;      # Earth
m2 = 7.34767309e22;   # Moon
r12 = 384400;
mu = 0.012277471;

# Compute Libration Points and positions of primary bodies
X1, X2, L1, L2, L3, L4, L5 = ComputeLibrationPoints(mu)

print 'X1', X1;
print 'X2', X2;
print 'L1', L1;
print 'L2', L2;
print 'L3', L3;
print 'L4', L4;
print 'L5', L5;

# analytic approximation from wikipedia
test = r12*(m2/(3.0*m1))**(1.0/3.0)

print test
print X2[0] - test/r12
print X2[0] + test/r12

# <codecell>



# dictionary test

Waypoints = dict();
Waypoints[0] = {'t': 0.0,
                'r': [20.0, 0.0, 0.0]};
Waypoints[1] = {'t': 60.0,
                'r': [10.0, 0.0, 0.0]};

print Waypoints[0]['r'][0]

# <codecell>

# set up 3x3 identity matrix and zeroes matrix
I3 = np.eye(3)
Z3 = np.zeros((3,3))
print I3
print Z3
test = np.array([[1,2,3],[4,5,6]])
print test
A = np.vstack([np.hstack([Z3, I3]),
               np.hstack([I3, Z3])])
print A

# <codecell>

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6])

b = a[0:3]

c = np.array([a[0:3],
              a[3:6],
              a[6:9]])

c = a[0:9].reshape(3,3)

print a
print b
print c

d = np.dot(c, c)

print len(d[0])

print d

print d.reshape(1,9)
print d.reshape(9,1)

Phi = a[6:42].reshape(6,6)

print Phi

print d.reshape(1,9)[0]

e = np.concatenate((b, d.reshape(1,9)[0]))

print e

# <codecell>

rVec = np.array([[ 0.02601457 , 0.0,          0.0        ]])
print rVec
mag = np.linalg.norm(rVec,2,None)
mag = np.linalg.norm(rVec,2,1)[:,None]
print mag
print rVec/mag

# <codecell>

Waypoints = {}
nextPoint = 0
Waypoints['r_RLP_achieved'] = np.array([ 5.8, 3.4, 5.3 ])

dxW = 0.0
dyW = 0.0
dzW = 0.0
    
# compute updated waypoint location in RIC and VNB
#dxW = Waypoints['r_RLP_achieved'][0]
#dyW = Waypoints['r_RLP_achieved'][1]
#dzW = Waypoints['r_RLP_achieved'][2]
#print dxW, dyW, dzW
    
# compute updated waypoint location in RIC and VNB
[dxW, dyW, dzW] = Waypoints['r_RLP_achieved']

print dxW, dyW, dzW

# <codecell>


