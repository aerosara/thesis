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

