# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np

# <codecell>


def SetInitialConditions(ICs, ICset = 'Howell', ICtestcase = 0, numPoints = 2000):
    
    # user inputs 
    #ICset = 'Howell'  # 'Sharp' 'Howell' 'Barbee'
    #ICtestcase = 0
    #numPoints = 2000

    # assign simulation variables using specified elements from dictionary of IC's
    mu = ICs[ICset]['mu'][ICtestcase]

    timespan = np.linspace(0, ICs[ICset]['T'][ICtestcase], numPoints)

    initialstate1 = [ICs[ICset]['X'][ICtestcase], 0,                              ICs[ICset]['Z'][ICtestcase],
                     0,                           ICs[ICset]['Ydot'][ICtestcase], 0]
    
    return mu, timespan, initialstate1
  

# <codecell>


def InputDataDictionary():
    
    # create a dictionary for the initial conditions
    ICs = dict()

    # From Sharp, A Collection of Restricted Three-Body Test Problems
    # For problems 1 to 15, mu = 0.012277471 and for problems 16 to 20, mu = 0.000953875
    ICs['Sharp'] = {'mu': np.ones(20),
                    'X':  np.zeros(20),
                    'Z':  np.zeros(20),
                    'Ydot': np.zeros(20),
                    'T':    np.zeros(20)}

    Sharp_X_Z_Ydot_T = np.matrix([[0.994000E+00, 0.0, -0.21138987966945026683E+01, 0.54367954392601899690E+01],
                                  [0.994000E+00, 0.0, -0.20317326295573368357E+01, 0.11124340337266085135E+02],
                                  [0.994000E+00, 0.0, -0.20015851063790825224E+01, 0.17065216560157962559E+02],
                                  [0.997000E+00, 0.0, -0.16251217072210773125E+01, 0.22929723423442969481E+02],
                                  [0.879962E+00, 0.0, -0.66647197988564140807E+00, 0.63006757422352314657E+01],
                                  [0.879962E+00, 0.0, -0.43965281709207999128E+00, 0.12729711861022426544E+02],
                                  [0.879962E+00, 0.0, -0.38089067106386964470E+00, 0.19138746281183026809E+02],
                                  [0.997000E+00, 0.0, -0.18445010489730401177E+01, 0.12353901248612092736E+02],
                                  [0.100000E+01, 0.0, -0.16018768253456252603E+01, 0.12294387796695023304E+02],
                                  [0.100300E+01, 0.0, -0.14465123738451062297E+01, 0.12267904265603897140E+02],
                                  [0.120000E+01, 0.0, -0.71407169828407848921E+00, 0.18337451820715063383E+02],
                                  [0.120000E+01, 0.0, -0.67985320356540547720E+00, 0.30753758552146029263E+02],
                                  [0.120000E+01, 0.0, -0.67153130632829144331E+00, 0.43214375227857454128E+02],
                                  [0.120000E+01, 0.0, -0.66998291305226832207E+00, 0.55672334134347612727E+02],
                                  [0.120000E+01, 0.0, -0.66975741517271092087E+00, 0.68127906604713772763E+02],
                                  [-0.102745E+01, 0.0, 0.40334488290490413053E-01, 0.18371316400018903965E+03],
                                  [-0.976680E+00, 0.0, -0.61191623926410837000E-01, 0.17733241131524483004E+03],
                                  [-0.766650E+00, 0.0, -0.51230158665978820282E+00, 0.17660722897242937108E+03],
                                  [-0.109137E+01, 0.0, 0.14301959822238380020E+00, 0.82949461922342093092E+02],
                                  [-0.110137E+01, 0.0, 0.15354250908611454510E+00, 0.60952121909407746612E+02]])

    ICs['Sharp']['mu'][0:15] *= 0.012277471
    ICs['Sharp']['mu'][15:20] *= 0.000953875
    ICs['Sharp']['X']    = np.array(Sharp_X_Z_Ydot_T[:,0])
    ICs['Sharp']['Z']    = np.array(Sharp_X_Z_Ydot_T[:,1])
    ICs['Sharp']['Ydot'] = np.array(Sharp_X_Z_Ydot_T[:,2])
    ICs['Sharp']['T']    = np.array(Sharp_X_Z_Ydot_T[:,3])

    # From Howell, Three-Dimensional, Periodic, 'Halo' Orbits
    ICs['Howell'] = {'mu':   [0.04, 0.04],
                     'X':    [0.723268,     0.723268],
                     'Z':    [0.040000,    -0.040000],
                     'Ydot': [0.198019,     0.198019],
                     'T':    [1.300177*2.0, 1.300177*2.0]}

    # From Barbee, Notional Mission 4 (Earth-Moon)
    ICs['Barbee'] = {'mu':   [0.012277471],
                     'X':    [0.862307159058101],
                     'Z':    [0.0],
                     'Ydot': [-0.187079489569182],
                     'T':    [2.79101343456226]}
    
    return ICs

# <codecell>


