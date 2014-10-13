# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


foo = pd.DataFrame({"x": [1,2,3], "y": [4,5,6]}, index=[0.0, 0.1, 0.2])

foo.join(bar)

bar.iloc[-1]

bar.loc[0.1]


# <codecell>


In [135]: %paste
foo = pd.DataFrame({
    "x": x,
    "y": y},
    index=[colors, time])
## -- End pasted text --

In [136]: foo
Out[136]:
          x  y
blue 0.1  1  4
     0.2  2  5
     0.3  3  6
red  0.1  4  7
     0.2  5  8
     0.3  6  9

In [137]: foo.loc["blue", 0.2]
Out[137]:
x    2
y    5
Name: (blue, 0.2), dtype: int64

In [138]: foo.loc["blue", 0.2].x
Out[138]: 2

In [139]: foo.loc["blue", 0.2]
Out[139]:
x    2
y    5
Name: (blue, 0.2), dtype: int64

In [140]: foo.loc["blue", :]
Out[140]:
     x  y
0.1  1  4
0.2  2  5
0.3  3  6

# <codecell>


In [154]: foo = pd.DataFrame({"x": [1, 2], "y": [7, 2]})
 
In [155]: foo
Out[155]:
   x  y
0  1  7
1  2  2
 
In [156]: np.linalg.in
np.linalg.info  np.linalg.inv
 
In [156]: np.linalg.inv(foo)
Out[156]:
array([[-0.16666667,  0.58333333],
       [ 0.16666667, -0.08333333]])
 
In [157]: foo
Out[157]:
   x  y
0  1  7
1  2  2
 
In [158]: foo.values
Out[158]:
array([[1, 7],
       [2, 2]])

# <codecell>


In [255]: foo
Out[255]:
     name  stuff
0  Stefan      1
1  Stefan      2
2  Stefan      3
3    Sara      4
4    Sara      5
5    Sara      6
6    Sara      7
7    Sara      8
8    Sara      9
 
In [256]: foo.groupby("name").apply(lambda x: x.stuff.sum())
Out[256]:
name
Sara      39
Stefan     6
dtype: int64
 
In [257]: foo.groupby("name").apply(lambda x: x.stuff.mean())
Out[257]:
name
Sara      6.5
Stefan    2.0
dtype: float64
 
In [258]: foo[foo.name == "Stefan"]
Out[258]:
     name  stuff
0  Stefan      1
1  Stefan      2
2  Stefan      3
 
In [259]: foo[(foo.name == "Stefan") & (foo.stuff > 2)]
Out[259]:
     name  stuff
2  Stefan      3

# <codecell>


In [42]: waypoints
Out[42]:
<class 'pandas.core.panel.Panel'>
Dimensions: 3 (items) x 6 (major_axis) x 3 (minor_axis)
Items axis: ric to vnb
Major_axis axis: 1.22276554245 to 1.67180373601
Minor_axis axis: x to z
 
In [43]: waypoints["ric"]
Out[43]:
          x       y  z
1.222766  0  100.00  0
1.305665  0   15.00  0
1.388564  0    5.00  0
1.529033  0    1.00  0
1.611932  0    0.03  0
1.671804  0    0.00  0
 
In [44]: waypoints["rlp"]
Out[44]:
                  x          y          z
1.222766 -85.026391 -39.016456  35.330285
1.305665   7.503328  12.880621   1.670232
1.388564  -0.115501  -4.738320   1.592162
1.529033  -0.302219   0.937056   0.174901
1.611932  -0.027508  -0.011441  -0.003525
1.671804   0.000000   0.000000   0.000000
 
In [45]: waypoints["vnb"]
Out[45]:
                  x          y          z
1.222766 -85.026391 -46.836424 -24.017955
1.305665  10.299739   8.487322  -6.846952
1.388564  -4.580250  -1.628272  -1.170488
1.529033   0.448119  -0.142235  -0.882587
1.611932  -0.027326   0.010141  -0.007104
1.671804   0.000000   0.000000   0.000000

# <codecell>


