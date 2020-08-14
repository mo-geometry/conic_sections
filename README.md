# Conic sections and non-linear maps
Cone-plane intersection formaulae - as described in "Sensor tilt via conic sections" IMVIP (2020).

<center>
<img src="https://user-images.githubusercontent.com/62537514/90258836-023cc900-de41-11ea-8f7b-cdecd6440604.png" width="linewidth"/>
</center>

In the figure a cone of principle angle β is intersected by 2 planes S0 and S1, which share a common origin at (x,y,z)=(0,0,1). 

The apex of the cone is at (0,0,0). 

```
* S0 (flat plane): unit normal parallel to the z-axis
* S1 (tilted plane): unit normal n = (nx, ny, nz)
```
<center>
<img src="https://user-images.githubusercontent.com/62537514/90262577-61e9a300-de46-11ea-9475-925b545b7114.png" width="200"/>
</center>

The S0 plane intersects the cone to create a circle of points p, with radius r=tanβ:

<center>
<img src="https://user-images.githubusercontent.com/62537514/90261899-81340080-de45-11ea-9995-f46f64e98822.png" width="500"/>
</center>

The S1 plane is related to the S0 plane via the rodrigues rotation matrix

<center>
<img src="https://user-images.githubusercontent.com/62537514/90264204-a8d89800-de48-11ea-9e5d-2b5c45d15c7f.png" width="350"/>
</center>

### Vector map:

<center>
<img src="https://user-images.githubusercontent.com/62537514/90265054-ed186800-de49-11ea-87d7-57760eb35109.png" width="600"/>
</center>

### Forward map: circle to (ellipse, parabola, hyperbola)

The projected point forms a vector λp touching the tilted plane S1, and connecting with the vector q in the local coordinates of S1. 

Taking the dot product of both sides of this vector equation, with the unit normal n, we arrive at an expression for the scaling factor λ (since n·q=0),

<center>
<img src="https://user-images.githubusercontent.com/62537514/90264530-23a1b300-de49-11ea-9ebf-5a5a9f768905.png" width="linewidth"/>
</center>

To perform the rotation around the projective plane origin, the points are first shifted to the coordinate centre, then rotated, then shifted back to the original location. The conic p' generated from an intersecting plane of normal n is:

<center>
<img src="https://user-images.githubusercontent.com/62537514/90264479-11c01000-de49-11ea-86a0-7666f235544b.png" width="250"/>
</center>

### Backward map: (ellipse, parabola, hyperbola) to circle

The mapping from the conic p′ to the circle p is obtained by first shifting the points to the coordinate origin, performing the inverse rotation, shifting back, and then rescaling through division by the z component.

<center>
<img src="https://user-images.githubusercontent.com/62537514/90265348-5dbf8480-de4a-11ea-9911-5c73f5643dbd.png" width="280"/>
</center>

where the scaling factor is the perspective projection:

<center>
<img src="https://user-images.githubusercontent.com/62537514/90265360-6021de80-de4a-11ea-8dbc-502936123830.png" width="296"/>
</center>

A circle p, is mapped to the conic p', as a function of the unit normal n to the intersecting plane, via:
<center>
<img src="https://user-images.githubusercontent.com/62537514/90267439-72514c00-de4d-11ea-854e-e2a5e59a22ec.png" width="linewidth"/>
</center>

### Animation

<center>
<img src="https://github.com/mo-geometry/conic_sections/blob/master/conic_sections.gif" width="linewidth"/>
</center>

### References 

* B. O'Sullivan and P. Stec, "Sensor tilt via conic sections" IMVIP (2020).

* P. Stec and B. O'Sullivan, ["Method for compensating for the off axis tilting of a lens"](https://patents.google.com/patent/US10356346) United states patent number: 10,356,346 B1, July (2019).
