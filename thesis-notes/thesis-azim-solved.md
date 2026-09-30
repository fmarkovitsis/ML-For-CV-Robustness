1. Roll Pitch Yaw (RPY):
	1. camera: $[c]$
	2. sun: $[s]$
	
2. Calculate rotation matrices:
	- source for rotation matrix calculation for RPY and rotation of a unit vector: [here](https://mtsu.pressbooks.pub/robotics/chapter/chapter-2/)
	1. Camera Rotation Matrix: $R_c = R_z(yaw) * R_y(pitch) * R_x(roll) = \begin{bmatrix}cos(yaw) & -sin(yaw) & 0 \\ sin(yaw) & cos(yaw)  & 0 \\ 0 & 0 & 1 \end{bmatrix} * \begin{bmatrix}cos(pitch) & 0 & sin(pitch) \\ 0 & 1  & 0 \\ -sin(pitch) & 0 & cos(pitch) \end{bmatrix} * \begin{bmatrix}1 & 0 & 0 \\ 0 & cos(roll) & -sin(roll) \\ 0 & sin(roll) & cos(roll) \end{bmatrix}$
	2. Sun Rotation Matrix: $R_s = R_z(yaw) * R_y(pitch) * R_x(roll)$
	
3. Assume unit vectors coming out of the camera, having the direction of the x, y and z axis of the world. By default they are like $\begin{bmatrix} 1 \\ 0 \\ 0\end{bmatrix}$, $\begin{bmatrix} 0 \\ 1 \\ 0\end{bmatrix}$ and $\begin{bmatrix} 0 \\ 0 \\ 1\end{bmatrix}$. Since in Unreal Engine (UE) x is forward and y is right, we need the first and second vectors to define the plane onto which we are projecting the sun. We need to rotate them first using the rotation matrix, so that they are matching the direction of the camera. The new x' and y' are perpendicular and they kind of short of function as a coordinate system whose center is the camera and have the direction of whatever its looking at: 
	1. Calculate $f_c = R_c * \begin{bmatrix} 1 \\ 0 \\ 0\end{bmatrix}$ (forward)
	2. and $r_c = R_c * \begin{bmatrix} 0 \\ 1 \\ 0\end{bmatrix}$ (right)
	
4. Similarly, we are finding the sun's x' (only x' and not y' because we are trying to find the difference between the sun's forward and the camera's forward):
	- $s = R_s * \begin{bmatrix} 1 \\ 0 \\ 0\end{bmatrix}$
	
5. In general, it is possible to calculate the angle between two vertices using <u>dot product</u>:
	- source: [here](https://en.wikipedia.org/wiki/Dot_product)
	- $θ = arccos(\frac{a \cdot b}{||a||*||b||})$ 
	- in our case, $f_c$, $r_c$ and $s$ are unit vectors, so their magnitudes are 1
	- the formula simplifies to $θ = arccos(a \cdot b)$
	1. Calculate how much of the sun is pointing towards camera's front: $\frac{s \cdot f_c}{||s||*||f_c||} = cos(θ) \Rightarrow s \cdot f_c = 1 * cos(θ) = cos(θ)$ 
	2. And towards camera's right, since the right axis is perpendicular to the forward: $s \cdot r_c = sinθ$
	
6. These can be plugged in the atan2 function defined [here](https://en.wikipedia.org/wiki/Atan2):
	- Definition: $θ = atan2(y, x)$
		- $x = rcosθ$
		- $y = rsinθ$
	