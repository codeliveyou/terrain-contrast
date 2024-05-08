import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline, splrep, splev
from scipy.integrate import cumtrapz, simps
import math
import random

from Convert import convert

number_of_points = 600
sec = 30
LT_N = convert("39°8'20.25")
LT_E = convert("125°50'24.44")
RB_N = convert("39°6'42.95")
RB_E = convert("125°52'22.37")

def get(x, y):
    ltn = LT_N
    lte = LT_E
    rbn = RB_N
    rbe = RB_E
    H = LT_N - RB_N
    W = RB_E - LT_E
    if W / H < 10 / 6:
        d = H - W / 10 * 6
        ltn -= d / 2
        rbn += d / 2
    else:
        d = W - H / 6 * 10
        lte += d / 2
        rbe -= d / 2
    ratio = (rbe - lte) / 1000
    return [rbn + y * ratio, lte + x * ratio]
    

# Define the sequence of points
path_manual = np.array([
    [917, 552], [949, 465], [903, 375], [779, 394], [670, 479],
    [553, 485], [546, 394], [669, 318], [714, 199], [607, 117],
    [394, 131], [302, 177], [236, 262], [214, 332], [191, 354],
    [161, 308], [157, 251], [154, 182]
])

out = open("output.txt", "w")

# Separate the points into x and y coordinates
x = path_manual[:, 0]
y = path_manual[:, 1]

# Parameterize the path based on cumulative distance between points
t = np.zeros(x.shape)
t[1:] = np.sqrt((np.diff(x)**2) + (np.diff(y)**2)).cumsum()
t_fine = np.linspace(0, t[-1], 500)

# Create cubic spline interpolations for x and y coordinates
spline_x = interp1d(t, x, kind='cubic')
spline_y = interp1d(t, y, kind='cubic')

# Create cubic spline interpolations for x and y coordinates
l_spline_x = UnivariateSpline(t, x, k=3, s=0)
l_spline_y = UnivariateSpline(t, y, k=3, s=0)

# Create spline representations for x and y coordinates
v_spline_x = splrep(t, x, k=3)
v_spline_y = splrep(t, y, k=3)

# Evaluate splines on the fine parameter range
x_smooth = spline_x(t_fine)
y_smooth = spline_y(t_fine)

# Calculate points at equal arc-length intervals
interval_length = t_fine.max() / (number_of_points - 1)
t_points = np.linspace(0, t_fine.max(), number_of_points)

# Get x and y coordinates of these points
x_points = spline_x(t_points)
y_points = spline_y(t_points)

v_x_points = splev(t_points, v_spline_x)
v_y_points = splev(t_points, v_spline_y)
dx_points = splev(t_points, v_spline_x, der=1)
dy_points = splev(t_points, v_spline_y, der=1)

# Calculate the derivative of x and y with respect to t
dx_dt = l_spline_x.derivative()(t_fine)
dy_dt = l_spline_y.derivative()(t_fine)

# Calculate the length of the curve
arc_length = simps(np.sqrt(dx_dt**2 + dy_dt**2), t_fine)

# Plotting the original path points and the interpolated smooth curve
plt.figure(figsize=(12, 8))
plt.plot(x_smooth, y_smooth, '-', label='Interpolated smooth curve', linewidth=2)
plt.scatter(x_points, y_points, color='red', zorder=5)


# Annotate points
for i, (xi, yi) in enumerate(zip(x_points, y_points)):
    plt.text(xi, yi, f'{i+1}', fontsize=9, ha='right')

# Annotate points and plot tangent vectors
for i, (xi, yi, dxi, dyi) in enumerate(zip(v_x_points, v_y_points, dx_points, dy_points)):
    plt.text(xi, yi, f'{i+1}', fontsize=9, ha='right')
    plt.quiver(xi, yi, dxi, dyi, color='blue', scale=10, width=0.003)

out.write(str(number_of_points) + '\n')
for i in range(number_of_points):
    radian = -(math.atan2(dy_points[i], dx_points[i]) - math.pi / 2)
    if radian < 0:
        radian += math.pi * 2
    azimuth = radian / math.pi * 180
    N, E = get(x_points[i], y_points[i])
    out.write(' '.join([str(N), str(E), '0', 
                        str(random.randint(43, 47)), str(azimuth), '0',
                        str(arc_length / sec), str(dx_points[i]), str(dy_points[i]), '0']) + '\n')
out.write(f"{arc_length:.2f}\n")

out.close()
print(list(dx_points))
print(list(dy_points))

plt.legend()
plt.title('Path Interpolation with Selected Points')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.axis('equal')  # Ensures that scale is the same on both axes
plt.grid(True)
plt.show()
