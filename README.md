# fhn-3comp-basin
Numerical calculation to solve extended 1D FitzHugh-Nagumo model


# The extended FitzHugh-Nagumo model
$
u_t = \phi (a u - \alpha u^3 -b v - c w) + d_1 u_{xx},\\
v_t = \phi \varepsilon_2 (u-v) + d_2 v_{xx},\\
w_t = \phi \varepsilon_3 (u-w) + d_3 w_{xx}.
$

## Grid
N = 1024, dx=0.005

T = 200, dt = 5e-5

## Boundary conditions
Neumann BC: $\frac{\partial u}{\partial x}\Big|_{x=0} = 0, \quad \frac{\partial u}{\partial x}\Big|_{x=L} = 0.$
## Initial conditions

$
u_{x,0}=\cos{(2\pi f (x-x_{mid}) },\\
v_{x,0}=\cos{(2\pi f (x-x_{mid}) + \theta \pi)},\\
w_{x,0}=0
$

with $x\in [0, L]$, where $L=(N-1)dx$ and $x_{mid} = L/2$

## Parameters
$(a,b,c) = (3.5, 3.0, 3.5),$ 

$(\alpha, \phi, \varepsilon_2, \varepsilon_3) = (1.5, 0.5, 1.0, 0.5),$

$(d_1, d_2, d_3) = (0.0, 0.0, 0.5)$

## Method

## Metrics to analyse results
### L

### SI

### $g_0$