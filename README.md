# fhn-3comp-basin

Numerical solver for the extended 1D three-component FitzHugh-Nagumo model. This code simulates spatiotemporal dynamics and analyzes chimera states using order parameters.

## Overview

The FitzHugh-Nagumo (FHN) model is a simplified version of the Hodgkin-Huxley model for neuronal dynamics. This extended three-component version exhibits rich spatiotemporal patterns including **chimera states** - phenomena where coherent (synchronized) and incoherent (desynchronized) regions coexist.

## The Extended FitzHugh-Nagumo Model

The system consists of three coupled PDEs:

```math
u_t = \phi (a u - \alpha u^3 - b v - c w) + d_1 u_{xx}
```

```math
v_t = \phi \varepsilon_2 (u - v) + d_2 v_{xx}
```

```math
w_t = \phi \varepsilon_3 (u - w) + d_3 w_{xx}
```

### Variables

- **u**: Activator variable (fast dynamics)
- **v**: First inhibitor variable (slow dynamics)
- **w**: Second inhibitor variable (slow dynamics, diffusive)

### Parameters

| Symbol | Default | Description                               |
| ------ | ------- | ----------------------------------------- |
| `a`    | 3.5     | Bifurcation parameter (controls dynamics) |
| `b`    | 3.0     | Coupling strength from v to u             |
| `c`    | 3.5     | Coupling strength from w to u             |
| `α`    | 1.5     | Cubic nonlinearity coefficient            |
| `ϕ`    | 0.5     | Time scale separation parameter           |
| `ε₂`   | 1.0     | Relaxation rate for v                     |
| `ε₃`   | 0.5     | Relaxation rate for w                     |
| `d₁`   | 0.0     | Diffusion coefficient for u               |
| `d₂`   | 0.0     | Diffusion coefficient for v               |
| `d₃`   | 0.5     | Diffusion coefficient for w               |

## Numerical Method

### Discretization

**Spatial Grid:**

- `N = 1024` grid points
- `dx = 0.005` spatial step size
- Domain length: `L = (N-1) × dx ≈ 5.12`

**Temporal Discretization:**

- Total simulation time: `T = 200`
- Time step: `dt = 4 × dx² / max(d₁, d₂, d₃) ≈ 1×10⁻⁴`
- Number of steps: `~2×10⁶`

### Time-Stepping Scheme

The code uses **operator splitting** to separate reaction and diffusion terms:

1. **Reaction Step (RK4)**: 4th order Runge-Kutta for the nonlinear reaction terms
   - Explicit scheme with 4 stages
   - Pre-allocated buffers to minimize memory allocation

2. **Diffusion Step (Crank-Nicolson)**: Implicit scheme for diffusion terms
   - Second-order accurate in space and time
   - Unconditionally stable
   - Solved efficiently using the **Thomas Algorithm (TDMA)**

### Boundary Conditions

**Neumann (zero-flux) boundary conditions:**

```math
\frac{\partial u}{\partial x}\bigg|_{x=0} = 0, \quad \frac{\partial u}{\partial x}\bigg|_{x=L} = 0
```

Implemented using ghost points at boundaries.

### Initial Conditions

```math
u(x, 0) = \cos\left(\pi \cdot 2f \cdot (x - x_{mid})\right)
```

```math
v(x, 0) = \cos\left(\pi \cdot 2f \cdot (x - x_{mid}) + \theta\pi\right)
```

```math
w(x, 0) = 0
```

where:

- `f`: spatial frequency (default: 0.4)
- `θ`: phase shift (default: 0.0)
- `x_mid = L/2`: center of domain

## Metrics for Analysis

### L - Local Order Parameter

Quantifies the degree of local phase synchronization.

**Range:** `L ∈ [0, 1]`

- `L ≈ 1`: Fully coherent (all phases aligned)
- `L ≈ 0`: Fully incoherent (random phases)

**Algorithm:**

1. Compute local phase: `φᵢ = atan(uᵢ, wᵢ)`
2. Convert to complex phase vectors: `zᵢ = exp(i·φᵢ)`
3. Sum magnitudes in local neighborhoods (size 3)
4. Normalize by `3N`

### SI - Strength of Incoherence

Measures spatial incoherence through variance analysis.

**Range:** `SI ∈ [0, 1]`

- `SI = 0`: Fully coherent
- `SI = 1`: Fully incoherent

**Algorithm:**

1. Compute spatial differences between neighbors
2. Divide domain into `M = 16` bins
3. Compute variance in each bin
4. `SI = 1 - (fraction of bins with σ < δ)`

### g₀ - Coherence Classification Metric

Classifies chimera states based on spatial smoothness.

**Range:** `g₀ ∈ [0, 1]`

- `g₀ ≈ 1`: Fully coherent (smooth profile)
- `g₀ ≈ 0`: Incoherent or chimera state

**Algorithm:**

1. Compute second spatial derivative (curvature)
2. Find maximum absolute curvature `Dm`
3. Count fraction of points with `|D| < δ` where `δ = 0.01 × Dm`

**Reference:** Kemeth et al., "A classification scheme for chimera states", _Chaos_ **26**, 094815 (2016). DOI: [10.1063/1.4959804](https://doi.org/10.1063/1.4959804)

## Installation

### Requirements

- Julia 1.8 or later
- Required packages (install automatically on first run):
  ```julia
  using Pkg
  Pkg.add(["CairoMakie", "ColorSchemes", "BenchmarkTools", "LinearAlgebra",
           "StatsBase", "Printf", "DataFrames", "CSV", "JLD2", "CodecZlib"])
  ```

## Usage

### Basic Run

Execute with default parameters:

```bash
julia main.jl
```

### Custom Parameters

```bash
julia main.jl <a_parameter> <frequency>
```

**Arguments:**

- `a_parameter`: Bifurcation parameter (default: 3.5)
- `frequency`: Spatial frequency of initial condition (default: 0.4)

**Examples:**

```bash
# Custom bifurcation parameter
julia main.jl 3.8

# Custom frequency
julia main.jl 3.5 0.6

# Both parameters
julia main.jl 4.0 0.3
```

### Output

1. **Console Output:**
   - Simulation execution time
   - Final metrics: `L`, `SI`, `g₀`

2. **Visualization:**
   - Multi-panel figure displaying:
     - Spacetime evolution `u(x,t)`
     - Initial profiles `[u, v, w]`
     - Final profiles `[u, v, w]`
     - Metrics evolution over time
   - Saved to: `./data/uvw/fig_a_X.XX_phase_X.XXXX_freq_X.XXXX.png`

### Batch Processing (Cluster Mode)

The code includes commented-out sections for parameter sweeps:

1. **Uncomment** the "Batch Processing" section in `main.jl`
2. **Configure** parameter ranges:
   ```julia
   phase_array = range(start=-1, stop=1, step=0.02)
   freq = 0.45
   ```
3. **Run** with multi-threading:
   ```bash
   julia -t auto main.jl
   ```

**Output Files:**

- CSV files with metrics vs. parameters
- JLD2 compressed files with full simulation data

## Project Structure

```
julia-basin-fhn/
├── main.jl              # Main simulation code
├── README.md            # This documentation
├── data/
│   └── uvw/             # Output figures and data
├── cpu_test.jl          # CPU performance tests
└── gpu_main.jl          # GPU-accelerated version (CUDA)
```

## Performance Notes

- **Memory optimization**: Pre-allocated arrays minimize garbage collection overhead
- **Loop optimization**: Explicit loops with `@inbounds` for performance
- **Parallel processing**: Thread-based parallelization for parameter sweeps
- **GPU acceleration**: See `gpu_main.jl` and `OneDimFHN.cu` for CUDA implementation

## Interpretation of Results

### State Classification

| State Type     | L            | SI           | g₀           | Description          |
| -------------- | ------------ | ------------ | ------------ | -------------------- |
| **Coherent**   | High (>0.9)  | Low (<0.1)   | High (>0.9)  | Fully synchronized   |
| **Incoherent** | Low (<0.2)   | High (>0.9)  | Low (<0.2)   | Fully desynchronized |
| **Chimera**    | Intermediate | Intermediate | Intermediate | Coexistence of both  |

### Typical Patterns

- **Traveling waves**: Periodic spatiotemporal patterns
- **Standing waves**: Stationary oscillating patterns
- **Chimera states**: Coherent regions coexisting with incoherent regions
- **Turbulence**: Spatiotemporal chaos

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce `N` or increase downsampling in batch mode
2. **Slow performance**: Enable multi-threading (`julia -t auto`)
3. **Numerical instability**: Decrease `dt` or check parameter ranges
