# ==============================================================================
# main.jl - Numerical solver for the extended 1D FitzHugh-Nagumo model
# ==============================================================================
# This script simulates a three-component FitzHugh-Nagumo system using:
# - 4th order Runge-Kutta (RK4) for reaction terms
# - Crank-Nicolson scheme with Thomas algorithm (TDMA) for diffusion terms
#
# The model exhibits chimera states - coexistence of coherent and incoherent
# regions - which are analyzed using metrics: L (local order), SI (strength
# of incoherence), and g₀ (classification metric).
#
# Author: [Your Name]
# License: [Your License]
# ==============================================================================

using CairoMakie, ColorSchemes   # Visualization and colormaps
using BenchmarkTools            # Performance benchmarking
using LinearAlgebra             # Linear algebra utilities
using Base.Threads              # Multi-threading support
using StatsBase                 # Statistical functions
using Printf                    # Formatted output
using DataFrames, CSV           # Data manipulation and CSV I/O
using JLD2, CodecZlib           # Compressed binary data storage

# ==============================================================================
# Parameter Structures
# ==============================================================================

"""
    ConstParams{T}

Immutable structure holding all constant parameters for the FHN system.

# Fields
- `a, b, c`: Reaction kinetics parameters controlling the u-component dynamics
- `α`: Cubic nonlinearity coefficient
- `ϕ`: Time scale separation parameter
- `ϵ₂, ϵ₃`: Coupling strengths for v and w components
- `D1, D2, D3`: Diffusion coefficients for u, v, w components respectively
- `dt`: Time step size
- `dx`: Spatial grid spacing
- `inv_dx2`: Pre-computed 1/dx² for efficiency
- `steps`: Total number of time steps
- `save_step`: Interval for saving snapshots (every save_step steps)
- `N`: Number of spatial grid points

# Type Parameters
- `T`: Floating point type (typically Float64)
"""
struct ConstParams{T}
    a::T
    b::T
    c::T
    α::T
    ϕ::T
    ϵ₂::T
    ϵ₃::T
    D1::T
    D2::T
    D3::T
    dt::T
    dx::T
    inv_dx2::T
    steps::Int
    save_step::Int        # Save snapshot every save_step iterations
    N::Int
end

# Pre-allocate arrays for RK4 to avoid allocation
struct RK4Buffers
    k1::Matrix{Float64}
    k2::Matrix{Float64}
    k3::Matrix{Float64}
    k4::Matrix{Float64}
end;

"""
    RK4Buffers(N::Int)

Outer constructor for RK4Buffers that pre-allocates stage arrays for RK4.

Pre-allocating these buffers avoids memory allocation during the time-stepping
loop, significantly improving performance for long simulations.

# Arguments
- `N`: Number of spatial grid points

# Returns
- `RK4Buffers` instance with pre-allocated 3×N matrices for each RK4 stage
"""
function RK4Buffers(N::Int)
    k1 = Matrix{Float64}(undef, 3, N)
    k2 = Matrix{Float64}(undef, 3, N)
    k3 = Matrix{Float64}(undef, 3, N)
    k4 = Matrix{Float64}(undef, 3, N)
    return RK4Buffers(k1, k2, k3, k4)
end;

"""
    reaction_fhn!(du, u)

Compute the reaction terms of the FitzHugh-Nagumo system.

Evaluates the local (non-diffusive) dynamics at each spatial point:
- du₁/dt = ϕ(a·u₁ - α·u₁³ - b·u₂ - c·u₃)
- du₂/dt = ϕ·ϵ₂·(u₁ - u₂)
- du₃/dt = ϕ·ϵ₃·(u₁ - u₃)

# Arguments
- `du`: Output matrix (3×N) to store the reaction term derivatives
- `u`: Current state matrix (3×N) where rows are [u, v, w] components

# Notes
- Uses `@inbounds` for performance (bounds checking disabled)
- Reads global `params` constant for parameter values
"""
function reaction_fhn!(du, u)
    a, b, c, α, ϕ, ϵ₂, ϵ₃ = params.a, params.b, params.c, params.α, params.ϕ, params.ϵ₂, params.ϵ₃
    N = params.N
    @inbounds for i in 1:N
        u1 = u[1, i]
        u2 = u[2, i]
        u3 = u[3, i]

        du[1, i] = ϕ * (a * u1 - α * u1^3 - b * u2 - c * u3)
        du[2, i] = ϕ * ϵ₂ * (u1 - u2)
        du[3, i] = ϕ * ϵ₃ * (u1 - u3)
    end
end;

"""
    runge_kutta_4!(u_next, u_current, buffers::RK4Buffers)

Perform one step of the 4th order Runge-Kutta (RK4) method for the reaction terms.

Implements the classic RK4 scheme:
- k₁ = f(uⁿ)
- k₂ = f(uⁿ + dt/2 · k₁)
- k₃ = f(uⁿ + dt/2 · k₂)
- k₄ = f(uⁿ + dt · k₃)
- uⁿ⁺¹ = uⁿ + dt/6 · (k₁ + 2k₂ + 2k₃ + k₄)

# Arguments
- `u_next`: Output matrix (3×N) for the state at the next time step
- `u_current`: Current state matrix (3×N)
- `buffers`: Pre-allocated RK4Buffers for storing intermediate stages k1-k4

# Notes
- Uses explicit loops instead of broadcasting (@.) for better performance
- Only handles reaction terms; diffusion is applied separately via Crank-Nicolson
- Uses `@inbounds` for bounds-checking elimination
"""
function runge_kutta_4!(u_next, u_current, buffers::RK4Buffers)
    k1, k2, k3, k4 = buffers.k1, buffers.k2, buffers.k3, buffers.k4
    N = params.N

    # k1: evaluate reaction terms at current state
    reaction_fhn!(k1, u_current)
    # k2: evaluate at midpoint using k1
    @inbounds for j in 1:N, i in 1:3
        u_next[i, j] = u_current[i, j] + 0.5 * params.dt * k1[i, j]
    end
    reaction_fhn!(k2, u_next)
    # k3: evaluate at midpoint using k2
    @inbounds for j in 1:N, i in 1:3
        u_next[i, j] = u_current[i, j] + 0.5 * params.dt * k2[i, j]
    end
    reaction_fhn!(k3, u_next)
    # k4: evaluate at endpoint using k3
    @inbounds for j in 1:N, i in 1:3
        u_next[i, j] = u_current[i, j] + params.dt * k3[i, j]
    end
    reaction_fhn!(k4, u_next)

    # Combine stages with RK4 weights
    dt_6 = params.dt / 6.0
    @inbounds for j in 1:N, i in 1:3
        u_next[i, j] = u_current[i, j] + dt_6 * (k1[i, j] + 2.0 * k2[i, j] + 2.0 * k3[i, j] + k4[i, j])
    end
end;

"""
    right_hand!(output, input, r)

Compute the right-hand side of the Crank-Nicolson scheme for diffusion.

Applies the explicit part of the Crank-Nicolson discretization for the
diffusion equation with Neumann boundary conditions (zero flux at boundaries).

The stencil is: output[i] = input[i] + r·(input[i-1] - 2·input[i] + input[i+1])
where r = D·dt/(2·dx²)

# Arguments
- `output`: Output vector (N,) for the RHS result
- `input`: Input vector (N,) with current state
- `r`: Coefficient r = D·dt/(2·dx²)

# Boundary Conditions
- Left boundary (i=1): Uses ghost point to enforce ∂u/∂x = 0
- Right boundary (i=N): Uses ghost point to enforce ∂u/∂x = 0
"""
function right_hand!(output, input, r)
    N = params.N
    @inbounds begin
        # Left boundary: Neumann BC with ghost point
        output[1] = input[1] + 2 * r * (input[2] - input[1])
        # Right boundary: Neumann BC with ghost point
        output[N] = input[N] + 2 * r * (input[N-1] - input[N])
        # Interior points: standard 3-point stencil
        for i in 2:N-1
            output[i] = input[i] + r * (input[i-1] - 2 * input[i] + input[i+1])
        end
    end
end;

"""
    thomas_solver!(d, TDMA, c_prime, d_prime)

Solve a tridiagonal system using the Thomas algorithm (TDMA).

Solves the linear system Ax = d where A is tridiagonal with:
- Subdiagonal: sub[i] = -ξ
- Diagonal: diag[i] = 1 + 2ξ (or 1 + ξ at boundaries)
- Superdiagonal: sup[i] = -ξ

This arises from the implicit part of the Crank-Nicolson discretization.

# Arguments
- `d`: Right-hand side vector (N,); overwritten with solution
- `TDMA`: Tuple (sub, diag, sup) containing tridiagonal matrix coefficients
- `c_prime`: Pre-allocated workspace for modified superdiagonal
- `d_prime`: Pre-allocated workspace for modified RHS

# Algorithm
1. Forward sweep: eliminate subdiagonal, modify diagonal and RHS
2. Backward substitution: recover solution from bottom to top

# Notes
- O(N) complexity vs O(N³) for general Gaussian elimination
- Uses pre-allocated buffers to avoid allocation in time-stepping loop
"""
function thomas_solver!(d, TDMA, c_prime, d_prime)
    N = params.N
    sub, diag, sup = TDMA
    # Forward sweep: eliminate subdiagonal
    c_prime[1] = sup[1] / diag[1]
    d_prime[1] = d[1] / diag[1]
    @inbounds for i in 2:N-1
        denom = diag[i] - sub[i-1] * c_prime[i-1]
        c_prime[i] = sup[i] / denom
        d_prime[i] = (d[i] - sub[i-1] * d_prime[i-1]) / denom
    end
    # Last row: no superdiagonal element
    denom = diag[N] - sub[N-1] * c_prime[N-1]
    d_prime[N] = (d[N] - sub[N-1] * d_prime[N-1]) / denom
    # Backward substitution: recover solution
    d[N] = d_prime[N]
    @inbounds for i in N-1:-1:1
        d[i] = d_prime[i] - c_prime[i] * d[i+1]
    end
end;

# ==============================================================================
# Metrics Functions
# ==============================================================================
# These functions quantify the spatiotemporal patterns and classify chimera
# states. Chimera states are characterized by coexisting coherent (synchronized)
# and incoherent (desynchronized) regions.
# ==============================================================================

"""
    metric_local_order(u_comp, w_comp)

Compute the local order parameter L for quantifying spatial coherence.

Measures the degree of local phase synchronization by computing the average
magnitude of the sum of complex phase vectors in local neighborhoods (size 3).

# Arguments
- `u_comp`: First component field (N,) - typically the activator u
- `w_comp`: Second component field (N,) - typically the slow variable w

# Returns
- `local_R`: Local order parameter ∈ [0, 1]
  - L ≈ 1: Fully coherent (all phases aligned)
  - L ≈ 0: Fully incoherent (random phases)

# Algorithm
1. Compute local phase φᵢ = atan(uᵢ, wᵢ) at each point
2. Convert to complex phase vector: zᵢ = exp(i·φᵢ)
3. For each point, sum phase vectors in neighborhood (j-1, j, j+1)
4. Average the magnitudes and normalize by 3N
"""
function metric_local_order(u_comp, w_comp)
    N = size(u_comp, 1)
    exp_phase = zeros(ComplexF64, N)
    local_R = 0.0
    for i in 1:N
        ϕ = atan(u_comp[i], w_comp[i])
        exp_phase[i] = cis(ϕ) # cis(x) = exp(i*x)
    end
    @inbounds for j in 1:N
        j_prev = max(1, j - 1)
        j_next = min(N, j + 1)
        local_R += abs(exp_phase[j] + exp_phase[j_prev] + exp_phase[j_next])
    end
    return local_R / (3.0 * N)
end;

"""
    metric_si(u, M, delta)

Compute the Strength of Incoherence (SI) metric.

Quantifies the degree of spatial incoherence by analyzing the variance of
differences between neighboring points across spatial bins.

# Arguments
- `u`: Spatiotemporal data (T×N) where T is time, N is space
- `M`: Number of spatial bins for analysis
- `delta`: Threshold for classifying bins as coherent/incoherent

# Returns
- `SI`: Strength of incoherence ∈ [0, 1]
  - SI = 0: Fully coherent (all bins have small variance)
  - SI = 1: Fully incoherent (all bins have large variance)

# Algorithm
1. Compute spatial differences: w[t,i] = u[t,i+1] - u[t,i]
2. Divide domain into M bins of size m = (N-1)÷M
3. For each bin and time, compute variance of differences
4. Average variance over time for each bin
5. SI = 1 - (fraction of bins with σ < δ)

# Reference
Based on methods for detecting chimera states in spatiotemporal systems.
"""
function metric_si(u, M, delta)
    # Strength of incoherence
    T, N = size(u)
    m = (N - 1) ÷ M
    w = u[:, 2:end] .- u[:, 1:end-1]
    sigma_m_t = zeros(M, T)
    for idx in 1:M
        for t in 1:T
            bin_t = w[t, idx*m-m+1:idx*m]
            mean_t = mean(w[t, :])
            var = mean((bin_t .- mean_t) .^ 2)
            sigma_m_t[idx, t] = sqrt(var)
        end
    end
    sigma_m_avg = mean(sigma_m_t, dims=2)
    SI = 1 - sum(sigma_m_avg .< delta) / M
    return SI
end;

"""
    metric_g0(u; delta_factor=0.01)

Compute the g₀ metric for classifying chimera states.

Measures the fraction of points where the second spatial derivative is small,
indicating smooth (coherent) regions. High g₀ indicates coherent states,
while low g₀ indicates incoherent or chimera states.

# Arguments
- `u`: Spatial profile (N,)
- `delta_factor`: Fraction of maximum curvature used as threshold (default: 0.01)

# Returns
- `g0`: Coherence fraction ∈ [0, 1]
  - g₀ ≈ 1: Fully coherent (smooth spatial profile)
  - g₀ ≈ 0: Fully incoherent (highly irregular profile)

# Algorithm
1. Compute second derivative: D[i] = u[i+2] - 2·u[i+1] + u[i]
2. Find maximum absolute curvature: Dm = max|D|
3. Set threshold: δ = delta_factor × Dm
4. g₀ = fraction of points where |D| < δ

# Reference
Kemeth et al., "A classification scheme for chimera states",
Chaos 26, 094815 (2016). DOI: 10.1063/1.4959804
"""
function metric_g0(u; delta_factor=0.01)
    # A classification scheme for chimera states. Kemeth et al DOI: 10.1063/1.4959804
    D = (u[3:end] - 2 * u[2:end-1] + u[1:end-2])
    Dm = maximum(abs, D)
    if isapprox(Dm, 0; atol=1e-5)
        return 1.0
    end
    delta = delta_factor * Dm
    return sum(abs.(D) .< delta) / length(D)
end;


"""
    one_calculation_step(freq::Float64, phase::Float64)

Run a complete simulation of the 3-component FHN system.

This is the main simulation function that:
1. Initializes the system with cosine initial conditions
2. Time-steps using operator splitting: RK4 for reaction + Crank-Nicolson for diffusion
3. Tracks metrics (L, g₀) during evolution
4. Saves snapshots for post-processing

# Arguments
- `freq`: Spatial frequency of initial cosine pattern
- `phase`: Phase shift between u and v components (in units of π)

# Returns
- `u_history`: (T_save×N) array of u-component snapshots at save times
- `initlast`: (2×3×N) array with initial [u,v,w] and final [u,v,w] states
- `loc_history`: Time series of local order parameter L
- `g0_history`: Time series of g₀ metric

# Initial Conditions
- u(x,0) = cos(π·2f·(x - x_mid))
- v(x,0) = cos(π·2f·(x - x_mid) + phase)
- w(x,0) = 0

# Time-Stepping Scheme
Operator splitting with:
1. RK4 step for reaction terms (explicit, 4th order)
2. Crank-Nicolson for diffusion (implicit, 2nd order) using Thomas algorithm

# Performance Notes
- Uses `@time` macro to report total simulation time
- Pre-allocates all arrays to minimize GC overhead
- Uses views to avoid array copies in diffusion solver
"""
function one_calculation_step(freq::Float64, phase::Float64)
    x = (0:dx:(N-1)*dx)
    xmid = x[end] / 2
    u = zeros(Float64, 3, N)
    @. u[1, :] = cospi((x - xmid) * freq * 2f0) # Initial condition for u
    @. u[2, :] = cospi((x - xmid) * freq * 2f0 + phase) # Initial condition for v with phase shift

    u_next = copy(u) # allocate auxiliary array for time stepping

    # Store initial and final states for visualization
    initlast = zeros(Float64, 2, 3, N)
    initlast[1, :, :] = Array(u)

    # Pre-allocate history arrays
    u_history = Matrix{Float64}(undef, length(save_range), N)
    loc_history = empty([])
    g0_history = empty([])

    buffers = RK4Buffers(N) # allocate help buffers for RK4

    @time begin
        history_idx = 0
        for step in 1:steps
            runge_kutta_4!(u_next, u, buffers) # Do Runge-Kutta 4th order
            # * trick to skip diffusion solver if D=0
            for (k, D_val) in enumerate([params.D1, params.D2, params.D3])
                if D_val != 0
                    # Do Crank-Nicolson
                    # explicit step: compute RHS
                    right_hand!(view(u, k, :), view(u_next, k, :), KSI[k])
                    # implicit step: solve tridiagonal system
                    thomas_solver!(view(u, k, :), TDMA_COEFFS[k], view(buffers.k1, k, :), view(buffers.k2, k, :))
                else
                    # swap main and auxiliary arrays for next iteration if diffusion skipped
                    u[k, :], u_next[k, :] = u_next[k, :], u[k, :]
                end
            end
            # keep evolution of metrics
            if step % 2_000 == 0
                append!(loc_history, metric_local_order(view(u, 1, :), view(u, 3, :)))
                append!(g0_history, metric_g0(view(u, 1, :)))
            end
            # save first component for SI metric
            if step ∈ save_range
                history_idx += 1
                u_history[history_idx, :] = u[1, :]
            end
        end
    end
    initlast[2, :, :] = u
    return Float32.(u_history), Float32.(initlast), Float32.(loc_history), Float32.(g0_history)
end



# ==============================================================================
# Visualization Functions
# ==============================================================================

"""
    create_theme()

Create a custom CairoMakie theme for publication-quality figures.

Combines LaTeX fonts, dark theme, and custom styling for:
- Figure size and font sizes
- Colormaps (berlin for heatmaps, seaborn_muted for lines)
- Axis styling (grid lines, tick alignment, label sizes)
- Legend and colorbar appearance

# Returns
- A merged CairoMakie.Theme object for use with `with_theme()`
"""
create_theme() =
    let
        merge(theme_latexfonts(), theme_black(),
            CairoMakie.Theme(
                # font="CMU Serif",
                # figure_padding=(5, 5, 10, 10),
                size=(1200, 900),
                fontsize=20,
                colormap=:berlin,
                color=cgrad(:seaborn_muted, categorical=true),
                markersize=12,
                linewidth=0.2,
                Axis=(xlabelsize=20, xlabelpadding=-5,
                    xgridstyle=:dash, ygridstyle=:dash,
                    xtickalign=1, ytickalign=1,
                    # yticksize=10, xticksize=10,
                ),
                Legend=(;
                    backgroundcolor=:transparent,
                    framecolor=:gray,
                    valign=:center,
                    tellheight=false,
                ),
                Colorbar=(ticksize=16, tickalign=1, spinewidth=0.5),
            ))
    end

"""
    plot_plot(data, text_with_meta, savename)

Create a multi-panel visualization of simulation results.

Generates a figure with 4 panels:
1. Spacetime heatmap of u(x,t) evolution
2. Initial profiles of all three components [u, v, w]
3. Final profiles of all three components
4. Time evolution of metrics L and g₀

# Arguments
- `data`: Tuple (u_history, initlast, loc_history, g0_history) from simulation
- `text_with_meta`: Title string with parameters and metrics
- `savename`: File path for saving the figure (currently commented out)

# Layout
- Left panel (ga): Full spacetime evolution heatmap
- Right panels (gb, top): Initial state profiles
- Right panels (gb, middle): Final state profiles
- Right panels (gb, bottom): Metrics evolution over time
"""
function plot_plot(data, text_with_meta, savename)
    history, initlast, loc, g0 = data
    maxu = maximum(abs, history) * 1.1 + 0.01
    la = ["u", "v", "w"]
    cmap = cgrad(:Set1, categorical=true)
    fig = Figure()
    ga = fig[1:2, 1] = GridLayout()
    gb = fig[1:2, 2] = GridLayout()
    ax = Axis(ga[1, 1], xlabel="Space", ylabel="Time step", titlealign=:right, subtitle="u(x,t)")
    bx = Axis(gb[1, 1], titlealign=:right, subtitle="nt=0", limits=((0, N), nothing), yaxisposition=:right)
    cx = Axis(gb[2, 1], titlealign=:right, subtitle="nt=$(steps)", limits=((0, N), (-maxu, maxu)), yaxisposition=:right, xlabel="Space")
    dx = Axis(gb[3, 1], titlealign=:right, subtitle="metrics", xlabel="Time step", yaxisposition=:right)

    hm = heatmap!(ax, 1:N, save_range, history')
    Colorbar(ga[0, 1], hm, vertical=false)
    scl_init = [lines!(bx, initlast[1, v, :], label=la[v], color=cmap[v], linewidth=6) for v in 1:3]
    hidexdecorations!(bx, grid=false)
    scatterlines!(cx, initlast[2, 3, :], color=cmap[3], alpha=0.7,)
    scatterlines!(cx, initlast[2, 2, :], color=cmap[2], alpha=0.7,)
    scl_last = scatterlines!(cx, initlast[2, 1, :], color=cmap[1], strokecolor=:white, strokewidth=0.3,)

    Legend(gb[1:2, 2], scl_init, la)
    l = scatter!(dx, 1:2000:steps, loc, color=cmap[4], markersize=4)
    g = scatter!(dx, 1:2000:steps, g0, color=cmap[6], markersize=4)
    Legend(gb[3, 2], [l => (; markersize=20), g => (; markersize=20),], ["L", "g₀"])

    rowgap!(gb, 2)
    colgap!(gb, 2)
    colsize!(fig.layout, 1, Auto(1.0))
    Label(fig[0, :], text_with_meta,)

    fig
    # save(savename, fig)
end

# ==============================================================================
# Global Constants and Parameters
# ==============================================================================

# Spatial discretization
const N = 1024                    # Number of grid points
const dx = 0.005                  # Spatial step size

# Diffusion coefficients
const D1 = 0.0                    # Diffusion coefficient for u-component
const D2 = 0.0                    # Diffusion coefficient for v-component
const D3 = 0.5                    # Diffusion coefficient for w-component

# Time discretization
# dt chosen to satisfy stability: Courant number ≤ 0.5 for explicit schemes
# Crank-Nicolson is unconditionally stable but accuracy still matters
const dt = 4 * dx^2 / max(D1, D2, D3)

const steps = round(Int, 200 / dt)  # Total number of time steps (T=200)
const save_step = 200               # Save every save_step steps
const save_range = range(stop=steps, step=save_step, length=1001)  # Save indices

# Command-line parameter: 'a' controls the bifurcation parameter
VAR_A = isempty(ARGS) ? 3.5 : parse(Float64, ARGS[1])

# Pack all parameters into immutable struct
const params = ConstParams{Float64}(
    VAR_A,    # a: bifurcation parameter
    3.0,      # b: coupling to v
    3.5,      # c: coupling to w
    1.5,      # α: cubic nonlinearity
    0.5,      # ϕ: time scale separation
    1.0,      # ϵ₂: v relaxation rate
    0.5,      # ϵ₃: w relaxation rate
    D1, D2, D3,  # diffusion coefficients
    dt, dx, dt / dx^2,  # time/space steps and precomputed inverse
    steps, Int(save_step), Int(N)  # iteration counts
)

# Pre-computed coefficients for Crank-Nicolson diffusion solver
# KSI = D·dt/(2·dx²) - should be in [0.25, 0.5] for stability/accuracy
const KSI = 0.5 * params.inv_dx2 * [D1, D2, D3]

# Pre-computed tridiagonal matrix coefficients for Thomas algorithm
# For each component k: (subdiagonal, diagonal, superdiagonal)
const TDMA_COEFFS = map(KSI) do ξ
    return (
        fill(-ξ, N - 1),  # Subdiagonal (constant)
        [j ∈ (1, N) ? 1 + ξ : 1 + 2 * ξ for j in 1:N],  # Diagonal (boundary vs interior)
        fill(-ξ, N - 1)   # Superdiagonal (constant)
    )
end

# ==============================================================================
# Main Execution: Single Simulation Run
# ==============================================================================
# Run one instance of the FHN simulation with specified frequency and phase.
# Results are visualized and metrics are printed to console.
#
# Usage:
#   julia main.jl [a_param] [freq]
#   - a_param: bifurcation parameter (default: 3.5)
#   - freq: spatial frequency of initial condition (default: 0.4)
# ==============================================================================

# Initial condition spatial frequency
freq = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 0.4
phase = 0.0  # Phase shift (currently fixed, can be parameterized)

# Run simulation and measure execution time
@info "Starting simulation" freq phase
arr = one_calculation_step(freq, phase)  # MAIN FUNCTION RUN

# Compute final metrics from the simulation results
loc_value = metric_local_order(view(arr[2], 2, 1, :), view(arr[2], 2, 3, :))
si_value = metric_si(arr[1], 16, 0.2)
g0_value = metric_g0(view(arr[2], 2, 1, :))

# Print metrics to console
@printf("Metrics: L=%.3f SI=%.3f g₀=%.3f\n", loc_value, si_value, g0_value)

# Create formatted title string with parameters and metrics
text_with_meta = @sprintf(
    """Parameters: a=%.2f δx=%.1e δt=%.1e  || θ=%.4fπ f=%.4f
    Metrics: L=%.3f SI=%.3f g₀=%.3f""", 
    params.a, dx, dt, phase, freq, loc_value, si_value, g0_value
)

# Generate output filename
svnm = @sprintf("./data/uvw/fig_a_%.2f_phase_%.4f_freq_%.4f.png", VAR_A, phase, freq)

# Create and display the figure with custom theme
with_theme(create_theme()) do
    plot_plot(arr, text_with_meta, svnm)
end

# ==============================================================================
# Batch Processing: Parameter Sweep (Cluster Mode)
# ==============================================================================
# This section contains commented-out code for running parameter sweeps
# across multiple frequencies and phases, typically on an HPC cluster.
#
# Features:
# - Parallel execution using Threads.@threads
# - Saves metrics to CSV for analysis
# - Saves full simulation data to compressed JLD2 format
#
# To use: uncomment desired sections and adjust parameters as needed
# ==============================================================================

# Example: Single frequency sweep over phases
# freq = isempty(ARGS) ? 0.45 : parse(Float64, ARGS[1])
# results_dir = "results33"
# isdir(results_dir) || mkdir(results_dir)
# for freq in range(0.025, 0.5, step=0.025)
# Phase sweep parameters
# phase_array = range(start=-1, stop=1, step=0.02)
# n_phases = length(phase_array)
# locs = Vector{Float64}(undef, n_phases)
# si_array = Vector{Float64}(undef, n_phases)
# g_null = Vector{Float64}(undef, n_phases)

# Storage for histories
# u_histories = Vector{Any}(undef, n_phases)  # Any
# lg_histories = Vector{Any}(undef, n_phases)  # Any

# Parallel loop over phases
# Threads.@threads for i in 1:n_phases
#     phase = phase_array[i]
#     arr = one_calculation_step(freq, phase) #! MAIN FUNCTION RUN in parallel
#     locs[i] = metric_local_order(view(arr[2], 2, 1, :), view(arr[2], 2, 3, :))
#     si_array[i] = metric_si(arr[1], 16, 0.2)
#     g_null[i] = metric_g0(view(arr[2], 2, 1, :))
#     @printf("θ=%.4fπ f=%.4f || Metrics: L=%.3f SI=%.3f g₀=%.3f\n", phase, freq, locs[i], si_array[i], g_null[i])
#     u_histories[i] = (phase, arr[1][1:10:end, :])
#     lg_histories[i] = (phase, arr[3], arr[4])
# end

# Save metrics to CSV file
# metric_file = joinpath(results_dir, @sprintf("freq_%.4f_metrics.csv", freq))
# results_df = DataFrame(
#     phase=collect(phase_array),
#     freq=freq,
#     loc=locs,
#     si=si_array,
#     g0=g_null
# )

# CSV.write(metric_file, results_df, append=isfile(metric_file))
# end
# ==============================================================================
# Save simulation data to JLD2 format (compressed binary)
# data_file = joinpath(results_dir, @sprintf("data_freq_%.4f.jld2", freq))
# jldopen(data_file, "w"; compress=true) do file
#     # Save metadata
#     file["metadata/dx"] = dx
#     file["metadata/dt"] = dt
#     file["metadata/d1"] = params.D1
#     file["metadata/d2"] = params.D2
#     file["metadata/d3"] = params.D3
#     file["metadata/a"] = params.a
#     file["metadata/b"] = params.b
#     file["metadata/c"] = params.c
#     file["metadata/alpha"] = params.α
#     file["metadata/phi"] = params.ϕ
#     file["metadata/eps2"] = params.ϵ₂
#     file["metadata/eps3"] = params.ϵ₃
#     file["metadata/Nx"] = N
#     file["metadata/Nt"] = steps
#     file["metadata/phase_array"] = phase_array
#     # Save metrics
#     file["loc/values"] = locs
#     file["si/values"] = si_array
#     file["g0/values"] = g_null
#     # Save u-component histories (downsampled)
#     for (_, (phase, u_history)) in enumerate(u_histories)
#         file["u_history/$(phase)"] = u_history
#     end
#     # Save metric time series
#     for (_, (phase, l, g)) in enumerate(lg_histories)
#         file["metric/$(phase)/l"] = l
#         file["metric/$(phase)/g"] = g
#     end
# end
# ==============================================================================
