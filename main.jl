using CairoMakie, ColorSchemes
# using Random
using BenchmarkTools
using LinearAlgebra
using Base.Threads
using StatsBase
using Printf
using DataFrames, CSV
using JLD2, CodecZlib
using ProgressMeter

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
    save_step::Int        # каждые save_step шагов пишем снимок
    N::Int
end

# Pre-allocate arrays for RK4 to avoid allocation
struct RK4Buffers
    k1::Matrix{Float64}
    k2::Matrix{Float64}
    k3::Matrix{Float64}
    k4::Matrix{Float64}
end;

function RK4Buffers(N::Int)
    k1 = Matrix{Float64}(undef, 3, N)
    k2 = Matrix{Float64}(undef, 3, N)
    k3 = Matrix{Float64}(undef, 3, N)
    k4 = Matrix{Float64}(undef, 3, N)
    return RK4Buffers(k1, k2, k3, k4)
end;

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

function runge_kutta_4!(u_next, u_current, buffers::RK4Buffers)
    # ? idk why explicit for-loops work faster @. and [:]
    k1, k2, k3, k4 = buffers.k1, buffers.k2, buffers.k3, buffers.k4
    N = params.N

    # k1
    reaction_fhn!(k1, u_current)
    # k2
    @inbounds for j in 1:N, i in 1:3
        u_next[i, j] = u_current[i, j] + 0.5 * params.dt * k1[i, j]
    end
    reaction_fhn!(k2, u_next)
    # k3  
    @inbounds for j in 1:N, i in 1:3
        u_next[i, j] = u_current[i, j] + 0.5 * params.dt * k2[i, j]
    end
    reaction_fhn!(k3, u_next)
    # k4
    @inbounds for j in 1:N, i in 1:3
        u_next[i, j] = u_current[i, j] + params.dt * k3[i, j]
    end
    reaction_fhn!(k4, u_next)

    dt_6 = params.dt / 6.0
    @inbounds for j in 1:N, i in 1:3
        u_next[i, j] = u_current[i, j] + dt_6 * (k1[i, j] + 2.0 * k2[i, j] + 2.0 * k3[i, j] + k4[i, j])
    end
end;

function right_hand!(output, input, r)
    N = params.N
    @inbounds begin
        output[1] = input[1] + 2 * r * (input[2] - input[1])
        output[N] = input[N] + 2 * r * (input[N-1] - input[N])
        for i in 2:N-1
            output[i] = input[i] + r * (input[i-1] - 2 * input[i] + input[i+1])
        end
    end
end;

function thomas_solver!(d, TDMA, c_prime, d_prime)
    N = params.N
    sub, diag, sup = TDMA
    # Forward sweep
    c_prime[1] = sup[1] / diag[1]
    d_prime[1] = d[1] / diag[1]
    @inbounds for i in 2:N-1
        denom = diag[i] - sub[i-1] * c_prime[i-1]
        c_prime[i] = sup[i] / denom
        d_prime[i] = (d[i] - sub[i-1] * d_prime[i-1]) / denom
    end
    # Last row
    denom = diag[N] - sub[N-1] * c_prime[N-1]
    d_prime[N] = (d[N] - sub[N-1] * d_prime[N-1]) / denom
    # Backward substitution
    d[N] = d_prime[N]
    @inbounds for i in N-1:-1:1
        d[i] = d_prime[i] - c_prime[i] * d[i+1]
    end
end;

# =============================================
# Metrics functions
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


function one_calculation_step(freq::Float64, phase::Float64)
    x = (0:dx:(N-1)*dx)
    xmid = x[end] / 2
    u = zeros(Float64, 3, N)
    @. u[1, :] = cospi((x - xmid) * freq * 2f0) #
    @. u[2, :] = cospi((x - xmid) * freq * 2f0 + phase) #

    u_next = copy(u) # allocate auxiliary array

    initlast = zeros(Float64, 2, 3, N)
    initlast[1, :, :] = Array(u)
    u_history = Matrix{Float64}(undef, length(save_range), N)
    buffers = RK4Buffers(N) # allocate help buffers for RK4
    @time begin
        history_idx = 0
        for step in 1:steps
            runge_kutta_4!(u_next, u, buffers) # Do Runge-Kutta 4th order
            # * trick to skip diffusion solver if D=0
            for (k, D_val) in enumerate([params.D1, params.D2, params.D3])
                if D_val != 0
                    # Do Crank-Nicolson
                    # craft explicit step
                    right_hand!(view(u, k, :), view(u_next, k, :), KSI[k])
                    # do implicit step, tridiagonal solver
                    thomas_solver!(view(u, k, :), TDMA_COEFFS[k], view(buffers.k1, k, :), view(buffers.k2, k, :))
                else
                    # swap main and auxiliary arrays for next iteration if diffusion skipped
                    u[k, :], u_next[k, :] = u_next[k, :], u[k, :]
                end
            end
            if step ∈ save_range
                history_idx += 1
                u_history[history_idx, :] = u[1, :]
            end
        end
    end
    initlast[2, :, :] = u
    return u_history, initlast
end


# Make a theme for prettier plot
create_theme() =
    let
        merge(theme_latexfonts(), theme_black(),
            CairoMakie.Theme(
                # font="CMU Serif",
                # figure_padding=(5, 5, 10, 10),
                size=(900, 800),
                fontsize=20,
                colormap=:berlin,
                # color=cgrad(:seaborn_muted, categorical=true),
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
# Plot function that uses local `text_with_meta`
function plot_plot(data)
    history, initlast = data
    maxu = maximum(abs, history) * 1.1 + 0.01
    la = ["u", "v", "w"]
    cmap = cgrad(:Set1, categorical=true)

    fig = Figure()
    ga = fig[1:3, 1:2] = GridLayout()
    gb = fig[4:5, 1:2] = GridLayout()
    ax = Axis(ga[1, 1], xlabel="Space", ylabel="Time step", titlealign=:right, subtitle="u(x,t)")
    bx = Axis(gb[1, 1], titlealign=:right, subtitle="nt=0", limits=((0, N), nothing),)
    cx = Axis(gb[2, 1], titlealign=:right, subtitle="nt=$(steps)", limits=((0, N), (-maxu, maxu)))

    hm = heatmap!(ax, 1:N, save_range, history')
    cb = Colorbar(ga[1, 2], hm,)
    cb.alignmode = Mixed(right=0)

    scl_init = [lines!(bx, initlast[1, v, :], label=la[v], color=cmap[v], linewidth=6) for v in 1:3]
    hidexdecorations!(bx, grid=false)
    leg = Legend(gb[1:2, 2], scl_init, la)
    scatterlines!(cx, initlast[2, 3, :], color=cmap[3], alpha=0.7,)
    scatterlines!(cx, initlast[2, 2, :], color=cmap[2], alpha=0.7,)
    scl_last = scatterlines!(cx, initlast[2, 1, :], color=cmap[1], strokecolor=:white, strokewidth=0.2,)
    rowgap!(gb, 4)
    colgap!(gb, 5)

    Label(fig[0, :], text_with_meta)
    display(fig)
end

# CONSTANTS CONSTANTS
const N = 1024
const dx = 0.005
const D1 = 0.0
const D2 = 0.0
const D3 = 0.5
const dt = 2 * dx^2 / max(D1, D2, D3) # keep Courant number ≤ 0.5 but in CN-mthd can be above
const steps = round(Int, 3000 / dt)
const save_step = 200#steps ÷ 4 ÷ 1000
const save_range = range(stop=steps, step=save_step, length=1001)

VAR_A = isempty(ARGS) ? 2.0 : parse(Float64, ARGS[1]) #! READ _A_ PARAMETER
const params = ConstParams{Float64}(VAR_A, 3.0, 3.5, 1.5, 0.5, 1.0, 0.5,
    D1, D2, D3, dt, dx, dt / dx^2,
    steps, Int(save_step), Int(N))

# common constants for calculation
const KSI = 0.5 * params.inv_dx2 * [D1, D2, D3] # prefer ∈ [0.25, 0.5]
const TDMA_COEFFS = map(KSI) do ξ
    return (
        fill(-ξ, N - 1),
        [j ∈ (1, N) ? 1 + ξ : 1 + 2 * ξ for j in 1:N],
        fill(-ξ, N - 1)
    )
end


# =============================================
# Run one instance
# =============================================
freq = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 3.25 #! READ FREQUENCY PARAMETER
phase = 0.15

arr = one_calculation_step(freq, phase) #! MAIN FUNCTION RUN

loc_value = metric_local_order(view(arr[2], 2, 1, :), view(arr[2], 2, 3, :))
si_value = metric_si(arr[1], 16, 0.2)
g0_value = metric_g0(view(arr[2], 2, 1, :))

@printf("Metrics: L=%.3f SI=%.3f g₀=%.3f\n", loc_value, si_value, g0_value)
text_with_meta = @sprintf("""Parameters: a=%.2f δx=%.1e δt=%.1e  || θ=%.4fπ f=%.4f 
Metrics: L=%.3f SI=%.3f g₀=%.3f""", params.a, dx, dt, phase, freq, loc_value, si_value, g0_value)
with_theme(create_theme()) do
    plot_plot(arr)
end


# =============================================
# Run on cluster
# =============================================
# freq = isempty(ARGS) ? 0.51 : parse(Float64, ARGS[1])
# results_dir = "results"
# isdir(results_dir) || mkdir(results_dir)
# for freq in range(0.025, 0.5, step=0.025)
# phase_array = range(start=-1, stop=1, step=0.025)
# n_phases = length(phase_array)
# locs = Vector{Float64}(undef, n_phases)
# si_array = Vector{Float64}(undef, n_phases)
# g_null = Vector{Float64}(undef, n_phases)

# u_histories = Vector{Any}(undef, n_phases)  # Any 

# Threads.@threads for i in 1:n_phases
#     phase = phase_array[i]
#     arr = one_calculation_step(freq, phase) #! MAIN FUNCTION RUN in parallel
#     locs[i] = metric_local_order(view(arr[2], 2, 1, :), view(arr[2], 2, 3, :))
#     si_array[i] = metric_si(arr[1], 16, 0.2)
#     g_null[i] = metric_g0(view(arr[2], 2, 1, :))
#     @printf("θ=%.4fπ f=%.4f || Metrics: L=%.3f SI=%.3f g₀=%.3f\n", phase, freq, locs[i], si_array[i], g_null[i])
#     # u_histories[i] = (phase, arr[1][1:10:end, :])
# end

# =============================================
# Save metrics into CSV
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
# =============================================
# Save data into JLD2
# data_file = joinpath(results_dir, @sprintf("data_freq_%.4f.jld2", freq))
# jldopen(data_file, "w"; compress=true) do file
#     file["metadata/dx"] = dx
#     file["metadata/dt"] = dt
#     file["metadata/t_step"] = save_step
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
#     file["loc/values"] = locs
#     file["si/values"] = si_array
#     file["g0/values"] = g_null
#     for (_, (phase, u_history)) in enumerate(u_histories)
#         file["u_history/$(phase)"] = u_history
#     end
# end
# =============================================
