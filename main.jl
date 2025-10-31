using CairoMakie
using ColorSchemes
# using Random
using BenchmarkTools
using LinearAlgebra
using Base.Threads
using Printf
using DataFrames
using CSV
using JLD2
using CodecZlib
using ProgressMeter
using Statistics


struct FHNParams
    a::Float32
    b::Float32
    c::Float32
    alpha::Float32
    phi::Float32
    eps2::Float32
    eps3::Float32
    D1::Float32
    D2::Float32
    D3::Float32
end;

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

function reaction_fhn!(du, u, p::FHNParams)
    a, b, c = p.a, p.b, p.c
    α, ϕ, ϵ₂, ϵ₃ = p.alpha, p.phi, p.eps2, p.eps3
    u1 = @view u[1, :]
    u2 = @view u[2, :]
    u3 = @view u[3, :]
    @. du[1, :] = ϕ * (a * u1 - α * u1 * u1 * u1 - b * u2 - c * u3)
    @. du[2, :] = ϕ * ϵ₂ * (u1 - u2)
    @. du[3, :] = ϕ * ϵ₃ * (u1 - u3)
end;

function runge_kutta_4!(u_next, u_current, p::FHNParams, buffers::RK4Buffers, dt)
    @inbounds begin
        k1, k2, k3, k4 = buffers.k1, buffers.k2, buffers.k3, buffers.k4
        reaction_fhn!(k1, u_current, p)
        @. u_next = u_current + 0.5 * dt * k1
        reaction_fhn!(k2, u_next, p)
        @. u_next = u_current + 0.5 * dt * k2
        reaction_fhn!(k3, u_next, p)
        @. u_next = u_current + dt * k3
        reaction_fhn!(k4, u_next, p)
        dt_6 = dt / 6
        @. u_next = u_current + dt_6 * (k1 + 2 * k2 + 2 * k3 + k4)
    end
end;

function right_hand!(output, input, r)
    N = size(input, 1)
    @inbounds begin
        output[1] = input[1] + 2 * r * (input[2] - input[1])
        output[N] = input[N] + 2 * r * (input[N-1] - input[N])
        for i in 2:N-1
            output[i] = input[i] + r * (input[i-1] - 2 * input[i] + input[i+1])
        end
    end
end;

function thomas_solver!(d, TDMA, c_prime, d_prime)
    N = size(d, 1)

    sub, diag, sup = TDMA.dl, TDMA.d, TDMA.du
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

    temp_real = zeros(N)
    temp_imag = zeros(N)
    for i in 1:N
        ϕ = atan(u_comp[i], w_comp[i])
        temp_real[i] = cos(ϕ)
        temp_imag[i] = sin(ϕ)
    end

    local_R = 0.0
    @inbounds for j in 1:N
        j_prev = max(1, j - 1)
        j_next = min(N, j + 1)
        sum_r = temp_real[j] + temp_real[j_prev] + temp_real[j_next]
        sum_i = temp_imag[j] + temp_imag[j_prev] + temp_imag[j_next]
        local_R += hypot(sum_r, sum_i)
    end
    return sum(local_R) / 3N
end;

function metric_si(u, M, delta)
    # Strength of incoherence with
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
    sigma_m_avg = mean(sigma_m_t, dims=2)[:]
    SI = 1 - sum(sigma_m_avg .< delta) / M
    return SI
end;


# parameters of the model
#                         a    b    c    α    ϕ    ϵ₂   ϵ₃   d1   d2   d3
const params = FHNParams(3.5, 3.0, 3.5, 1.5, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5)

const N = 1024
const dx = 5e-3
const dt = dx * dx / maximum([params.D1, params.D2, params.D3])

const steps = round(Int, 200 / dt)
const start_save = steps * 4 ÷ 5
const sample_interval = (steps - start_save) ÷ 1000  # Store ~X time points in last 20% of time
const save_steps = range(start_save, steps, step=sample_interval)

function main(; phase=0.0, freq=2)

    ksiD = 0.5 * (dt / (dx * dx)) * [params.D1, params.D2, params.D3] # prefer ∈ [0.25, 0.5]

    x = (0:dx:(N-1)*dx)

    # Initial Conditions
    u = zeros(Float64, 3, N)
    @. u[1, :] = sin(x * freq * 2π)
    # @. u[2, :] += 0.55
    @. u[3, :] = sin(x * freq * 2π + π * phase)
    u_init = copy(u) # keep initial for plot_fig

    # Helping vectors for Thomas algorithm
    tdma_coeffs = map(ksiD) do ξ
        return Tridiagonal(fill(-ξ, N - 1), [j ∈ (1, N) ? 1 + ξ : 1 + 2 * ξ for j in 1:N], fill(-ξ, N - 1))
    end

    buffers = RK4Buffers(N) # Helping buffers for RK4
    u_next = copy(u) # temp array

    u_history = Matrix{Float64}(undef, length(save_steps), N) # store for u-component
    # =============================================
    # Main loop
    @time begin
        progress = Progress(steps;
            dt=0.5, desc="Computing...",
            barglyphs=BarGlyphs('|', '█', ['▁', '▂', '▃', '▄', '▅', '▆', '▇'], ' ', '|',),
            barlen=40, showspeed=true
        )
        local history_idx = 0
        for step in 1:steps
            ProgressMeter.update!(progress, step)
            # =============================================
            # Reaction part solver
            # using Runge-Kutta 4th order method
            # =============================================
            runge_kutta_4!(u_next, u, params, buffers, dt)
            # =============================================
            # Diffusion part solver
            # using Thomas algorithm and Neumann BC
            # =============================================
            for k in 1:3
                right_hand!(view(u, k, :), view(u_next, k, :), ksiD[k])
                thomas_solver!(view(u, k, :), tdma_coeffs[k], view(buffers.k1, k, :), view(buffers.k2, k, :))
            end
            if step ∈ save_steps
                history_idx += 1
                u_history[history_idx, :] = u[1, :]
            end
        end
    end

    si = metric_si(u_history, 16, 0.4)
    loc = metric_local_order(view(u, 1, :), view(u, 2, :))
    # End main loop and calculate metrics SI and L
    # =============================================

    text_with_meta = @sprintf("""Simulation with a=%.2f d₃=%.2f δx=%.1ef δt=%.1e
    θ=%.2fπ f=%.2f L=%.3f SI=%.3f""", params.a, params.D3, dx, dt, phase, freq, loc, si)
    fname = @sprintf("freq_%.2f_phase_%.2f", phase, freq)

    function plot_fig()
        # with_theme(fontsize=24, markersize=12, merge(theme_latexfonts(), theme_minimal())) do
        nt, nx = size(u_history)
        x = 1:nx
        t = 1:nt
        maxu = maximum(u) * 1.1 + 0.01
        cmap = cgrad(:seaborn_muted, categorical=true)
        fig = Figure(size=(1100, 1000); colormap=:berlin)
        ax = Axis(fig[1:3, 1:2], xlabel="Space", ylabel="Time step")
        bx = Axis(fig[4, 1:3], title="initial", titlealign=:right)
        cx = Axis(fig[5, 1:3], title="last", titlealign=:right, limits=(nothing, (-maxu, maxu)))
        Label(fig[0, :], text_with_meta)

        hm = heatmap!(ax, x, save_steps, u_history',)
        Colorbar(fig[1:3, 3], hm)

        scl_init = [scatterlines!(bx, u_init[v, :], color=cmap[v], linewidth=0.25) for v in 3:-1:1]
        hidexdecorations!(bx)
        axislegend(bx, reverse(scl_init), ["u", "v", "w"], orientation=:vertical, position=:rt, framevisible=true)

        scl_last = [scatterlines!(cx, u[v, :], color=cmap[v], linewidth=0.5) for v in 3:-1:1]

        # save("./results/fig_$(fname).png", fig)
        display(fig)
    end

    function plot_video()
        # with_theme(merge(theme_latexfonts(), theme_black())) do
        nt, _ = size(u_history)
        crange = extrema(u_history) .* 1.1
        data_vid = Observable(u_history[1, :])
        title_text = Observable("timestep: 0")
        fig = Figure(size=(600, 300); colormap=:berlin)
        ax = Axis(fig[1, 1:2], title=title_text, titlealign=:left, xlabel="Space", ylabel="Value u")
        ylims!(ax, crange)
        Label(fig[0, :], text_with_meta)
        scatterlines!(ax, data_vid, marker=:circle, markersize=12, linestyle=:dash, color=data_vid)

        record(fig, "./results/vid_$fname.mp4", 1:2:nt; framerate=30) do i
            data_vid[] = u_history[i, :]
            title_text[] = "timestep: $((i-1)*sample_interval)"
        end
        # display(fig)
    end

    # with_theme(plot_fig, fontsize=24, markersize=12, merge(theme_latexfonts(), theme_minimal()))
    # with_theme(plot_video, merge(theme_latexfonts(), theme_black()))
    # return loc, si, u_history
end

# =============================================
# Run one instance
# =============================================
main(; phase=0.7, freq=0.55)

# =============================================
# Run in parallel
# =============================================
# Threads.@threads for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
#     main(; phase=0.7, freq=i)
# end

# =============================================
# Run on cluster
# =============================================
# fr = parse(Float64, ARGS[1])
# results_dir = "results"
# if !isdir(results_dir)
#     mkdir(results_dir)
# end
# phase_array = range(start=-1, stop=1, step=0.1)
# n_phases = length(phase_array)
# locs = Vector{Float64}(undef, n_phases)
# si_array = Vector{Float64}(undef, n_phases)

# u_histories = Vector{Any}(undef, n_phases)  # Any 

# Threads.@threads for i in 1:n_phases
#     println(i)
#     phase = phase_array[i]
#     local_order, si_val, history = main(; phase=phase, freq=fr)
#     locs[i] = local_order
#     si_array[i] = si_val
#     u_histories[i] = (phase, history)
# end

# freq_file = joinpath(results_dir, @sprintf("results_freq_%.4f.jld2", fr))
# metric_file = joinpath(results_dir, @sprintf("metrics_freq_%.4f.csv", fr))

# jldopen(freq_file, "w"; compress=true) do file
#     file["metadata/dx"] = dx
#     file["metadata/dt"] = dt
#     file["metadata/t_step"] = sample_interval
#     file["metadata/d1"] = params.D1
#     file["metadata/d2"] = params.D2
#     file["metadata/d3"] = params.D3
#     file["metadata/a"] = params.a
#     file["metadata/b"] = params.b
#     file["metadata/c"] = params.c
#     file["metadata/alpha"] = params.alpha
#     file["metadata/phi"] = params.phi
#     file["metadata/eps2"] = params.eps2
#     file["metadata/eps3"] = params.eps3
#     file["metadata/Nx"] = N
#     file["metadata/Nt"] = steps
#     file["metadata/phase_array"] = phase_array
#     file["loc/values"] = locs
#     file["si/values"] = si_array
#     for (_, (phase, u_history)) in enumerate(u_histories)
#         file["u_history/$(phase)"] = u_history
#     end
# end

# results_df = DataFrame(
#     phase=collect(phase_array),
#     freq=fr,
#     loc=locs,
#     si=si_array
# )
# CSV.write(metric_file, results_df, append=isfile(metric_file))
