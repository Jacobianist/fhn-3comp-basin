using CairoMakie
using BenchmarkTools
using LinearAlgebra
using Base.Threads
using Printf
using CUDA

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
end

function gpu_rk4_kernel!(u_next, u_current, p::FHNParams, dt, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    a, b, c = p.a, p.b, p.c
    α, ϕ, ϵ₂, ϵ₃ = p.alpha, p.phi, p.eps2, p.eps3
    if i <= N

        u1 = u_current[1, i]
        u2 = u_current[2, i]
        u3 = u_current[3, i]

        # k1
        k1_1 = ϕ * (a * u1 - α * u1 * u1 * u1 - b * u2 - c * u3)
        k1_2 = ϕ * ϵ₂ * (u1 - u2)
        k1_3 = ϕ * ϵ₃ * (u1 - u3)

        # k2
        y2_1 = u1 + 0.5 * dt * k1_1
        y2_2 = u2 + 0.5 * dt * k1_2
        y2_3 = u3 + 0.5 * dt * k1_3

        k2_1 = ϕ * (a * y2_1 - α * y2_1 * y2_1 * y2_1 - b * y2_2 - c * y2_3)
        k2_2 = ϕ * ϵ₂ * (y2_1 - y2_2)
        k2_3 = ϕ * ϵ₃ * (y2_1 - y2_3)

        # k3
        y3_1 = u1 + 0.5 * dt * k2_1
        y3_2 = u2 + 0.5 * dt * k2_2
        y3_3 = u3 + 0.5 * dt * k2_3

        k3_1 = ϕ * (a * y3_1 - α * y3_1 * y3_1 * y3_1 - b * y3_2 - c * y3_3)
        k3_2 = ϕ * ϵ₂ * (y3_1 - y3_2)
        k3_3 = ϕ * ϵ₃ * (y3_1 - y3_3)

        # k4
        y4_1 = u1 + dt * k3_1
        y4_2 = u2 + dt * k3_2
        y4_3 = u3 + dt * k3_3

        k4_1 = ϕ * (a * y4_1 - α * y4_1 * y4_1 * y4_1 - b * y4_2 - c * y4_3)
        k4_2 = ϕ * ϵ₂ * (y4_1 - y4_2)
        k4_3 = ϕ * ϵ₃ * (y4_1 - y4_3)

        dt_6 = dt / 6
        u_next[1, i] = u1 + dt_6 * (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1)
        u_next[2, i] = u2 + dt_6 * (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2)
        u_next[3, i] = u3 + dt_6 * (k1_3 + 2 * k2_3 + 2 * k3_3 + k4_3)
    end
    return nothing
end

function gpu_diffusion_kernel!(u_next, u, D, dt, dx, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    r = D * dt / dx^2

    if i == 1
        u_next[i] = u[i] + r * (u[i+1] - u[i])
    elseif i == N
        u_next[i] = u[i] + r * (u[i-1] - u[i])
    elseif 2 <= i <= N - 1
        u_next[i] = u[i] + r * (u[i-1] - 2 * u[i] + u[i+1])
    end
    return nothing
end

# parameters of the model
#                         a    b    c    α    ϕ    ϵ₂   ϵ₃   d1   d2   d3
const params = FHNParams(3.5, 3.0, 3.5, 1.5, 0.5, 1.0, 0.5, 0.0, 0.0, 0.5)

const N = 1024
const dx = 5e-3
const dt = 0.5 * dx * dx / maximum([params.D1, params.D2, params.D3])

const steps = round(Int, 50 / dt)
const start_save = steps * 4 ÷ 5
const sample_interval = (steps - start_save) ÷ 1000  # Store ~X time points in last 20% of time
const save_steps = range(start_save, steps, step=sample_interval)

function main(; phase=0.0, freq=1)
    x = CuArray((0:dx:(N-1)*dx))
    u = CUDA.zeros(Float32, 3, N)
    u[1, :] .= sin.(x .* freq * 2π)
    u[2, :] .= sin.(x .* freq * 2π .+ π * phase)

    u_init = Array(u) # keep initial for plot_fig
    u_next = similar(u)

    u_history_device = CuArray{Float32}(undef, length(save_steps), N) # store for u-component
    # =============================================
    # Main loop
    threads = 128
    blocks = cld(N, threads)

    @time begin
        local history_idx = 0
        for step in 1:steps
            # =============================================
            # Reaction part solver
            # using Runge-Kutta 4th order method
            # =============================================            
            @cuda blocks = blocks threads = threads gpu_rk4_kernel!(u_next, u, params, dt, N)
            # =============================================
            # Diffusion part solver
            # using explicit method and Neumann BC
            # =============================================
            @cuda blocks = blocks threads = threads gpu_diffusion_kernel!(view(u, 1, :), view(u_next, 1, :), params.D1, dt, dx, N)
            @cuda blocks = blocks threads = threads gpu_diffusion_kernel!(view(u, 2, :), view(u_next, 2, :), params.D2, dt, dx, N)
            @cuda blocks = blocks threads = threads gpu_diffusion_kernel!(view(u, 3, :), view(u_next, 3, :), params.D3, dt, dx, N)
            # =============================================
            # Save only several time snapshots for visualize
            # =============================================
            if step ∈ save_steps
                history_idx += 1
                u_history_device[history_idx, :] = u[1, :]
            end
        end
    end

    u_final = Array(u)
    u_history_host = Array(u_history_device)

    text_with_meta = @sprintf("""Simulation with a=%.2f d₃=%.2f δx=%.1ef δt=%.1e
    θ=%.2fπ f=%.2f""", params.a, params.D3, dx, dt, phase, freq)
    fname = @sprintf("freq_%.2f_phase_%.2f", phase, freq)

    function plot_fig()
        # with_theme(fontsize=24, markersize=12, merge(theme_latexfonts(), theme_minimal())) do
        nt, nx = size(u_history_host)
        x = 1:nx
        t = 1:nt
        maxu = abs(maximum(u_final)) * 1.1 + 0.01
        cmap = cgrad(:seaborn_muted, categorical=true)
        fig = Figure(size=(1100, 1000); colormap=:berlin)
        ax = Axis(fig[1:3, 1:2], xlabel="Space", ylabel="Time step")
        bx = Axis(fig[4, 1:3], title="initial", titlealign=:right)
        cx = Axis(fig[5, 1:3], title="last", titlealign=:right, limits=(nothing, (-maxu, maxu)))
        Label(fig[0, :], text_with_meta)

        hm = heatmap!(ax, x, save_steps, u_history_host',)
        Colorbar(fig[1:3, 3], hm)

        scl_init = [scatterlines!(bx, u_init[v, :], color=cmap[v], linewidth=0.25) for v in 3:-1:1]
        hidexdecorations!(bx)
        axislegend(bx, reverse(scl_init), ["u", "v", "w"], orientation=:vertical, position=:rt, framevisible=true)

        scl_last = [scatterlines!(cx, u_final[v, :], color=cmap[v], linewidth=0.5) for v in 3:-1:1]

        # save("./results/fig_$(fname).png", fig)
        display(fig)
    end
    with_theme(plot_fig, fontsize=24, markersize=12, merge(theme_latexfonts(), theme_minimal()))
end

main()
