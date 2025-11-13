using CairoMakie
using ColorSchemes
using BenchmarkTools
using LinearAlgebra
using Base.Threads
using Printf
using CUDA
using StatsBase

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
    steps::Int32
    save_step::Int32        # каждые save_step шагов пишем снимок
    N::Int32
end

function gpu_kernel_rk_diffusion!(u_next, u_current)
    gidx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    gidx > params.N && return
    a, b, c, α, ϕ, ϵ₂, ϵ₃ = params.a, params.b, params.c, params.α, params.ϕ, params.ϵ₂, params.ϵ₃

    # --------- границы Neumann ----------
    left = gidx == 1 ? 2 : gidx - 1
    right = gidx == params.N ? params.N - 1 : gidx + 1

    # --------- текущие значения ----------
    u1 = u_current[1, gidx]
    u2 = u_current[2, gidx]
    u3 = u_current[3, gidx]
    um1 = u_current[1, left]
    up1 = u_current[1, right]
    um2 = u_current[2, left]
    up2 = u_current[2, right]
    um3 = u_current[3, left]
    up3 = u_current[3, right]
    @inline function rhs(u1, u2, u3)
        du1 = ϕ * (a * u1 - α * u1^3 - b * u2 - c * u3)
        du2 = ϕ * ϵ₂ * (u1 - u2)
        du3 = ϕ * ϵ₃ * (u1 - u3)
        return du1, du2, du3
    end
    # --------- RK4 ----------
    k1_1, k1_2, k1_3 = rhs(u1, u2, u3)
    y1 = u1 + 0.5f0 * params.dt * k1_1
    y2 = u2 + 0.5f0 * params.dt * k1_2
    y3 = u3 + 0.5f0 * params.dt * k1_3
    k2_1, k2_2, k2_3 = rhs(y1, y2, y3)
    y1 = u1 + 0.5f0 * params.dt * k2_1
    y2 = u2 + 0.5f0 * params.dt * k2_2
    y3 = u3 + 0.5f0 * params.dt * k2_3
    k3_1, k3_2, k3_3 = rhs(y1, y2, y3)
    y1 = u1 + params.dt * k3_1
    y2 = u2 + params.dt * k3_2
    y3 = u3 + params.dt * k3_3
    k4_1, k4_2, k4_3 = rhs(y1, y2, y3)

    dt6 = params.dt / 6.0f0
    u1 += dt6 * (k1_1 + 2.0f0 * (k2_1 + k3_1) + k4_1)
    u2 += dt6 * (k1_2 + 2.0f0 * (k2_2 + k3_2) + k4_2)
    u3 += dt6 * (k1_3 + 2.0f0 * (k2_3 + k3_3) + k4_3)

    lap1 = up1 + um1 - 2.0f0 * u_current[1, gidx]
    lap2 = up2 + um2 - 2.0f0 * u_current[2, gidx]
    lap3 = up3 + um3 - 2.0f0 * u_current[3, gidx]

    u1 += params.inv_dx2 * params.D1 * lap1
    u2 += params.inv_dx2 * params.D2 * lap2
    u3 += params.inv_dx2 * params.D3 * lap3

    @inbounds begin
        u_next[1, gidx] = u1
        u_next[2, gidx] = u2
        u_next[3, gidx] = u3
    end
    return nothing
end

# =============================================
# Metrics functions
function metric_local_order(u_comp, w_comp)
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
    return sum(local_R) / (3 * N)
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

function metric_g0(u; delta_factor=0.01, nbins=50)
    # A classification scheme for chimera states. Kemeth et al DOI: 10.1063/1.4959804 
    # ? shorter version without destribution
    # metric_g0(u, dx; δ_factor=0.01) =
    #     let D = abs.(@view(u[3:end]) - 2 * @view(u[2:end-1]) + @view(u[1:end-2])) ./ (dx)^2
    #         δ = δ_factor * maximum(D)
    #         sum(D .<= δ) / length(D)
    #     end
    LaplaceNodx = (u[3:end] - 2 * u[2:end-1] + u[1:end-2])
    Dm = maximum(abs, LaplaceNodx)
    if isapprox(Dm, 0; atol=1e-5)
        return 1.0
    end
    delta = delta_factor * Dm
    edges = range(0.0, stop=Dm, length=nbins + 1)
    hist = fit(Histogram, LaplaceNodx, edges)
    bin_width = edges[2] - edges[1]
    g = hist.weights ./ (sum(hist.weights) * bin_width)
    # interval 0 to delta
    idx_delta = findlast(edges .<= delta)
    if isnothing(idx_delta) || idx_delta == 0
        g0 = 0.0
    else
        g0 = sum(g[1:idx_delta]) * bin_width
    end
    return g0
end


# Main calculation function with one kernel run
function cuda_run_calculation(freq::Float32, phase::Float32)
    x = CUDA.range(0f0, (params.N - 1) * params.dx, length=params.N)
    u = CUDA.zeros(Float32, 3, params.N)
    @. u[1, :] = sin(x * freq * 2f0 * π - 0.5 * π * phase)
    @. u[2, :] = sin(x * freq * 2f0 * π + 0.5 * π * phase)
    u_next = similar(u)
    initlast = zeros(Float32, 2, 3, N)
    initlast[1, :, :] = Array(u)
    u_history = Matrix{Float64}(undef, 1001, N)

    CUDA.@time begin
        history_idx = 0
        for step in 1:params.steps
            @cuda threads = cuda_threads blocks = cuda_blocks kernel_run_two!(u_next, u)
            u, u_next = u_next, u
            if step % 1000 == 0
                CUDA.synchronize()
            end
            if step ∈ save_range
                history_idx += 1
                u_history[history_idx, :] = Array(u[1, :])
            end
        end
    end
    initlast[2, :, :] = Array(u)
    return u_history, initlast
end

const N::Int = 1024
const dx::Float32 = 0.005f0
const D1::Float32 = 0.0f0
const D2::Float32 = 0.0f0
const D3::Float32 = 0.5f0
const dt::Float32 = 0.5f0 * dx^2 / max(D1, D2, D3)
const steps = round(Int, 50 / dt)
const save_step = steps ÷ 4 ÷ 1000
const save_range = range(stop=steps, step=save_step, length=1001)

const params = ConstParams{Float32}(3.5, 3.0, 3.5, 1.5, 0.5, 1.0, 0.5,
    D1, D2, D3, dt, dx, dt / dx^2,
    steps, Int32(save_step), Int32(N))

const cuda_threads = 256
const cuda_blocks = cld(N, cuda_threads)


# Make a theme for prettier plot
create_theme() =
    let
        merge(theme_latexfonts(), theme_black(),
            CairoMakie.Theme(
                # font="CMU Serif",
                # figure_padding=(5, 5, 10, 10),
                size=(900, 800),
                fontsize=20,
                # colormap=:berlin,
                # color=cgrad(:seaborn_muted, categorical=true),
                markersize=12,
                linewidth=8,
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

    scl_init = [lines!(bx, initlast[1, v, :], label=la[v], color=cmap[v],) for v in 1:3]
    hidexdecorations!(bx, grid=false)
    leg = Legend(gb[1:2, 2], scl_init, la)
    lines!(cx, initlast[2, 3, :], color=cmap[3], alpha=0.7)
    lines!(cx, initlast[2, 2, :], color=cmap[2], alpha=0.7)
    scl_last = scatterlines!(cx, initlast[2, 1, :], color=cmap[1], strokecolor=:white, strokewidth=0.1, linewidth=0.2)
    rowgap!(gb, 4)
    colgap!(gb, 5)

    Label(fig[0, :], text_with_meta)
    display(fig)
end

freq = isempty(ARGS) ? 0.6f0 : parse(Float32, ARGS[1])
phase = 0.45f0

arr = cuda_run_calculation(freq, phase) #! MAIN FUNCTION RUN

loc_value = metric_local_order(view(arr[2], 2, 1, :), view(arr[2], 2, 3, :))
si_value = metric_si(arr[1], 16, 0.2)
g0_value = metric_g0(view(arr[2], 2, 1, :))
@printf("Metrics: L=%.3f SI=%.3f g₀=%.3f", loc, si, g0)

text_with_meta = @sprintf("""Parameters: δx=%.1e δt=%.1e  || θ=%.4fπ f=%.4f 
Metrics: L=%.3f SI=%.3f g₀=%.3f""", dx, dt, phase, freq, loc, si, g0)
with_theme(create_theme()) do
    plot_plot(arr)
end


