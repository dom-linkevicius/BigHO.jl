using CairoMakie
using JLD2
using Statistics: median, quantile

const RESULTS_PATH = joinpath(@__DIR__, "results.jld2")
const OUTDIR = joinpath(@__DIR__, "..", "docs", "benchmarks")
mkpath(OUTDIR)

const REGRET_GRID_POINTS = 40
const SAMPLER_NAMES = ("Random", "Hyperband", "ASHA")
const SAMPLER_COLORS = Dict(zip(SAMPLER_NAMES, Makie.wong_colors()))
const SHA_SAMPLER_NAMES = ("Hyperband", "ASHA")   # wrap an inner per-draw sampler, named in the legend
const Y_UPPER_LIMIT = 10^-1.2   # headroom so the top-right legend doesn't occlude any lines
const X_LOWER_LIMIT = 1e-1
const X_UPPER_LIMIT = 1000

"""
    _resample_to_grid(times, running_min, grid)
"""
function _resample_to_grid(times, running_min, grid)
    out = fill(NaN, length(grid))
    j = 0
    for (i, g) in enumerate(grid)
        while j < length(times) && times[j+1] <= g
            j += 1
        end
        j > 0 && (out[i] = running_min[j])
    end
    return out
end

# One row per executor: 8 overlapping lines/bands on one axis was unreadable. Both rows share axis
# limits and colors. Re-runnable on its own -- the curves come from collect_results.jl's results.jld2.
function plot_results(runs, metadata)
    regret_repeats = metadata.regret_repeats
    all_values = Float64[e for curves in values(runs) for (_, best) in curves for e in best]
    global_best = minimum(all_values)

    # Precompute each (sampler, executor)'s own grid/regret/band once, and track the combined
    # data extent so both rows can share identical axis limits.
    computed = Dict{Tuple{String,Symbol},NamedTuple}()
    ymin = Inf
    for name in SAMPLER_NAMES, ex_name in (:Serial, :Threaded)
        curves = runs[(name, ex_name)]
        # Own x-range per combo, not a shared one: a shared endpoint holds a fast combo's line flat
        # out to the slowest one's finish, implying it kept running when it had long since stopped.
        own_min = minimum(times[1] for (times, _) in curves if !isempty(times))
        own_max = maximum(times[end] for (times, _) in curves if !isempty(times))
        grid = exp10.(range(log10(own_min), log10(own_max); length=REGRET_GRID_POINTS))
        curves_on_grid = Matrix{Float64}(undef, length(curves), length(grid))
        for (r, (times, best)) in enumerate(curves)
            curves_on_grid[r, :] = _resample_to_grid(times, best, grid)
        end
        # Median + 25/75 band, not mean+-std: std is symmetric and one outlier repeat widens it
        # enough to bury the lines in shading.
        regret = [any(isnan, col) ? NaN : max(median(col) - global_best, 1e-4) for col in eachcol(curves_on_grid)]
        lower = [any(isnan, col) ? NaN : max(quantile(col, 0.25) - global_best, 1e-4) for col in eachcol(curves_on_grid)]
        upper = [any(isnan, col) ? NaN : max(quantile(col, 0.75) - global_best, 1e-4) for col in eachcol(curves_on_grid)]
        computed[(name, ex_name)] = (; grid, regret, lower, upper)
        finite_lower = filter(!isnan, lower)
        isempty(finite_lower) || (ymin = min(ymin, minimum(finite_lower)))
    end

    fig = Figure(size=(750, 950))
    for (row, ex_name, title_word) in ((1, :Serial, "Serial"), (2, :Threaded, "Threaded"))
        ax = Axis(fig[row, 1]; xlabel="wall-clock time (s)", ylabel="regret (best validation loss - global best)",
                  xscale=log10, yscale=log10,
                  xminorticksvisible=true, xminorgridvisible=true, xminorticks=IntervalsBetween(9),
                  yminorticksvisible=true, yminorgridvisible=true, yminorticks=IntervalsBetween(9),
                  title="Wall-clock regret comparison -- $title_word ($regret_repeats repeats/combo, $(metadata.nthreads) threads)")
        for name in SAMPLER_NAMES
            c = computed[(name, ex_name)]
            color = SAMPLER_COLORS[name]
            label_name = name in SHA_SAMPLER_NAMES ? "$name/$(metadata.sha_inner_sampler)" : name
            lines!(ax, c.grid, c.regret; label=label_name, color=color)
            band!(ax, c.grid, c.lower, c.upper; color=(color, 0.15))
        end
        xlims!(ax, X_LOWER_LIMIT, X_UPPER_LIMIT)
        ylims!(ax, ymin, Y_UPPER_LIMIT)
        axislegend(ax; position=:rt, nbanks=1)
    end
    save(joinpath(OUTDIR, "wallclock_regret_comparison.png"), fig)
end

@info "Loading results from $RESULTS_PATH..."
runs, metadata = JLD2.load(RESULTS_PATH, "runs", "metadata")
plot_results(runs, metadata)
@info "Plot written to $OUTDIR"
