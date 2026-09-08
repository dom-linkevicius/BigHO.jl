using CairoMakie
using JLD2
using Statistics: median, quantile

const RESULTS_PATH = joinpath(@__DIR__, "results.jld2")
const OUTDIR = joinpath(@__DIR__, "..", "docs", "benchmarks")
mkpath(OUTDIR)

const REGRET_GRID_POINTS = 40
const SAMPLER_NAMES = ("Random", "Hyperband", "ASHA")
const SAMPLER_COLORS = Dict(zip(SAMPLER_NAMES, Makie.wong_colors()))
const SHA_SAMPLER_NAMES = ("Hyperband", "ASHA")   # these wrap an inner per-draw sampler -- named in the legend
const Y_UPPER_LIMIT = 10^-1.2   # fixed headroom above the highest curve so the top-right legend doesn't occlude any lines
const X_LOWER_LIMIT = 1e-1
const X_UPPER_LIMIT = 1000

"""
    _resample_to_grid(times, running_min, grid)

Step-function resample: value of the running-min curve at each grid time, `NaN` before
this repeat's first completion (rather than e.g. `Inf`) -- `NaN` lets Makie render a gap
and lets `mean`/`std` honestly propagate "not enough repeats have data yet" instead of
polluting the y-axis autoscale the way an `Inf` sentinel would.
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

# ---- wall-clock regret comparison (BOHB paper, Fig. 1 style) ----
# One row per executor (Serial on top, Threaded below) rather than both on one axis -- 8
# overlapping lines/bands on a single plot was too cluttered to read. Both rows share the same
# x/y axis limits and per-sampler colors, so they're directly comparable at a glance. Loads raw
# per-repeat curves collected by `collect_results.jl` -- edit/re-run this file alone to tweak
# the plot without retraining.
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
        # Each (sampler, executor) gets its OWN x-range, ending at its own last completion time --
        # NOT a shared grid ending at the slowest combo's finish time. A shared endpoint would hold
        # a fast combo's line flat all the way out to match the slowest one, visually implying it
        # kept running just as long, when it actually finished (and stopped mattering) much earlier.
        own_min = minimum(times[1] for (times, _) in curves if !isempty(times))
        own_max = maximum(times[end] for (times, _) in curves if !isempty(times))
        grid = exp10.(range(log10(own_min), log10(own_max); length=REGRET_GRID_POINTS))
        curves_on_grid = Matrix{Float64}(undef, length(curves), length(grid))
        for (r, (times, best)) in enumerate(curves)
            curves_on_grid[r, :] = _resample_to_grid(times, best, grid)
        end
        # Median + 25th/75th percentile band instead of mean+-std -- std is symmetric around the
        # mean and blows up with even one slow/fast repeat, making bands wide enough to bury the
        # lines in overlapping shading. Quantiles show the actual spread of the 10 repeats instead.
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
