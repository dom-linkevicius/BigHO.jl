module BigHOMakieExt

using BigHO
using CairoMakie
import AlgebraOfGraphics
using AlgebraOfGraphics: data, mapping, visual, draw!

function _draw_marginal_scatter!(fig, row, col, completed, p; axis_kwargs, scatter_kwargs, histogram_kwargs)
    gl = fig[row, col] = GridLayout()
    ax_top = Axis(gl[1, 1]; axis_kwargs...)
    ax_main = Axis(gl[2, 1]; xlabel=string(p), ylabel="value", axis_kwargs...)
    ax_right = Axis(gl[2, 2]; axis_kwargs...)
    draw!(ax_top, data(completed) * mapping(p) * AlgebraOfGraphics.histogram() * visual(BarPlot; histogram_kwargs...))
    draw!(ax_main, data(completed) * mapping(p, :value) * visual(Scatter; scatter_kwargs...))
    draw!(ax_right, data(completed) * mapping(:value) * AlgebraOfGraphics.histogram() * visual(BarPlot; direction=:x, histogram_kwargs...))
    linkxaxes!(ax_top, ax_main)
    linkyaxes!(ax_right, ax_main)
    hidedecorations!(ax_top, grid=false)
    hidedecorations!(ax_right, grid=false)
    colsize!(gl, 2, Relative(0.2))
    rowsize!(gl, 1, Relative(0.2))
    return nothing
end

function _draw_value_over_id!(fig, row, col, completed; axis_kwargs, scatter_kwargs)
    ax = Axis(fig[row, col]; xlabel="id", ylabel="value", title="objective value over id", axis_kwargs...)
    draw!(ax, data(completed) * mapping(:id, :value) * visual(Scatter; scatter_kwargs...))
    return nothing
end

function _draw_running_min!(fig, row, col, completed; axis_kwargs, line_kwargs)
    ax = Axis(fig[row, col]; xlabel="id", ylabel="best value so far", title="minimum over runs so far", axis_kwargs...)
    lines!(ax, completed.id, accumulate(min, completed.value); line_kwargs...)
    return nothing
end

function BigHO.summaryplot(ho::BigHO.Hyperoptimizer; figure_kwargs=NamedTuple(), axis_kwargs=NamedTuple(),
                            scatter_kwargs=NamedTuple(), histogram_kwargs=NamedTuple(), line_kwargs=NamedTuple())
    isempty(ho.completed) &&
        error("summaryplot: this Hyperoptimizer has no completed runs")
    completed = sort(BigHO.DataFrames.DataFrame(ho)[ho.completed, :], :id)

    panels = Any[(fig, row, col) -> _draw_marginal_scatter!(fig, row, col, completed, p; axis_kwargs, scatter_kwargs, histogram_kwargs) for p in ho.params]
    push!(panels, (fig, row, col) -> _draw_value_over_id!(fig, row, col, completed; axis_kwargs, scatter_kwargs))
    push!(panels, (fig, row, col) -> _draw_running_min!(fig, row, col, completed; axis_kwargs, line_kwargs))

    # As square as possible (e.g. 4 -> 2x2, 6 -> 2x3), rather than one long column.
    total = length(panels)
    nrows = max(1, floor(Int, sqrt(total)))
    ncols = cld(total, nrows)

    fig = Figure(; size=(400 * ncols, 400 * nrows), figure_kwargs...)
    for (i, panel) in enumerate(panels)
        row = (i - 1) ÷ ncols + 1
        col = (i - 1) % ncols + 1
        panel(fig, row, col)
    end

    return fig
end

end # module
