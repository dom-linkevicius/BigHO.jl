module BigHOMakieExt

using BigHO
using CairoMakie
import AlgebraOfGraphics
using AlgebraOfGraphics: data, mapping, visual, draw!

function BigHO.summaryplot(ho::BigHO.Hyperoptimizer)
    isempty(ho.completed) &&
        error("summaryplot: this Hyperoptimizer has no completed runs")
    completed = sort(BigHO.DataFrames.DataFrame(ho)[ho.completed, :], :id)

    fig = Figure(size=(700, 300 * length(ho.params) + 500))

    for (i, p) in enumerate(ho.params)
        gl = fig[i, 1] = GridLayout()
        ax_top = Axis(gl[1, 1])
        ax_main = Axis(gl[2, 1]; xlabel=string(p), ylabel="value")
        ax_right = Axis(gl[2, 2])
        draw!(ax_top, data(completed) * mapping(p) * AlgebraOfGraphics.histogram())
        draw!(ax_main, data(completed) * mapping(p, :value) * visual(Scatter))
        draw!(ax_right, data(completed) * mapping(:value) * AlgebraOfGraphics.histogram() * visual(BarPlot; direction=:x))
        linkxaxes!(ax_top, ax_main)
        linkyaxes!(ax_right, ax_main)
        hidedecorations!(ax_top, grid=false)
        hidedecorations!(ax_right, grid=false)
        colsize!(gl, 2, Relative(0.2))
        rowsize!(gl, 1, Relative(0.2))
    end

    n = length(ho.params)
    ax_id = Axis(fig[n+1, 1]; xlabel="id", ylabel="value", title="objective value over id")
    draw!(ax_id, data(completed) * mapping(:id, :value) * visual(Scatter))

    ax_running = Axis(fig[n+2, 1]; xlabel="id", ylabel="best value so far", title="minimum over runs so far")
    lines!(ax_running, completed.id, accumulate(min, completed.value))

    return fig
end

end # module
