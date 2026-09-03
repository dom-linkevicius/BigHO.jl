"""
    summaryplot(ho::Hyperoptimizer; figure_kwargs=(;), axis_kwargs=(;), scatter_kwargs=(;), histogram_kwargs=(;), line_kwargs=(;))

One figure summarizing a completed run: for each hyperparameter, its value
against the objective value with marginal histograms; plus the objective
value over trial id, and the best value found so far over trial id.
Requires `CairoMakie` and `AlgebraOfGraphics` to be loaded (implemented as a
package extension) -- returns a `Makie.Figure`.

Each `_kwargs` argument is forwarded (as keyword arguments) to the
corresponding call: `figure_kwargs` to `Figure`, `axis_kwargs` to every
`Axis`, `scatter_kwargs`/`histogram_kwargs` to the scatter/histogram marks'
`visual`, and `line_kwargs` to the running-minimum line.
"""
function summaryplot end
