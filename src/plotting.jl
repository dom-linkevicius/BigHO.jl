"""
    summaryplot(ho::Hyperoptimizer)

One figure summarizing a completed run: for each hyperparameter, its value
against the objective value with marginal histograms; plus the objective
value over trial id, and the best value found so far over trial id.
Requires `CairoMakie` and `AlgebraOfGraphics` to be loaded (implemented as a
package extension) -- returns a `Makie.Figure`.
"""
function summaryplot end
