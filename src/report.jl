"""
    history(ho) -> Vector

The params of every `Completed` trial, in `tell!` order.
"""
history(ho::Hyperoptimizer) = [ho.runs[i].params for i in ho.completed]

"""
    results(ho) -> Vector

The objective value of every `Completed` trial, aligned index-for-index
with `history(ho)`.
"""
results(ho::Hyperoptimizer) = [ho.runs[i].value for i in ho.completed]

_no_completed_runs_error(fname) = error("$fname is undefined: this Hyperoptimizer has no completed runs")

"""
    minimum(ho)

The smallest recorded objective value. Throws if no trial has completed yet.
"""
function Base.minimum(ho::Hyperoptimizer)
    ho.best_min_id === nothing && _no_completed_runs_error("minimum")
    return ho.runs[ho.best_min_id].value
end

"""
    minimizer(ho)

The params of the trial that achieved `minimum(ho)`. Throws if no trial has
completed yet.
"""
function minimizer(ho::Hyperoptimizer)
    ho.best_min_id === nothing && _no_completed_runs_error("minimizer")
    return collect(ho.runs[ho.best_min_id].params)
end

# A Continuous domain's `values` isn't necessarily evenly spaced (e.g.
# log-spaced), so there's no single `dt` to show -- don't assume it's a
# range, dispatch on `d.type` instead.
_domain_summary(d::Domain) = _domain_summary(d, Val(d.type))
_domain_summary(d::Domain, ::Val{:continuous}) = "[$(first(d.values)), $(last(d.values))], $(length(d.values)) points"
_domain_summary(d::Domain, ::Union{Val{:nominal},Val{:ordinal}}) = length(d.values) <= 5 ? string(d.values) : "length: $(length(d.values))"

function Base.show(io::IO, ho::Hyperoptimizer)
    println(io, "Hyperoptimizer with")
    candstrings = map(1:length(ho.candidates)) do i
        k, c = ho.params[i], ho.candidates[i]
        "  " * string(k) * " " * _domain_summary(c)
    end
    println(io, join(candstrings, "\n"))
    _show_optimum(io, ho, ho.best_min_id)
end

_show_optimum(io::IO, ho::Hyperoptimizer, ::Nothing) = println(io, "  no completed runs yet")
function _show_optimum(io::IO, ho::Hyperoptimizer, ::Int)
    println(io, "  minimum: $(minimum(ho))")
    println(io, "  minimizer:")
    mzer = minimizer(ho)
    for i in eachindex(mzer)
        @printf(io, "%9s ", string(ho.params[i]))
    end
    println(io)
    for v in mzer
        _print_value(io, v)
    end
    println(io)
end

_print_value(io::IO, v::Number) = @printf(io, "%9.4g ", v)
_print_value(io::IO, v) = @printf(io, "%9s ", v)

"""
    printmin([io=stdout,] ho)

Prints the parameters that minimized the function.
"""
printmin(ho::Hyperoptimizer) = printmin(stdout, ho)
function printmin(io::IO, ho::Hyperoptimizer)
    for (param, value) in zip(ho.params, minimizer(ho))
        println(io, param, " = ", value)
    end
end

