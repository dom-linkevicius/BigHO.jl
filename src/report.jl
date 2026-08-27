"""
    history(ho) -> Vector

The params tuple of every `Completed` trial, in `tell!` order. Derived from
`ho.completed`, so it can never desync from `results(ho)`.
"""
history(ho::Hyperoptimizer) = [ho.runs[i].params for i in ho.completed]

"""
    results(ho) -> Vector

The objective value of every `Completed` trial, in `tell!` order, aligned
index-for-index with `history(ho)`.
"""
results(ho::Hyperoptimizer) = [ho.runs[i].value for i in ho.completed]

_no_completed_runs_error(fname) = error("$fname is undefined: this Hyperoptimizer has no completed runs")

"""
    minimum(ho)

The smallest recorded objective value among `Completed` trials (`NaN` is
excluded as a valid optimization value -- an objective returning `NaN`
produces a `Failed` trial, never a `Completed` one, so it's never a
candidate here). Throws if no trial has completed yet (there is no value to
report -- deliberately not a `NaN`/sentinel return).

Hyperopt only ever minimizes; to maximize an objective `f`, minimize `-f(...)`.
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

_domain_summary(d::Continuous) = "[$(d.min), $(d.max)], dt=$(d.dt)"
_domain_summary(d::Categorical) = _values_summary(d.values, nlevels(d))
_values_summary(::Nothing, n::Int) = "length: $n"
_values_summary(values::Vector, n::Int) = length(values) <= 3 ? string(values) : "length: $n"

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

"""
    warn_on_boundary(ho)

Prints a warning message for each parameter where the optimum was obtained on
an extreme point of the sampled space. Only meaningful for `Continuous`/
`Ordinal` domains -- `Nominal` domains have no notion of "boundary" at all
(their levels aren't ordered), so they're skipped entirely.

Example: If parameter `a` can take values in 1:10 and the optimum was obtained
at `a = 1`, it's an indication that the parameter was constrained by the
search space. The warning is effective even if the lowest value of `a` that
was sampled was higher than 1, but the optimum occured on the lowest sampled
value -- deliberately checked against the *sampled* extreme, not the
domain's declared bounds, since a small sample may never actually reach the
true min/max even though the optimizer is still visibly pushing toward one.
"""
_has_boundary(::Continuous) = true
_has_boundary(::Ordinal) = true
_has_boundary(::Nominal) = false

# A value's position for boundary-extremum comparison purposes: itself, when
# numeric order already means the right thing (Continuous, or an Ordinal
# with numeric values, or an Ordinal sampled as a bare level index) -- or,
# for a non-numeric Ordinal, its position in the domain's own (asserted, at
# construction) order, since e.g. `extrema(["low","medium","high"])` would
# silently use alphabetical order instead (see Ordinal's own docstring).
_boundary_rank(::Continuous, v) = v
_boundary_rank(::Ordinal, v::Real) = v
_boundary_rank(d::Ordinal, v) = findfirst(==(v), d.values)

function warn_on_boundary(ho::Hyperoptimizer)
    m = minimizer(ho)
    hist = history(ho)
    for i in eachindex(m)
        c = ho.candidates[i]
        (_has_boundary(c) && nlevels(c) > 3) || continue
        ranks = [_boundary_rank(c, h[i]) for h in hist]
        if _boundary_rank(c, m[i]) ∈ extrema(ranks)
            println("Parameter $(ho.params[i]) obtained its optimum on an extremum of the sampled region: $(m[i])")
        end
    end
end
