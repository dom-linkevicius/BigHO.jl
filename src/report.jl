"""
    history(ho) -> Vector

The params tuple of every `Completed` trial, in `tell!` order. Derived from
`ho.completed`, so it can never desync from `results(ho)`.
"""
history(ho::Hyperoptimizer) = [ho.trials[i].params for i in ho.completed]

"""
    results(ho) -> Vector

The objective value of every `Completed` trial, in `tell!` order, aligned
index-for-index with `history(ho)`.
"""
results(ho::Hyperoptimizer) = [ho.results[i].value for i in ho.completed]

_no_completed_runs_error(fname) = error("$fname is undefined: this Hyperoptimizer has no completed runs")

"""
    minimum(ho)

The smallest recorded objective value among `Completed` trials (`NaN` never
wins — a legitimate `NaN` return is recorded but excluded from ranking).
Throws if no trial has completed yet (there is no value to report — this is
deliberately not a `NaN`/sentinel return, since a completed trial may
legitimately have recorded `NaN` as its value).

Hyperopt only ever minimizes; to maximize an objective `f`, minimize `-f(...)`.
"""
function Base.minimum(ho::Hyperoptimizer)
    ho.best_min_id === nothing && _no_completed_runs_error("minimum")
    return ho.results[ho.best_min_id].value
end

"""
    minimizer(ho)

The params of the trial that achieved `minimum(ho)`. Throws if no trial has
completed yet.
"""
function minimizer(ho::Hyperoptimizer)
    ho.best_min_id === nothing && _no_completed_runs_error("minimizer")
    return collect(ho.trials[ho.best_min_id].params)
end

_domain_summary(d::Continuous) = "[$(d.min), $(d.max)], dt=$(d.dt)"
_domain_summary(d::Union{Levels,Categorical}) = _values_summary(d.values, nlevels(d))
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
an extreme point of the sampled space.

Example: If parameter `a` can take values in 1:10 and the optimum was obtained
at `a = 1`, it's an indication that the parameter was constrained by the
search space. The warning is effective even if the lowest value of `a` that
was sampled was higher than 1, but the optimum occured on the lowest sampled
value.
"""
_isnumeric_domain(::Continuous) = true
_isnumeric_domain(d::Union{Levels,Categorical}) = _isnumeric_values(d.values)
_isnumeric_values(::Nothing) = false
_isnumeric_values(values::Vector) = eltype(values) <: Real

function warn_on_boundary(ho::Hyperoptimizer)
    m = minimizer(ho)
    hist = history(ho)
    n_params = length(m)
    extremas = map(1:n_params) do i
        c = ho.candidates[i]
        if _isnumeric_domain(c)
            extrema(getindex.(hist, i))
        else
            (m[i],)
        end
    end
    for i in eachindex(m)
        c = ho.candidates[i]
        if m[i] ∈ extremas[i] && nlevels(c) > 3
            println("Parameter $(ho.params[i]) obtained its optimum on an extremum of the sampled region: $(m[i])")
        end
    end
end
