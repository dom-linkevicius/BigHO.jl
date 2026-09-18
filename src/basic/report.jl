"""
    history(ho) -> Vector
"""
history(ho::Hyperoptimizer) = [ho.runs[i].params for i in ho.completed]

"""
    results(ho) -> Vector
"""
results(ho::Hyperoptimizer) = [ho.runs[i].value for i in ho.completed]

_no_completed_runs_error(fname) = error("$fname is undefined: this Hyperoptimizer has no completed runs")

"""
    minimum(ho)
"""
function Base.minimum(ho::Hyperoptimizer)
    ho.best_min_id === nothing && _no_completed_runs_error("minimum")
    return ho.runs[ho.best_min_id].value
end

"""
    minimizer(ho)
"""
function minimizer(ho::Hyperoptimizer)
    ho.best_min_id === nothing && _no_completed_runs_error("minimizer")
    return collect(ho.runs[ho.best_min_id].params)
end

_domain_summary(d::Continuous) = "[$(d.min), $(d.max)]" * (d.transform === identity ? "" : " via $(d.transform)")
_domain_summary(d::Union{Nominal,Ordinal}) = length(d) <= 5 ? string(d.values) : "length: $(length(d))"

_show_schedule(io::IO, ::Sampler) = nothing
function _show_schedule(io::IO, s::SuccessiveHalving)
    levels = [_resource(s.R, s.r_min, s.η, 1, rung) for rung in 1:_n_rungs(s.R, s.r_min, s.η, 1)]
    println(io, "  resource levels: " * join(levels, ", "))
    println(io, "  total trials: $(_total_trials(s))")
end

function Base.show(io::IO, ho::Hyperoptimizer)
    println(io, "Hyperoptimizer with")
    candstrings = map(1:length(ho.candidates)) do i
        k, c = ho.params[i], ho.candidates[i]
        "  " * string(k) * " " * _domain_summary(c)
    end
    println(io, join(candstrings, "\n"))
    _show_schedule(io, ho.sampler)
    _show_optimum(io, ho, ho.best_min_id)
end

_show_optimum(io::IO, ho::Hyperoptimizer, ::Nothing) = println(io, "  no completed runs yet")
function _show_optimum(io::IO, ho::Hyperoptimizer, ::Int)
    println(io, "  minimum: $(minimum(ho))")
    println(io, "  minimizer:")
    mzer = minimizer(ho)
    for k in _minimizer_names(ho)
        @printf(io, "%9s ", string(k))
    end
    println(io)
    for v in mzer
        _print_value(io, v)
    end
    println(io)
end

_minimizer_names(ho::Hyperoptimizer) = keys(ho.runs[ho.best_min_id].params)

_print_value(io::IO, v::Number) = @printf(io, "%9.4g ", v)
_print_value(io::IO, v) = @printf(io, "%9s ", v)

"""
    printmin([io=stdout,] ho)
"""
printmin(ho::Hyperoptimizer) = printmin(stdout, ho)
function printmin(io::IO, ho::Hyperoptimizer)
    mzer = minimizer(ho)
    for (param, value) in zip(_minimizer_names(ho), mzer)
        println(io, param, " = ", value)
    end
end
