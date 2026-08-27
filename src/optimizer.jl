mutable struct Hyperoptimizer{S<:Sampler,F}
    params::Vector{Symbol}
    candidates::Tuple
    sampler::S
    objective::F
    n::Union{Int,Nothing}
    trials::Vector{Trial}
    results::Vector{Union{Result,Nothing}}
    completed::Vector{Int}
    n_pending::Int
    done::Bool
    best_min_id::Union{Int,Nothing}
    lock::ReentrantLock
end

"""
    Hyperoptimizer(objective, candidates::NamedTuple; sampler=RandomSampler(), n=nothing)

Construct an ask-tell hyperparameter optimizer. `candidates` gives one
[`Domain`](@ref) per parameter name (e.g. `Continuous(1,5,0.1)`,
`Levels([true,false])`, `Categorical([tanh,exp])`); `objective` is called as
`objective(params...)` (no macro, no iteration-index argument). `n`, if
given, bounds the total number of trials.

Hyperopt only ever minimizes `objective` — to maximize, minimize `-objective(...)`.
"""
function Hyperoptimizer(objective, candidates::NamedTuple; sampler::Sampler=RandomSampler(), n::Union{Int,Nothing}=nothing)
    cands = values(candidates)
    all(d -> d isa Domain, cands) ||
        throw(ArgumentError("every candidate must be a Domain (Continuous/Levels/Categorical), got types: $(typeof.(cands))"))
    params = collect(Symbol, keys(candidates))
    ho = Hyperoptimizer(params, cands, sampler, objective, n,
                         Trial[], Union{Result,Nothing}[], Int[], 0, false,
                         nothing, ReentrantLock())
    init!(sampler, AskContext(cands, 0, n))
    return ho
end

reached_target(ho::Hyperoptimizer) = _reached_target(ho.n, ho.trials)
_reached_target(::Nothing, trials) = false
_reached_target(n::Int, trials) = length(trials) >= n

"""
    settarget!(ho, n)

Raise the planned total number of trials to `n`. This is how you resume a
run: call `settarget!(ho, n)` with a larger `n` and call `run!(ho)` again --
there is no other special "resume" feature, since all state already lives in
`ho`.

Only raising the target is supported: samplers may size internal state off
`ho.n` (e.g. a Latin hypercube design matrix built for exactly `n` rows), so
lowering it after trials have already been planned against the old target is
not allowed and throws `ArgumentError`.
"""
function settarget!(ho::Hyperoptimizer, n::Int)
    _check_target_increase(ho.n, n)
    ho.n = n
    @info "Hyperoptimizer target set to $(ho.n) trials"
    return ho
end

_check_target_increase(::Nothing, n::Int) = nothing
function _check_target_increase(old::Int, n::Int)
    n < old && throw(ArgumentError("settarget!: new target ($n) is less than the current target ($old) -- settarget! can only raise the target"))
    return nothing
end

"""
    ask(ho) -> Trial

Draw the next candidate from `ho.sampler` and register a pending `Trial`.
Only ever called from the single driver task in `run!` — workers never call
this directly, which is what keeps sampler state single-writer by construction.
"""
function ask(ho::Hyperoptimizer)
    lock(ho.lock) do
        exhausted(ho.sampler, ho) && error("Hyperoptimizer's sampler is exhausted: no more candidates available")
        ctx = AskContext(ho.candidates, length(ho.trials), ho.n)
        raw = ho.sampler(ctx)
        id = length(ho.trials) + 1
        trial = Trial(id, Tuple(raw))
        push!(ho.trials, trial)
        push!(ho.results, nothing)
        ho.n_pending += 1
        return trial
    end
end

_rank_min(v::Real) = isnan(v) ? Inf : v
_rank_min(v) = v

function update_best!(ho::Hyperoptimizer, trial::Trial, result::Result)
    if _is_new_best(ho.best_min_id, ho.results, result.value)
        ho.best_min_id = trial.id
    end
    return ho
end

_is_new_best(::Nothing, results, value) = true
_is_new_best(best_min_id::Int, results, value) = _rank_min(value) < _rank_min(results[best_min_id].value)

"""
    tell!(ho, trial, outcome)

The sole mutation point for `ho.results`/`ho.completed`/the cached optimum.
`outcome` is either the objective's return value or a caught exception (or, from
a fault-tolerant executor, an explicit failure/abandonment marker) — `tell!`
classifies it into a `Result` and never writes a `NaN` placeholder for a
non-`Completed` trial.
"""
function tell!(ho::Hyperoptimizer, trial::Trial, outcome)
    lock(ho.lock) do
        result = to_result(outcome) # dispatches on outcome's type -- passes a Result through unchanged, wraps anything else
        ho.results[trial.id] = result
        ho.n_pending -= 1
        if result.status === Completed
            push!(ho.completed, trial.id)
            update_best!(ho, trial, result)
        end
        on_tell!(ho.sampler, trial, result)
        return ho
    end
end

"""
    run!(ho; executor=Serial())

Drive `ho` to completion (or until `ho.n` trials have completed, or the
sampler is exhausted, or the run is interrupted), dispatching evaluations
through `executor`. Swapping `executor` requires no change to `ho.sampler`
or `ho.objective`.

Resuming a previous run is not a special feature: call `settarget!(ho, n)` (or
leave the target `nothing`) and call `run!` again on the same object.
"""
function run!(ho::Hyperoptimizer; executor::AbstractExecutor=Serial())
    start!(executor, ho)
    try
        while true
            while !ho.done && !reached_target(ho) && !exhausted(ho.sampler, ho) && capacity(executor) > 0
                trial = ask(ho)
                submit!(executor, trial, ho.objective)
            end
            for (trial, outcome) in poll(executor)
                tell!(ho, trial, outcome)
            end
            (ho.done || reached_target(ho) || exhausted(ho.sampler, ho)) && ho.n_pending == 0 && break
        end
    catch e
        _handle_run_error(e, ho)
    finally
        shutdown!(executor)
    end
    return ho
end

_handle_run_error(e, ho::Hyperoptimizer) = rethrow(e)
function _handle_run_error(::InterruptException, ho::Hyperoptimizer)
    ho.done = true
    @info "Aborting hyperoptimization, returning partial results"
end
