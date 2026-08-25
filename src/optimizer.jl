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

Construct an ask-tell hyperparameter optimizer. `candidates` gives one entry per
parameter name; `objective` is called as `objective(params...)` (no macro, no
iteration-index argument). `n`, if given, bounds the total number of trials.

Hyperopt only ever minimizes `objective` — to maximize, minimize `-objective(...)`.
"""
function Hyperoptimizer(objective, candidates::NamedTuple; sampler::Sampler=RandomSampler(), n::Union{Int,Nothing}=nothing)
    params = collect(Symbol, keys(candidates))
    cands = values(candidates)
    ho = Hyperoptimizer(params, cands, sampler, objective, n,
                         Trial[], Union{Result,Nothing}[], Int[], 0, false,
                         nothing, ReentrantLock())
    init!(sampler, AskContext(cands, 0, n))
    return ho
end

reached_target(ho::Hyperoptimizer) = ho.n !== nothing && length(ho.trials) >= ho.n

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
    if ho.n !== nothing && n < ho.n
        throw(ArgumentError("settarget!: new target ($n) is less than the current target ($(ho.n)) -- settarget! can only raise the target"))
    end
    ho.n = n
    @info "Hyperoptimizer target set to $(ho.n) trials"
    return ho
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

_rank_min(v) = (v isa Real && isnan(v)) ? Inf : v

function update_best!(ho::Hyperoptimizer, trial::Trial, result::Result)
    if ho.best_min_id === nothing || _rank_min(result.value) < _rank_min(ho.results[ho.best_min_id].value)
        ho.best_min_id = trial.id
    end
    return ho
end

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
        result = outcome isa Result ? outcome : to_result(outcome)
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
        if e isa InterruptException
            ho.done = true
            @info "Aborting hyperoptimization, returning partial results"
        else
            rethrow()
        end
    finally
        shutdown!(executor)
    end
    return ho
end
