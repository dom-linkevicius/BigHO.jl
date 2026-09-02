"""
    OptimizerStatus

A [`Hyperoptimizer`](@ref)'s lifecycle: `Initialized`, `Running`, `Finished` (resumable via `settarget!`+`run!`), or `Errored` (permanent).
`Errored` covers any exception escaping `run!`'s orchestration -- in-flight trials abandoned, the exception rethrown to the caller.
"""
@enum OptimizerStatus Initialized Running Errored Finished

mutable struct Hyperoptimizer{S<:Sampler,F}
    params::Vector{Symbol}
    candidates::Tuple
    sampler::S
    objective::F
    n::Union{Int,Nothing}
    runs::Vector{RunEntry}
    completed::Vector{Int}
    n_pending::Int
    status::OptimizerStatus
    best_min_id::Union{Int,Nothing}
    lock::ReentrantLock
end

"""
    Hyperoptimizer(objective, candidates::NamedTuple; sampler=RandomSampler(), n=nothing)

Construct a hyperparameter optimizer. `candidates` gives one [`Domain`](@ref) per parameter name; `n` bounds the total trials.
`objective` is called as `objective(params...)`, unless wrapped in [`Stateful`](@ref); only ever minimizes.
"""
function Hyperoptimizer(objective, candidates::NamedTuple; sampler::Sampler=RandomSampler(), n::Union{Int,Nothing}=nothing)
    n === nothing || n >= 0 || throw(ArgumentError("n must be non-negative, got $n"))
    cands = values(candidates)
    all(d -> d isa Domain, cands) ||
        throw(ArgumentError("every candidate must be a Domain (Continuous/Nominal/Ordinal), got types: $(typeof.(cands))"))
    params = collect(Symbol, keys(candidates))
    ho = Hyperoptimizer(params, cands, sampler, objective, n,
                         RunEntry[], Int[], 0, Initialized,
                         nothing, ReentrantLock())
    init!(sampler, AskContext(cands, 0, n))
    return ho
end

reached_target(ho::Hyperoptimizer) = ho.n !== nothing && length(ho.runs) >= ho.n

# Trials ever told an outcome (Completed/Failed/Abandoned), regardless of
# how many separate run! calls it took to get there -- used to decide when
# save_every worth of new trials have accumulated since the last checkpoint.
n_told(ho::Hyperoptimizer) = length(ho.runs) - ho.n_pending

"""
    settarget!(ho, n)

Raise the planned total number of trials to `n` -- how you resume a run.
Throws if lowering, if `ho.status` is `Errored`, or the sampler has a fixed plan; warns if trials are still pending.
"""
function settarget!(ho::Hyperoptimizer, n::Int)
    ho.status == Errored &&
        throw(ArgumentError("settarget!: this Hyperoptimizer already errored and cannot be resumed -- construct a new Hyperoptimizer to continue"))
    ho.n_pending > 0 &&
        @warn "settarget!: $(ho.n_pending) trial(s) still pending -- changing the target while trials are in flight may race with them"
    ho.n !== nothing && n < ho.n &&
        throw(ArgumentError("settarget!: new target ($n) is less than the current target ($(ho.n)) -- settarget! can only raise the target"))
    ho.sampler isa FixedPlanSampler &&
        throw(ArgumentError("settarget!: $(typeof(ho.sampler)) fixes its plan to the sample count given at construction and can't respond to a new target"))
    ho.n = n
    @info "Hyperoptimizer target set to $(ho.n) trials"
    return ho
end

"""
    ask!(ho) -> RunEntry

Draw the next candidate from `ho.sampler` and register a `Pending` [`RunEntry`](@ref).
Throws if the sampler is exhausted, `ho.n` is already reached, or `ho.status` is `Errored`.
"""
function ask!(ho::Hyperoptimizer)
    lock(ho.lock) do
        ho.status == Errored &&
            throw(ArgumentError("ask!: this Hyperoptimizer already errored and cannot produce new trials -- construct a new Hyperoptimizer to continue"))
        exhausted(ho.sampler, ho) && throw(ArgumentError("Hyperoptimizer's sampler is exhausted: no more candidates available"))
        reached_target(ho) && throw(ArgumentError("Hyperoptimizer has already reached its target of $(ho.n) trials; call settarget! to raise it before asking for more"))
        ctx = AskContext(ho.candidates, length(ho.runs), ho.n)
        raw = ho.sampler(ctx)
        id = length(ho.runs) + 1
        params = NamedTuple{Tuple(ho.params)}(Tuple(raw)) # e.g. (a = 1.5, b = true) -- labeled everywhere, not just in warnings
        entry = RunEntry(id, params)
        push!(ho.runs, entry)
        ho.n_pending += 1
        return entry
    end
end

# finalize_entry already excludes NaN outcomes as Failed, so a Completed value is never NaN here.
function update_best!(ho::Hyperoptimizer, entry::RunEntry)
    if ho.best_min_id === nothing || entry.value < ho.runs[ho.best_min_id].value
        ho.best_min_id = entry.id
    end
    return ho
end

"""
    tell!(ho, entry, outcome)

Record `outcome` for `entry` via [`finalize_entry`](@ref), updating the cached optimum.
Throws if `ho.status` is `Errored` (every `Pending` entry was already abandoned).
"""
function tell!(ho::Hyperoptimizer, entry::RunEntry, outcome)
    lock(ho.lock) do
        ho.status == Errored &&
            throw(ArgumentError("tell!: this Hyperoptimizer already errored and cannot record new outcomes -- construct a new Hyperoptimizer to continue"))
        told = finalize_entry(ho.runs[entry.id], outcome)
        ho.runs[told.id] = told
        ho.n_pending -= 1
        if told.status === Completed
            push!(ho.completed, told.id)
            update_best!(ho, told)
        end
        on_tell!(ho.sampler, told)
        return ho
    end
end

"""
    run!(ho; executor=Serial(), save_every=nothing, save_path=nothing)

Drive `ho` to completion, dispatching evaluations through `executor`.
Any exception escaping `run!`'s own orchestration (not the objective) is rethrown and sets `ho.status = Errored` (see [`OptimizerStatus`](@ref)).
To resume, call `settarget!(ho, n)` then `run!` again; throws if already `Errored`.

`save_path`, if given, checkpoints `ho` there (minus its objective -- see
[`load_hyperoptimizer`](@ref)) every `save_every` trials told (any outcome,
not just `Completed`), overwriting the same file each time. Omit
`save_every` to only save once, when the run ends naturally (target
reached or sampler exhausted) -- and even when `save_every` is given, one
last checkpoint is always taken at that point too, so the file on disk
always reflects the true end state rather than being stuck at the last
`save_every` boundary before it. `save_every` without `save_path` is an
error, and so is `save_path` on a `ho` whose `objective` is already
`nothing` (checkpointing substitutes `nothing` for the objective in the
saved file, so a real one is required to make that substitution meaningful).
Each write is atomic (a temp file renamed over `save_path`), so a crash
mid-write can't corrupt the last good checkpoint. Note this final save does
not happen if `run!` errors (`ho.status = Errored`) -- only the periodic
`save_every` checkpoints, if any, capture a run that didn't finish normally.
"""
_should_stop_asking(ho::Hyperoptimizer) = reached_target(ho) || exhausted(ho.sampler, ho)

function run!(ho::Hyperoptimizer; executor::AbstractExecutor=Serial(),
              save_every::Union{Int,Nothing}=nothing, save_path::Union{AbstractString,Nothing}=nothing)
    save_every !== nothing && save_path === nothing &&
        throw(ArgumentError("run!: save_every requires save_path"))
    save_every !== nothing && save_every < 1 &&
        throw(ArgumentError("run!: save_every must be >= 1, got $save_every"))
    save_path !== nothing && ho.objective === nothing &&
        throw(ArgumentError("run!: save_path requires a real ho.objective -- checkpointing substitutes `nothing` for it in the saved file (to be replaced with a fresh objective via load_hyperoptimizer), which would be ambiguous if it was already `nothing`"))
    ho.status == Errored &&
        throw(ArgumentError("run!: this Hyperoptimizer already errored and cannot be resumed -- construct a new Hyperoptimizer to continue"))
    if reached_target(ho)
        @warn "run!: Hyperoptimizer has already reached its target of $(ho.n) trials; call settarget! to raise it before calling run! again"
        return ho
    end
    if exhausted(ho.sampler, ho)
        if ho.sampler isa FixedPlanSampler
            @warn "$(typeof(ho.sampler)) has a fixed plan and is already exhausted; run! won't produce any more trials -- change params or use a different sampler"
        else
            @warn "run!: $(typeof(ho.sampler)) is exhausted; run! won't produce any more trials"
        end
        return ho
    end
    ho.status = Running
    start!(executor, ho)
    last_saved = n_told(ho)
    try
        while true
            while !_should_stop_asking(ho) && capacity(executor) > 0
                entry = ask!(ho)
                submit!(executor, entry, ho.objective)
            end
            for (entry, outcome) in poll(executor)
                tell!(ho, entry, outcome)
            end
            if save_every !== nothing && n_told(ho) - last_saved >= save_every
                _save_checkpoint(ho, save_path)
                last_saved = n_told(ho)
            end
            _should_stop_asking(ho) && ho.n_pending == 0 && break
        end
        ho.status = Finished
        save_path !== nothing && _save_checkpoint(ho, save_path)
    catch e
        _handle_run_error(e, ho)
    finally
        shutdown!(executor)
    end
    return ho
end

# Any exception is treated identically: Errored, in-flight trials abandoned, rethrown unchanged.
function _handle_run_error(e, ho::Hyperoptimizer)
    ho.status = Errored
    lock(ho.lock) do
        for (i, entry) in enumerate(ho.runs)
            if entry.status === Pending
                ho.runs[i] = _with_result(entry, Abandoned, missing, entry.post_artefact)
                ho.n_pending -= 1
            end
        end
    end
    rethrow(e)
end
