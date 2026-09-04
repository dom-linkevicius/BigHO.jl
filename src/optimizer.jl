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
    n !== nothing || !(sampler isa FixedPlanSampler) ||
        throw(ArgumentError("$(typeof(sampler)) needs n -- pass n explicitly, or construct via Hyperoptimizer(objective, candidates, sampler; n=...)"))
    cands = values(candidates)
    all(d -> d isa Domain, cands) ||
        throw(ArgumentError("every candidate must be a Domain (Continuous/Nominal/Ordinal), got types: $(typeof.(cands))"))
    params = collect(Symbol, keys(candidates))
    initialized_sampler = init(sampler, cands, n)
    return Hyperoptimizer(params, cands, initialized_sampler, objective, n,
                           RunEntry[], Int[], 0, Initialized,
                           nothing, ReentrantLock())
end

"""
    Hyperoptimizer(objective, candidates::NamedTuple, sampler::LHSampler; n::Int)

Construct with `n` (the trial budget) given directly. Every `Continuous(min,max,dt)` domain's grid is rebuilt to exactly `n` linearly-spaced values over its original range, overriding whatever resolution it was originally given; `Continuous(values)` (arbitrary spacing) is rejected -- apply any nonlinear transform (e.g. log-scale) inside the objective instead.
`Nominal`/`Ordinal` domains are left untouched.
"""
function Hyperoptimizer(objective, candidates::NamedTuple, sampler::LHSampler; n::Int)
    cands = values(candidates)
    all(d -> d isa Domain, cands) ||
        throw(ArgumentError("every candidate must be a Domain (Continuous/Nominal/Ordinal), got types: $(typeof.(cands))"))
    return Hyperoptimizer(objective, _linearize_for_lhs(candidates, n); sampler=sampler, n=n)
end

reached_target(ho::Hyperoptimizer) = ho.n !== nothing && length(ho.runs) >= ho.n

# Trials ever told an outcome, regardless of how many run! calls it took -- used for save_every's cadence.
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
        raw = ho.sampler(ho.candidates, ho.runs)
        id = length(ho.runs) + 1
        params = NamedTuple{Tuple(ho.params)}(Tuple(raw)) # e.g. (a = 1.5, b = true) -- labeled everywhere, not just in warnings
        entry = create_run_entry(ho.sampler, ho, id, params)
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
    run!(ho; executor=Serial(), save_every=nothing, save_path=nothing, show_progress=true)

Drive `ho` to completion, dispatching evaluations through `executor`.
Any exception escaping `run!`'s own orchestration (not the objective) is rethrown and sets `ho.status = Errored` (see [`OptimizerStatus`](@ref)).
To resume, call `settarget!(ho, n)` then `run!` again; throws if already `Errored`.

`save_path`, if given, checkpoints `ho` there every `save_every` trials told, overwriting the same file (minus the objective -- see [`load_hyperoptimizer`](@ref)); a final checkpoint always runs when the run ends normally too, even if `save_every` was given.
`save_every` requires `save_path`; `save_path` requires a real `ho.objective`. Writes are atomic (temp file renamed over `save_path`).
Not saved if `run!` errors -- only the periodic `save_every` checkpoints, if any, capture an unfinished run.

`show_progress` (on by default) shows a `ProgressMeter` bar tracking trials told against `ho.n`, which must be set for it (pass `show_progress=false` to run without a target).
The bar always finishes, even on error, so a partially-drawn one is never left in the terminal.
"""
_should_stop_asking(ho::Hyperoptimizer) = reached_target(ho) || exhausted(ho.sampler, ho)

function run!(ho::Hyperoptimizer; executor::AbstractExecutor=Serial(),
              save_every::Union{Int,Nothing}=nothing, save_path::Union{AbstractString,Nothing}=nothing,
              show_progress::Bool=true)
    save_every !== nothing && save_path === nothing &&
        throw(ArgumentError("run!: save_every requires save_path"))
    save_every !== nothing && save_every < 1 &&
        throw(ArgumentError("run!: save_every must be >= 1, got $save_every"))
    save_path !== nothing && ho.objective === nothing &&
        throw(ArgumentError("run!: save_path requires a real ho.objective -- checkpointing substitutes `nothing` for it in the saved file (to be replaced with a fresh objective via load_hyperoptimizer), which would be ambiguous if it was already `nothing`"))
    show_progress && ho.n === nothing &&
        throw(ArgumentError("run!: show_progress requires ho.n to be set"))
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
    progress = nothing
    if show_progress
        progress = ProgressMeter.Progress(ho.n)
        # force=true syncs the bar to a resumed run's real starting position (Progress's `start` kwarg doesn't actually seed it)...
        # ...and ensures finish! below actually prints, which ProgressMeter otherwise skips if nothing was ever printed.
        ProgressMeter.update!(progress, n_told(ho); force=true)
    end
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
            progress !== nothing && ProgressMeter.update!(progress, n_told(ho))
            _should_stop_asking(ho) && ho.n_pending == 0 && break
        end
        ho.status = Finished
        save_path !== nothing && _save_checkpoint(ho, save_path)
    catch e
        _handle_run_error(e, ho)
    finally
        shutdown!(executor)
        progress !== nothing && ProgressMeter.finish!(progress)
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
