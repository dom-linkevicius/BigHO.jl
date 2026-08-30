"""
    OptimizerStatus

A [`Hyperoptimizer`](@ref)'s lifecycle: `Initialized` (never run), `Running`
(inside a `run!` call), `Finished` (a `run!` call ended naturally -- target
reached or sampler exhausted -- and can still be resumed via `settarget!`
then `run!` again), or `Interrupted` (a `run!` call was stopped by a genuine
interrupt -- permanent, since any trials still in flight at that moment are
abandoned rather than recoverable; `run!` refuses to run again on an
`Interrupted` optimizer).
"""
@enum OptimizerStatus Initialized Running Interrupted Finished

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

Construct a hyperparameter optimizer. `candidates` gives one [`Domain`](@ref)
per parameter name (e.g. `Continuous(1,5,0.1)`, `Nominal([true,false])`,
`Ordinal([1,2,5,10])`); `objective` is called as `objective(params...)`
unless wrapped in [`Stateful`](@ref). `n`, if given, bounds the total number
of trials. Only ever minimizes -- to maximize, minimize `-objective(...)`.
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

Raise the planned total number of trials to `n` -- how you resume a run:
`settarget!(ho, n)` then `run!(ho)` again. Only raising is allowed; lowering
throws `ArgumentError`. Also throws if `ho.sampler` fixes its plan to the
sample count given at construction (e.g. a Latin Hypercube sampler) -- such
a sampler can't respond to a new target at all -- or if `ho.status ==
Interrupted`, since a prior interrupt already abandoned any trials that were
still in flight and `run!` will refuse to resume regardless. Warns if `ho`
still has trials pending, since changing the target while trials are in
flight isn't synchronized against them.
"""
function settarget!(ho::Hyperoptimizer, n::Int)
    ho.status == Interrupted &&
        throw(ArgumentError("settarget!: this Hyperoptimizer was interrupted and cannot be resumed -- construct a new Hyperoptimizer to continue"))
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

Draw the next candidate from `ho.sampler` and register a `Pending`
[`RunEntry`](@ref). Throws if `ho`'s sampler is exhausted, or if `ho.n` has
already been reached -- call `settarget!` to raise it first.
"""
function ask!(ho::Hyperoptimizer)
    lock(ho.lock) do
        exhausted(ho.sampler, ho) && error("Hyperoptimizer's sampler is exhausted: no more candidates available")
        reached_target(ho) && error("Hyperoptimizer has already reached its target of $(ho.n) trials; call settarget! to raise it before asking for more")
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

# No NaN handling needed here: finalize_entry already excludes NaN outcomes
# as Failed, so a Completed entry's value is never NaN by the time it's ranked.
function update_best!(ho::Hyperoptimizer, entry::RunEntry)
    if ho.best_min_id === nothing || entry.value < ho.runs[ho.best_min_id].value
        ho.best_min_id = entry.id
    end
    return ho
end

"""
    tell!(ho, entry, outcome)

Record `outcome` (the objective's return value, or a caught exception) for
`entry`, classifying it via [`finalize_entry`](@ref) and updating the cached
optimum. The sole mutation point for `ho.runs`/`ho.completed`.
"""
function tell!(ho::Hyperoptimizer, entry::RunEntry, outcome)
    lock(ho.lock) do
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

Drive `ho` to completion (until `ho.n` trials have completed, the sampler is
exhausted, or the run is interrupted), dispatching evaluations through
`executor`. To resume a finished run, call `settarget!(ho, n)` then `run!`
again. Throws if `ho.status == Interrupted`, since a prior interrupt already
abandoned any trials that were still in flight -- construct a new
`Hyperoptimizer` instead. Warns and returns without doing anything if `ho`
already has nothing to do: its target already reached (call `settarget!`
first), or its sampler exhausted (with a fixed plan, no further call could
ever produce more trials for it; see [`FixedPlanSampler`](@ref)).

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
not happen on interrupt (see [`OptimizerStatus`](@ref)) -- only the
periodic `save_every` checkpoints, if any, capture an interrupted run.

`show_progress` (on by default) displays a `ProgressMeter` bar tracking
trials told (any outcome) against `ho.n`, which must therefore be set --
there's currently no sampler that determines its own total independently
of `ho.n`; pass `show_progress=false` to run without a target instead. The
bar always finishes (even on interrupt or error) so a partially-drawn one
is never left behind in the terminal.
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
    ho.status == Interrupted &&
        throw(ArgumentError("run!: this Hyperoptimizer was interrupted and cannot be resumed -- construct a new Hyperoptimizer to continue"))
    if reached_target(ho)
        @warn "run!: Hyperoptimizer has already reached its target of $(ho.n) trials; call settarget! to raise it before calling run! again"
        return ho
    end
    if exhausted(ho.sampler, ho)
        ho.sampler isa FixedPlanSampler &&
            @warn "$(typeof(ho.sampler)) has a fixed plan and is already exhausted; run! won't produce any more trials -- change params or use a different sampler"
        return ho
    end
    ho.status = Running
    start!(executor, ho)
    last_saved = n_told(ho)
    progress = nothing
    if show_progress
        progress = ProgressMeter.Progress(ho.n)
        # Progress's own `start` kwarg doesn't actually seed its internal
        # counter (confirmed empirically: it stays 0 regardless), so a
        # resumed run whose first update! jumps straight from 0 to a large
        # n_told(ho) would otherwise show a wrong/stale position until the
        # next natural redraw. force=true here -- only for this one initial
        # sync, not the in-loop updates below -- makes the correct starting
        # position visible immediately instead of waiting on dt throttling.
        # It also has a second, necessary effect: ProgressMeter's own
        # completion message (from finish! below) only ever prints if some
        # earlier update actually printed (checked directly in its source --
        # not itself controllable via force), so without this, a run whose
        # every trial resolves faster than dt apart would finish having
        # displayed nothing at all, start to end.
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

_handle_run_error(e, ho::Hyperoptimizer) = rethrow(e)
function _handle_run_error(::InterruptException, ho::Hyperoptimizer)
    # Trials still in flight at the moment of interrupt are abandoned, not
    # recoverable (their eventual outcome, if any, is never told) -- give
    # them an honest terminal status instead of leaving them looking
    # indistinguishable from "still in flight" forever.
    lock(ho.lock) do
        for (i, entry) in enumerate(ho.runs)
            if entry.status === Pending
                ho.runs[i] = _with_result(entry, Abandoned, missing, entry.post_artefact)
                ho.n_pending -= 1
            end
        end
    end
    ho.status = Interrupted
    @info "Aborting hyperoptimization, returning partial results"
end
