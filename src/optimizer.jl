"""
    OptimizerStatus

A [`Hyperoptimizer`](@ref)'s lifecycle: `Initialized` (never run), `Running`
(inside a `run!` call), `Finished` (a `run!` call ended naturally -- target
reached or sampler exhausted -- and can still be resumed via `settarget!`
then `run!` again), `Interrupted` (a `run!` call was stopped by a genuine
interrupt), or `Errored` (a `run!` call was stopped by any other exception
escaping its own orchestration -- e.g. a bug in a custom `Sampler` -- as
opposed to an exception from the objective itself, which `safe_call` already
catches and records as an ordinary `Failed` trial without ever reaching this
far). Both `Interrupted` and `Errored` are permanent: any trials still in
flight at that moment are abandoned rather than recoverable, and `run!`
refuses to run again on either. They're kept as separate statuses rather
than one, since conflating "the user asked to stop" with "something in the
optimizer's own code is broken" would make the latter harder to notice and
debug -- checking `ho.status` after a crash should say which one happened.
"""
@enum OptimizerStatus Initialized Running Interrupted Errored Finished

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

# Interrupted and Errored are both permanent -- any trial still in flight at
# that moment was already forcibly abandoned (see _abandon_pending!), so
# nothing (run!, settarget!, or the manual ask!/tell! API) can meaningfully
# continue from here. Shared so all four call sites describe it identically.
_is_terminal(ho::Hyperoptimizer) = ho.status in (Interrupted, Errored)
_terminal_description(ho::Hyperoptimizer) = ho.status == Interrupted ? "interrupted" : "stopped by an error"

"""
    settarget!(ho, n)

Raise the planned total number of trials to `n` -- how you resume a run:
`settarget!(ho, n)` then `run!(ho)` again. Only raising is allowed; lowering
throws `ArgumentError`. Also throws if `ho.sampler` fixes its plan to the
sample count given at construction (e.g. a Latin Hypercube sampler) -- such
a sampler can't respond to a new target at all -- or if `ho.status` is
`Interrupted` or `Errored`, since either one already abandoned any trials
that were still in flight and `run!` will refuse to resume regardless. Warns
if `ho` still has trials pending, since changing the target while trials are
in flight isn't synchronized against them.
"""
function settarget!(ho::Hyperoptimizer, n::Int)
    _is_terminal(ho) &&
        throw(ArgumentError("settarget!: this Hyperoptimizer was $(_terminal_description(ho)) and cannot be resumed -- construct a new Hyperoptimizer to continue"))
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
[`RunEntry`](@ref). Throws if `ho`'s sampler is exhausted, if `ho.n` has
already been reached (call `settarget!` to raise it first), or if `ho.status`
is `Interrupted` or `Errored` -- both are permanent, so this manual API can't
be used to sneak new trials onto an optimizer `run!`/`settarget!` already
refuse to touch.
"""
function ask!(ho::Hyperoptimizer)
    lock(ho.lock) do
        _is_terminal(ho) &&
            error("ask!: this Hyperoptimizer was $(_terminal_description(ho)) and cannot produce new trials -- construct a new Hyperoptimizer to continue")
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
optimum. The sole mutation point for `ho.runs`/`ho.completed` outside of
`_abandon_pending!`'s direct reclassification on interrupt/error. Throws if
`ho.status` is `Interrupted` or `Errored` -- both already forcibly abandoned
every still-Pending entry, so recording a fresh outcome here (e.g. for one
of those very entries) would silently resurrect it and could double-count
`ho.n_pending`.
"""
function tell!(ho::Hyperoptimizer, entry::RunEntry, outcome)
    lock(ho.lock) do
        _is_terminal(ho) &&
            error("tell!: this Hyperoptimizer was $(_terminal_description(ho)) and cannot record new outcomes -- construct a new Hyperoptimizer to continue")
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
    run!(ho; executor=Serial())

Drive `ho` to completion (until `ho.n` trials have completed, the sampler is
exhausted, the run is interrupted, or some other exception escapes `run!`'s
own orchestration -- e.g. a bug in a custom `Sampler`; an exception from the
objective itself never reaches this far, since `safe_call` already catches
and records it as an ordinary `Failed` trial), dispatching evaluations
through `executor`. To resume a finished run, call `settarget!(ho, n)` then
`run!` again. Throws if `ho.status` is already `Interrupted` or `Errored`,
since either one already abandoned any trials that were still in flight --
construct a new `Hyperoptimizer` instead. Warns and returns without doing
anything if `ho` already has nothing to do: its target already reached
(call `settarget!` first), or its sampler exhausted (with a fixed plan, no
further call could ever produce more trials for it; see
[`FixedPlanSampler`](@ref)).
"""
_should_stop_asking(ho::Hyperoptimizer) = reached_target(ho) || exhausted(ho.sampler, ho)

function run!(ho::Hyperoptimizer; executor::AbstractExecutor=Serial())
    _is_terminal(ho) &&
        throw(ArgumentError("run!: this Hyperoptimizer was $(_terminal_description(ho)) and cannot be resumed -- construct a new Hyperoptimizer to continue"))
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
    try
        while true
            while !_should_stop_asking(ho) && capacity(executor) > 0
                entry = ask!(ho)
                submit!(executor, entry, ho.objective)
            end
            for (entry, outcome) in poll(executor)
                tell!(ho, entry, outcome)
            end
            _should_stop_asking(ho) && ho.n_pending == 0 && break
        end
        ho.status = Finished
    catch e
        _handle_run_error(e, ho)
    finally
        shutdown!(executor)
    end
    return ho
end

# Trials still in flight at the moment ho stops abnormally (interrupt or
# error) are abandoned, not recoverable (their eventual outcome, if any, is
# never told) -- give them an honest terminal status instead of leaving them
# looking indistinguishable from "still in flight" forever. Shared by both
# _handle_run_error methods below.
function _abandon_pending!(ho::Hyperoptimizer)
    lock(ho.lock) do
        for (i, entry) in enumerate(ho.runs)
            if entry.status === Pending
                ho.runs[i] = _with_result(entry, Abandoned, missing, entry.post_artefact)
                ho.n_pending -= 1
            end
        end
    end
    return nothing
end

# Any exception other than InterruptException reaching this far came from
# run!'s own orchestration (ask!/poll!/tell!), not the objective -- safe_call
# already catches and records an objective's own exception as an ordinary
# Failed trial, so this is a bug in the optimizer's own code (most likely a
# custom Sampler). Unlike InterruptException, this is never meant to be
# silently absorbed: ho is still stopped and its in-flight trials abandoned
# (so a caller who blindly retries doesn't busy-loop on stale state), but the
# exception itself still propagates, since swallowing a genuine bug would
# make it far harder to notice or debug.
function _handle_run_error(e, ho::Hyperoptimizer)
    _abandon_pending!(ho)
    ho.status = Errored
    rethrow(e)
end
function _handle_run_error(::InterruptException, ho::Hyperoptimizer)
    _abandon_pending!(ho)
    ho.status = Interrupted
    @info "Aborting hyperoptimization, returning partial results"
end
