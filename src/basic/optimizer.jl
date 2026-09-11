"""
    OptimizerStatus
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
"""
function Hyperoptimizer(objective, candidates::NamedTuple, sampler::LHSampler; n::Int)
    cands = values(candidates)
    all(d -> d isa Domain, cands) ||
        throw(ArgumentError("every candidate must be a Domain (Continuous/Nominal/Ordinal), got types: $(typeof.(cands))"))
    return Hyperoptimizer(objective, candidates; sampler=sampler, n=n)
end

"""
    Hyperoptimizer(objective, candidates::NamedTuple, sampler::SuccessiveHalving; kwargs...)
"""
function Hyperoptimizer(objective, candidates::NamedTuple, sampler::SuccessiveHalving; kwargs...)
    haskey(candidates, :r) &&
        throw(ArgumentError("Hyperoptimizer: `:r` is reserved for $(typeof(sampler))'s resource level -- rename your `:r` candidate"))
    # `nothing` is exempt: there's no objective to wrap, so the warning would have nothing to act on.
    objective isa Stateful || objective === nothing || @warn "$(typeof(sampler)) with a non-Stateful objective: promoted trials can't " *
                                                             "resume from a previous trial's state, so each promotion re-pays all the resource already spent on it -- " *
                                                             "wrap the objective in `Stateful` to make promotions continue instead of restart"
    haskey(kwargs, :n) &&
        throw(ArgumentError("Hyperoptimizer: $(typeof(sampler))'s trial count is fully determined by R/η/r_min -- don't pass n explicitly"))
    n = _total_trials(sampler.R, sampler.r_min, sampler.η)
    return Hyperoptimizer(objective, candidates; sampler=sampler, n=n, kwargs...)
end

reached_target(ho::Hyperoptimizer) = ho.n !== nothing && length(ho.runs) >= ho.n

# Trials ever told an outcome, regardless of how many run! calls it took -- used for save_every's cadence.
n_told(ho::Hyperoptimizer) = length(ho.runs) - ho.n_pending

"""
    settarget!(ho, n)
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
"""
function ask!(ho::Hyperoptimizer)
    lock(ho.lock) do
        ho.status == Errored &&
            throw(ArgumentError("ask!: this Hyperoptimizer already errored and cannot produce new trials -- construct a new Hyperoptimizer to continue"))
        exhausted(ho.sampler, ho) && throw(ArgumentError("Hyperoptimizer's sampler is exhausted: no more candidates available"))
        reached_target(ho) && throw(ArgumentError("Hyperoptimizer has already reached its target of $(ho.n) trials; call settarget! to raise it before asking for more"))
        unit_params = ho.sampler(ho.candidates, ho.runs)
        id = length(ho.runs) + 1
        decoded = Tuple(from_unit(d, u) for (d, u) in zip(ho.candidates, unit_params))
        params = NamedTuple{Tuple(ho.params)}(decoded) # e.g. (a = 1.5, b = true) -- labeled everywhere, not just in warnings
        entry = create_run_entry(ho.sampler, ho, id, params, unit_params)
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
        on_tell!(ho.sampler, ho.runs, told)
        return ho
    end
end

"""
    run!(ho; executor=Serial(), save_every=nothing, save_path=nothing, show_progress=true)
"""
_should_stop_asking(ho::Hyperoptimizer) = reached_target(ho) || exhausted(ho.sampler, ho) || blocked(ho.sampler, ho)

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
                save_hyperoptimizer(ho, save_path)
                last_saved = n_told(ho)
            end
            progress !== nothing && ProgressMeter.update!(progress, n_told(ho))
            _should_stop_asking(ho) && ho.n_pending == 0 && break
        end
        ho.status = Finished
        save_path !== nothing && save_hyperoptimizer(ho, save_path)
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
