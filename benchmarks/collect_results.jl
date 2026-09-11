using BigHO
using JLD2

include("nn_objective.jl")

# @info is buffered when stdout is redirected to a file (the CI log), so a long run shows nothing
# until it exits -- println+flush prints immediately.
_log(msg) = (println(msg); flush(stdout))

const RESULTS_PATH = joinpath(@__DIR__, "results.jld2")

const CANDIDATES = (
    lr=Continuous(1e-3, 5e-2),
    n_dense_layers=Ordinal([1, 2, 3]),
    hidden=Ordinal([8, 16, 32, 64]),
    activation=Nominal([tanh, relu]),
    reg=Continuous(0.0, 1e-2),
)

const ETA = 3            # Hyperband/ASHA η (R comes from nn_objective.jl's R_MAX)

# _total_draws counts fresh rung-1 draws only (not promotions), so Random tries the same number of
# distinct configs as Hyperband/ASHA. BIGHO_BENCHMARK_REPEATS overrides the repeat count directly.
const N_TRIALS = DEPLOY ? BigHO._total_draws(R_MAX, R_MIN, ETA) : 4
const REGRET_REPEATS = parse(Int, get(ENV, "BIGHO_BENCHMARK_REPEATS", DEPLOY ? "10" : "2"))

# Give Random the same total epoch budget as the bracket schedule: capacity per rung times the
# INCREMENTAL resource, since warm-started promotions never re-pay earlier epochs.
if DEPLOY
    smax = BigHO._smax(R_MAX, R_MIN, ETA)
    total_resource_units = sum(
        BigHO._capacity(R_MAX, R_MIN, ETA, k, i) * (BigHO._resource(R_MAX, R_MIN, ETA, k, i) - (i > 1 ? BigHO._resource(R_MAX, R_MIN, ETA, k, i - 1) : 0))
        for k in 1:(smax+1) for i in 1:k
    )
    hyperband_total_epochs = total_resource_units * EPOCHS_PER_RESOURCE
    RANDOM_FULL_EPOCHS[] = round(Int, hyperband_total_epochs / N_TRIALS)
    random_total_epochs = RANDOM_FULL_EPOCHS[] * N_TRIALS
    _log("Matched Random's per-trial epochs to Hyperband/ASHA's total budget: random_full_epochs=$(RANDOM_FULL_EPOCHS[]) random_total_epochs=$random_total_epochs hyperband_total_epochs=$hyperband_total_epochs")
end

const PLAIN_OBJ = Stateful(nn_objective)
const SH_OBJ = Stateful(nn_objective_stateful)

const SAMPLER_NAMES = ("Random", "Hyperband", "ASHA")
const EXECUTORS = (Serial=Serial(), Threaded=Threaded())

# Named rather than left to the constructors' default so it can be recorded in metadata for the legend.
const SHA_INNER = RandomSampler()

const MAKE_HYPEROPTIMIZER = Dict(
    "Random" => () -> Hyperoptimizer(PLAIN_OBJ, CANDIDATES; sampler=RandomSampler(), n=N_TRIALS),
    "Hyperband" => () -> Hyperoptimizer(SH_OBJ, CANDIDATES, Hyperband(R=R_MAX, η=ETA, r_min=R_MIN, inner=SHA_INNER)),
    "ASHA" => () -> Hyperoptimizer(SH_OBJ, CANDIDATES, ASHA(R=R_MAX, η=ETA, r_min=R_MIN, inner=SHA_INNER)),
)

# Absorb JIT cost up front, or whichever executor runs first carries it. Uses the 1-epoch warmup
# objectives -- the real ones have their epoch count baked in and would cost full deploy scale.
_log("Warming up (JIT compilation)...")
const WARMUP_OBJ = Stateful(_warmup_objective)
const WARMUP_SH_OBJ = Stateful(_warmup_objective_stateful)
for executor in EXECUTORS
    run!(Hyperoptimizer(WARMUP_OBJ, CANDIDATES; sampler=RandomSampler(), n=2); executor=executor, show_progress=false)
end
for sh_sampler in (Hyperband(R=1, η=3, r_min=1), ASHA(R=1, η=3, r_min=1))
    run!(Hyperoptimizer(WARMUP_SH_OBJ, CANDIDATES, sh_sampler); executor=Serial(), show_progress=false)
end

_finished_at(entry) = entry.post_artefact[4]

"""
    _timed_results(ho, t0)
"""
function _timed_results(ho::Hyperoptimizer, t0::Float64)
    completed = filter(e -> !ismissing(e.value), ho.runs)
    pairs = [(_finished_at(e) - t0, e.value) for e in completed]
    return sort(pairs; by=first)
end

function _running_min_curve(pairs)
    times, values = first.(pairs), last.(pairs)
    best = accumulate(min, values)
    return times, best
end

# Wall-clock regret comparison (BOHB paper, Fig. 1 style): every sampler under both executors,
# REGRET_REPEATS times each. Saves only the (times, best-so-far) curves, never the trained models.
function collect_results()
    runs = Dict((name, ex) => Tuple{Vector{Float64},Vector{Float64}}[] for name in SAMPLER_NAMES, ex in keys(EXECUTORS))
    total_combos = REGRET_REPEATS * length(EXECUTORS) * length(SAMPLER_NAMES)
    combo = 0

    for repeat in 1:REGRET_REPEATS, ex_name in keys(EXECUTORS), name in SAMPLER_NAMES
        combo += 1
        _log("[$combo/$total_combos] repeat=$repeat sampler=$name executor=$ex_name -- starting")
        ho = MAKE_HYPEROPTIMIZER[name]()
        t0 = time()
        run!(ho; executor=EXECUTORS[ex_name], show_progress=false)
        pairs = _timed_results(ho, t0)
        elapsed = time() - t0
        best = isempty(pairs) ? NaN : minimum(last, pairs)
        _log("[$combo/$total_combos] repeat=$repeat sampler=$name executor=$ex_name -- done elapsed=$elapsed best=$best")
        push!(runs[(name, ex_name)], _running_min_curve(pairs))
    end

    metadata = (regret_repeats=REGRET_REPEATS, n_trials=N_TRIALS, r_max=R_MAX,
                deploy=DEPLOY, nthreads=Threads.nthreads(), sha_inner_sampler=nameof(typeof(SHA_INNER)))
    JLD2.save(RESULTS_PATH, "runs", runs, "metadata", metadata)
    return runs, metadata
end

_log("Collecting wall-clock regret comparison results...")
collect_results()
_log("Results written to $RESULTS_PATH")
