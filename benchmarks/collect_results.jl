using BigHO
using JLD2

include("nn_objective.jl")

# @info is buffered (often for minutes) when stdout/stderr are redirected to a file, e.g. the
# CI workflow's log -- println+flush prints immediately, so progress is visible while a long
# deploy-scale run is still in progress, not just after it exits.
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

# nn_objective.jl's DEPLOY flag also gates these -- one full trial now costs seconds (a real
# MLP on Titanic), not milliseconds, so trial/repeat counts must shrink for local dev
# iteration and grow deliberately for the real published (CI) run. See BIGHO_BENCHMARK_DEPLOY.
# BIGHO_BENCHMARK_REPEATS overrides the repeat count directly -- e.g. a single deploy-scale
# repeat to sanity-check params/results before committing to the full REGRET_REPEATS sweep.
# Random's trial budget -- matched to Hyperband/ASHA's own count of unique hyperparameter
# configurations actually tried (BigHO._total_draws counts only fresh rung-1 draws, not
# promotions/continuations of an already-drawn config to a higher resource level; _total_trials
# counts every dispatch including those, which is a different, larger number). For a clean
# "same number of configs tried, does total training budget matter" comparison.
const N_TRIALS = DEPLOY ? BigHO._total_draws(R_MAX, R_MIN, ETA) : 4
const REGRET_REPEATS = parse(Int, get(ENV, "BIGHO_BENCHMARK_REPEATS", DEPLOY ? "10" : "2"))

# Match Random's full-trial epoch count to Hyperband/ASHA's actual total training
# budget (from their real bracket schedule -- capacity at each rung times the INCREMENTAL
# resource over the previous rung, since warm-started promotions never re-pay earlier epochs),
# divided evenly across N_TRIALS trials. Without this, changing EPOCHS_PER_RESOURCE, R_MAX, or
# N_TRIALS independently would silently change how the two total budgets compare -- this keeps
# "same total compute, which sampler does better" meaningful regardless of those tunings.
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

const SHA_INNER = RandomSampler()   # Hyperband/ASHA's inner per-draw sampler -- named here (rather
# than relying on the constructors' own default) so its name can be recorded in metadata for the plot legend.

const MAKE_HYPEROPTIMIZER = Dict(
    "Random" => () -> Hyperoptimizer(PLAIN_OBJ, CANDIDATES; sampler=RandomSampler(), n=N_TRIALS),
    "Hyperband" => () -> Hyperoptimizer(SH_OBJ, CANDIDATES, Hyperband(R=R_MAX, η=ETA, r_min=R_MIN, inner=SHA_INNER)),
    "ASHA" => () -> Hyperoptimizer(SH_OBJ, CANDIDATES, ASHA(R=R_MAX, η=ETA, r_min=R_MIN, inner=SHA_INNER)),
)

# Absorb Julia's one-time JIT compilation cost (Flux, BigHO's run!/executors/Hyperband/ASHA
# dispatch) up front -- otherwise whichever executor/sampler happens to run first in the
# real, timed comparisons below would unfairly carry that one-time cost, not a genuine
# per-trial speed difference. Uses the dedicated 1-epoch warmup objectives, NOT PLAIN_OBJ/
# SH_OBJ -- nn_objective's epoch count is baked in (R_MAX * EPOCHS_PER_RESOURCE), so warming
# up with the real objective would mean every "just 2 trials" call pays the full deploy-scale
# training cost instead of a few seconds.
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

"one repeat's (elapsed_time_since_run_start, value) pairs, sorted by actual completion time"
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

# ---- wall-clock regret comparison (BOHB paper, Fig. 1 style) ----
# Every sampler runs under BOTH executors here: Serial and Threaded. Repeated REGRET_REPEATS
# times per (sampler, executor) pair to average out run-to-run noise. Only ever saves the
# per-repeat (times, best-so-far) curves -- never the trained models themselves (large,
# closure-laden Flux objects not worth persisting) -- to `results.jld2`, so `plot_results.jl`
# can be re-run (and edited) without repeating this expensive training.
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
