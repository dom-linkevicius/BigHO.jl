using Flux
using MLDatasets: Titanic
using DataFrames
using Statistics: median, mean, std
using StableRNGs: StableRNG
using Random: shuffle
using LinearAlgebra: BLAS

# Each trial's own matrix ops must stay single-threaded -- BigHO's Threaded() executor is
# the outer parallelism (many concurrent trials); if BLAS also spawns its own internal
# threads per trial, the two layers of parallelism oversubscribe the real core count and
# fight each other, making Threaded() *slower* than Serial() instead of faster.
BLAS.set_num_threads(1)

# Dev (default): tuned so one full-budget training run (nn_objective) takes ~10s -- long
# enough that per-trial compute genuinely dominates fixed dispatch overhead, while staying
# fast for local iteration. Deploy (BIGHO_BENCHMARK_DEPLOY=true, set by the CI workflow):
# same shape, a larger resource ladder -- but nowhere near MNIST+CNN's multipliers, since
# Titanic (891 rows, plain MLP, no convolutions) is cheap enough per-epoch that we don't
# need extreme multipliers to reach multi-second/multi-minute trials.
const DEPLOY = get(ENV, "BIGHO_BENCHMARK_DEPLOY", "false") == "true"
const R_MIN = DEPLOY ? 5 : 1      # bottom resource level, in resource UNITS
const R_MAX = DEPLOY ? 1215 : 27   # top resource level, in resource UNITS -- both r_min and R_MAX
# are exact powers of the default η=3 apart (1215 = 5*3^5), giving a clean 6-bracket schedule
# with no "short of requested R" shortfall.
# Each Hyperband/ASHA resource unit r is EPOCHS_PER_RESOURCE real training epochs, not 1 --
# R_MAX resource units alone isn't much real training; the promotion/rung math itself still
# operates on resource units, only the actual epoch count trained is scaled up. Titanic's tiny
# dataset + plain MLP means epochs are ~1000x cheaper than MNIST+CNN's were, so these
# multipliers land at ~10s/trial (dev) and well under a minute/trial (deploy) instead of
# MNIST's hours. Set so the bottom rung (r_min=5) trains at least ~30 epochs -- 5*6=30 exactly,
# a clean integer, so no rounding is needed anywhere in the schedule.
const EPOCHS_PER_RESOURCE = DEPLOY ? 6 : 250

# Random's full-trial epoch count -- deliberately NOT just R_MAX*EPOCHS_PER_RESOURCE. That would
# scale Random's total budget (N_TRIALS * epochs/trial) in lockstep with EPOCHS_PER_RESOURCE, but
# Hyperband/ASHA's total budget is fixed by their bracket schedule (independent of N_TRIALS) -- so
# bumping EPOCHS_PER_RESOURCE would silently change how the two compare. Defaults to the old direct
# formula; collect_results.jl overrides this after N_TRIALS is known, to match Random's total
# epochs to Hyperband/ASHA's actual total budget.
const RANDOM_FULL_EPOCHS = Ref(round(Int, R_MAX * EPOCHS_PER_RESOURCE))
const BATCHSIZE = 32

"""
    Dataset

Titanic passenger records (891 total), a fixed train/validation split (80/20) built once
with a fixed seed so every sampler/executor benchmark sees identical data. 9 features:
`Pclass`, `Sex` (binary), `Age`, `SibSp`, `Parch`, `Fare` (numeric, standardized using
train-set statistics only, to avoid leaking validation-set information into the scaling),
`Embarked` (one-hot, 3 categories). Missing `Age`/`Fare`/`Embarked` are imputed (median/mode)
before standardization. `.*_onehot` labels are precomputed once since every trial needs them.
"""
struct Dataset
    Xtrain::Matrix{Float32}
    ytrain_onehot::Flux.OneHotArray
    Xval::Matrix{Float32}
    yval_onehot::Flux.OneHotArray
end

function _build_features(df::DataFrame)
    n = nrow(df)
    X = Matrix{Float32}(undef, 9, n)
    X[1, :] = Float32.(df.Pclass)
    X[2, :] = Float32.(df.Sex .== "female")
    X[3, :] = Float32.(df.Age)
    X[4, :] = Float32.(df.SibSp)
    X[5, :] = Float32.(df.Parch)
    X[6, :] = Float32.(df.Fare)
    X[7, :] = Float32.(df.Embarked .== "S")
    X[8, :] = Float32.(df.Embarked .== "C")
    X[9, :] = Float32.(df.Embarked .== "Q")
    return X
end

function make_dataset(; seed::Int=1)
    rng = StableRNG(seed)
    t = Titanic()
    df = copy(t.features)
    df.Age = coalesce.(df.Age, median(skipmissing(df.Age)))
    df.Fare = coalesce.(df.Fare, median(skipmissing(df.Fare)))
    df.Embarked = coalesce.(df.Embarked, "S")
    y = t.targets.Survived

    n = nrow(df)
    perm = shuffle(rng, 1:n)
    n_train = round(Int, 0.8n)
    train_idx, val_idx = perm[1:n_train], perm[n_train+1:end]

    X = _build_features(df)
    # Standardize numeric rows (1=Pclass,3=Age,4=SibSp,5=Parch,6=Fare; 2,7,8,9 are already
    # 0/1) using train-set mean/std only, then apply the same scaling to validation.
    numeric_rows = (1, 3, 4, 5, 6)
    mu = mean(X[collect(numeric_rows), train_idx]; dims=2)
    sigma = std(X[collect(numeric_rows), train_idx]; dims=2)
    X[collect(numeric_rows), :] .= (X[collect(numeric_rows), :] .- mu) ./ sigma

    return Dataset(X[:, train_idx], Flux.onehotbatch(y[train_idx], 0:1),
                   X[:, val_idx], Flux.onehotbatch(y[val_idx], 0:1))
end

const DATASET = make_dataset()

"""
    build_model(n_dense_layers, hidden, activation)

`n_dense_layers` `Dense(...=>hidden)` layers, then a final `Dense(...=>2)` classifier (no
activation -- see the module docs on `logitcrossentropy`). Plain MLP, no convolutions --
Titanic's 9 tabular features have no spatial structure for a conv kernel to exploit.
"""
function build_model(n_dense_layers::Int, hidden::Int, activation)
    layers = Any[]
    in_dim = 9
    for _ in 1:n_dense_layers
        push!(layers, Flux.Dense(in_dim => hidden, activation))
        in_dim = hidden
    end
    push!(layers, Flux.Dense(in_dim => 2))
    return Flux.Chain(layers...)
end

_loss(model, x, y) = Flux.Losses.logitcrossentropy(model(x), y)
_val_loss(model) = _loss(model, DATASET.Xval, DATASET.yval_onehot)
function _setup(n_dense_layers, hidden, activation, lr, reg)
    m = build_model(n_dense_layers, hidden, activation)
    return m, Flux.setup(Flux.OptimiserChain(Flux.WeightDecay(reg), Flux.Adam(lr)), m)
end

function _train_epochs!(model, opt_state, epochs::Int)
    data = Flux.DataLoader((DATASET.Xtrain, DATASET.ytrain_onehot); batchsize=BATCHSIZE, shuffle=true)
    for _ in 1:epochs
        # gc_interval=0 disables Flux.train!'s built-in paced GC (GC.gc(false) on a schedule) --
        # it exists to reclaim GPU buffers, is pure overhead on CPU, and under Threaded() each
        # concurrent trial's own periodic GC.gc() call stalls every other thread at a safepoint,
        # which made Threaded() *slower* than Serial() until this was disabled (verified).
        Flux.train!(_loss, model, data, opt_state; gc_interval=0)
    end
    return model
end

"""
    nn_objective(lr, n_dense_layers, hidden, activation, reg; pre_artefact=nothing)
        -> (val_loss, (model, nothing, R_MAX, finished_at))

Full-budget (`RANDOM_FULL_EPOCHS[]` epochs) training from scratch -- used (wrapped in
`Stateful`) by `RandomSampler`, which has no notion of a partial/resource-limited trial
(`pre_artefact` is always `nothing` for it, so always ignored here). Wrapped in
`Stateful` purely so the trained model, epoch count, and completion timestamp are retrievable
afterward via `post_artefact`, on the same `(model, opt_state_or_nothing, epochs_trained,
finished_at)` shape [`nn_objective_stateful`](@ref) uses -- letting callers extract these
identically regardless of which sampler produced the run. `finished_at` (`time()`, called
right as this objective call itself finishes, i.e. from inside the executor that actually
ran it) is what gives the wall-clock/regret benchmark real per-trial completion timestamps
without needing to drive `ask!`/`tell!` manually or add a callback hook to `run!`.
"""
function nn_objective(lr, n_dense_layers, hidden, activation, reg; pre_artefact=nothing)
    model, opt_state = _setup(n_dense_layers, hidden, activation, lr, reg)
    _train_epochs!(model, opt_state, RANDOM_FULL_EPOCHS[])
    return _val_loss(model), (model, nothing, R_MAX, time())
end

"""
    nn_objective_stateful(r, lr, n_dense_layers, hidden, activation, reg; pre_artefact)
        -> (val_loss, (model, opt_state, r, finished_at))

Hyperband/ASHA path: a promoted trial resumes training from its own checkpoint
(`pre_artefact`, threaded automatically by `SuccessiveHalving`) instead of retraining from
scratch -- so the next promotion only trains the incremental resource units (each
`EPOCHS_PER_RESOURCE` real epochs) needed to reach the new resource level `r`. See
[`nn_objective`](@ref) for why `finished_at` is captured here.
"""
function nn_objective_stateful(r, lr, n_dense_layers, hidden, activation, reg; pre_artefact=nothing)
    if pre_artefact === nothing
        model, opt_state = _setup(n_dense_layers, hidden, activation, lr, reg)
        resource_to_run = r
    else
        model, opt_state, resource_trained, _ = pre_artefact
        resource_to_run = r - resource_trained
    end
    resource_to_run > 0 && _train_epochs!(model, opt_state, round(Int, resource_to_run * EPOCHS_PER_RESOURCE))
    return _val_loss(model), (model, opt_state, r, time())
end

"""
    _warmup_objective / _warmup_objective_stateful

Same call signature/shape as [`nn_objective`](@ref)/[`nn_objective_stateful`](@ref) (so
warming up with these still JIT-compiles the same `Hyperoptimizer`/`run!`/executor/`Stateful`
dispatch machinery for the real ones' argument types), but always trains exactly 1 epoch --
`nn_objective`'s epoch count (`R_MAX * EPOCHS_PER_RESOURCE`) is baked in, not overridable per
call, so warming up with the real objective would mean every "just 2 trials" warmup call pays
the full deploy-scale training cost instead of a few seconds.
"""
function _warmup_objective(lr, n_dense_layers, hidden, activation, reg; pre_artefact=nothing)
    model, opt_state = _setup(n_dense_layers, hidden, activation, lr, reg)
    _train_epochs!(model, opt_state, 1)
    return _val_loss(model), (model, nothing, 1, time())
end
function _warmup_objective_stateful(r, lr, n_dense_layers, hidden, activation, reg; pre_artefact=nothing)
    model, opt_state = pre_artefact === nothing ? _setup(n_dense_layers, hidden, activation, lr, reg) : pre_artefact[1:2]
    _train_epochs!(model, opt_state, 1)
    return _val_loss(model), (model, opt_state, r, time())
end
