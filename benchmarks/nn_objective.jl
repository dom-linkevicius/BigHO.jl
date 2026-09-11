using Flux
using MLDatasets: Titanic
using DataFrames
using Statistics: median, mean, std
using StableRNGs: StableRNG
using Random: shuffle
using LinearAlgebra: BLAS

# Threaded() is the outer parallelism -- BLAS threads on top of it oversubscribe the cores and
# make Threaded() slower than Serial().
BLAS.set_num_threads(1)

# Dev (default): ~10s per full-budget training run, so compute dominates dispatch overhead.
# Deploy (BIGHO_BENCHMARK_DEPLOY=true, set by the CI workflow): same shape, longer ladder.
const DEPLOY = get(ENV, "BIGHO_BENCHMARK_DEPLOY", "false") == "true"
const R_MIN = DEPLOY ? 5 : 1      # bottom resource level, in resource UNITS
const R_MAX = DEPLOY ? 1215 : 27   # top resource level; 1215 = 5*3^5, an exact η=3 ladder from R_MIN
# One resource unit is this many real epochs -- the rung math stays in units, only the training
# scales. Set so the bottom rung trains a meaningful number of epochs (deploy: 5*6=30).
const EPOCHS_PER_RESOURCE = DEPLOY ? 6 : 250

# Overridden by collect_results.jl once N_TRIALS is known, to match Random's total epochs to
# Hyperband/ASHA's -- theirs is fixed by the bracket schedule, so it can't be derived from R_MAX.
const RANDOM_FULL_EPOCHS = Ref(round(Int, R_MAX * EPOCHS_PER_RESOURCE))
const BATCHSIZE = 32

"""
    Dataset
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
    # Standardize the numeric rows (2,7,8,9 are already 0/1) on train-set statistics only.
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
        # gc_interval=0 disables Flux's paced GC: it's for GPU buffers, and under Threaded() each
        # trial's GC.gc() stalls every other thread at a safepoint (measured slower than Serial()).
        Flux.train!(_loss, model, data, opt_state; gc_interval=0)
    end
    return model
end

"""
    nn_objective(params; pre_artefact=nothing)
        -> (val_loss, (model, nothing, R_MAX, finished_at))
"""
function nn_objective(p; pre_artefact=nothing)
    model, opt_state = _setup(p.n_dense_layers, p.hidden, p.activation, p.lr, p.reg)
    _train_epochs!(model, opt_state, RANDOM_FULL_EPOCHS[])
    return _val_loss(model), (model, nothing, R_MAX, time())
end

"""
    nn_objective_stateful(params; pre_artefact)
        -> (val_loss, (model, opt_state, r, finished_at))
"""
function nn_objective_stateful(p; pre_artefact=nothing)
    if pre_artefact === nothing
        model, opt_state = _setup(p.n_dense_layers, p.hidden, p.activation, p.lr, p.reg)
        resource_to_run = p.r
    else
        model, opt_state, resource_trained, _ = pre_artefact
        resource_to_run = p.r - resource_trained
    end
    resource_to_run > 0 && _train_epochs!(model, opt_state, round(Int, resource_to_run * EPOCHS_PER_RESOURCE))
    return _val_loss(model), (model, opt_state, p.r, time())
end

"""
    _warmup_objective / _warmup_objective_stateful
"""
function _warmup_objective(p; pre_artefact=nothing)
    model, opt_state = _setup(p.n_dense_layers, p.hidden, p.activation, p.lr, p.reg)
    _train_epochs!(model, opt_state, 1)
    return _val_loss(model), (model, nothing, 1, time())
end
function _warmup_objective_stateful(p; pre_artefact=nothing)
    model, opt_state = pre_artefact === nothing ? _setup(p.n_dense_layers, p.hidden, p.activation, p.lr, p.reg) : pre_artefact[1:2]
    _train_epochs!(model, opt_state, 1)
    return _val_loss(model), (model, opt_state, p.r, time())
end
