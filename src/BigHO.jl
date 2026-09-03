module BigHO

export Hyperoptimizer, run!, settarget!
export Stateful
export RandomSampler, LHSampler, get_lhs_optim_history
export Serial, Threaded, DistributedQueue
export minimizer, history, results, printmin
export Nominal, Ordinal, Continuous
export load_hyperoptimizer

using Random
using Printf
using Distributed
using StableRNGs: StableRNG
using StatsBase: Weights, sample
import JLD2
import ProgressMeter
import DataFrames
import LatinHypercubeSampling

include("domains.jl")
include("types.jl")

include("samplers/sampler.jl")
include("samplers/random.jl")
include("samplers/lhs.jl")
"""
    FixedPlanSampler

Sampler types whose plan is fixed to `n` at construction (e.g. a Latin Hypercube design matrix), so `settarget!` can't work for them.
"""
const FixedPlanSampler = Union{LHSampler}

include("executors/executor.jl")
include("executors/serial.jl")
include("executors/threaded.jl")
include("executors/distributed_queue.jl")

include("optimizer.jl")

include("report.jl")
include("persistence.jl")
include("dataframe.jl")

end # module
