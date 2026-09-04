module BigHO

export Hyperoptimizer, run!, settarget!
export Stateful
export RandomSampler, LHSampler, get_lhs_optim_history
export Hyperband, ASHA
export Serial, Threaded, DistributedQueue
export minimizer, history, results, printmin
export summaryplot
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
include("samplers/sha_based/sh.jl")
include("samplers/sha_based/hyperband.jl")
include("samplers/sha_based/asha.jl")
"""
    FixedPlanSampler

Sampler types whose plan is fixed at construction (e.g. a Latin Hypercube design matrix, or Hyperband/ASHA's own bracket schedule), so `settarget!` can't work for them.
"""
const FixedPlanSampler = Union{LHSampler,Hyperband,ASHA}

include("executors/executor.jl")
include("executors/serial.jl")
include("executors/threaded.jl")
include("executors/distributed_queue.jl")

include("optimizer.jl")

include("report.jl")
include("persistence.jl")
include("dataframe.jl")
include("plotting.jl")

end # module
