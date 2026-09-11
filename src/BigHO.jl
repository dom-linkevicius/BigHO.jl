module BigHO

export Hyperoptimizer, run!, settarget!
export Stateful
export RandomSampler, LHSampler
export Hyperband, ASHA
export Serial, Threaded, DistributedQueue
export minimizer, history, results, printmin
export summaryplot
export Nominal, Ordinal, Continuous
export save_hyperoptimizer, load_hyperoptimizer

using Random
using Printf
using Distributed
using StableRNGs: StableRNG
import JLD2
import ProgressMeter
import DataFrames
import QuasiMonteCarlo

include("basic/domains.jl")
include("basic/types.jl")

include("samplers/sampler.jl")
include("samplers/random.jl")
include("samplers/lhs.jl")
include("samplers/sha_based/sh.jl")
include("samplers/sha_based/hyperband.jl")
include("samplers/sha_based/asha.jl")
"""
    FixedPlanSampler
"""
const FixedPlanSampler = Union{LHSampler,Hyperband,ASHA}

include("executors/executor.jl")
include("executors/serial.jl")
include("executors/threaded.jl")
include("executors/distributed_queue.jl")

include("basic/optimizer.jl")

include("basic/report.jl")
include("basic/persistence.jl")
include("basic/dataframe.jl")
include("basic/plotting.jl")

end # module
