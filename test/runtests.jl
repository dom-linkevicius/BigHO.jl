using Test, Random, Logging, Distributed
using StableRNGs: StableRNG
using BigHO
using DataFrames
using CairoMakie
using AlgebraOfGraphics

const CONCERNS = Dict(
    "basic" => ["basic/domains.jl", "basic/artefacts.jl", "basic/failures.jl", "basic/manual.jl", "basic/persistence.jl", "basic/progress.jl", "basic/dataframe.jl", "basic/plotting.jl"],
    "samplers" => ["samplers/random.jl", "samplers/lhs.jl", "samplers/sh_based.jl", "samplers/dehb.jl"],
    "serial" => ["executors/serial.jl"],
    "threaded" => ["executors/threaded.jl"],
    "distributed" => ["executors/distributed_queue.jl", "executors/equivalence.jl"],
)

concerns_to_run = isempty(ARGS) ? sort(collect(keys(CONCERNS))) : ARGS

for concern in concerns_to_run
    haskey(CONCERNS, concern) ||
        error("unknown concern \"$concern\" -- choose from: $(join(sort(collect(keys(CONCERNS))), ", "))")
    for file in CONCERNS[concern]
        include(file)
    end
end
