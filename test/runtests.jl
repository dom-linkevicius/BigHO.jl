using Test, Random, Logging, Distributed
using StableRNGs: StableRNG
using BigHO

# Run only the concern(s) named in ARGS -- e.g. `julia test/runtests.jl basic
# samplers` or `Pkg.test(; test_args=["threaded"])`. With no ARGS at all (e.g.
# plain `Pkg.test()`/`] test`), run every concern, so the standard invocation
# still works out of the box; callers that only care about a subset -- like
# CI, which doesn't need to re-verify non-threading concerns under multiple
# Julia threads -- can still narrow it down explicitly via test_args.
const CONCERNS = Dict(
    "basic" => ["basic/domains.jl", "basic/artefacts.jl", "basic/failures.jl", "basic/manual.jl"],
    "samplers" => ["samplers/random.jl"],
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
