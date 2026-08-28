using Test, Random, Logging
using StableRNGs: StableRNG
using BigHO

# Run only the concern(s) named in ARGS -- e.g. `julia test/runtests.jl basic
# samplers` or `Pkg.test(; test_args=["threaded"])`. With no ARGS at all (e.g.
# plain `Pkg.test()`/`] test`), run every concern, so the standard invocation
# still works out of the box; callers that only care about a subset -- like
# CI, which doesn't need to re-verify non-threading concerns under multiple
# Julia threads -- can still narrow it down explicitly via test_args.
const CONCERNS = Dict(
    "basic" => ["basic/test_domains.jl", "basic/test_artefacts.jl", "basic/test_failures.jl", "basic/test_manual.jl"],
    "samplers" => ["samplers/test_random.jl"],
    "serial" => ["executors/test_serial.jl"],
    "threaded" => ["executors/test_threaded.jl"],
)

concerns_to_run = isempty(ARGS) ? sort(collect(keys(CONCERNS))) : ARGS

for concern in concerns_to_run
    haskey(CONCERNS, concern) ||
        error("unknown concern \"$concern\" -- choose from: $(join(sort(collect(keys(CONCERNS))), ", "))")
    for file in CONCERNS[concern]
        include(file)
    end
end
