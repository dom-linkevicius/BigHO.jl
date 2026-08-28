using Test, Random, Logging
using StableRNGs: StableRNG
using Hyperopt

# Run only the concern(s) named in ARGS -- e.g. `julia test/runtests.jl basic
# samplers` or `Pkg.test(; test_args=["threaded"])`. No default "run
# everything" concern: threading concerns (Threaded) are pointless to
# re-verify under every Julia thread-count configuration, so callers (CI
# included) pick exactly the concerns that matter for a given run rather than
# always paying for the full suite.
const CONCERNS = Dict(
    "basic" => ["basic/test_domains.jl", "basic/test_artefacts.jl", "basic/test_failures.jl", "basic/test_manual.jl"],
    "samplers" => ["samplers/test_random.jl"],
    "serial" => ["executors/test_serial.jl"],
    "threaded" => ["executors/test_threaded.jl"],
)

isempty(ARGS) &&
    error("runtests.jl needs at least one concern to run -- choose from: $(join(sort(collect(keys(CONCERNS))), ", "))")

for concern in ARGS
    haskey(CONCERNS, concern) ||
        error("unknown concern \"$concern\" -- choose from: $(join(sort(collect(keys(CONCERNS))), ", "))")
    for file in CONCERNS[concern]
        include(file)
    end
end
