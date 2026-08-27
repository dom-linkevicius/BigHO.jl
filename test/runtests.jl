using Test, Random, Logging
using StableRNGs: StableRNG
using Hyperopt

@testset "Hyperopt" begin
    include("test_domains.jl")
    include("test_random.jl")
    include("test_manual.jl")
    include("test_failures.jl")
    include("test_artefacts.jl")
end
