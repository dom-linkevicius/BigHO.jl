using Test, Random
using Hyperopt

@testset "Hyperopt" begin
    include("test_random.jl")
    include("test_manual.jl")
end
