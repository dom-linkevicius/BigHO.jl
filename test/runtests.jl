using Test, Random, Logging
using StableRNGs: StableRNG
using Hyperopt

@testset "Hyperopt" begin
    @testset "Basic" begin
        include("basic/test_domains.jl")
        include("basic/test_artefacts.jl")
        include("basic/test_failures.jl")
        include("basic/test_manual.jl")
    end
    @testset "Samplers" begin
        include("samplers/test_random.jl")
    end
    @testset "Executors" begin
        @testset "Non-parallel" begin
            include("executors/test_serial.jl")
        end
        @testset "Parallel" begin
            include("executors/test_threaded.jl")
        end
    end
end
