@testset "DEHB constructor validation" begin
    @info "Testing DEHB's F/crossover validation and the inherited R/η/r_min validation"

    @test_throws ArgumentError DEHB(R=27, F=0.0)
    @test_throws ArgumentError DEHB(R=27, F=-0.5)
    @test_throws ArgumentError DEHB(R=27, F=1.5)
    @test_throws ArgumentError DEHB(R=27, crossover=-0.1)
    @test_throws ArgumentError DEHB(R=27, crossover=1.1)
    @test_throws ArgumentError DEHB(R=0)
    @test_throws ArgumentError DEHB(R=27, η=1)
    @test_throws ArgumentError DEHB(R=27, r_min=0)
    @test_throws ArgumentError DEHB(R=1, r_min=9)
    @test_throws ArgumentError DEHB(R=27, iterations=0)

    DEHB(R=27, F=1.0, crossover=0.0)
    DEHB(R=27, F=0.5, crossover=1.0)
end

@testset "DEHBSampler cannot be used on its own" begin
    @info "Testing that a bare DEHBSampler rejects every Sampler entry point"

    de = BigHO.DEHBSampler()
    @test BigHO.init(de, (a=Nominal([1]),), 5) === de

    @test_throws ArgumentError BigHO.on_tell!(de, BigHO.RunEntry[], nothing)
    @test_throws ArgumentError BigHO.exhausted(de, nothing)
    @test_throws ArgumentError BigHO.blocked(de, nothing)
    @test_throws ArgumentError BigHO.create_run_entry(de, nothing, 1, (a=1,), [0.5])
    @test_throws ArgumentError de((a=Nominal([1]),), BigHO.RunEntry[])

    ho = Hyperoptimizer(p -> p.a, (a=Nominal([1, 2]),), DEHB(R=9, η=3, r_min=1))
    @test ho.sampler.inner isa BigHO.DEHBSampler
end

@testset "DEHB subpopulations are keyed by resource" begin
    @info "Testing that DEHB allocates one subpopulation per resource level, sized by that bracket's first rung"

    s = DEHB(R=27, η=3, r_min=1)
    de = BigHO._subpopulations!(s)

    @test sort(collect(keys(de.subpops))) == [1, 3, 9, 27]
    @test sort(collect(keys(de.total_dispatched))) == [1, 3, 9, 27]
    @test [length(de.subpops[r]) for r in (1, 3, 9, 27)] == [27, 12, 6, 4]
    @test all(slots -> all(==(0), slots), values(de.subpops))
    @test all(==(0), values(de.total_dispatched))

    @test BigHO._subpopulations!(s) === de
    @test length(de.subpops) == 4
end

@testset "DEHB rolling slot pointer wraps within a subpopulation" begin
    @info "Testing that DEHB's target pointer walks a subpopulation and resets at its end"

    s = DEHB(R=27, η=3, r_min=1)
    de = BigHO._subpopulations!(s)

    @test BigHO._next_slot(s, 3) == 1
    de.total_dispatched[3] = 11
    @test BigHO._next_slot(s, 3) == 12
    de.total_dispatched[3] = 12
    @test BigHO._next_slot(s, 3) == 1
    de.total_dispatched[3] = 13
    @test BigHO._next_slot(s, 3) == 2
end

@testset "DEHB parent pool comes from the next lower resource" begin
    @info "Testing that DEHB's parent pool is the subpopulation one resource level down, skipping empty slots"

    s = DEHB(R=27, η=3, r_min=1)
    de = BigHO._subpopulations!(s)
    runs = [BigHO.RunEntry(i, (r=1, a=Float64(i)), [0.1i, 0.2i]) for i in 1:3]

    de.subpops[1][1] = 1
    de.subpops[1][5] = 3
    de.subpops[9][2] = 2

    @test BigHO._parent_pool(s, runs, 3) == [runs[1].unit_params, runs[3].unit_params]
    @test BigHO._parent_pool(s, runs, 27) == [runs[2].unit_params]

    pool = BigHO._global_pool(s, runs)
    @test length(pool) == 3
    @test sort(pool) == sort([runs[i].unit_params for i in 1:3])
end

@testset "DEHB trial generation mixes mutant and target" begin
    @info "Testing DEHB's mutation/crossover: the forced dimension, the crossover rate, and the range reset"

    target = [0.5, 0.5, 0.5, 0.5]
    parents = [[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6], [0.15, 0.25, 0.35, 0.45]]

    no_cross = BigHO.DEHBSampler(; F=0.5, crossover=0.0, rng=StableRNG(7))
    for _ in 1:20
        trial = BigHO._de_trial(no_cross, target, copy(parents))
        @test length(trial) == length(target)
        @test all(0 .<= trial .<= 1)
        @test count(trial .!= target) == 1
    end

    all_cross = BigHO.DEHBSampler(; F=0.5, crossover=1.0, rng=StableRNG(7))
    for _ in 1:20
        trial = BigHO._de_trial(all_cross, target, copy(parents))
        @test all(0 .<= trial .<= 1)
        @test count(trial .!= target) == length(target)
    end

    wide = BigHO.DEHBSampler(; F=1.0, crossover=1.0, rng=StableRNG(3))
    for _ in 1:50
        @test all(0 .<= BigHO._de_trial(wide, [0.99, 0.01], [[0.99, 0.01], [0.01, 0.99], [0.5, 0.5]]) .<= 1)
    end
end

@testset "DEHB warns only on a Stateful objective" begin
    @info "Testing that DEHB warns about pre_artefact for Stateful objectives and stays quiet otherwise"

    sampler() = DEHB(R=9, η=3, r_min=1)
    stateful = Stateful((p; pre_artefact=nothing) -> (p.a, nothing))

    @test_logs (:warn, r"first bracket of the first iteration") match_mode = :any Hyperoptimizer(stateful, (a=Nominal([1]),), sampler())
    @test_logs min_level = Logging.Warn Hyperoptimizer(p -> p.a, (a=Nominal([1]),), sampler())
    @test_logs min_level = Logging.Warn Hyperoptimizer(nothing, (a=Nominal([1]),), sampler())
end

@testset "DEHB construction and basic run" begin
    @info "Testing a full DEHB run: schedule shape, resource levels, and per-trial bookkeeping"

    toy(p) = (p.a - 3.0)^2 + (p.b - 1.0)^2 + 1.0 / p.r

    s = DEHB(R=27, η=3, r_min=1)
    ho = Hyperoptimizer(toy, (a=Continuous(0, 10), b=Continuous(0, 5)), s)
    run!(ho; show_progress=false)

    @test ho.n == BigHO._total_trials(27, 1, 3)
    @test length(ho.runs) == ho.n
    @test length(ho.completed) == ho.n
    @test ho.status == BigHO.Finished

    @test :r ∉ ho.params
    @test all(e -> e.params.r in (1, 3, 9, 27), ho.runs)
    @test all(e -> length(e.unit_params) == 2, ho.runs)
    @test all(e -> haskey(e.metadata, :slot), ho.runs)

    de = ho.sampler.inner
    for r in (1, 3, 9, 27)
        @test de.total_dispatched[r] == count(e -> e.params.r == r, ho.runs)
        @test all(id -> 1 <= id <= ho.n, filter(!=(0), de.subpops[r]))
    end

    @test_throws ArgumentError settarget!(ho, ho.n + 10)
    @test ho.sampler isa BigHO.FixedPlanSampler
end

@testset "DEHB keeps the best trial in each subpopulation slot" begin
    @info "Testing DEHB's selection step: a slot is only overwritten by a better-scoring trial"

    toy(p) = (p.a - 3.0)^2 + 1.0 / p.r

    ho = Hyperoptimizer(toy, (a=Continuous(0, 10),), DEHB(R=27, η=3, r_min=1))
    run!(ho; show_progress=false)

    de = ho.sampler.inner
    n_checked = 0
    for (r, slots) in de.subpops, (slot, id) in enumerate(slots)
        id == 0 && continue
        entry = ho.runs[id]
        @test entry.status == BigHO.Completed
        @test entry.params.r == r
        @test entry.metadata[:slot] == slot

        rivals = [e for e in ho.runs
                  if e.status == BigHO.Completed && e.params.r == r && e.metadata[:slot] == slot]
        @test entry.value == minimum(e.value for e in rivals)
        n_checked += 1
    end
    @test n_checked > 0
end

@testset "DEHB promotes unchanged only in the initialization bracket" begin
    @info "Testing that DEHB copies promoted configurations in bracket 1 of iteration 1 and evolves them everywhere else"

    toy(p) = (p.a - 3.0)^2 + (p.b - 1.0)^2 + 1.0 / p.r

    ho = Hyperoptimizer(toy, (a=Continuous(0, 10), b=Continuous(0, 5)), DEHB(R=27, η=3, r_min=1))
    run!(ho; show_progress=false)

    n_copied, n_evolved = 0, 0
    for e in ho.runs
        e.metadata[:rung] == 1 && continue
        if e.metadata[:bracket] == 1
            @test haskey(e.metadata, :promoted_from)
            parent = ho.runs[e.metadata[:promoted_from]]
            @test e.unit_params == parent.unit_params
            @test e.params.r > parent.params.r
            n_copied += 1
        else
            @test !haskey(e.metadata, :promoted_from)
            n_evolved += 1
        end
    end
    @test n_copied > 0
    @test n_evolved > 0
end

@testset "DEHB carries pre_artefact only through the initialization bracket" begin
    @info "Testing that only bracket 1 of iteration 1 resumes from a promoted trial's state"

    call_id = Ref(0)
    function stateful_obj(p; pre_artefact=nothing)
        call_id[] += 1
        return (p.a - 3.0)^2 + 1.0 / p.r, (call_id[], pre_artefact === nothing ? nothing : pre_artefact[1])
    end

    ho = @test_logs (:warn, r"first bracket of the first iteration") match_mode = :any Hyperoptimizer(
        Stateful(stateful_obj), (a=Continuous(0, 10),), DEHB(R=27, η=3, r_min=1))
    run!(ho; show_progress=false)

    resumed = filter(e -> e.pre_artefact !== nothing, ho.runs)
    @test !isempty(resumed)
    @test all(e -> e.metadata[:bracket] == 1 && e.metadata[:iteration] == 1, resumed)
    @test all(e -> e.metadata[:rung] > 1, resumed)
    @test all(e -> e.post_artefact !== nothing, ho.runs)
end

@testset "DEHB multi-iteration schedules" begin
    @info "Testing that DEHB replays the bracket schedule and stops copying after the first iteration"

    toy(p) = (p.a - 3.0)^2 + 1.0 / p.r
    one_pass = BigHO._total_trials(9, 1, 3)

    ho = Hyperoptimizer(toy, (a=Continuous(0, 10),), DEHB(R=9, η=3, r_min=1, iterations=3))
    run!(ho; show_progress=false)

    @test ho.n == 3 * one_pass
    @test length(ho.runs) == ho.n
    @test ho.status == BigHO.Finished

    shapes = [sort([(e.metadata[:bracket], e.metadata[:rung])
                    for e in ho.runs if e.metadata[:iteration] == t]) for t in 1:3]
    @test shapes[2] == shapes[1]
    @test shapes[3] == shapes[1]

    @test all(e -> e.metadata[:iteration] == 1, filter(e -> haskey(e.metadata, :promoted_from), ho.runs))
end

@testset "DEHB under the Threaded executor" begin
    @info "Testing DEHB's subpopulation bookkeeping under real concurrency"

    toy(p) = (p.a - 3.0)^2 + (p.b - 1.0)^2 + 1.0 / p.r

    ho = Hyperoptimizer(toy, (a=Continuous(0, 10), b=Continuous(0, 5)), DEHB(R=27, η=3, r_min=1))
    run!(ho; executor=Threaded(4), show_progress=false)

    @test length(ho.runs) == ho.n
    @test length(ho.completed) == ho.n
    @test ho.n_pending == 0
    @test ho.status == BigHO.Finished
    @test sort([e.id for e in ho.runs]) == collect(1:ho.n)

    de = ho.sampler.inner
    for r in keys(de.subpops)
        @test de.total_dispatched[r] == count(e -> e.params.r == r, ho.runs)
    end
end
