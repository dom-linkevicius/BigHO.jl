"""
    load_hyperoptimizer(objective, path) -> Hyperoptimizer
"""
function load_hyperoptimizer(objective, path::AbstractString)
    saved = JLD2.jldopen(file -> file["ho"], path, "r")
    return Hyperoptimizer(saved.params, saved.candidates, saved.sampler, objective, saved.n,
                           saved.runs, saved.completed, saved.n_pending, saved.status,
                           saved.best_min_id, ReentrantLock())
end

"""
    save_hyperoptimizer(ho, path)
"""
function save_hyperoptimizer(ho::Hyperoptimizer, path::AbstractString)
    sanitized = Hyperoptimizer(ho.params, ho.candidates, ho.sampler, nothing, ho.n,
                                ho.runs, ho.completed, ho.n_pending, ho.status,
                                ho.best_min_id, ReentrantLock())
    tmp_path = path * ".tmp"
    JLD2.jldopen(tmp_path, "w") do file
        file["ho"] = sanitized
    end
    mv(tmp_path, path; force=true)
    return nothing
end
