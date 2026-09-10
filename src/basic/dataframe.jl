"""
    DataFrame(ho::Hyperoptimizer) -> DataFrame

One row per trial (`id`, `status`, one column per hyperparameter, `value`,
`error`, `pre_artefact`, `post_artefact`), in `ask!` order -- covers every
trial regardless of status, so failed or still-pending trials can be
filtered on `status` rather than being silently dropped like
[`history`](@ref)/[`results`](@ref).
"""
function DataFrames.DataFrame(ho::Hyperoptimizer)
    rows = [merge((id=e.id, status=e.status), e.params,
                   (value=e.value, error=e.error, pre_artefact=e.pre_artefact, post_artefact=e.post_artefact))
            for e in ho.runs]
    return DataFrames.DataFrame(rows)
end
