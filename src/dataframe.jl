"""
    DataFrame(ho::Hyperoptimizer) -> DataFrame

One row per trial (`id`, `status`, one column per hyperparameter, `value`),
in `ask!` order -- covers every trial regardless of status, so failed or
still-pending trials can be filtered on `status` rather than being silently
dropped like [`history`](@ref)/[`results`](@ref).
"""
function DataFrames.DataFrame(ho::Hyperoptimizer)
    rows = [merge((id=e.id, status=e.status), e.params, (value=e.value,)) for e in ho.runs]
    return DataFrames.DataFrame(rows)
end
