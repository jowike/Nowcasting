with cte as (
    select distinct 
        v.VariableId
        , min(ReferenceDate) as min_reference_date
        , max(ReferenceDate) as max_reference_date
    from Variable v
    inner join VintageData vd on v.VariableId=vd.VariableId
    where IsDiscontinued=0
        and NewestDataSource='FRED'
        and VariableValue is not null
    group by v.VariableId
)
select distinct
    v.VariableCode
    , v.Description
    , v.Category
    , v.Region
    , v.Unit
    , v.Adjustment
    , v.FrequencyDescription
    , LastUpdatedOnSource
    , ReleaseName
    , ReleaseLink
    , SourceName
    , SourceLink
    , vd.*
from cte
inner join Variable v on v.VariableId=cte.VariableId
left join Sources on v.VariableCode=Sources.VariableCode
left join VintageData vd on vd.VariableId=cte.VariableId
    -- additional constraint not to extract rows with nulls
    where VariableValue is not null