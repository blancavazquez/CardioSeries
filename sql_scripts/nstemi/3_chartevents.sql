--- Script for extracting information of table of chartevents for NSTEMI patients
drop view NSTEMI_chartevents;
create view NSTEMI_chartevents as
with temp_convertion as -- creo vista temporal para conversión de units
(
select
    ch.subject_id
    ,ch.hadm_id
    ,ch.icustay_id
    ,ch.itemid
    ,ch.charttime
    ,ch.storetime
    ,ch.value
    ,case --- converting
    when itemid IN (678, 679, 223761) and (valuenum>50) then (valuenum - 32) * 5/9 -- From F to C
    when itemid IN (226707) and (valuenum<100) then valuenum * 2.54 -- from inch to cm
    else valuenum
    end as measure

    ,case --- updating valueuom
    when LOWER(ch.valueuom) = 'inch' then 'cm'
    when LOWER(ch.valueuom) = 'deg. f' then 'Deg. C'
    when LOWER(ch.valueuom) = '?f' then 'Deg. C'
    else valueuom
    end as unit-- unidad de medida

    ,ch.valuenum --- en teoría, esta variable  debe ser sustituida por valuenum
    ,ch.valueuom
    ,ch.warning
    ,ch.error
    ,ch.resultstatus
    ,ch.stopped

  from chartevents as ch
  inner join NSTEMI_patients  st
  on ch.subject_id=st.subject_id
  where (ch.error IS NULL OR ch.error = 0)
  order by ch.subject_id
)
select
    ch.subject_id as "SUBJECT_ID"
    ,ch.hadm_id as "HADM_ID"
    ,ch.icustay_id as "ICUSTAY_ID"
    ,ch.itemid as "ITEMID"
    ,ch.charttime as "CHARTTIME"
    ,ch.storetime as "STORETIME"
    ,round(cast(measure as numeric),2) as "VALUE"
    ,ch.unit as "VALUEUOM"
    ,ch.warning as "WARNING"
    ,ch.resultstatus as "RESULTSTATUS"
    ,ch.stopped as "STOPPED"
from temp_convertion as ch
where
  (ch.valuenum IS NOT NULL and ch.valuenum > 0)  -- chartvalues cannot be 0 and cannot be negative
order by ch.subject_id;
\copy (SELECT * FROM NSTEMI_chartevents) to '/tmp/NSTEMI_CHARTEVENTS.csv' CSV HEADER;
drop view NSTEMI_chartevents;
