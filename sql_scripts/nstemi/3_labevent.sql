--- Script for extracting information of table of labevents for NSTEMI patients
drop view NSTEMI_labevents;
create view NSTEMI_labevents as
with temp_convertion as -- creo vista temporal para conversión de units
(
select
    lab.subject_id
    ,lab.hadm_id
    ,lab.itemid
    ,lab.charttime
    ,lab.value

    ,case --- converting
    when itemid IN (678, 679, 223761) and (valuenum>50) then (valuenum - 32) * 5/9 -- From F to C
    when itemid IN (226707) and (valuenum<100) then valuenum * 2.54 -- from inch to cm
    else valuenum
    end as measure

    ,case --- updating valueuom
    when LOWER(lab.valueuom) = 'inch' then 'cm'
    when LOWER(lab.valueuom) = 'deg. f' then 'Deg. C'
    else valueuom
    end as unit-- unidad de medida

    ,lab.valuenum --- en teoría, esta variable  debe ser sustituida por valuenum

from labevents lab
inner join NSTEMI_patients st
on lab.subject_id = st.subject_id
order by lab.subject_id
)
select
    lab.subject_id as "SUBJECT_ID"
    ,lab.hadm_id as "HADM_ID"
    ,lab.itemid as "ITEMID"
    ,lab.charttime as "CHARTTIME"
    ,round(cast(measure as numeric),2) as "VALUE"
    ,lab.unit as "VALUEUOM"

from temp_convertion as lab
where
  (lab.valuenum IS NOT NULL and lab.valuenum > 0)  -- chartvalues cannot be 0 and cannot be negative
order by lab.subject_id;
\copy (SELECT * FROM NSTEMI_labevents) to '/tmp/NSTEMI_LABEVENTS.csv' CSV HEADER;
drop view NSTEMI_labevents;
