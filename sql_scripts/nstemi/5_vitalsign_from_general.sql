----script to extract vital signs from noteevents (category: general)

--*******************************
--Paso 1: extracción de insulin
drop view general_insulin_ns cascade;
create view general_insulin_ns as--extract medications from general notes
with temp_extraccion as 
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(nt.text, 'Flowsheet Data as of  [\[\]0-9*-]+ ([0-9:]+)') as hora
 ,substring(nt.text, 'Insulin - Regular - (.*?)\n') as insulin
 ,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General'
),
temp_unit as (
select
   nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
 ,cast(split_part(nt."insulin",' ',1) as text) as valor
 ,split_part(nt."insulin",' ',2) as unit
 ,concat(nt."fecha", ' ', nt."hora",':00') as charttime

from temp_extraccion nt
order by nt.subject_id
),
temp_valor as (
select
   nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."charttime"
  ,nt."valor"
  ,nt."unit"
  ,case
  when nt."unit" = 'units/hour' then nt."valor"
  else null end as value

from temp_unit nt
order by nt.subject_id
),
temp_valueom as (
select
   nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."charttime"
  ,cast(nt."value" as integer)
  ,case
    when LOWER(nt."unit") = 'units/hour' then 'unit'
  else null end as valueuom

from temp_valor nt
where nt."value" is not null
order by nt.subject_id
)
select
   nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,nt."charttime" as "CHARTTIME"
  ,case
    when nt."value" > 0 then 30045
  else null end as "ITEMID"
  ,nt."value" as "VALUE" 
  ,nt."valueuom" as "VALUEUOM"

from temp_valueom nt
where nt."value" is not null
order by nt.subject_id;

--*******************************
--Paso 2: extracción de heparin
drop view general_heparin_ns cascade;
create view general_heparin_ns as--extract medications from general notes
with temp_extraccion as 
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(nt.text, 'Flowsheet Data as of  [\[\]0-9*-]+ ([0-9:]+)') as hora
 ,substring(lower(nt.text), 'heparin sodium - (.*?)\n') as heparin
 ,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General'
),
temp_unit as (
select
   nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
 ,split_part(nt."heparin",' ',1) as dose
 ,split_part(nt."heparin",' ',2) as unit
 ,concat(nt."fecha", ' ', nt."hora",':00') as charttime

from temp_extraccion nt
where nt."heparin" is not null
order by nt.subject_id
),
temp_value as
(
select
   nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."charttime"
  ,REPLACE(nt."dose",',','') as value 
 ,case
   when LOWER(nt."unit") = 'units/hour' then 'unit'
  else null end as valueuom

from temp_unit nt
order by nt.subject_id
),
temp_valor as 
(
select
   nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."charttime"
  ,cast(REPLACE(nt."value",'[**2133-12-29**]','0') as integer) as valor
 ,nt."valueuom"

from temp_value nt
order by nt.subject_id
)
select
    nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,nt."charttime" as "CHARTTIME"
  ,case
    when nt."valor">0 then 217
  else null end as "ITEMID"
  ,nt."valor" as "VALUE" 
  ,nt."valueuom" as "VALUEUOM"

from temp_valor nt
where nt."valor" is not null
order by nt.subject_id;

--**********************************
--Paso 3: extracción de amiodarone
drop view general_amiodarone_ns cascade;
create view general_amiodarone_ns as--extract medications from general notes
with temp_extraccion as 
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(nt.text, 'Flowsheet Data as of  [\[\]0-9*-]+ ([0-9:]+)') as hora
 ,substring(nt.text, 'Amiodarone - (.*?)\n') as amiodarone
 ,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General'
),
temp_unit as (
select
   nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
 ,cast(split_part(nt."amiodarone",' ',1) as text) as valor
 ,split_part(nt."amiodarone",' ',2) as unit
 ,concat(nt."fecha", ' ', nt."hora",':00') as charttime

from temp_extraccion nt
order by nt.subject_id
),
temp_valor as (
select
   nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."charttime"
  ,nt."valor"
  ,nt."unit"
  ,case
  when nt."unit" = 'mg/min' then nt."valor"
  else null end as value

from temp_unit nt
order by nt.subject_id
),
temp_valueom as (
select
   nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."charttime"
  ,cast(nt."value" as double precision)
  ,case
    when LOWER(nt."unit") = 'mg/min' then 'unit'
  else null end as valueuom

from temp_valor nt
where nt."value" is not null
order by nt.subject_id
)
select
    nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,nt."charttime" as "CHARTTIME"
  ,case
    when nt."value" > 0 then 30112
  else null end as "ITEMID"
  ,nt."value" as "VALUE" 
  ,nt."valueuom" as "VALUEUOM"

from temp_valueom nt
where nt."value" is not null
order by nt.subject_id;

--**********************************
--Paso N: join views

drop view general_data_ns cascade;
create view general_data_ns as

select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE" 
,"VALUEUOM"
from general_insulin_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE" 
,"VALUEUOM"
from general_heparin_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE" 
,"VALUEUOM"
from general_amiodarone_ns
order by "SUBJECT_ID";

