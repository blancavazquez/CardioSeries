----script to extract vital signs from noteevents (category: echo)

--*******************************
--Paso 1: extracción de height
drop view echo_height;
create view echo_height as
with temp_value as --extract values from echo notes
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(nt.text, 'Date/Time: [\[\]0-9*-]+ at ([0-9:]+)') as hora
 ,cast(substring(nt.text, 'Height: \(in\)(.*?)\n') as double precision) *2.54 as height
 ,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Echo'
),
temp_itemid as
(
select --- add itemid for Weight
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."height" ---226730
  ,case
    when nt."height" > 100 and nt."height" < 240 then 226730
  else null end as itemid
  ,concat(nt."fecha", ' ', nt."hora",':00') as charttime
from temp_value nt
order by nt.subject_id
)
select --- add itemid for Weight
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."charttime"
  ,nt."height" ---226730
  ,nt."itemid"
  ,case
    when nt."height" is not null then  nt."height"
  ----  when nt."weight" > 50 and nt."weight" < 250 then 226846
  else null end as value
from temp_itemid nt
where "itemid" is not null
order by nt.subject_id;

--*******************************
--- Paso2: extracción de weight
drop view echo_weight;
create view echo_weight as
with temp_value as --extract values from echo notes
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(nt.text, 'Date/Time: [\[\]0-9*-]+ at ([0-9:]+)') as hora
 ,cast(substring(nt.text, 'Weight \(lb\): ([0-9]+)\n') as double precision) * 0.45 as weight
 ,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Echo'
),
temp_itemid as
(
select --- add itemid for Weight
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."weight"
  ,case
   when nt."weight" > 50 and nt."weight" < 250 then 226846
  else null end as itemid
  ,concat(nt."fecha", ' ', nt."hora",':00') as charttime
from temp_value nt
order by nt.subject_id
)
select --- add itemid for Weight
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."charttime"
  ,nt."weight"
  ,nt."itemid"
  ,case
    when nt."weight" is not null then  nt."weight"
  else null end as value
from temp_itemid nt
where "itemid" is not null
order by nt.subject_id;

--***************************************
-- Paso 3: extracción de sysbp
drop view echo_sysbp;
create view echo_sysbp as
with temp_value as --extract values from echo notes
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(nt.text, 'Date/Time: [\[\]0-9*-]+ at ([0-9:]+)') as hora
 ---,substring(nt.text, 'BP \(mm Hg\): (.*?)\n') as Blood pressure original
 ,cast(substring(nt.text, 'BP \(mm Hg\): ([0-9]+)') as double precision) as sysbp
 ,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Echo'
),
temp_itemid as
(
select --- add itemid for sysbp
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."sysbp"
  ,case
    when nt."sysbp" > 0 then 51
  else null end as itemid
  ,concat(nt."fecha", ' ', nt."hora",':00') as charttime
from temp_value nt
order by nt.subject_id
)
select --- add itemid for sysbp
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."charttime"
  ,nt."sysbp" ---226730
  ,nt."itemid"
  ,case
    when nt."sysbp" is not null then  nt."sysbp"
  else null end as value
from temp_itemid nt
where "itemid" is not null
order by nt.subject_id;

--******************************
--Paso 4: extracción diasbp
drop view echo_diasbp;
create view echo_diasbp as
with temp_value as --extract values from echo notes
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(nt.text, 'Date/Time: [\[\]0-9*-]+ at ([0-9:]+)') as hora
 ---,substring(nt.text, 'BP \(mm Hg\): (.*?)\n') as Blood pressure original
 ,cast(substring(nt.text, 'BP \(mm Hg\): [0-9]+/([0-9]+?)') as double precision) as diasbp
 ,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Echo'
),
temp_itemid as
(
select --- add itemid for diasp
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."diasbp"
  ,case
    when nt."diasbp" > 0 then 8368
  else null end as itemid
  ,concat(nt."fecha", ' ', nt."hora",':00') as charttime
from temp_value nt
order by nt.subject_id
)
select --- add itemid for diasp
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."charttime"
  ,nt."diasbp" ---8368
  ,nt."itemid"
  ,case
    when nt."diasbp" is not null then  nt."diasbp"
  else null end as value
from temp_itemid nt
where "itemid" is not null
order by nt.subject_id;

--*****************************
-- Paso 5: extracción heart_rate
drop view echo_heart_rate;
create view echo_heart_rate as
with temp_value as --extract values from echo notes
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
 ,substring(nt.text, 'Date/Time: [\[\]0-9*-]+ at ([0-9:]+)') as hora
 ,cast(substring(nt.text, 'HR \(bpm\): (.*?)\n') as double precision) as heart_rate
 ,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Echo'
),
temp_itemid as
(
select --- add itemid for heart_rate
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."heart_rate"
  ,case
    when nt."heart_rate" > 0 then 211
  else null end as itemid
  ,concat(nt."fecha", ' ', nt."hora",':00') as charttime
from temp_value nt
order by nt.subject_id
)
select --- add itemid for heart_rate
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."charttime"
  ,nt."heart_rate" ---211
  ,nt."itemid"
  ,case
    when nt."heart_rate" is not null then  nt."heart_rate"
  else null end as value
from temp_itemid nt
where "itemid" is not null
order by nt.subject_id;

--***************************************
--- Paso 6: join itemid (height, weight,sysbp,diasbp,heart_rate)
drop view echo_join cascade;
create view echo_join as

select
subject_id
,hadm_id
,chartdate
,"charttime"
,"itemid"
,"value"
from echo_height
UNION
select
subject_id
,hadm_id
,chartdate
,"charttime"
,"itemid"
,"value"
from echo_weight
UNION
select
subject_id
,hadm_id
,chartdate
,"charttime"
,"itemid"
,"value"
from echo_sysbp
UNION
select
subject_id
,hadm_id
,chartdate
,"charttime"
,"itemid"
,"value"
from echo_diasbp
UNION
select
subject_id
,hadm_id
,chartdate
,"charttime"
,"itemid"
,"value"
from echo_heart_rate
order by subject_id;

--*******************************
--Paso 7: add valueom
drop view echo_data cascade;
create view echo_data as
select
  subject_id as "SUBJECT_ID"
  ,hadm_id as "HADM_ID"
  ,chartdate as "CHARTDATE"
  ,"charttime" as "CHARTTIME"
  ,"itemid" as "ITEMID"
  ,cast("value" as double precision) as "VALUE"

,case
when "itemid" in (51, 8368) then 'mmHg'
when "itemid" in (211) then 'bpm'
when "itemid" in (226730) then 'cm'
when "itemid" in (226846) then 'kg'
else null end as "VALUEUOM"
from echo_join
where hadm_id is not null
order by subject_id;
