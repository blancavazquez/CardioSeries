----script to extract medications from noteevents
-- (categories: General, Nursing, Nursing/other, Physician)
-- Importante: solo se extraen los medicamentos de los 4 grupos de meds:
-- aspirina, beta-bloqueador, estatinas e IECA.
--******************************************
--Paso previo: delete all related views
drop view general_insulin_ns cascade;
drop view general_heparin_ns cascade;
drop view general_amiodarone_ns cascade;
drop view general_data_ns cascade;

drop view all_lisinopril_ns cascade;
drop view all_enalapril_ns cascade;
drop view all_captopril_ns cascade;
drop view all_atorvastatin_ns cascade;
drop view all_simvastatin_ns cascade;
drop view all_pravastatin_ns cascade;
drop view all_rosuvastatin_ns cascade;
drop view all_metoprolol_ns cascade;
drop view all_atenolol_ns cascade;
drop view all_carvedilol_ns cascade;
drop view all_sotalol_ns cascade;
drop view all_aspirin_ns cascade;
drop view consult_data_ns cascade;
drop view all_medicaments_ns cascade;

drop view echo_data_ns cascade;

---------------------------

--Paso 1: extracción de lisinopril
drop view all_lisinopril_ns cascade;
create view all_lisinopril_ns as
with temp_extract_dosis as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(LOWER(nt.text), 'lisinopril ([0-9:\.]+)(.*)') as lisi_dose
 ,substring(LOWER(nt.text), 'lisinopril dose to ([0-9:\.]+)') as lisi_dose2
 ,substring(LOWER(nt.text), 'lisinopril ([0-9:]+)') as lisi_dose3
 ,substring(LOWER(nt.text), 'lisinopril ([0-9:]+\.[0-9:]+)(.*)') as lisi_dose4

,substring(LOWER(nt.text), 'started on lisinopril [0-9:]mg - given at ([0-9:]+)') as hora
,substring(LOWER(nt.text), 'lisinopril [0-9:] mg at ([0-9:]+)') as hora_2
,substring(LOWER(nt.text), 'lisinopril at ([0-9:]+)\.') as hora_3
,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General' or
      category = 'Nursing' or
      category = 'Nursing/other' or
      category = 'Physician ' or
      category = 'Consultant'
),
temp_fix_doses as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"

  ,case
  	when nt."lisi_dose"!='' then nt."lisi_dose"
    when nt."lisi_dose2"!='' then nt."lisi_dose2"
    when nt."lisi_dose3"!='' then nt."lisi_dose3"
    when nt."lisi_dose4"!='' then nt."lisi_dose4"
  	else null end as value

  ,case
  	when (nt."hora_3"!='' and CHAR_LENGTH(nt."hora_3")=4) then nt."hora_3"
  	when (nt."hora_2"!='' and CHAR_LENGTH(nt."hora_2")=4) then nt."hora_2"
  else hora end as hora_final

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_hours_minute as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."value"
  ,nt."hora_final"
  ,substring(nt."hora_final",1,2) as hours
  ,substring(nt."hora_final",3,4) as minutes

FROM temp_fix_doses nt
where nt."value"!='.'
order by nt.subject_id
),
temp_charttime as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,cast(nt."value" as double precision)
  ,nt."hours"
  ,nt."minutes"
  ,concat(nt."fecha", ' ', nt."hours",':',nt."minutes",':00') as date_hour

FROM temp_hours_minute nt
order by nt.subject_id
),
temp_longitud as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt.date_hour
  ,CHAR_LENGTH(nt."date_hour") as longitud
  ,nt."value"
  ,case
    when nt."value"> 1000 then nt."value"/1000
    else nt."value" end as value2 -- corrección de valores atípicos!

FROM temp_charttime nt
order by nt.subject_id
)
select
  nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,case
  	when nt."longitud" = '15' then nt.chartdate
  	else nt.date_hour::timestamp end as "CHARTTIME"

  ,case
    when nt."value2">=0 then 65374
  else null end as "ITEMID"

  ,nt."value2"::double precision as "VALUE"
  ,case
    when nt."value2">0 then 'mg'
  else null end as "VALUEUOM"

FROM temp_longitud nt
where "value2" is not null
order by nt.subject_id;

--**************************************
--Paso2: extracción de enalapril
drop view all_enalapril_ns cascade;
create view all_enalapril_ns as
with temp_extract_dosis as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(LOWER(nt.text), 'enalapril iv (0\.[0-9:]+)(.*)') as ena_dose
 ,substring(LOWER(nt.text), 'enalapril to ([0-9:\.]+)') as ena_dose2
 ,substring(LOWER(nt.text), 'enalapril ([0-9:]+)') as ena_dose3
 ,substring(LOWER(nt.text), 'enalapril ([0-9:]+\.[0-9:]+)(.*)') as ena_dose4
 ,substring(LOWER(nt.text), 'given .* ([0-9:]+).* iv vasotec') as vaso_dose
 ,substring(LOWER(nt.text), 'on vasotec ([0-9:]+)') as vaso_dose2

,substring(LOWER(nt.text), 'started on enalapril [0-9:]mg - given at ([0-9:]+)') as hora
,substring(LOWER(nt.text), 'enalapril [0-9:] mg at ([0-9:]+)') as hora_2
,substring(LOWER(nt.text), 'enalapril at ([0-9:]+)\.') as hora_3
,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General' or
      category = 'Nursing' or
      category = 'Nursing/other' or
      category = 'Physician ' or
      category = 'Consultant'
),
temp_fix_doses as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,case
    when nt."ena_dose"!='' then nt."ena_dose"
    when nt."ena_dose2"!='' then nt."ena_dose2"
    when nt."ena_dose3"!='' then nt."ena_dose3"
    when nt."ena_dose4"!='' then nt."ena_dose4"
    when nt."vaso_dose"!='' then nt."vaso_dose"
    when nt."vaso_dose2"!='' then nt."vaso_dose2"
    else null end as value

  ,case
    when (nt."hora_3"!='' and CHAR_LENGTH(nt."hora_3")=4) then nt."hora_3"
    when (nt."hora_2"!='' and CHAR_LENGTH(nt."hora_2")=4) then nt."hora_2"
  else hora end as hora_final

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_hours_minute as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."value"
  ,nt."hora_final"
  ,substring(nt."hora_final",1,2) as hours
  ,substring(nt."hora_final",3,4) as minutes

FROM temp_fix_doses nt
where nt."value"!='.'
order by nt.subject_id
),
temp_charttime as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,cast(nt."value" as double precision)
  ,nt."hours"
  ,nt."minutes"
  ,concat(nt."fecha", ' ', nt."hours",':',nt."minutes",':00') as date_hour

FROM temp_hours_minute nt
order by nt.subject_id
),
temp_longitud as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt.date_hour
  ,CHAR_LENGTH(nt."date_hour") as longitud
  ,nt."value"

  ,case
    when nt."value"> 1000 then nt."value"/1000
    else nt."value" end as value2 -- corrección de valores atípicos!

FROM temp_charttime nt
order by nt.subject_id
)
select
   nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,case
    when nt."longitud" = '15' then nt.chartdate
    else nt.date_hour::timestamp end as "CHARTTIME"

  ,case
    when nt."value2">0 then 42648
  else null end as "ITEMID"

  ,nt."value2"::double precision as "VALUE"
  ,case
    when nt."value2">0 then 'mg'
  else null end as "VALUEUOM"

FROM temp_longitud nt
where "value2" is not null
order by nt.subject_id;

--**************************************
--Paso3: extracción de captopril
drop view all_captopril_ns cascade;
create view all_captopril_ns as
with temp_extract_dosis as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(LOWER(nt.text), 'captopril ([0-9:]\.[0-9:]+)') as capt_doses
 ,substring(LOWER(nt.text), 'captopril ([0-9:]+)') as capt_doses2
 ,substring(LOWER(nt.text), 'captopril to ([0-9:]\.[0-9:]+)') as capt_doses3
 ,substring(LOWER(nt.text), 'captopril to ([0-9:]+)') as capt_doses4
 ,substring(LOWER(nt.text), 'captopril ([0-9:]+\.[0-9:]+)(.*)') as capt_doses5
 ,substring(LOWER(nt.text), 'capoten ([0-9:]\.[0-9:]+)') as capo_doses
 ,substring(LOWER(nt.text), 'capoten ([0-9:]+)\.') as capo_doses2

,substring(LOWER(nt.text), 'started on captopril [0-9:]mg - given at ([0-9:]+)\.') as capt_hour
,substring(LOWER(nt.text), 'captopril [0-9:] mg at ([0-9:]+)\.') as capt_hour2
,substring(LOWER(nt.text), 'started on capoten [0-9:] - given at ([0-9:]+)\.') as capo_hour
,substring(LOWER(nt.text), 'capoten [0-9:+].* po tid started at ([0-9:]+)\,') as capo_hour2

,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General' or
      category = 'Nursing' or
      category = 'Nursing/other' or
      category = 'Physician ' or
      category = 'Consultant'
),
temp_add_value as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."capt_doses"
  ,nt."capt_doses2"
  ,nt."capt_doses3"
  ,nt."capt_doses4"
  ,nt."capo_doses"
  ,nt."capo_doses2"
  ,nt."capt_hour"
  ,nt."capt_hour2"
  ,nt."capo_hour"
  ,nt."capo_hour2"

  ,case
    when nt."capt_doses"!='' then nt."capt_doses"
    when nt."capt_doses2"!='' then nt."capt_doses2"
    when nt."capt_doses3"!='' then nt."capt_doses3"
    when nt."capt_doses4"!='' then nt."capt_doses4"
    when nt."capt_doses5"!='' then nt."capt_doses5"
    when nt."capo_doses"!='' then nt."capo_doses"
    when nt."capo_doses2"!='' then nt."capo_doses2"
    else null end as value -- added value

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_add_vartime as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."capt_doses"
  ,nt."capt_doses2"
  ,nt."capt_doses3"
  ,nt."capt_doses4"
  ,nt."capo_doses"
  ,nt."capo_doses2"
  ,nt."capt_hour"
  ,nt."capt_hour2"
  ,nt."capo_hour"
  ,nt."capo_hour2"
  ,nt.value

  ,case
    when nt."capt_hour"!='' then nt."capt_hour"
    when nt."capt_hour2"!='' then nt."capt_hour2"
    when nt."capo_hour"!='' then nt."capo_hour"
    when nt."capo_hour2"!='' then nt."capo_hour2"
    else null end as var_time -- added time

FROM temp_add_value nt
where nt.value!='.'
order by nt.subject_id
),
temp_add_hours_minutes as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."capt_doses"
  ,nt."capt_doses2"
  ,nt."capt_doses3"
  ,nt."capt_doses4"
  ,nt."capo_doses"
  ,nt."capo_doses2"
  ,nt."capt_hour"
  ,nt."capt_hour2"
  ,nt."capo_hour"
  ,nt."capo_hour2"
  ,nt."value"
  ,substring(nt."var_time",1,2) as hours
  ,substring(nt."var_time",3,4) as minutes

FROM temp_add_vartime nt
order by nt.subject_id
),
temp_add_date_hour as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,concat(nt."fecha", ' ', nt."hours",':',nt."minutes",':00') as date_hour
  ,cast(nt."value" as double precision)

FROM temp_add_hours_minutes nt
order by nt.subject_id
),
temp_add_charttime as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,case
    when CHAR_LENGTH(nt."date_hour") = '15' then nt.chartdate
  else nt.date_hour::timestamp end as charttime
  ,nt."value"
  ,case
    when nt."value"> 1000 then nt."value"/1000
    else nt."value" end as value2 -- corrección de valores atípicos!

FROM temp_add_date_hour nt
where nt."value" is not null
order by nt.subject_id
)
select
  nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,nt."charttime" as "CHARTTIME"
  --,nt."value"
  ,case
    when nt."value2">0 then 3349
  else null end as "ITEMID"

  ,nt."value2"::double precision as "VALUE"

  ,case
    when nt."value2">0 then 'mg'
  else null end as "VALUEUOM"

FROM temp_add_charttime nt
where nt."value2" is not null
order by nt.subject_id;

--**************************************
--Paso3: extracción de atorvastatin

drop view all_atorvastatin_ns cascade;
create view all_atorvastatin_ns as
with temp_extract_dosis as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(LOWER(nt.text), 'atorvastatin ([0-9:]+)(.*)') as ator_dose --0\.[0-9:]+
 ,substring(LOWER(nt.text), 'atorvastatin to ([0-9:]+)') as ator_dose2
 ,substring(LOWER(nt.text), 'atorvastatin ([0-9:]+)') as ator_dose3
 ,substring(LOWER(nt.text), 'atorvastatin ([0-9:]+)') as ator_dose4
 ,substring(LOWER(nt.text), 'atorvastatin ([0-9:]+) ') as ator_dose5
 ,substring(LOWER(nt.text), '([0-9:]+) mg atorvastatin ') as ator_dose6
 ,substring(LOWER(nt.text), 'atorvastatin ([0-9:]+\.[0-9:]+)(.*)') as ator_dose7
 ,substring(LOWER(nt.text), 'lipitor ([0-9:\.]+)(.*)') as lip_dose
 ,substring(LOWER(nt.text), 'lipitor to ([0-9:\.]+)') as lip_dose2
 ,substring(LOWER(nt.text), 'lipitor ([0-9:]+)') as lip_dose3
 ,substring(LOWER(nt.text), 'lipitor ([0-9:]+\.[0-9:]+)(.*)') as lip_dose4

,substring(LOWER(nt.text), 'started on atorvastatin [0-9:]mg - given at ([0-9:]+)') as hora
,substring(LOWER(nt.text), 'atorvastatin [0-9:] mg at ([0-9:]+)') as hora_2
,substring(LOWER(nt.text), 'atorvastatin .* at ([0-9:]+)\.') as hora_3
,substring(LOWER(nt.text), 'started on lipitor [0-9:]mg - given at ([0-9:]+)') as hora_4
,substring(LOWER(nt.text), 'lipitor [0-9:] mg at ([0-9:]+)') as hora_5
,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General' or
      category = 'Nursing' or
      category = 'Nursing/other' or
      category = 'Physician ' or
      category = 'Consultant'
),
temp_fix_doses as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."ator_dose"
  ,nt."ator_dose2"

  ,case
    when nt."ator_dose"!='' then nt."ator_dose"
    when nt."ator_dose2"!='' then nt."ator_dose2"
    when nt."ator_dose3"!='' then nt."ator_dose3"
    when nt."ator_dose4"!='' then nt."ator_dose4"
    when nt."ator_dose5"!='' then nt."ator_dose5"
    when nt."ator_dose6"!='' then nt."ator_dose6"
    when nt."ator_dose7"!='' then nt."ator_dose7"
    when nt."lip_dose"!='' then nt."lip_dose"
    when nt."lip_dose2"!='' then nt."lip_dose2"
    when nt."lip_dose3"!='' then nt."lip_dose3"
    when nt."lip_dose4"!='' then nt."lip_dose4"
    else null end as value

  ,case
    when (nt."hora"!='' and CHAR_LENGTH(nt."hora")=4) then nt."hora"
    when (nt."hora_2"!='' and CHAR_LENGTH(nt."hora_2")=4) then nt."hora_2"
    when (nt."hora_3"!='' and CHAR_LENGTH(nt."hora_3")=4) then nt."hora_3"
    when (nt."hora_4"!='' and CHAR_LENGTH(nt."hora_4")=4) then nt."hora_4"
    when (nt."hora_5"!='' and CHAR_LENGTH(nt."hora_5")=4) then nt."hora_5"
  else null end as hora_final

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_hours_minute as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."value"
  ,nt."hora_final"
  ,substring(nt."hora_final",1,2) as hours
  ,substring(nt."hora_final",3,4) as minutes

FROM temp_fix_doses nt
order by nt.subject_id
),
temp_charttime as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."value"
  ,nt."hours"
  ,nt."minutes"
  ,concat(nt."fecha", ' ', nt."hours",':',nt."minutes",':00') as date_hour

FROM temp_hours_minute nt
order by nt.subject_id
),
temp_longitud as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt.date_hour
  ,CHAR_LENGTH(nt."date_hour") as longitud
  ,nt."value"

FROM temp_charttime nt
order by nt.subject_id
)
select
  nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,case
    when nt."longitud" = '15' then nt.chartdate
    else nt.date_hour::timestamp end as "CHARTTIME"

  ,case
    when nt."value"!='' then 286651
  else null end as "ITEMID"

  ,nt."value"::double precision as "VALUE"

  ,case
    when nt."value"!='' then 'mg'
  else null end as "VALUEUOM"

FROM temp_longitud nt
where "value" is not null and "value"!='.'
order by nt.subject_id;


--*************************************************
--Paso5: extracción de simvastatin
drop view all_simvastatin_ns cascade;
create view all_simvastatin_ns as
with temp_extract_dosis as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(LOWER(nt.text), 'simvastatin ([0-9:]+)(.*)') as sim_dose --0\.[0-9:]+
 ,substring(LOWER(nt.text), 'simvastatin to ([0-9:]+)') as sim_dose2
 ,substring(LOWER(nt.text), 'simvastatin ([0-9:]+)') as sim_dose3
 ,substring(LOWER(nt.text), 'simvastatin ([0-9:]+)') as sim_dose4
 ,substring(LOWER(nt.text), 'simvastatin ([0-9:]+) ') as sim_dose5
 ,substring(LOWER(nt.text), 'simvastatin ([0-9:]+\.[0-9:]+)(.*)') as sim_dose6
 ,substring(LOWER(nt.text), 'lovastatin ([0-9:]+)') as sim_dose7
 ,substring(LOWER(nt.text), 'lovastatin ([0-9:]+\.[0-9:]+)(.*)') as sim_dose8

,substring(LOWER(nt.text), 'started on simvastatin [0-9:]mg - given at ([0-9:]+)') as hora
,substring(LOWER(nt.text), 'simvastatin [0-9:] mg at ([0-9:]+)') as hora_2
,substring(LOWER(nt.text), 'simvastatin at ([0-9:]+)\.') as hora_3
,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General' or
      category = 'Nursing' or
      category = 'Nursing/other' or
      category = 'Physician ' or
      category = 'Consultant'
),
temp_fix_doses as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."sim_dose"
  ,nt."sim_dose2"

  ,case
    when nt."sim_dose"!='' then nt."sim_dose"
    when nt."sim_dose2"!='' then nt."sim_dose2"
    when nt."sim_dose3"!='' then nt."sim_dose3"
    when nt."sim_dose4"!='' then nt."sim_dose4"
    when nt."sim_dose5"!='' then nt."sim_dose5"
    when nt."sim_dose6"!='' then nt."sim_dose6"
    when nt."sim_dose7"!='' then nt."sim_dose7"
    when nt."sim_dose8"!='' then nt."sim_dose8"
    else null end as value

  ,case
    when (nt."hora_3"!='' and CHAR_LENGTH(nt."hora_3")=4) then nt."hora_3"
    when (nt."hora_2"!='' and CHAR_LENGTH(nt."hora_2")=4) then nt."hora_2"
  else hora end as hora_final

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_hours_minute as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."value"
  ,nt."hora_final"
  ,substring(nt."hora_final",1,2) as hours
  ,substring(nt."hora_final",3,4) as minutes

FROM temp_fix_doses nt
order by nt.subject_id
),
temp_charttime as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."value"
  ,nt."hours"
  ,nt."minutes"
  ,concat(nt."fecha", ' ', nt."hours",':',nt."minutes",':00') as date_hour

FROM temp_hours_minute nt
order by nt.subject_id
),
temp_longitud as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt.date_hour
  ,CHAR_LENGTH(nt."date_hour") as longitud
  ,nt."value"

FROM temp_charttime nt
order by nt.subject_id
)
select
  nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,case
    when nt."longitud" = '15' then nt.chartdate
    else nt.date_hour::timestamp end as "CHARTTIME"

  ,case
    when nt."value"!='' then 74554
  else null end as "ITEMID"

  ,nt."value"::double precision as "VALUE"

  ,case
    when nt."value"!='' then 'mg'
  else null end as "VALUEUOM"

FROM temp_longitud nt
where "value" is not null and "value"!='.'
order by nt.subject_id;

--*******************************************************************
--Paso6: extracción de pravastatin
drop view all_pravastatin_ns cascade;
create view all_pravastatin_ns as
with temp_extract_dosis as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(LOWER(nt.text), 'pravastatin ([0-9:]+)(.*)') as pra_dose
 ,substring(LOWER(nt.text), 'pravastatin to ([0-9:]+)') as pra_dose2
 ,substring(LOWER(nt.text), 'pravastatin (0\.[0-9:]+)') as pra_dose3
 ,substring(LOWER(nt.text), 'pravastatin ([0-9:]+)') as pra_dose4
 ,substring(LOWER(nt.text), 'pravastatin ([0-9:]+) ') as pra_dose5
 ,substring(LOWER(nt.text), '([0-9:]+) mg pravastatin') as pra_dose6
 ,substring(LOWER(nt.text), 'pravastatin ([0-9:]+\.[0-9:]+)(.*)') as pra_dose7

 ,substring(LOWER(nt.text), 'pravachol ([0-9:]+) ') as pra_dose8
 ,substring(LOWER(nt.text), 'pravachol ([0-9:]+)(.*)') as pra_dose9
 ,substring(LOWER(nt.text), 'pravachol ([0-9:]+\.[0-9:]+)(.*)') as pra_dose10

,substring(LOWER(nt.text), 'started on pravastatin [0-9:]mg - given at ([0-9:]+)') as hora
,substring(LOWER(nt.text), 'pravastatin [0-9:] mg at ([0-9:]+)') as hora_2
,substring(LOWER(nt.text), 'pravastatin at ([0-9:]+)\.') as hora_3
,substring(LOWER(nt.text), 'started on pravachol [0-9:] - given at ([0-9:]+)') as hora_4
,substring(LOWER(nt.text), 'pravachol [0-9:] mg at ([0-9:]+)') as hora_5
,split_part(nt.chartdate::text,' ',1) as fecha


FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General' or
      category = 'Nursing' or
      category = 'Nursing/other' or
      category = 'Physician ' or
      category = 'Consultant'
),
temp_fix_doses as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."pra_dose"
  ,nt."pra_dose2"

  ,case
    when nt."pra_dose"!='' then nt."pra_dose"
    when nt."pra_dose2"!='' then nt."pra_dose2"
    when nt."pra_dose3"!='' then nt."pra_dose3"
    when nt."pra_dose4"!='' then nt."pra_dose4"
    when nt."pra_dose5"!='' then nt."pra_dose5"
    when nt."pra_dose6"!='' then nt."pra_dose6"
    when nt."pra_dose7"!='' then nt."pra_dose7"
    when nt."pra_dose8"!='' then nt."pra_dose8"
    when nt."pra_dose9"!='' then nt."pra_dose9"
    when nt."pra_dose10"!='' then nt."pra_dose10"
    else null end as value

  ,case
    when (nt."hora"!='' and CHAR_LENGTH(nt."hora")=4) then nt."hora"
    when (nt."hora_2"!='' and CHAR_LENGTH(nt."hora_2")=4) then nt."hora_2"
    when (nt."hora_3"!='' and CHAR_LENGTH(nt."hora_3")=4) then nt."hora_3"
    when (nt."hora_4"!='' and CHAR_LENGTH(nt."hora_4")=4) then nt."hora_4"
    when (nt."hora_5"!='' and CHAR_LENGTH(nt."hora_5")=4) then nt."hora_5"
  else null end as hora_final

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_hours_minute as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."value"
  ,nt."hora_final"
  ,substring(nt."hora_final",1,2) as hours
  ,substring(nt."hora_final",3,4) as minutes

FROM temp_fix_doses nt
order by nt.subject_id
),
temp_charttime as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."value"
  ,nt."hours"
  ,nt."minutes"
  ,concat(nt."fecha", ' ', nt."hours",':',nt."minutes",':00') as date_hour

FROM temp_hours_minute nt
order by nt.subject_id
),
temp_longitud as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt.date_hour
  ,CHAR_LENGTH(nt."date_hour") as longitud
  ,nt."value"

FROM temp_charttime nt
order by nt.subject_id
)
select
  nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,case
    when nt."longitud" = '15' then nt.chartdate
    else nt.date_hour::timestamp end as "CHARTTIME"

  ,case
    when nt."value"!='' then 85542
  else null end as "ITEMID"

  ,nt."value"::double precision as "VALUE"

  ,case
    when nt."value"!='' then 'mg'
  else null end as "VALUEUOM"

FROM temp_longitud nt
where "value" is not null and "value"!='.'
order by nt.subject_id;

--************************************************
--Paso7: extracción de rosuvastatin
drop view all_rosuvastatin_ns cascade;
create view all_rosuvastatin_ns as
with temp_extract_dosis as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(LOWER(nt.text), 'rosuvastatin ([0-9:]+)(.*)') as ros_dose
 ,substring(LOWER(nt.text), 'rosuvastatin to ([0-9:]+)') as ros_dose2
 ,substring(LOWER(nt.text), 'rosuvastatin (0\.[0-9:]+)') as ros_dose3
 ,substring(LOWER(nt.text), 'rosuvastatin ([0-9:]+)') as ros_dose4
 ,substring(LOWER(nt.text), 'rosuvastatin ([0-9:]+).*') as ros_dose5
 ,substring(LOWER(nt.text), '([0-9:]+) mg rosuvastatin') as ros_dose6
 ,substring(LOWER(nt.text), 'rosuvastatin calcium ([0-9:]+).*') as ros_dose7
 ,substring(LOWER(nt.text), 'rosuvastatin ([0-9:]+\.[0-9:]+)(.*)') as ros_dose8
 ,substring(LOWER(nt.text), 'crestor ([0-9:]+) ') as ros_dose9
 ,substring(LOWER(nt.text), 'crestor ([0-9:]+)(.*)') as ros_dose10

,substring(LOWER(nt.text), 'started on rosuvastatin [0-9:]mg - given at ([0-9:]+)') as hora
,substring(LOWER(nt.text), 'rosuvastatin [0-9:] mg at ([0-9:]+)') as hora_2
,substring(LOWER(nt.text), 'rosuvastatin at ([0-9:]+)\.') as hora_3
,substring(LOWER(nt.text), 'started on crestor [0-9:] - given at ([0-9:]+)') as hora_4
,substring(LOWER(nt.text), 'crestor [0-9:] mg at ([0-9:]+)') as hora_5
,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General' or
      category = 'Nursing' or
      category = 'Nursing/other' or
      category = 'Physician ' or
      category = 'Consultant'
),
temp_fix_doses as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."ros_dose"
  ,nt."ros_dose2"

  ,case
    when nt."ros_dose"!='' then nt."ros_dose"
    when nt."ros_dose2"!='' then nt."ros_dose2"
    when nt."ros_dose3"!='' then nt."ros_dose3"
    when nt."ros_dose4"!='' then nt."ros_dose4"
    when nt."ros_dose5"!='' then nt."ros_dose5"
    when nt."ros_dose6"!='' then nt."ros_dose6"
    when nt."ros_dose7"!='' then nt."ros_dose7"
    when nt."ros_dose8"!='' then nt."ros_dose8"
    when nt."ros_dose9"!='' then nt."ros_dose9"
    when nt."ros_dose10"!='' then nt."ros_dose10"
    else null end as value

  ,case
    when (nt."hora"!='' and CHAR_LENGTH(nt."hora")=4) then nt."hora"
    when (nt."hora_2"!='' and CHAR_LENGTH(nt."hora_2")=4) then nt."hora_2"
    when (nt."hora_3"!='' and CHAR_LENGTH(nt."hora_3")=4) then nt."hora_3"
    when (nt."hora_4"!='' and CHAR_LENGTH(nt."hora_4")=4) then nt."hora_4"
    when (nt."hora_5"!='' and CHAR_LENGTH(nt."hora_5")=4) then nt."hora_5"
  else null end as hora_final

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_hours_minute as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."value"
  ,nt."hora_final"
  ,substring(nt."hora_final",1,2) as hours
  ,substring(nt."hora_final",3,4) as minutes

FROM temp_fix_doses nt
order by nt.subject_id
),
temp_charttime as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."value"
  ,nt."hours"
  ,nt."minutes"
  ,concat(nt."fecha", ' ', nt."hours",':',nt."minutes",':00') as date_hour

FROM temp_hours_minute nt
order by nt.subject_id
),
temp_longitud as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt.date_hour
  ,CHAR_LENGTH(nt."date_hour") as longitud
  ,nt."value"

FROM temp_charttime nt
order by nt.subject_id
)
select
  nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,case
    when nt."longitud" = '15' then nt.chartdate
    else nt.date_hour::timestamp end as "CHARTTIME"

  ,case
    when nt."value"!='' then 1101751
  else null end as "ITEMID"

  ,nt."value"::double precision as "VALUE"
  ,case
    when nt."value"!='' then 'mg'
  else null end as "VALUEUOM"

FROM temp_longitud nt
where "value" is not null and "value"!='.'
order by nt.subject_id;

--*********************************************
--Paso8: extracción de metoprolol
drop view all_metoprolol_ns cascade;
create view all_metoprolol_ns as
with temp_extract_dosis as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(LOWER(nt.text), 'metoprolol to ([0-9:]+)') as met_dose
 ,substring(LOWER(nt.text), 'metoprolol (0\.[0-9:]+)') as met_dose2
 ,substring(LOWER(nt.text), 'metoprolol ([0-9:]+).*') as met_dose3
 ,substring(LOWER(nt.text), '([0-9:]+) mg metoprolol') as met_dose4
 ,substring(LOWER(nt.text), '([0-9:]+\.[0-9:]+)mg metoprolol') as met_dose5
 ,substring(LOWER(nt.text), 'metoprolol ([0-9:]+\.[0-9:]+).*') as met_dose6
 ,substring(LOWER(nt.text), 'metoprolol tartrate ([0-9:]+\.[0-9:]+)(.*)') as met_dose7
 ,substring(LOWER(nt.text), 'metoprolol tartrate ([0-9:]+).*') as met_dose8
 ,substring(LOWER(nt.text), 'crestor ([0-9:]+)*.') as met_dose9
 ,substring(LOWER(nt.text), 'lopressor ([0-9:]+)*.') as met_dose10
 ,substring(LOWER(nt.text), 'lopressor ([0-9:]+\.[0-9:]+).*') as met_dose11
 ,substring(LOWER(nt.text), 'toprol ([0-9:]+)*.') as met_dose12
 ,substring(LOWER(nt.text), 'toprol xl ([0-9:]+)*.') as met_dose13
 ,substring(LOWER(nt.text), '([0-9:]+)mg toprol') as met_dose14
 ,substring(LOWER(nt.text), 'metoprolol succinate ([0-9:]+).*') as met_dose15
 ,substring(LOWER(nt.text), 'metoprolol succinate ([0-9:]+\.[0-9:]+).*') as met_dose16
 ,substring(LOWER(nt.text), 'metoprolol succinate \(([0-9:]+\.[0-9:]+).*') as met_dose17
 ,substring(LOWER(nt.text), 'metoprolol xl ([0-9:]+\.[0-9:]+).*') as met_dose18
 ,substring(LOWER(nt.text), 'metoprolol xl ([0-9:]+).*') as met_dose19

,substring(LOWER(nt.text), 'started on metoprolol [0-9:]mg - given at ([0-9:]+)') as hora
,substring(LOWER(nt.text), 'metoprolol [0-9:]mg at ([0-9:]+)') as hora_2
,substring(LOWER(nt.text), 'metoprolol at ([0-9:]+)\.') as hora_3
,substring(LOWER(nt.text), 'started on crestor [0-9:] - given at ([0-9:]+)') as hora_4
,substring(LOWER(nt.text), 'crestor [0-9:] mg at ([0-9:]+)') as hora_5
,substring(LOWER(nt.text), 'lopressor [0-9:] mg at ([0-9:]+)') as hora_6
,substring(LOWER(nt.text), 'toprol [0-9:] mg xl @ ([0-9:]+)') as hora_7
,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General' or
      category = 'Nursing' or
      category = 'Nursing/other' or
      category = 'Physician ' or
      category = 'Consultant'
),
temp_fix_doses as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"

  ,case
    when nt."met_dose"!='' then nt."met_dose"
    when nt."met_dose2"!='' then nt."met_dose2"
    when nt."met_dose3"!='' then nt."met_dose3"
    when nt."met_dose4"!='' then nt."met_dose4"
    when nt."met_dose5"!='' then nt."met_dose5"
    when nt."met_dose6"!='' then nt."met_dose6"
    when nt."met_dose7"!='' then nt."met_dose7"
    when nt."met_dose8"!='' then nt."met_dose8"
    when nt."met_dose9"!='' then nt."met_dose9"
    when nt."met_dose10"!='' then nt."met_dose10"
    when nt."met_dose11"!='' then nt."met_dose11"
    when nt."met_dose12"!='' then nt."met_dose12"
    when nt."met_dose13"!='' then nt."met_dose13"
    when nt."met_dose14"!='' then nt."met_dose14"
    when nt."met_dose15"!='' then nt."met_dose15"
    when nt."met_dose16"!='' then nt."met_dose16"
    when nt."met_dose17"!='' then nt."met_dose17"
    when nt."met_dose18"!='' then nt."met_dose18"
    when nt."met_dose19"!='' then nt."met_dose19"
    else null end as value

  ,case
    when (nt."hora"!='' and CHAR_LENGTH(nt."hora")=4) then nt."hora"
    when (nt."hora_2"!='' and CHAR_LENGTH(nt."hora_2")=4) then nt."hora_2"
    when (nt."hora_3"!='' and CHAR_LENGTH(nt."hora_3")=4) then nt."hora_3"
    when (nt."hora_4"!='' and CHAR_LENGTH(nt."hora_4")=4) then nt."hora_4"
    when (nt."hora_5"!='' and CHAR_LENGTH(nt."hora_5")=4) then nt."hora_5"
    when (nt."hora_6"!='' and CHAR_LENGTH(nt."hora_6")=4) then nt."hora_6"
    when (nt."hora_7"!='' and CHAR_LENGTH(nt."hora_7")=4) then nt."hora_7"
  else null end as hora_final

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_hours_minute as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."value"
  ,nt."hora_final"
  ,substring(nt."hora_final",1,2) as hours
  ,substring(nt."hora_final",3,4) as minutes

FROM temp_fix_doses nt
where nt."value"!='.'
order by nt.subject_id
),
temp_charttime as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,cast(nt."value" as double precision)
  ,nt."hours"
  ,nt."minutes"
  ,concat(nt."fecha", ' ', nt."hours",':',nt."minutes",':00') as date_hour

FROM temp_hours_minute nt
order by nt.subject_id
),
temp_longitud as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt.date_hour
  ,CHAR_LENGTH(nt."date_hour") as longitud
  ,nt."value"
  ,case
    when nt."value"> 1000 then nt."value"/1000
    else nt."value" end as value2 -- corrección de valores atípicos!

FROM temp_charttime nt
order by nt.subject_id
)
select
  nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,case
    when nt."longitud" = '15' then nt.chartdate
    else nt.date_hour::timestamp end as "CHARTTIME"

  ,case
    when nt."value2">0 then 42647
  else null end as "ITEMID"

  ,nt."value2"::double precision as "VALUE"

  ,case
    when nt."value2">0 then 'mg'
  else null end as "VALUEUOM"

FROM temp_longitud nt
where "value2" is not null
order by nt.subject_id;

--*********************************************
--Paso9: extracción de atenolol
drop view all_atenolol_ns cascade;
create view all_atenolol_ns as
with temp_extract_dosis as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(LOWER(nt.text), 'atenolol ([0-9:]+\.[0-9:]+).*') as ate_dose
 ,substring(LOWER(nt.text), 'atenolol (0\.[0-9:]+)') as ate_dose2
 ,substring(LOWER(nt.text), 'atenolol ([0-9:]+).*') as ate_dose3
 ,substring(LOWER(nt.text), '([0-9:]+) mg po atenolol') as ate_dose4
 ,substring(LOWER(nt.text), '([0-9:]+\.[0-9:]+)mg atenolol') as ate_dose5

,substring(LOWER(nt.text), 'started on atenolol [0-9:]mg - given at ([0-9:]+)') as hora
,substring(LOWER(nt.text), 'atenolol [0-9:] mg at ([0-9:]+)') as hora_2
,substring(LOWER(nt.text), 'atenolol [0-9:]+ at ([0-9:]+)\.') as hora_3

,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General' or
      category = 'Nursing' or
      category = 'Nursing/other' or
      category = 'Physician ' or
      category = 'Consultant'
),
temp_fix_doses as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"

  ,case
    when nt."ate_dose"!='' then nt."ate_dose"
    when nt."ate_dose2"!='' then nt."ate_dose2"
    when nt."ate_dose3"!='' then nt."ate_dose3"
    when nt."ate_dose4"!='' then nt."ate_dose4"
    when nt."ate_dose5"!='' then nt."ate_dose5"
    else null end as value

  ,case
    when (nt."hora"!='' and CHAR_LENGTH(nt."hora")=4) then nt."hora"
    when (nt."hora_2"!='' and CHAR_LENGTH(nt."hora_2")=4) then nt."hora_2"
    when (nt."hora_3"!='' and CHAR_LENGTH(nt."hora_3")=4) then nt."hora_3"
  else null end as hora_final

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_hours_minute as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."value"
  ,nt."hora_final"
  ,substring(nt."hora_final",1,2) as hours
  ,substring(nt."hora_final",3,4) as minutes

FROM temp_fix_doses nt
where nt."value"!='.'
order by nt.subject_id
),
temp_charttime as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,cast(nt."value" as double precision)
  ,nt."hours"
  ,nt."minutes"
  ,concat(nt."fecha", ' ', nt."hours",':',nt."minutes",':00') as date_hour

FROM temp_hours_minute nt
order by nt.subject_id
),
temp_longitud as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt.date_hour
  ,CHAR_LENGTH(nt."date_hour") as longitud
  ,nt."value"
  ,case
    when nt."value"> 1000 then nt."value"/1000
    else nt."value" end as value2 -- corrección de valores atípicos!

FROM temp_charttime nt
order by nt.subject_id
)
select
  nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,case
    when nt."longitud" = '15' then nt.chartdate
    else nt.date_hour::timestamp end as "CHARTTIME"

  ,case
    when nt."value">0 then 4147
  else null end as "ITEMID"

  ,nt."value2"::double precision as "VALUE"

  ,case
    when nt."value">0 then 'mg'
  else null end as "VALUEUOM"

FROM temp_longitud nt
where "value" is not null
order by nt.subject_id;

--*********************************************
--Paso10: extracción de carvedilol
drop view all_carvedilol_ns cascade;
create view all_carvedilol_ns as
with temp_extract_dosis as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(LOWER(nt.text), 'carvedilol ([0-9:]+\.[0-9:]+).*') as ate_dose
 ,substring(LOWER(nt.text), 'carvedilol (0\.[0-9:]+)') as ate_dose2
 ,substring(LOWER(nt.text), 'carvedilol ([0-9:]+).*') as ate_dose3
 ,substring(LOWER(nt.text), '([0-9:]+) mg po carvedilol') as ate_dose4
 ,substring(LOWER(nt.text), '([0-9:]+\.[0-9:]+)mg carvedilol') as ate_dose5
 ,substring(LOWER(nt.text), 'coreg ([0-9:]+\.[0-9:]+).*') as ate_dose6
 ,substring(LOWER(nt.text), 'coreg (0\.[0-9:]+)') as ate_dose7
 ,substring(LOWER(nt.text), 'coreg ([0-9:]+).*') as ate_dose8

,substring(LOWER(nt.text), 'started on carvedilol [0-9:]mg - given at ([0-9:]+)') as hora
,substring(LOWER(nt.text), 'carvedilol [0-9:] at ([0-9:]+)') as hora_2
,substring(LOWER(nt.text), 'carvedilol at ([0-9:]+)\.') as hora_3

,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General' or
      category = 'Nursing' or
      category = 'Nursing/other' or
      category = 'Physician ' or
      category = 'Consult'

),
temp_fix_doses as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"

  ,case
    when nt."ate_dose"!='' then nt."ate_dose"
    when nt."ate_dose2"!='' then nt."ate_dose2"
    when nt."ate_dose3"!='' then nt."ate_dose3"
    when nt."ate_dose4"!='' then nt."ate_dose4"
    when nt."ate_dose5"!='' then nt."ate_dose5"
    when nt."ate_dose6"!='' then nt."ate_dose6"
    when nt."ate_dose7"!='' then nt."ate_dose7"
    when nt."ate_dose8"!='' then nt."ate_dose8"
    else null end as value

  ,case
    when (nt."hora"!='' and CHAR_LENGTH(nt."hora")=4) then nt."hora"
    when (nt."hora_2"!='' and CHAR_LENGTH(nt."hora_2")=4) then nt."hora_2"
    when (nt."hora_3"!='' and CHAR_LENGTH(nt."hora_3")=4) then nt."hora_3"
  else null end as hora_final

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_hours_minute as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."value"
  ,nt."hora_final"
  ,substring(nt."hora_final",1,2) as hours
  ,substring(nt."hora_final",3,4) as minutes

FROM temp_fix_doses nt
where nt."value"!='.'
order by nt.subject_id
),
temp_charttime as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,cast(nt."value" as double precision)
  ,nt."hours"
  ,nt."minutes"
  ,concat(nt."fecha", ' ', nt."hours",':',nt."minutes",':00') as date_hour

FROM temp_hours_minute nt
order by nt.subject_id
),
temp_longitud as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt.date_hour
  ,CHAR_LENGTH(nt."date_hour") as longitud
  ,nt."value"
  ,case
    when nt."value"> 1000 then nt."value"/1000
    else nt."value" end as value2 -- corrección de valores atípicos!

FROM temp_charttime nt
order by nt.subject_id
)
select
  nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,case
    when nt."longitud" = '15' then nt.chartdate
    else nt.date_hour::timestamp end as "CHARTTIME"

  ,case
    when nt."value2">0 then 54836
  else null end as "ITEMID"

  ,nt."value2"::double precision as "VALUE"

  ,case
    when nt."value2">0 then 'mg'
  else null end as "VALUEUOM"

FROM temp_longitud nt
where "value2" is not null
order by nt.subject_id;

--*********************************************
--Paso11: extracción de sotalol
drop view all_sotalol_ns cascade;
create view all_sotalol_ns as
with temp_extract_dosis as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(LOWER(nt.text), 'sotalol ([0-9:]+\.[0-9:]+).*') as sot_dose
 ,substring(LOWER(nt.text), 'sotalol (0\.[0-9:]+)') as sot_dose2
 ,substring(LOWER(nt.text), 'sotalol ([0-9:]+).*') as sot_dose3

,substring(LOWER(nt.text), 'started on sotalol [0-9:]mg - given at ([0-9:]+)') as hora
,substring(LOWER(nt.text), 'sotalol [0-9:] mg at ([0-9:]+)') as hora_2
,substring(LOWER(nt.text), 'sotalol [0-9:]+ at ([0-9:]+)\.') as hora_3

,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General' or
      category = 'Nursing' or
      category = 'Nursing/other' or
      category = 'Physician ' or
      category = 'Consultant'
),
temp_fix_doses as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"

  ,case
    when nt."sot_dose"!='' then nt."sot_dose"
    when nt."sot_dose2"!='' then nt."sot_dose2"
    when nt."sot_dose3"!='' then nt."sot_dose3"
    else null end as value

  ,case
    when (nt."hora"!='' and CHAR_LENGTH(nt."hora")=4) then nt."hora"
    when (nt."hora_2"!='' and CHAR_LENGTH(nt."hora_2")=4) then nt."hora_2"
    when (nt."hora_3"!='' and CHAR_LENGTH(nt."hora_3")=4) then nt."hora_3"
  else null end as hora_final

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_hours_minute as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."value"
  ,nt."hora_final"
  ,substring(nt."hora_final",1,2) as hours
  ,substring(nt."hora_final",3,4) as minutes

FROM temp_fix_doses nt
where nt."value"!='.'
order by nt.subject_id
),
temp_charttime as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,cast(nt."value" as double precision)
  ,nt."hours"
  ,nt."minutes"
  ,concat(nt."fecha", ' ', nt."hours",':',nt."minutes",':00') as date_hour

FROM temp_hours_minute nt
order by nt.subject_id
),
temp_longitud as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt.date_hour
  ,CHAR_LENGTH(nt."date_hour") as longitud
  ,nt."value"
  ,case
    when nt."value"> 1000 then nt."value"/1000
    else nt."value" end as value2 -- corrección de valores atípicos!

FROM temp_charttime nt
order by nt.subject_id
)
select
  nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,case
    when nt."longitud" = '15' then nt.chartdate
    else nt.date_hour::timestamp end as "CHARTTIME"

  ,case
    when nt."value2">0 then 37707
  else null end as "ITEMID"

  ,nt."value2"::double precision as "VALUE"

  ,case
    when nt."value2">0 then 'mg'
  else null end as "VALUEUOM"

FROM temp_longitud nt
where "value2" is not null
order by nt.subject_id;

--*********************************************
--Paso12: extracción de aspirin
drop view all_aspirin_ns cascade;
create view all_aspirin_ns as
with temp_extract_dosis as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

 ,substring(LOWER(nt.text), 'aspirin ([0-9:]+\.[0-9:]+).*') as asp_dose
 ,substring(LOWER(nt.text), 'aspirin (0\.[0-9:]+)') as asp_dose2
 ,substring(LOWER(nt.text), 'aspirin ([0-9:]+).*') as asp_dose3
 ,substring(LOWER(nt.text), 'aas ([0-9:]+).*') as asp_dose4
 ,substring(LOWER(nt.text), 'asa ([0-9:]+).*') as asp_dose5
 ,substring(LOWER(nt.text), 'asa ([0-9:]+\.[0-9:]+).*') as asp_dose6
 ,substring(LOWER(nt.text), 'aspirin (buffered) ([0-9:]+).*') as asp_dose7
 ,substring(LOWER(nt.text), 'aspirin (rectal) ([0-9:]+).*') as asp_dose8
 ,substring(LOWER(nt.text), 'acetylsalicylic acid ([0-9:]+).*') as asp_dose9
 ,substring(LOWER(nt.text), 'aspirin ec ([0-9:]+).*') as asp_dose10

,substring(LOWER(nt.text), 'started on aspirin [0-9:]mg - given at ([0-9:]+)') as hora
,substring(LOWER(nt.text), 'aspirin [0-9:] mg at ([0-9:]+)') as hora_2
,substring(LOWER(nt.text), 'aspirin [0-9:]+ at ([0-9:]+)\.') as hora_3

,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General' or
      category = 'Nursing' or
      category = 'Nursing/other' or
      category = 'Physician ' or
      category = 'Consult'
),
temp_fix_doses as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"

  ,case
    when nt."asp_dose"!='' then nt."asp_dose"
    when nt."asp_dose2"!='' then nt."asp_dose2"
    when nt."asp_dose3"!='' then nt."asp_dose3"
    when nt."asp_dose4"!='' then nt."asp_dose4"
    when nt."asp_dose5"!='' then nt."asp_dose5"
    when nt."asp_dose6"!='' then nt."asp_dose6"
    when nt."asp_dose7"!='' then nt."asp_dose7"
    when nt."asp_dose8"!='' then nt."asp_dose8"
    when nt."asp_dose9"!='' then nt."asp_dose9"
    when nt."asp_dose10"!='' then nt."asp_dose10"
    else null end as value

  ,case
    when (nt."hora"!='' and CHAR_LENGTH(nt."hora")=4) then nt."hora"
    when (nt."hora_2"!='' and CHAR_LENGTH(nt."hora_2")=4) then nt."hora_2"
    when (nt."hora_3"!='' and CHAR_LENGTH(nt."hora_3")=4) then nt."hora_3"
  else null end as hora_final

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_hours_minute as (
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt."fecha"
  ,nt."value"
  ,nt."hora_final"
  ,substring(nt."hora_final",1,2) as hours
  ,substring(nt."hora_final",3,4) as minutes

FROM temp_fix_doses nt
where "value"!='.'
order by nt.subject_id
),
temp_charttime as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,cast(nt."value" as double precision)
  ,nt."hours"
  ,nt."minutes"
  ,concat(nt."fecha", ' ', nt."hours",':',nt."minutes",':00') as date_hour

FROM temp_hours_minute nt
order by nt.subject_id
),
temp_longitud as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt.date_hour
  ,CHAR_LENGTH(nt."date_hour") as longitud
  ,nt."value"
  ,case
    when nt."value">1000 then nt."value"/1000
  else nt."value" end as value2 -- corrección de valores atípicos!

FROM temp_charttime nt
order by nt.subject_id
)
select
   nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,case
    when nt."longitud" = '15' then nt.chartdate
    else nt.date_hour::timestamp end as "CHARTTIME"

  ,case
    when nt."value">0 then 7325
  else null end as "ITEMID"

  ,nt."value2"::double precision as "VALUE"

  ,case
    when nt."value">0 then 'mg'
  else null end as "VALUEUOM"

FROM temp_longitud nt
where "value" is not null
order by nt.subject_id;

--*****************************************************
--*****************************************************
----script to extract meds from noteevents (category: consultant)
--- Dado que este scritp, es únicamente para extracción de medicamentos,
--- Se adjunta script para extrar amiodarone from category: consultant
--*******************************
--Paso 13: extracción de amiodarone
drop view consult_data_ns;
create view consult_data_ns as--extract medications from consultant notes
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
where category = 'Consult'
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
    when LOWER(nt."unit") = 'mg/min' then 'mg'
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
  ,nt."value"::double precision as "VALUE"
  ,nt."valueuom" as "VALUEUOM"

from temp_valueom nt
where nt."value" is not null
order by nt.subject_id;

--***********************************
--Paso 14: join views

drop view all_medicaments_ns cascade;
create view all_medicaments_ns as
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from all_lisinopril_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from all_enalapril_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from all_captopril_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from all_atorvastatin_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from all_simvastatin_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from all_pravastatin_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from all_rosuvastatin_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from all_metoprolol_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from all_atenolol_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from all_carvedilol_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from all_sotalol_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from all_aspirin_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from consult_data_ns --amiodarone data from category:consult
order by "SUBJECT_ID";
