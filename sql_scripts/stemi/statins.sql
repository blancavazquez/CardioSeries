--*********************************************
--Paso12: extracción de aspirin
drop view all_aspirin cascade;
create view all_aspirin as
with temp_extract_dosis as
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate

  ---ya tengo la dosis
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

---vamos por la hora
,substring(LOWER(nt.text), 'started on aspirin [0-9:].* - given at ([0-9:]+)') as hora
,substring(LOWER(nt.text), 'aspirin [0-9:].* mg at ([0-9:]+)') as hora_2
,substring(LOWER(nt.text), 'aspirin .* at ([0-9:]+)\.') as hora_3

,split_part(nt.chartdate::text,' ',1) as fecha

FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General' or
      category = 'Nursing' or
      category = 'Nursing/other' or
      category = 'Physician '
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
  ,concat(nt."fecha", ' ', nt."hours",':',nt."minutes",':00') as dasp_hour

FROM temp_hours_minute nt
order by nt.subject_id
),
temp_longitud as 
(
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt.dasp_hour
  ,CHAR_LENGTH(nt."dasp_hour") as longitud
  ,nt."value"

FROM temp_charttime nt
order by nt.subject_id
)
select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,case
    when nt."longitud" = '15' then nt.chartdate
    else nt.dasp_hour::timestamp end as charttime

  ,case
    when nt."value"!='' then 7325
  else null end as itemid

  ,nt."value"::double precision
  ,case
    when nt."value"!='' then 'mg'
  else null end as valueuom

FROM temp_longitud nt
where "value" is not null and "value"!='.'
order by nt.subject_id;

select * from all_aspirin;
