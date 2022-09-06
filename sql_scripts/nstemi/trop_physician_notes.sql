--**********************************************
--Extracción de troponin from physician notes
--estrategia: separar en varias vistas de acuerdo al número de regex usadas 
-- y luego unir las vistas

--*****************regex 1
drop view physician_temp1_troponin_ns cascade;
create view physician_temp1_troponin_ns as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Physician '
),
temp_sin_espacios as(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,replace(replace(replace(texto, ' ', '*^'), '^*', ''), '*^', ' ') as texto_final
FROM temp_note_sin_saltos nt
order by nt.subject_id
),
temp_extract_fecha_hora_valor as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
     ,regexp_matches(LOWER(texto_final), '[c]?trop[n]?[t]?[-]? ?[>]?[<]? ?([0-9:]+.[0-9:]+)', 'g') as tropo
     ,regexp_matches(LOWER(texto_final), 'flowsheet data as of ([\[\]0-9*-]+)', 'g') as fecha
     ,regexp_matches(LOWER(texto_final), 'flowsheet data as of [\[\]0-9*-]+ ([0-9:]+)', 'g') as hora

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,tropo
      ,fecha
      ,hora
      ,REPLACE(REPLACE(REPLACE(REPLACE(tropo::text,'{',''),'}',''),'":',''),'"','')as tropo_clean
      ,REPLACE(REPLACE(REPLACE(fecha::text,'{[',''),']}',''),'*','')as fecha_clean
      ,REPLACE(REPLACE(hora::text,'{',''),'}','')as hora_clean


FROM temp_extract_fecha_hora_valor nt
order by nt.subject_id
),
temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,tropo_clean::double precision
      ,fecha_clean
      ,hora_clean
      ,case when (tropo_clean::double precision>0 and tropo_clean::double precision<100)  then tropo_clean::double precision end as value

FROM temp_replace_caracteres nt
order by nt.subject_id
),
temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,fecha_clean
      ,hora_clean
      ,value      
FROM temp_asignacion_value nt
where value!=0
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case when ((value>0 and value<100 and hora_clean!='')) then concat(fecha_clean,' ',hora_clean,':00')::timestamp else nt.chartdate end as "CHARTTIME"
      ,case when (value>0) then 227429 else null end as "ITEMID"
      ,value as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"
FROM temp_seleccion_value_empty nt
order by nt.subject_id;

--**********************regex 2
drop view physician_temp2_troponin_ns;
create view physician_temp2_troponin_ns as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Physician '
),
temp_sin_espacios as(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,replace(replace(replace(texto, ' ', '*^'), '^*', ''), '*^', ' ') as texto_final
FROM temp_note_sin_saltos nt
order by nt.subject_id
),
temp_extract_fecha_hora_valor as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
     ,regexp_matches(LOWER(texto_final), 'trop[o]?[n]?[i]?[n]?[-]? ?[t]? ?[:]?[0-9:]+[.]?[0-9:]*/[0-9:]+[.]?[0-9:]*/([0-9:]+.[0-9:]+)', 'g') as tropo
     ,regexp_matches(LOWER(texto_final), 'flowsheet data as of ([\[\]0-9*-]+)', 'g') as fecha
     ,regexp_matches(LOWER(texto_final), 'flowsheet data as of [\[\]0-9*-]+ ([0-9:]+)', 'g') as hora

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,tropo
      ,fecha
      ,hora
      ,REPLACE(REPLACE(REPLACE(REPLACE(tropo::text,'{',''),'}',''),'7/0','0'),'"','')as tropo_clean
      ,REPLACE(REPLACE(REPLACE(fecha::text,'{[',''),']}',''),'*','')as fecha_clean
      ,REPLACE(REPLACE(hora::text,'{',''),'}','')as hora_clean


FROM temp_extract_fecha_hora_valor nt
order by nt.subject_id
),
temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,tropo_clean::double precision
      ,fecha_clean
      ,hora_clean
      ,case when (tropo_clean::double precision>0 and tropo_clean::double precision<100)  then tropo_clean::double precision end as value

FROM temp_replace_caracteres nt
order by nt.subject_id
),
temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,tropo_clean
      ,fecha_clean
      ,hora_clean
      ,value      
FROM temp_asignacion_value nt
where value!=0
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case when ((value>0 and value<100 and hora_clean!='')) then concat(fecha_clean,' ',hora_clean,':00')::timestamp + '1 hour'::interval else nt.chartdate end as "CHARTTIME"
      ,case when (value>0) then 227429 else null end as "ITEMID"
      ,value as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
--*************************************************


------- join notas physician ----
drop view  physician_troponin_regex_ns cascade;
create view  physician_troponin_regex_ns as

select --- echo notes
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from physician_temp1_troponin_ns
UNION ALL
select --- general notes
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from physician_temp2_troponin_ns
order by "SUBJECT_ID";

--select * from physician_troponin_regex_ns;

