--Paso4: extracción de troponin from clinical notes
--1) from phy notes
drop view physician_troponin_ns cascade;
create view physician_troponin_ns as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join STEMI_patients st
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
temp_extract_dosis as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      --este patrón extrae de la forma: Troponin-T:250/31/0.36 (solo extrae el 3er término:0.36)
      ,substring(texto_final, 'Troponin-T:[0-9:]+/[0-9:]+/([0-9:].[0-9:]+)') as tropo_slash
      --este patrón extrae de la forma: TropT 0.38
      ,substring(texto_final, 'TropT ([0-9:]+.[0-9:]+)') as tropo_onevalue
      --este patrón extrae de la forma: Troponin T <0.01
      ,substring(texto_final, 'Troponin T <([0-9:].[0-9:]+)') as tropo_menor
      --este patrón extrae de la forma: Troponin-T:51//<0.01
      ,substring(texto_final, 'Troponin-T:[0-9:]+//<([0-9:].[0-9:]+)') as tropo_slash_menor
      --este patrón extrae de la forma: Repeat troponin
      ,substring(texto_final, 'Repeat troponin here was ([0-9:].[0-9:]+)') as tropo_as_texto
      --este patrón extrae de la forma: Troponin T 0.11 0.05
      ,substring(texto_final, 'Troponin T ([0-9:].[0-9:]+)') as tropo_espacio
      --este patrón extrae de la forma: Troponin T:167/4/0.05
      ,substring(texto_final, 'Troponin T:[0-9:]+/[0-9:]+/([0-9:].[0-9:]+)') as tropo_espacio2
      ,substring(texto_final, 'Flowsheet Data as of ([\[\]0-9*-]+ [0-9:]+)') as fecha_hora
      ,substring(texto_final, 'Flowsheet Data as of [\[\]0-9*-]+ ([0-9:]+)') as hora
      ,split_part(nt.chartdate::text,' ',1) as fecha

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_extra_value_troponin_t as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,fecha
      ,fecha_hora
      ,hora
      ,case
      when hora!='' then cast(concat(fecha, ' ', hora,':00') as timestamp)
      else nt.chartdate end as charttime
      ,tropo_slash
      ,tropo_onevalue
      ,tropo_menor
      ,tropo_slash_menor
      ,tropo_as_texto
      ,tropo_espacio
      ,tropo_espacio2
      ,GREATEST(tropo_slash, tropo_onevalue,tropo_menor,tropo_slash_menor,
                tropo_as_texto,tropo_espacio,tropo_espacio2) as value_trop_t

FROM temp_extract_dosis nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id  as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,charttime as "CHARTTIME"
      ,value_trop_t::double precision as "VALUE"
      ,case
        when value_trop_t!='' then 227429
      else null end as "ITEMID"
      ,case
        when value_trop_t!='' then 'ng/ml'
      else null end as "VALUEUOM"

FROM temp_extra_value_troponin_t nt
where value_trop_t is not null
order by nt.subject_id;

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
from physician_temp1_troponin_ns
order by "SUBJECT_ID";

--**************************************
--Paso4: extracción de troponin from clinical notes
--1) from nursing notes
drop view nursing_troponin_ns cascade;
create view nursing_troponin_ns as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nursing'
),
temp_sin_espacios as(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,replace(replace(replace(texto, ' ', '*^'), '^*', ''), '*^', ' ') as texto_final
FROM temp_note_sin_saltos nt
order by nt.subject_id
),
temp_extract_dosis as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,substring(LOWER(texto_final), 'troponin ([0-9:]+.[0-9:]+)') as tropo_slash
      ,substring(LOWER(texto_final), 'trop ([0-9:]+.[0-9:]+)') as tropo_onevalue
      ,substring(LOWER(texto_final), 'troponin<([0-9:].[0-9:]+)') as tropo_menor
      ,substring(LOWER(texto_final), 'trop t: ([0-9:]+.[0-9:]+)') as tropo_min
      ,substring(LOWER(texto_final), 'troponin of ([0-9:].[0-9:]+)') as tropo_as_texto
      ,substring(LOWER(texto_final), 'troponin (.[0-9:]+)') as tropo_punto
      ,substring(LOWER(texto_final), 'troponin ([0-9:].[0-9:]+)') as tropo_punto2
      ,substring(LOWER(texto_final), 'troponin: ([0-9:]+.[0-9:]+)') as tropo_dospuntos
      ,substring(LOWER(texto_final), 'trop >([0-9:]+)') as trop_mayor
      ,substring(LOWER(texto_final), 'troponin \^([0-9].[0-9:]+)') as trop_casa

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_extra_value_troponin_t as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,tropo_slash,tropo_onevalue,tropo_menor,tropo_min,trop_casa,tropo_punto2,
                tropo_as_texto,tropo_punto,tropo_dospuntos,trop_mayor
      ,GREATEST(tropo_slash,tropo_onevalue,tropo_menor,tropo_min,trop_casa,tropo_punto2,
                tropo_as_texto,tropo_punto,tropo_dospuntos,trop_mayor) as value_trop_t

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_replace as (
  select nt.subject_id
        ,nt.hadm_id
        ,nt.chartdate
        ,value_trop_t
        ,tropo_slash,tropo_onevalue,tropo_menor,tropo_min,trop_casa,tropo_punto2,
                  tropo_as_texto,tropo_punto,tropo_dospuntos,trop_mayor
        ,REPLACE(REPLACE(REPLACE(REPLACE(value_trop_t,'>',''),'^',''),'<',''),'#','') as value
FROM temp_extra_value_troponin_t nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id  as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case
        when value!='' then nt.chartdate
      else null end as "CHARTTIME"
      ,case
        when value!='' then 227429
      else null end as "ITEMID"
      ,value::double precision  as "VALUE"
      ,case
        when value!='' then 'ng/ml'
      else null end as "VALUEUOM"

FROM temp_replace nt
where value is not null
order by nt.subject_id;

--*****************************************************
--Paso4: extracción de troponin from clinical notes
--1) from nursing_other notes
drop view nursing_other_troponin_ns cascade;
create view nursing_other_troponin_ns as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nursing/other'
),
temp_sin_espacios as(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,replace(replace(replace(texto, ' ', '*^'), '^*', ''), '*^', ' ') as texto_final
FROM temp_note_sin_saltos nt
order by nt.subject_id
),
temp_extract_dosis as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,substring(LOWER(texto_final), 'troponin ([0-9:]+.[0-9:]+)') as tropo_slash
      ,substring(LOWER(texto_final), 'trop ([0-9:]+.[0-9:]+)') as tropo_onevalue
      ,substring(LOWER(texto_final), 'troponin<([0-9:].[0-9:]+)') as tropo_menor
      ,substring(LOWER(texto_final), 'trop t: ([0-9:]+.[0-9:]+)') as tropo_min
      ,substring(LOWER(texto_final), 'troponin of ([0-9:].[0-9:]+)') as tropo_as_texto
      ,substring(LOWER(texto_final), 'troponin (.[0-9:]+)') as tropo_punto
      ,substring(LOWER(texto_final), 'troponin ([0-9:].[0-9:]+)') as tropo_punto2
      ,substring(LOWER(texto_final), 'troponin: ([0-9:]+.[0-9:]+)') as tropo_dospuntos
      ,substring(LOWER(texto_final), 'trop >([0-9:]+)') as trop_mayor
      ,substring(LOWER(texto_final), 'trop>([0-9:]+)') as trop_mayor2
      ,substring(LOWER(texto_final), 'troponin \^([0-9].[0-9:]+)') as trop_casa

      ----roponin >25

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_extra_value_troponin_t as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,GREATEST(tropo_slash,tropo_onevalue,tropo_menor,tropo_min,trop_casa,tropo_punto2,
                tropo_as_texto,tropo_punto,tropo_dospuntos,trop_mayor,trop_mayor2) as value_trop_t

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_replace as (
  select nt.subject_id
        ,nt.hadm_id
        ,nt.chartdate
        ,value_trop_t
        ,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(value_trop_t,'>',''),'^',''),'<',''),'#',''),'q','') as value
FROM temp_extra_value_troponin_t nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id  as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case
        when value!='' then nt.chartdate
      else null end as "CHARTTIME"
      ,case
        when value!='' then 227429
      else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case
        when value!='' then 'ng/ml'
      else null end as "VALUEUOM"

FROM temp_replace nt
where value is not null
order by nt.subject_id;

--******************************************************
--Paso4: extracción de troponin from clinical notes
--1) from general notes
drop view general_troponin_ns cascade;
create view general_troponin_ns as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'General'
),
temp_sin_espacios as(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,replace(replace(replace(texto, ' ', '*^'), '^*', ''), '*^', ' ') as texto_final
FROM temp_note_sin_saltos nt
order by nt.subject_id
),
temp_extract_dosis as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,substring(LOWER(texto_final), 'troponin-t:[0-9:]+/[0-9:]+/([0-9:].[0-9:]+)') as tropo_slash2
      ,substring(LOWER(texto_final), 'troponin ([0-9:]+.[0-9:]+)') as tropo_slash
      ,substring(LOWER(texto_final), 'trop ([0-9:]+.[0-9:]+)') as tropo_onevalue
      ,substring(LOWER(texto_final), 'troponin<([0-9:].[0-9:]+)') as tropo_menor
      ,substring(LOWER(texto_final), 'trop t: ([0-9:]+.[0-9:]+)') as tropo_min
      ,substring(LOWER(texto_final), 'trop-t: ([0-9:]+.[0-9:]+)') as tropo_min2
      ,substring(LOWER(texto_final), 'troponin of ([0-9:].[0-9:]+)') as tropo_as_texto
      ,substring(LOWER(texto_final), 'troponin t:[0-9:]+/[0-9:]+/([0-9:]+.[0-9:]+)')  as tropo_punto
      ,substring(LOWER(texto_final), 'troponin ([0-9:].[0-9:]+)') as tropo_punto2
      ,substring(LOWER(texto_final), 'troponin: ([0-9:]+.[0-9:]+)') as tropo_dospuntos
      ,substring(LOWER(texto_final), 'trop >([0-9:]+)') as trop_mayor
      ,substring(LOWER(texto_final), 'trop>([0-9:]+)') as trop_mayor2
      ,substring(LOWER(texto_final), 'troponin \^([0-9].[0-9:]+)') as trop_casa

--- Troponin T:4839/376/24.20

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_extra_value_troponin_t as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,tropo_slash,tropo_onevalue,tropo_menor,tropo_min,trop_casa,tropo_punto2,tropo_min2,
                tropo_as_texto,tropo_punto,tropo_dospuntos,trop_mayor,trop_mayor2,tropo_slash2
      ,GREATEST(tropo_slash,tropo_onevalue,tropo_menor,tropo_min,trop_casa,tropo_punto2,tropo_min2,
                tropo_as_texto,tropo_punto,tropo_dospuntos,trop_mayor,trop_mayor2,tropo_slash2) as value_trop_t

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_replace as (
  select nt.subject_id
        ,nt.hadm_id
        ,nt.chartdate
        ,tropo_slash,tropo_onevalue,tropo_menor,tropo_min,trop_casa,tropo_punto2,tropo_min2,
                  tropo_as_texto,tropo_punto,tropo_dospuntos,trop_mayor,trop_mayor2,tropo_slash2
        ,value_trop_t
        ,REPLACE(REPLACE(REPLACE(REPLACE(value_trop_t,'>',''),'^',''),'<',''),'#','') as value
FROM temp_extra_value_troponin_t nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id  as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case
        when value!='' then nt.chartdate
      else null end as "CHARTTIME"
      ,case
        when value!='' then 227429
      else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case
        when value!='' then 'ng/ml'
      else null end as "VALUEUOM"

FROM temp_replace nt
where value is not null
order by nt.subject_id;

--**********************************************
--Paso4: extracción de troponin from clinical notes
--1) from rehab services notes
drop view rehab_services_troponin_ns cascade;
create view rehab_services_troponin_ns as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Rehab Services'
),
temp_sin_espacios as(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,replace(replace(replace(texto, ' ', '*^'), '^*', ''), '*^', ' ') as texto_final
FROM temp_note_sin_saltos nt
order by nt.subject_id
),
temp_extract_dosis as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,substring(LOWER(texto_final), 'troponin ([0-9:]+.[0-9:]+)') as tropo_slash
      ,substring(LOWER(texto_final), 'trop ([0-9:]+.[0-9:]+)') as tropo_onevalue
      ,substring(LOWER(texto_final), 'troponin<([0-9:].[0-9:]+)') as tropo_menor
      ,substring(LOWER(texto_final), 'trop t: ([0-9:]+.[0-9:]+)') as tropo_min
      ,substring(LOWER(texto_final), 'troponin of ([0-9:].[0-9:]+)') as tropo_as_texto
      ,substring(LOWER(texto_final), 'troponin (.[0-9:]+)') as tropo_punto
      ,substring(LOWER(texto_final), 'troponin ([0-9:].[0-9:]+)') as tropo_punto2
      ,substring(LOWER(texto_final), 'troponin: ([0-9:]+.[0-9:]+)') as tropo_dospuntos
      ,substring(LOWER(texto_final), 'trop >([0-9:]+)') as trop_mayor
      ,substring(LOWER(texto_final), 'trop>([0-9:]+)') as trop_mayor2
      ,substring(LOWER(texto_final), 'troponin \^([0-9].[0-9:]+)') as trop_casa

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_extra_value_troponin_t as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,GREATEST(tropo_slash,tropo_onevalue,tropo_menor,tropo_min,trop_casa,tropo_punto2,
                tropo_as_texto,tropo_punto,tropo_dospuntos,trop_mayor,trop_mayor2) as value_trop_t

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_replace as (
  select nt.subject_id
        ,nt.hadm_id
        ,nt.chartdate
        ,value_trop_t
        ,REPLACE(REPLACE(value_trop_t,'>',''),'^','') as value
FROM temp_extra_value_troponin_t nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id  as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case
        when value!='' then nt.chartdate
      else null end as "CHARTTIME"
      ,case
        when value!='' then 227429
      else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case
        when value!='' then 'ng/ml'
      else null end as "VALUEUOM"

FROM temp_replace nt
where value is not null
order by nt.subject_id;

--*******************************************************
--**********************************************
--Paso4: extracción de troponin from clinical notes
--1) from consult notes
drop view consult_troponin_ns cascade;
create view consult_troponin_ns as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Consult'
),
temp_sin_espacios as(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,replace(replace(replace(texto, ' ', '*^'), '^*', ''), '*^', ' ') as texto_final
FROM temp_note_sin_saltos nt
order by nt.subject_id
),
temp_extract_dosis as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,substring(LOWER(texto_final), 'troponin-t:[0-9:]+/[0-9:]+/([0-9:].[0-9:]+)') as tropo_slash
      ,substring(LOWER(texto_final), 'tropt ([0-9:]+.[0-9:]+)') as tropo_onevalue

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_extra_value_troponin_t as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,GREATEST(tropo_slash,tropo_onevalue) as value_trop_t

FROM temp_extract_dosis nt
order by nt.subject_id
),
temp_replace as (
  select nt.subject_id
        ,nt.hadm_id
        ,nt.chartdate
        ,value_trop_t
        ,REPLACE(REPLACE(value_trop_t,'>',''),'^','') as value
FROM temp_extra_value_troponin_t nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id  as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case
        when value!='' then nt.chartdate
      else null end as "CHARTTIME"
      ,case
        when value!='' then 227429
      else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case
        when value!='' then 'ng/ml'
      else null end as "VALUEUOM"

FROM temp_replace nt
where value is not null
order by nt.subject_id;
--*************************************************
--**********************************************
--Extracción de troponin from discharge notes
drop view discharge_troponin_ns cascade;
create view discharge_troponin_ns as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Discharge summary'
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
     ,regexp_matches(LOWER(texto_final), '[c]?trop[n]?[t]?[-]?[>]?[<]? ?([0-9:]+.[0-9:]+[\*]?[0-9]? [\[\]0-9*-]+ [0-9:]+)', 'g') as tropo
     ,regexp_matches(LOWER(texto_final), '[c]?trop[n]?[t]?[onin was ]?[-]?([0-9:]+.[0-9:]+)', 'g') as tropo1
     ,regexp_matches(LOWER(texto_final), 'trop[o]?[n]?[i]?[n]?[t]? ?[w]?[a]?[s]?[o]?[f]?[t]?[o]? [~]?([0-9:]+.[0-9:]+)', 'g') as tropo2

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,tropo
      ,tropo1
      ,tropo2
      ,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(tropo::text,'*',''),'{"',''),'"}',''),'[',''),']','') as tropo_clean
      ,REPLACE(REPLACE(REPLACE(REPLACE(tropo1::text,'{',''),'}',''),'": ',''),'"','') as tropo1_clean
      ,REPLACE(REPLACE(tropo2::text,'{',''),'}','') as tropo2_clean

FROM temp_extract_fecha_hora_valor nt
order by nt.subject_id
),
temp_split_tropo as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,tropo_clean
      ,tropo1_clean
      ,tropo2_clean

      ,split_part(tropo_clean::text,' ',1) as tropo_valor
      ,split_part(tropo_clean::text,' ',2) as tropo_fecha
      ,split_part(tropo_clean::text,' ',3) as tropo_hora
      ,EXTRACT(YEAR FROM nt.chartdate) as temp_year

from temp_replace_caracteres nt
order by nt.subject_id
),
temp_split_fecha as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,tropo_clean
      ,tropo1_clean
      ,tropo2_clean
      ,tropo_valor
      ,tropo_fecha
      ,temp_year

      ,split_part(tropo_fecha::text,'-',1) as ano
      ,split_part(tropo_fecha::text,'-',2) as mes
      ,split_part(tropo_fecha::text,'-',3) as dia
      ,concat(tropo_hora,':00') as date_hour

from temp_split_tropo nt
order by nt.subject_id
),
temp_join_charttime as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,tropo_clean
      ,tropo_valor
      ,tropo1_clean
      ,tropo2_clean
      ,date_hour
      ,ano,mes,dia
      ,tropo_fecha
      ,temp_year
      ,case when char_length(mes)=1 then concat('0',mes) else mes end as mes_final
      ,case when char_length(dia)=1 then concat('0',dia) else dia end as dia_final

from temp_split_fecha nt
order by nt.subject_id
),
temp_sinfecha_seleccionar_valormayor as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,tropo_valor
      ,case when char_length(ano)<4 then concat(temp_year,'-','0',ano,'-',mes,' ',date_hour ) else concat(ano,'-',mes_final,'-',dia_final,' ',date_hour ) end as charttime1

      ,ano,mes,dia
      ,tropo1_clean
      ,tropo2_clean
      ,tropo_fecha
      ,case when tropo_valor!='' then 0 else 1 end as flag

from temp_join_charttime nt
order by nt.subject_id
),
temp_add_new_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,tropo_valor::double precision
      ,charttime1
      ,ano,mes,dia
      ,flag
      ,tropo_fecha
      ,case when flag=1 then GREATEST(tropo1_clean::double precision,tropo2_clean::double precision) end as value_GREATEST
      ,case when flag=1 then nt.chartdate + '1 hour'::interval end as charttime2
FROM temp_sinfecha_seleccionar_valormayor nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case when flag=0 then charttime1::timestamp else charttime2::timestamp end as "CHARTTIME"
      ,case when (flag=0 or flag=1) then 227429 else null end as "ITEMID"
      ,case when flag=0 then tropo_valor::double precision else value_GREATEST::double precision end as "VALUE"
      ,case when (flag=0 or flag=1) then 'ng/ml' else null end as "VALUEUOM"

    
FROM temp_add_new_value nt
order by nt.subject_id;

--*************************************
--Paso N: join views

drop view troponin_data_ns cascade;
create view troponin_data_ns as

select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from physician_troponin_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from physician_troponin_regex_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from nursing_troponin_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from nursing_other_troponin_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from general_troponin_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from rehab_services_troponin_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from consult_troponin_ns
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from discharge_troponin_ns
order by "SUBJECT_ID";

select * from troponin_data_ns;
