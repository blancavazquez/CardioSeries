--**********************************************
--Dada las regex obtenidas previamente para las variables de laboratorio from nutrition notes
--se generará una vista por cada regex y luego se uniran por variable
--***********************************************************************************

--***********************************************************************************
--Regex: glucose
drop view glucose_nutrition_nstemi cascade;
create view glucose_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(glucose ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as glucose

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,glucose
      ,regexp_matches(glucose::text, 'glucose ?([0-9:]+)', 'g') as glucose_value
      ,split_part(nt.glucose::text,' ',3) as unit
      ,split_part(nt.glucose::text,' ',4) as fecha
      ,split_part(nt.glucose::text,' ',5) as hora
      ,split_part(nt.glucose::text,' ',6) as timezone

from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,glucose
      ,fecha,hora, unit

      ,REPLACE(REPLACE(glucose_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,glucose
      ,value, unit
      ,fecha2,hora
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 50809 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
--------------------------------------------------------------------------
--Regex: glucose finger stick
drop view glucose_finger_nutrition_nstemi cascade;
create view glucose_finger_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(glucose finger stick ?[0-9:]+ ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as glucose_finger

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,glucose_finger
      ,regexp_matches(glucose_finger::text, 'glucose finger stick ?([0-9:]+)', 'g') as glucose_finger_value
      ,split_part(nt.glucose_finger::text,' ',5) as fecha
      ,split_part(nt.glucose_finger::text,' ',6) as hora
      ,split_part(nt.glucose_finger::text,' ',7) as timezone

from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,glucose_finger
      ,fecha,hora

      ,REPLACE(REPLACE(glucose_finger_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,glucose_finger
      ,value
      ,fecha2,hora
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 225664 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'mg/dl' else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
------------------------------------
Regex: bun
drop view bun_nutrition_nstemi cascade;
create view bun_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(bun ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as bun

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,bun
      ,regexp_matches(bun::text, 'bun ?([0-9:]+)', 'g') as bun_value
      ,split_part(nt.bun::text,' ',3) as unit
      ,split_part(nt.bun::text,' ',4) as fecha
      ,split_part(nt.bun::text,' ',5) as hora
      ,split_part(nt.bun::text,' ',6) as timezone


      ---Glucose Finger Stick    75    [**2194-10-28**] 03:30 PM

from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,bun
      ,fecha,hora,unit,timezone

      ,REPLACE(REPLACE(bun_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,bun
      ,value
      ,fecha2,hora,unit,timezone
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 51006 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
-------------------------------------
--Regex: creatinine
drop view creatinine_nutrition_nstemi cascade;
create view creatinine_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(creatinine ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as creatinine

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,creatinine
      ,regexp_matches(creatinine::text, 'creatinine ?([0-9:]+.[0-9:]+)', 'g') as creatinine_value
      ,split_part(nt.creatinine::text,' ',3) as unit
      ,split_part(nt.creatinine::text,' ',4) as fecha
      ,split_part(nt.creatinine::text,' ',5) as hora
      ,split_part(nt.creatinine::text,' ',6) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,creatinine
      ,fecha,hora,unit,timezone

      ,REPLACE(REPLACE(creatinine_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,creatinine
      ,value
      ,fecha2,hora,unit,timezone
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 50912 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
------------------------------------------
--Regex sodium
drop view sodium_nutrition_nstemi cascade;
create view sodium_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(sodium ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as sodium

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,sodium
      ,regexp_matches(sodium::text, 'sodium ?([0-9:]+)', 'g') as sodium_value
      ,split_part(nt.sodium::text,' ',3) as unit
      ,split_part(nt.sodium::text,' ',4) as fecha
      ,split_part(nt.sodium::text,' ',5) as hora
      ,split_part(nt.sodium::text,' ',6) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,sodium
      ,fecha,hora,unit,timezone

      ,REPLACE(REPLACE(sodium_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,sodium
      ,value
      ,fecha2,hora,unit,timezone
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 50983 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
-------------------------------------
--Regex: potassium
drop view potassium_nutrition_nstemi cascade;
create view potassium_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(potassium ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as potassium

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,potassium
      ,regexp_matches(potassium::text, 'potassium ?([0-9:]+.[0-9:]+)', 'g') as potassium_value
      ,split_part(nt.potassium::text,' ',3) as unit
      ,split_part(nt.potassium::text,' ',4) as fecha
      ,split_part(nt.potassium::text,' ',5) as hora
      ,split_part(nt.potassium::text,' ',6) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,potassium
      ,fecha,hora,unit,timezone

      ,REPLACE(REPLACE(potassium_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,potassium
      ,value
      ,fecha2,hora,unit,timezone
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 50971 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
-------------------------------------------
--Regex: chloride
drop view chloride_nutrition_nstemi cascade;
create view chloride_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(chloride ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as chloride

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,chloride
      ,regexp_matches(chloride::text, 'chloride ?([0-9:]+)', 'g') as chloride_value
      ,split_part(nt.chloride::text,' ',3) as unit
      ,split_part(nt.chloride::text,' ',4) as fecha
      ,split_part(nt.chloride::text,' ',5) as hora
      ,split_part(nt.chloride::text,' ',6) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,chloride
      ,fecha,hora,unit,timezone

      ,REPLACE(REPLACE(chloride_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,chloride
      ,value
      ,fecha2,hora,unit,timezone
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 50902 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
--------------------------------------
--Regex: totalCO2 (tco2)
drop view total_co2_nutrition_nstemi cascade;
create view total_co2_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(tco2 ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as total_co2

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,total_co2
      ,regexp_matches(total_co2::text, 'tco2 ?([0-9:]+)', 'g') as total_co2_value
      ,split_part(nt.total_co2::text,' ',3) as unit
      ,split_part(nt.total_co2::text,' ',4) as fecha
      ,split_part(nt.total_co2::text,' ',5) as hora
      ,split_part(nt.total_co2::text,' ',6) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,total_co2
      ,fecha,hora,unit,timezone

      ,REPLACE(REPLACE(total_co2_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,total_co2
      ,value
      ,fecha2,hora,unit,timezone
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 50804 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
-------------------------------
--Regex: PO2 arterial 
drop view po2_arterial_nutrition_nstemi cascade;
create view po2_arterial_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(po2 \(arterial\) ?[0-9:]+ ?[a-z:]+? [a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as po2_arterial

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,po2_arterial
      ,regexp_matches(po2_arterial::text, 'po2 \(arterial\) ?([0-9:]+)', 'g') as po2_arterial_value
      --,split_part(nt.po2_arterial::text,' ',3) as unit
      ,split_part(nt.po2_arterial::text,' ',6) as fecha
      ,split_part(nt.po2_arterial::text,' ',7) as hora
      ,split_part(nt.po2_arterial::text,' ',8) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,po2_arterial
      ,fecha,hora,timezone
      --,unit

      ,REPLACE(REPLACE(po2_arterial_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,po2_arterial
      ,value
      ,fecha2,hora,timezone
      --,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 50821 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'mmHg' else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
----------------------------------------
--Regex: pco2 (Partial pressure of carbon dioxide)
drop view pco2_arterial_nutrition_nstemi cascade;
create view pco2_arterial_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(pco2 \(arterial\) ?[0-9:]+ ?[a-z:]+? ?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as pco2_arterial

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,pco2_arterial
      ,regexp_matches(pco2_arterial::text, 'pco2 \(arterial\) ?([0-9:]+)', 'g') as pco2_arterial_value
      --,split_part(nt.pco2_arterial::text,' ',4) as unit
      ,split_part(nt.pco2_arterial::text,' ',6) as fecha
      ,split_part(nt.pco2_arterial::text,' ',7) as hora
      ,split_part(nt.pco2_arterial::text,' ',8) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,pco2_arterial
      ,fecha,hora,timezone
      --,unit

      ,REPLACE(REPLACE(pco2_arterial_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,pco2_arterial
      ,value
      ,fecha2,hora,timezone
      --,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 50818 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'mmHg' else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
-----------------------------------------
--Regex: ph
drop view ph_nutrition_nstemi cascade;
create view ph_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(ph \(arterial\) ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as ph

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,ph
      ,regexp_matches(ph::text, 'ph \(arterial\) ?([0-9:]+.[0-9:]+)', 'g') as ph_value
      ,split_part(nt.ph::text,' ',4) as unit
      ,split_part(nt.ph::text,' ',5) as fecha
      ,split_part(nt.ph::text,' ',6) as hora
      ,split_part(nt.ph::text,' ',7) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,ph
      ,fecha,hora,timezone
      ,unit

      ,REPLACE(REPLACE(ph_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,ph
      ,value
      ,fecha2,hora,timezone
      ,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 50820 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
------------------------------------------
--Regex: co2
drop view co2_nutrition_nstemi cascade;
create view co2_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(co2 \(calc\) arterial ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as co2

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,co2
      ,regexp_matches(co2::text, 'co2 \(calc\) arterial ?([0-9:]+)', 'g') as co2_value
      ,split_part(nt.co2::text,' ',5) as unit
      ,split_part(nt.co2::text,' ',6) as fecha
      ,split_part(nt.co2::text,' ',7) as hora
      ,split_part(nt.co2::text,' ',8) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,co2
      ,fecha,hora,timezone
      ,unit

      ,REPLACE(REPLACE(co2_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,co2
      ,value
      ,fecha2,hora,timezone
      ,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 225698 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
---------------------------------
--Regex: albumin
drop view albumin_nutrition_nstemi cascade;
create view albumin_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(albumin ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as albumin

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,albumin
      ,regexp_matches(albumin::text, 'albumin ?([0-9:]+.[0-9:]+)', 'g') as albumin_value
      ,split_part(nt.albumin::text,' ',5) as unit
      ,split_part(nt.albumin::text,' ',6) as fecha
      ,split_part(nt.albumin::text,' ',7) as hora
      ,split_part(nt.albumin::text,' ',8) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,albumin
      ,fecha,hora,timezone
      ,unit

      ,REPLACE(REPLACE(albumin_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,albumin
      ,value
      ,fecha2,hora,timezone
      ,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 50862 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
-------------------------------------
--Regex: calcium non-ionized
drop view calcium_nutrition_nstemi cascade;
create view calcium_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(calcium non-ionized ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as calcium

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,calcium
      ,regexp_matches(calcium::text, 'calcium non-ionized ?([0-9:]+.[0-9:]+)', 'g') as calcium_value
      ,split_part(nt.calcium::text,' ',4) as unit
      ,split_part(nt.calcium::text,' ',5) as fecha
      ,split_part(nt.calcium::text,' ',6) as hora
      ,split_part(nt.calcium::text,' ',7) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,calcium
      ,fecha,hora,timezone
      ,unit

      ,REPLACE(REPLACE(calcium_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,calcium
      ,value
      ,fecha2,hora,timezone
      ,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 225625 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
---------------------------------------
--Regex: phosphorus
drop view phosphorus_nutrition_nstemi cascade;
create view phosphorus_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(phosphorus ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as phosphorus

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,phosphorus
      ,regexp_matches(phosphorus::text, 'phosphorus ?([0-9:]+.[0-9:]+)', 'g') as phosphorus_value
      ,split_part(nt.phosphorus::text,' ',3) as unit
      ,split_part(nt.phosphorus::text,' ',4) as fecha
      ,split_part(nt.phosphorus::text,' ',5) as hora
      ,split_part(nt.phosphorus::text,' ',6) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,phosphorus
      ,fecha,hora,timezone
      ,unit

      ,REPLACE(REPLACE(phosphorus_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,phosphorus
      ,value
      ,fecha2,hora,timezone
      ,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 225677 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
-------------------------------------
--Regex: magnesium
drop view magnesium_nutrition_nstemi cascade;
create view magnesium_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(magnesium ?[0-9:].[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as magnesium

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,magnesium
      ,regexp_matches(magnesium::text, 'magnesium ?([0-9:]+.[0-9:]+)', 'g') as magnesium_value
      ,split_part(nt.magnesium::text,' ',3) as unit
      ,split_part(nt.magnesium::text,' ',4) as fecha
      ,split_part(nt.magnesium::text,' ',5) as hora
      ,split_part(nt.magnesium::text,' ',6) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,magnesium
      ,fecha,hora,timezone
      ,unit

      ,REPLACE(REPLACE(magnesium_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,magnesium
      ,value
      ,fecha2,hora,timezone
      ,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 220635 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
--------------------------------------
--Regex: alkaline phosphate
drop view alkaline_phosphate_nutrition_nstemi cascade;
create view alkaline_phosphate_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(alkaline phosphate ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as alkaline_phosphate

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,alkaline_phosphate
      ,regexp_matches(alkaline_phosphate::text, 'alkaline phosphate ?([0-9:]+)', 'g') as alkaline_phosphate_value
      ,split_part(nt.alkaline_phosphate::text,' ',4) as unit
      ,split_part(nt.alkaline_phosphate::text,' ',5) as fecha
      ,split_part(nt.alkaline_phosphate::text,' ',6) as hora
      ,split_part(nt.alkaline_phosphate::text,' ',7) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,alkaline_phosphate
      ,fecha,hora,timezone
      ,unit

      ,REPLACE(REPLACE(alkaline_phosphate_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,alkaline_phosphate
      ,value
      ,fecha2,hora,timezone
      ,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 225612 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
--------------------------------------
--Regex: Aspartate Aminotransferase (ast)
drop view ast_nutrition_nstemi cascade;
create view ast_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(ast ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as ast

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,ast
      ,regexp_matches(ast::text, 'ast ?([0-9:]+)', 'g') as ast_value
      ,split_part(nt.ast::text,' ',3) as unit
      ,split_part(nt.ast::text,' ',4) as fecha
      ,split_part(nt.ast::text,' ',5) as hora
      ,split_part(nt.ast::text,' ',6) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,ast
      ,fecha,hora,timezone
      ,unit
      ,REPLACE(REPLACE(ast_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(REPLACE(REPLACE(fecha::text,'{"[**',''),'**]',''),'*',''),'[','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,ast
      ,value
      ,fecha2,hora,timezone
      ,REPLACE(REPLACE(REPLACE(REPLACE(hora2::text,'"',''),'}',''),'*',''),'[','') as hora3
      ,unit
      
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      --,fecha2,hora3
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora3)::timestamp else nt.chartdate end as "CHARTTIME"
 
      --,date_time as 
      ,case when (value>0) then 50878 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;

---------------------------------------
--Regex_ amylase
drop view amylase_nutrition_nstemi cascade;
create view amylase_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(amylase ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as amylase

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,amylase
      ,regexp_matches(amylase::text, 'amylase ?([0-9:]+)', 'g') as amylase_value
      ,split_part(nt.amylase::text,' ',3) as unit
      ,split_part(nt.amylase::text,' ',4) as fecha
      ,split_part(nt.amylase::text,' ',5) as hora
      ,split_part(nt.amylase::text,' ',6) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,amylase
      ,fecha,hora,timezone
      ,unit

      ,REPLACE(REPLACE(amylase_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,amylase
      ,value
      ,fecha2,hora,timezone
      ,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 50867 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
--------------------------------------
--Regex: total bilirubin
drop view total_bilirubin_nutrition_nstemi cascade;
create view total_bilirubin_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(total bilirubin ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as total_bilirubin

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,total_bilirubin
      ,regexp_matches(total_bilirubin::text, 'total bilirubin ?([0-9:]+.[0-9:]+)', 'g') as total_bilirubin_value
      ,split_part(nt.total_bilirubin::text,' ',4) as unit
      ,split_part(nt.total_bilirubin::text,' ',5) as fecha
      ,split_part(nt.total_bilirubin::text,' ',6) as hora
      ,split_part(nt.total_bilirubin::text,' ',7) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,total_bilirubin
      ,fecha,hora,timezone
      ,unit

      ,REPLACE(REPLACE(total_bilirubin_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,total_bilirubin
      ,value
      ,fecha2,hora,timezone
      ,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 238277 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
--------------------------------------
--Regex: wbc
drop view wbc_nutrition_nstemi cascade;
create view wbc_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(wbc ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as wbc

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,wbc
      ,regexp_matches(wbc::text, 'wbc ?([0-9:]+.[0-9:]+)', 'g') as wbc_value
      ,split_part(nt.wbc::text,' ',3) as unit
      ,split_part(nt.wbc::text,' ',4) as fecha
      ,split_part(nt.wbc::text,' ',5) as hora
      ,split_part(nt.wbc::text,' ',6) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,wbc
      ,fecha,hora,timezone
      ,unit

      ,REPLACE(REPLACE(wbc_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,wbc
      ,value
      ,fecha2,hora,timezone
      ,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 51301 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
--------------------------------------
--regex: hemoglobin
drop view hgb_nutrition_nstemi cascade;
create view hgb_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(hgb ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as hgb

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,hgb
      ,regexp_matches(hgb::text, 'hgb ?([0-9:]+.[0-9:]+)', 'g') as hgb_value
      ,split_part(nt.hgb::text,' ',3) as unit
      ,split_part(nt.hgb::text,' ',4) as fecha
      ,split_part(nt.hgb::text,' ',5) as hora
      ,split_part(nt.hgb::text,' ',6) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,hgb
      ,fecha,hora,timezone
      ,unit

      ,REPLACE(REPLACE(hgb_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,hgb
      ,value
      ,fecha2,hora,timezone
      ,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 50811 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
--------------------------------------
--regex: hematocrit
drop view hematocrit_nutrition_nstemi cascade;
create view hematocrit_nutrition_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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
      ,regexp_matches(LOWER(texto_final), '(hematocrit ?[0-9:]+.[0-9:]+ [%]? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as hematocrit

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,hematocrit
      ,regexp_matches(hematocrit::text, 'hematocrit ?([0-9:]+.[0-9:]+)', 'g') as hematocrit_value
      ,split_part(nt.hematocrit::text,' ',3) as unit
      ,split_part(nt.hematocrit::text,' ',4) as fecha
      ,split_part(nt.hematocrit::text,' ',5) as hora
      ,split_part(nt.hematocrit::text,' ',6) as timezone


from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,hematocrit
      ,fecha,hora,timezone
      ,unit

      ,REPLACE(REPLACE(hematocrit_value::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2
      ,concat(hora,timezone)  as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,hematocrit
      ,value
      ,fecha2,hora,timezone
      ,unit
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora2)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 51221 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then unit else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;


--*************************************
--Paso N: join views

drop view laboratorio_nutrition_nstemi cascade;
create view laboratorio_nutrition_nstemi as

select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from glucose_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from glucose_finger_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from bun_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from creatinine_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from sodium_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from potassium_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from chloride_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from total_co2_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from po2_arterial_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from pco2_arterial_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from ph_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from co2_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from albumin_nutrition_nstemi--ok 
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from calcium_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from phosphorus_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from magnesium_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from alkaline_phosphate_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from ast_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from amylase_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from total_bilirubin_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from wbc_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from hgb_nutrition_nstemi--ok
UNION ALL
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from hematocrit_nutrition_nstemi--ok
order by "SUBJECT_ID";

--select * from laboratorio_nutrition_nstemi;
