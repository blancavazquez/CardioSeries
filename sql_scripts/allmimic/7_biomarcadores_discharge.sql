--**********************************************
--Dada las regex obtenidas previamente por los 4 biomarcadores (ckmb, ck,troponin, mb)
--se generará una vista por cada regex y luego se uniran por biomarcador
--***********************************************************************************

--***********************************************************************************
drop view ckmb_discharge_temp1;
drop view ckmb_discharge_temp2;
drop view ckmb_discharge_temp3;
drop view ck_discharge_temp1;
drop view ck_discharge_temp2;
drop view troponin_discharge_temp1;
drop view troponin_discharge_temp2
drop view troponin_discharge_temp3
drop view troponin_discharge_temp4
drop view mioglobina_discharge_temp1
drop view mioglobina_discharge_temp2
drop view mioglobina_discharge_temp3

--Regex 1: creatinine kinase fracción mb
drop view ckmb_discharge_temp1 cascade;
create view ckmb_discharge_temp1 as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
--inner join STEMI_patients st
--on nt.subject_id = st.subject_id
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
      ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?[a-z]+? ?ck[-]?mb[-]? ?[>]?[0-9:]+)', 'g') as ckmb

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,ckmb
      ,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(ckmb::text,'[',''),']',''),'{"',''),'"}',''),'*',''),'blood',''),'>',''),'  ',' '),'ckmb ','ck-mb-'),'ck-mb ','ck-mb-') as ckmb_clean
FROM temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,ckmb
      ,REPLACE(ckmb_clean,'- ','-') as ckmb_clean2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,ckmb
      ,split_part(nt.ckmb_clean2,' ',1) as fecha
      ,split_part(nt.ckmb_clean2,' ',2) as hora
      ,split_part(nt.ckmb_clean2,' ',3) as value_previo
      

      -- ,value   
FROM temp_asignacion_value nt
--where value!=0
order by nt.subject_id
),
temp_process_fecha_hora as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,ckmb
      ,fecha
      ,REPLACE(REPLACE(hora,'1:23a','1:23am'),'amm','am') as hora2
      ,value_previo
      ,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(value_previo,'ck-mb-',''),'ckmb-',''),'ckmb',''),'','0'),'ck-mb:','')as value
FROM temp_seleccion_value_empty nt
order by nt.subject_id
),
temp_concat_fecha_hora as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,ckmb
      ,fecha
      ,hora2
      ,value
      --,case when (value>0) then value end as value2
      --,case when ((value>0 and char_length(fecha)>7)) then concat(fecha,' ',hora2)::timestamp else nt.chartdate end as date_time
FROM temp_process_fecha_hora nt
where value!=''
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case when ((value::double precision>0 and char_length(fecha)>7)) then concat(fecha,' ',hora2)::timestamp 
                  else nt.chartdate end as "CHARTTIME"
      ,case when (value::double precision>0) then 50911 else null end as "ITEMID"
      ,case when (value::double precision>0) then value::double precision end as "VALUE"
      ,case when (value::double precision>0) then 'ng/ml' else null end as "VALUEUOM"
      
FROM temp_concat_fecha_hora nt
where value::double precision>0
order by nt.subject_id;

--***********************************************************************************
--Regex 2: creatinine kinase fracción mb
drop view ckmb_discharge_temp2 cascade;
create view ckmb_discharge_temp2 as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
--inner join STEMI_patients st
--on nt.subject_id = st.subject_id
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
      ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ck[(]?[c]?[p]?[k]?[)]?[-]?[[0-9:]+]?[*]? ?ck-mb[-]? ?[>]?[0-9:]+)', 'g') as ckmb

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,ckmb
      ,regexp_matches(ckmb::text, 'ck-mb[-]? ?[>]?([0-9:]+)', 'g') as ckmb_clean
      ,split_part(nt.ckmb::text,' ',1) as fecha
      ,split_part(nt.ckmb::text,' ',2) as hora

from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,ckmb
      ,fecha,hora

      ,REPLACE(REPLACE(ckmb_clean::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2  

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,ckmb
      ,value
      ,fecha2,hora
      ,case when ((value>0 and char_length(fecha2)>7 and char_length(hora)<=7)) then concat(fecha2,' ',hora)::timestamp else nt.chartdate end as date_time
 
FROM temp_asignacion_value nt
order by nt.subject_id
),
temp_concat_fecha_hora as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,ckmb
      ,value
      ,fecha2,hora,date_time

FROM temp_seleccion_value_empty nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 50911 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"
      
FROM temp_concat_fecha_hora nt
where value::double precision>0
order by nt.subject_id;
--***********************************************************************************
--Regex 3: creatinine kinase fracción mb
drop view ckmb_discharge_temp3 cascade;
create view ckmb_discharge_temp3 as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
--inner join STEMI_patients st
--on nt.subject_id = st.subject_id
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
      ,regexp_matches(LOWER(texto_final), '[,]? ?ck[-]?mb[-]?[>]? ?[a-z]*? ([0-9:]+)', 'g') as ckmb
      ,regexp_matches(LOWER(texto_final), 'pertinent results: ?([\[\]0-9*-]+)', 'g') as fecha_proxima
      ,regexp_matches(LOWER(texto_final), 'pertinent results: ?[a-z:]+? ?[a-z:]+?[:]? ?([\[\]0-9*-]+)', 'g') as fecha_proxima2

FROM temp_sin_espacios nt
order by nt.subject_id
), temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate--::timestamp
      ,ckmb
      ,REPLACE(REPLACE(REPLACE(ckmb::text,'{',''),'}',''),':','') as value
      ,REPLACE(REPLACE(REPLACE(fecha_proxima::text,'{[**',''),'**]}',''),'*','') as fecha_proxima_clean
      ,REPLACE(REPLACE(REPLACE(fecha_proxima2::text,'{[**',''),'**]}',''),'**]-}','') as fecha_proxima2_clean

FROM temp_extract_fecha_hora_valor nt
order by nt.subject_id
),
temp_value_major_0 as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,value
      ,case when (char_length(fecha_proxima_clean)>7) then fecha_proxima_clean end as fecha_proxima_clean2
      ,case when (char_length(fecha_proxima2_clean)>7) then fecha_proxima2_clean end as fecha_proxima2_clean2


FROM temp_replace_caracteres nt
order by nt.subject_id
),
temp_asignacion_fecha as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,value::double precision
      ,fecha_proxima_clean2,fecha_proxima2_clean2
      ,case when (char_length(fecha_proxima2_clean2::text)<=10) then fecha_proxima2_clean2::timestamp else fecha_proxima_clean2::timestamp end as fecha_final


FROM temp_value_major_0 nt
where value!=''
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case when (char_length(fecha_final::text)>0) then fecha_final::timestamp 
                  else nt.chartdate::timestamp end as "CHARTTIME"
      ,case when (value>0) then 50911 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"

FROM temp_asignacion_fecha nt
where value::double precision>0
order by nt.subject_id;


--***********************************************************************************
--Regex 1: creatinine kinase
drop view ck_discharge_temp1 cascade;
create view ck_discharge_temp1 as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
--inner join STEMI_patients st
--on nt.subject_id = st.subject_id
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
      ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?ck[-]?[>]? ?[>]?[0-9:]+)', 'g') as ck

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,ck
      ,regexp_matches(ck::text, 'ck[-]?[>]? ?[>]?([0-9:]+)', 'g') as ck_clean
      ,regexp_matches(ck::text, '([\[\]0-9*-]+)', 'g') as fecha
      ,regexp_matches(ck::text, '[\[\]0-9*-]+ ?([0-9:]+[p]?[m]?[a]?[m]?)', 'g') as hora

      --,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(ck::text,'[',''),']',''),'{"',''),'"}',''),'*',''),'blood',''),'>',''),'  ',' '),'2am','02:00am'),'6:44a','6:44am') as ck_clean

FROM temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,ck
      ,ck_clean
      ,fecha
      ,hora
      ,REPLACE(REPLACE(REPLACE(ck_clean::text,'{',''),'}',''),':','') as value
      ,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(fecha::text,'{',''),'}',''),'*',''),'[',''),']','') as fecha2
      ,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(hora::text,'{',''),'}',''),'11:00a','11:00am'),
            '04:29a','04:29am'),'2:38p','2:38pm'),'12:29p','12:29pm'),'8:35p','8:35pm'),'6:44a','6:44am'),'9:44p','9:44pm'),
             '4:47a','4:47am'),'7:50p','7:50pm'),'7:15p','7:15pm'),'7:20p','7:20pm'),'7:45p','7:45pm'),'2:00p','2:00pm'),
            '430pm','04:30pm')as hora2

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,ck
      ,ck_clean
      ,fecha
      ,hora
      ,fecha2,hora2
      ,value
      ,case when (char_length(value::text)>0) then value::double precision end as value2
      ,case when (char_length(hora2::text)>4) then hora2 end as hora3  
        
FROM temp_asignacion_value nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case when ((value2>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora3)::timestamp 
                  else nt.chartdate end as "CHARTTIME"  
      ,case when (value2>0) then 50910 else null end as "ITEMID"
      ,value2::double precision as "VALUE"
      ,case when (value2>0) then 'iu/l' else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
where value2::double precision>0
order by nt.subject_id;


--***********************************************************************************
--Regex 2: creatinine kinase
drop view ck_discharge_temp2 cascade;
create view ck_discharge_temp2 as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
--inner join STEMI_patients st
--on nt.subject_id = st.subject_id
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
      ,regexp_matches(LOWER(texto_final), 'ck[-]?[>]? ?[a-z]*? ([0-9:]+)', 'g') as ck
      ,regexp_matches(LOWER(texto_final), 'pertinent results: ?([\[\]0-9*-]+)', 'g') as fecha_proxima
      ,regexp_matches(LOWER(texto_final), 'pertinent results: ?[a-z:]+? ?[a-z:]+?[:]? ?([\[\]0-9*-]+)', 'g') as fecha_proxima2

      ---,regexp_matches(ck::text, 'ck[-]?[>]? ?[>]?([0-9:]+)', 'g') as ck_clean

FROM temp_sin_espacios nt
order by nt.subject_id
), temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate--::timestamp
      ,ck
      ,fecha_proxima2
      ,REPLACE(REPLACE(REPLACE(ck::text,':',''),'{',''),'}','') as value
      ,REPLACE(REPLACE(REPLACE(fecha_proxima::text,'{[**',''),'**]}',''),'*','') as fecha_proxima_clean
      ,REPLACE(REPLACE(REPLACE(fecha_proxima2::text,'{[**',''),'**]}',''),'**]-}','') as fecha_proxima2_clean

FROM temp_extract_fecha_hora_valor nt
order by nt.subject_id
),
temp_value_major_0 as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,ck, value
      ,fecha_proxima_clean
      ,fecha_proxima2_clean
      ,case when (char_length(fecha_proxima_clean)>7 and char_length(fecha_proxima_clean)<=10) 
                  then fecha_proxima_clean end as fecha_proxima_clean2
      ,case when (char_length(fecha_proxima2_clean)>7 and char_length(fecha_proxima2_clean)<=10) 
                  then fecha_proxima2_clean end as fecha_proxima2_clean2


FROM temp_replace_caracteres nt
where value!=''
order by nt.subject_id
),
temp_asignacion_fecha as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,value::double precision
      ,case when (char_length(fecha_proxima_clean2)>0) 
                  then fecha_proxima_clean2::timestamp end as fecha_proxima_clean3
      --,fecha_proxima_clean2
      ,fecha_proxima2_clean2
      --,justify_hours(random() * (interval '24 hours'))

FROM temp_value_major_0 nt
order by nt.subject_id
),
temp_fecha_final as(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,value::double precision
      ,case when (char_length(fecha_proxima_clean3::text)>0) then fecha_proxima_clean3::timestamp 
                  else fecha_proxima2_clean2::timestamp end as fecha_final
FROM temp_asignacion_fecha nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case when (char_length(fecha_final::text)>0) then fecha_final::timestamp 
                  else nt.chartdate::timestamp end as "CHARTTIME"
      ,case when (value>0) then 50910 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'iu/l' else null end as "VALUEUOM"

FROM temp_fecha_final nt
where value::double precision>0
order by nt.subject_id;

--******************************************************************************************
--Regex 1: troponin T
drop view troponin_discharge_temp1 cascade;
create view troponin_discharge_temp1 as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
--inner join STEMI_patients st
--on nt.subject_id = st.subject_id
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
      ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?[a-z]+? ?[a-z]+?[-]?[a-z]+?[-]? ?[>]?[0-9:]+? ?[c]?trop[a-z]?[a-z]?[-]? ?[0-9:]+.[0-9:]+)', 'g') as troponin
     
FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,troponin
      ,regexp_matches(troponin::text, 'ctropnt[-]? ?[>]?([0-9:]+.[0-9:]+)', 'g') as troponin_clean
      ,split_part(nt.troponin::text,' ',1) as fecha
      ,split_part(nt.troponin::text,' ',2) as hora

from temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,troponin, troponin_clean,fecha,hora
      ,REPLACE(REPLACE(REPLACE(troponin_clean::text,'{',''),'}',''),':','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2  
      ,REPLACE(REPLACE(hora::text,'{"',''),'ck-mb-','') as hora2 

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,troponin, troponin_clean,fecha2,hora,value,hora2
      ,case when (char_length(fecha2)>7 and char_length(fecha2)<=10) then fecha2 end as fecha3
      ,case when (char_length(hora2)>=7 and char_length(hora2)<=10) then hora2 end as hora3
 
FROM temp_asignacion_value nt
where value::double precision >0
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      --,hora2,hora3,fecha3
      ,case when (char_length(fecha3)>=4 and char_length(hora3)>=4) then concat(fecha3,' ',hora3)::timestamp else nt.chartdate end as "CHARTTIME"
      ,case when (value>0) then 227429 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
where value::double precision>0
order by nt.subject_id;

--*********************************************************************
--Regex2: troponin T
drop view troponin_discharge_temp2 cascade;
create view troponin_discharge_temp2 as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
--inner join STEMI_patients st
--on nt.subject_id = st.subject_id
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
      ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?[c]?trop[a-z]?[a-z]?[-]? ?[>]?[0-9:]+.[0-9:]+)', 'g')  as troponin

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,troponin
      ,regexp_matches(troponin::text, 'ctropnt[-]? ?[>]?([0-9:]+.[0-9:]+)', 'g') as troponin_clean
      ,split_part(nt.troponin::text,' ',1) as fecha
      ,split_part(nt.troponin::text,' ',2) as hora

      --,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(troponin::text,'[',''),']',''),'{"',''),'"}',''),'*',''),'blood',''),'>',''),'  ',' '),'ctropnt ','ck-mb-'),'ck-mb ','ck-mb-') as troponin_clean
FROM temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,troponin,troponin_clean, fecha, hora
      ,REPLACE(REPLACE(REPLACE(troponin_clean::text,'{',''),'}',''),':','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2 

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,value, fecha2, hora
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora)::timestamp else nt.chartdate end as date_time
  
FROM temp_asignacion_value nt
--where value!=0
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 227429 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
where value::double precision>0
order by nt.subject_id;

--******************************************************************
--Regex 3: Troponin t
drop view troponin_discharge_temp3 cascade;
create view troponin_discharge_temp3 as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
--inner join STEMI_patients st
--on nt.subject_id = st.subject_id
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
      ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?[a-z]+? ?[a-z]+?[-]?[a-z]+?[-]? ?[a-z]+? ?[c]?trop[a-z]?[a-z]?[-]? ?[>]?[0-9:]+.[0-9:]+)', 'g')  as troponin

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,troponin
      ,regexp_matches(troponin::text, 'ctropnt[-]? ?[>]?([0-9:]+.[0-9:]+)', 'g') as troponin_clean
      ,split_part(nt.troponin::text,' ',1) as fecha
      ,split_part(nt.troponin::text,' ',2) as hora

      --,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(troponin::text,'[',''),']',''),'{"',''),'"}',''),'*',''),'blood',''),'>',''),'  ',' '),'ctropnt ','ck-mb-'),'ck-mb ','ck-mb-') as troponin_clean
FROM temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,troponin,troponin_clean, fecha, hora
      ,REPLACE(REPLACE(troponin_clean::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2 

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,value, fecha2, hora
      ,case when (value>0 and char_length(fecha2)>7 and char_length(fecha2)<=10 and char_length(hora)<=10) then concat(fecha2,' ',hora)::timestamp else nt.chartdate end as date_time
  
FROM temp_asignacion_value nt
--where value!=0
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 227429 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
where value::double precision>0
order by nt.subject_id;

--*************************************************************************
--Regex 4: troponin t
drop view troponin_discharge_temp4 cascade;
create view troponin_discharge_temp4 as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
--inner join STEMI_patients st
--on nt.subject_id = st.subject_id
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
      ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?[a-z]+? ?[c]?[k]?[-]?[a-z]+?[-]? ?[0-9:]+?[*]? ?[a-z]+? ?[a-z]+?[-]?[0-9:]+?[.]?[0-9:]+?[*]? ?[c]?trop[a-z]?[a-z]?[-]? ?[>]?[0-9:]+.[0-9:]+)', 'g')  as troponin

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,troponin
      ,regexp_matches(troponin::text, 'ctropnt[-]? ?[>]?([0-9:]+.[0-9:]+)', 'g') as troponin_clean
      ,split_part(nt.troponin::text,' ',1) as fecha
      ,split_part(nt.troponin::text,' ',2) as hora

FROM temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,troponin,troponin_clean, fecha, hora
      ,REPLACE(REPLACE(troponin_clean::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(REPLACE(fecha::text,'{"[**',''),'**]',''),'{"-','') as fecha2 

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,value, fecha2, hora
      ,case when (value>0 and char_length(fecha2)>7 and char_length(fecha2)<=10 and char_length(hora)<=10) 
                  then concat(fecha2,' ',hora)::timestamp else nt.chartdate end as date_time
  
FROM temp_asignacion_value nt
--where value!=0
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 227429 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
where value::double precision > 0
order by nt.subject_id;

--*******************************************************************
--Regex 1: mioglobina
drop view mioglobina_discharge_temp1 cascade;
create view mioglobina_discharge_temp1 as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
--inner join STEMI_patients st
--on nt.subject_id = st.subject_id
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
      ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?[a-z]+? ?[a-z]+?[-]?[a-z]+?[-]?[0-9:]+?[*]? ?mb indx[-]?[0-9:]+.[0-9:]+)', 'g') as mb

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,mb
      ,regexp_matches(mb::text, 'mb ?indx[-]? ?[>]?([0-9:]+.[0-9:]+)', 'g') as mb_clean
      ,split_part(nt.mb::text,' ',1) as fecha
      ,split_part(nt.mb::text,' ',2) as hora

      --,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(mb::text,'[',''),']',''),'{"',''),'"}',''),'*',''),'blood',''),'>',''),'  ',' '),'ctropnt ','ck-mb-'),'ck-mb ','ck-mb-') as mb_clean
FROM temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,mb,mb_clean
      mb_clean, fecha, hora
      ,REPLACE(REPLACE(mb_clean::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(REPLACE(fecha::text,'{"[**',''),'**]',''),'{"-','') as fecha2 

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,mb,mb_clean
      ,value, fecha2, hora
      ,case when (value>0 and char_length(fecha2)>7 and char_length(fecha2)<=10 and char_length(hora)<=10) 
                  then concat(fecha2,' ',hora)::timestamp else nt.chartdate end as date_time
  
FROM temp_asignacion_value nt
--where value!=0
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 51091 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
where value::double precision>0
order by nt.subject_id;

--********************************************************************************************************************
--Regex 2: mioglobina
drop view mioglobina_discharge_temp2 cascade;
create view mioglobina_discharge_temp2 as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
--inner join STEMI_patients st
--on nt.subject_id = st.subject_id
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
      ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:+]? ?[0-9:+]?[p]?[m]?[a]?[m]? ?ck[\(]?[a-z]+?[\)]?[-]?[0-9:]+?[*]? ?[a-z]+?[-]?[a-z]+?[-]?[0-9:]+?[*]? mb ?[a-z]+?[-]?[0-9:]+?[.]?[0-9:]+)', 'g') as mb


FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,mb
      ,regexp_matches(mb::text, 'mb ?indx[-]? ?[>]?([0-9:]+.[0-9:]+)', 'g') as mb_clean
      ,split_part(nt.mb::text,' ',1) as fecha

FROM temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,mb,mb_clean, fecha
      ,REPLACE(REPLACE(mb_clean::text,'{',''),'}','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2 

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,mb,value, fecha, fecha2
      ,case when ((value>0 and char_length(fecha2)>7)) then fecha2::timestamp else nt.chartdate + '2 hour'::interval end as date_time
  
FROM temp_asignacion_value nt
--where value!=0
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 51091 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
order by nt.subject_id;
--*****************************************************************************************
--Regex 3: mioglobina
drop view mioglobina_discharge_temp3 cascade;
create view mioglobina_discharge_temp3 as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
--inner join STEMI_patients st
--on nt.subject_id = st.subject_id
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
      ,regexp_matches(LOWER(texto_final),'mb [a-z]*? ?([0-9:]+[.]?[0-9:]+)','g') as mb


FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_replace_caracteres as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,mb
      ,REPLACE(REPLACE(REPLACE(mb::text,'{',''),'}',''),':','')::double precision as value

FROM temp_extract_fecha_hora_valor nt
order by nt.subject_id
),temp_asignacion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp
      ,mb, value

FROM temp_replace_caracteres nt
order by nt.subject_id
),temp_seleccion_value_empty as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,mb, value
      ,case when (value>0) then nt.chartdate + '2 hour'::interval end as date_time
  
FROM temp_asignacion_value nt
--where value!=0
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,date_time as "CHARTTIME"
      ,case when (value>0) then 51091 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"
      
FROM temp_seleccion_value_empty nt
where value::double precision >0
order by nt.subject_id;

--*************************************
--Paso N: join views

drop view biomarcadores_discharge_stemi cascade;
create view biomarcadores_discharge_stemi as

select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from ckmb_discharge_temp1 --ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from ckmb_discharge_temp2 --ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from ckmb_discharge_temp3--ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from ck_discharge_temp1--ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from ck_discharge_temp2--ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from troponin_discharge_temp1--ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from troponin_discharge_temp2--ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from troponin_discharge_temp3--ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from troponin_discharge_temp4--ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from mioglobina_discharge_temp1--ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from mioglobina_discharge_temp2 --ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from mioglobina_discharge_temp3--ok
order by "SUBJECT_ID";

--select * from biomarcadores_discharge_stemi;
