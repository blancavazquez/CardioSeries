--**********************************************
--Dada las regex obtenidas previamente por los 4 biomarcadores (ckmb, ck,troponin, mb)
--se generará una vista por cada regex y luego se uniran por biomarcador
--***********************************************************************************
--Regex 1: troponin (nursing other)
drop view nursing_other_troponin_temp1_nstemi cascade;
create view nursing_other_troponin_temp1_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
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
      ,regexp_matches(LOWER(texto_final), 'trop[a-z:]+? ?[-]?[a-z:]? ?[<]?[>]? ?([0-9:]+.[0-9:]+)', 'g') as tropo
      ,regexp_matches(LOWER(texto_final), 'flowsheet data as of ([\[\]0-9*-]+)', 'g') as fecha
      ,regexp_matches(LOWER(texto_final), 'flowsheet data as of [\[\]0-9*-]+ ([0-9:]+)', 'g') as hora

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_extra_value_troponin_t as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,tropo
      ,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(tropo::text,'{',''),'}',''),':',''),'" 0"','0'),'/','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2  
      ,hora 
      
FROM temp_extract_dosis nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id  as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora)::timestamp else nt.chartdate end as "CHARTTIME"
      ,case when (value>0) then 227429 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"

FROM temp_extra_value_troponin_t nt
where value>0
order by nt.subject_id;
--**********************************************************************************

--*****************************************************************************
--Regex 1 : ckmb (nursing)
drop view nursing_other_ckmb_temp1_nstemi cascade;
create view nursing_other_ckmb_temp1_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
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
      ,regexp_matches(LOWER(texto_final), '[,]? ?ck[-]?mb[-]?[>]? ?[a-z]*? ([0-9:]+)', 'g') as ckmb
      ,regexp_matches(LOWER(texto_final), 'flowsheet data as of ([\[\]0-9*-]+)', 'g') as fecha
      ,regexp_matches(LOWER(texto_final), 'flowsheet data as of [\[\]0-9*-]+ ([0-9:]+)', 'g') as hora

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_extra_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,ckmb
      ,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(ckmb::text,'{',''),'}',''),':',''),'"',''),'/',''),' ','')::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2  
      ,hora 
      
FROM temp_extract_dosis nt
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id  as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora)::timestamp else nt.chartdate end as "CHARTTIME"
      ,case when (value>0) then 50911 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"

FROM temp_extra_value nt
where value>0
order by nt.subject_id;

--*******************************************************************************
--Regex 1 : ck ((nursing other))
drop view nursing_other_ck_temp1_nstemi cascade;
create view nursing_other_ck_temp1_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
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
      ,regexp_matches(LOWER(texto_final), 'ck[-]?[>]? ?[a-z]*? ([0-9:]+)', 'g') as ck
      ,regexp_matches(LOWER(texto_final), 'flowsheet data as of ([\[\]0-9*-]+)', 'g') as fecha
      ,regexp_matches(LOWER(texto_final), 'flowsheet data as of [\[\]0-9*-]+ ([0-9:]+)', 'g') as hora

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_extra_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,ck
      ,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(ck::text,'{',''),'}',''),':',''),'"',''),'/',''),' ','')as value--::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2  
      ,hora 
      
FROM temp_extract_dosis nt
where char_length(ck::text)>0
order by nt.subject_id
),
temp_vales_zero as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,ck,value::double precision
      ,fecha2,hora 
      
FROM temp_extra_value nt
where char_length(value::text)>0
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id  as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora)::timestamp else nt.chartdate end as "CHARTTIME"
      ,case when (value>0) then 50910 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"

FROM temp_vales_zero nt
--where value>0
order by nt.subject_id;

--****************************************
--Regex 1 : mb ((nursing other))
drop view nursing_other_mb_temp1_nstemi cascade;
create view nursing_other_mb_temp1_nstemi as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join NSTEMI_patients st
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
      ,regexp_matches(LOWER(texto_final),'(mb [a-z]*? ?[0-9:]+[.]?[0-9:]+)','g') as mb0
      ,regexp_matches(LOWER(texto_final),'mb [a-z]*? ?([0-9:]+.[0-9:]+?)','g') as mb
      ,regexp_matches(LOWER(texto_final), 'flowsheet data as of ([\[\]0-9*-]+)', 'g') as fecha
      ,regexp_matches(LOWER(texto_final), 'flowsheet data as of [\[\]0-9*-]+ ([0-9:]+)', 'g') as hora

FROM temp_sin_espacios nt
order by nt.subject_id
),
temp_extra_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,mb0,mb
      ,REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(mb::text,'{',''),'}',''),':',''),'"',''),'/',''),' ',''),',','.'),'(',''),'>',''),'-','')as value--::double precision as value
      ,REPLACE(REPLACE(fecha::text,'{"[**',''),'**]','') as fecha2  
      ,hora 
      
FROM temp_extract_dosis nt
where char_length(mb::text)>0
order by nt.subject_id
),
temp_vales_zero as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,mb0,mb,value::double precision
      ,fecha2,hora 
      
FROM temp_extra_value nt
where char_length(value::text)>0
order by nt.subject_id
)
select nt.subject_id as "SUBJECT_ID"
      ,nt.hadm_id  as "HADM_ID"
      ,nt.chartdate as "CHARTDATE"
      --,mb0,mb
      ,case when ((value>0 and char_length(fecha2)>7)) then concat(fecha2,' ',hora)::timestamp else nt.chartdate end as "CHARTTIME"
      ,case when (value>0) then 51091 else null end as "ITEMID"
      ,value::double precision as "VALUE"
      ,case when (value>0) then 'ng/ml' else null end as "VALUEUOM"

FROM temp_vales_zero nt
--where value>0
order by nt.subject_id;


--*************************************
--Paso N: join views

drop view biomarcadores_nursing_other_nstemi cascade;
create view biomarcadores_nursing_other_nstemi as

select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from nursing_other_troponin_temp1_nstemi
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from nursing_other_ckmb_temp1_nstemi
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from nursing_other_ck_temp1_nstemi
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from nursing_other_mb_temp1_nstemi
order by "SUBJECT_ID";

--select * from biomarcadores_nursing_other_nstemi;

