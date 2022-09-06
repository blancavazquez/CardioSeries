----script to join all the values extracted from clinical notes
--*****************************************************
--- Paso 1: join all *_data
drop view notes_data cascade;
create view notes_data as

select --- echo notes
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from echo_data --- sign vital data: ok
UNION
select --- general notes
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from general_data ---insulin, heparin, amiodarone ok
UNION
select --- nursing_other notes
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from all_medicaments --ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_discharge_stemi --ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_physician_stemi --ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_nursing_stemi --ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_nursing_other_stemi --ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_consult_stemi --ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_rehab_services_stemi --ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_general_stemi ---ok
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from laboratorio_nutrition_stemi ---ok
where "VALUE">0
order by "SUBJECT_ID";

--select * from notes_data;
\copy (SELECT * FROM notes_data) to '/tmp/NOTEDATA.csv' CSV HEADER;
