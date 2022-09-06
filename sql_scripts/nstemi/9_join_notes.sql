----script to join all the values extracted from clinical notes
--*****************************************************
--- Paso 1: join all *_data
drop view notes_data_ns cascade;
create view notes_data_ns as

select --- echo notes
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from echo_data_ns --- sign vital data
UNION
select --- general notes
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from general_data_ns ---insulin, heparin, amiodarone
UNION
select --- nursing_other notes
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from all_medicaments_ns -- 4 categories of meds
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_discharge_nstemi
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_physician_nstemi
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_nursing_nstemi
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_nursing_other_nstemi
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_consult_nstemi
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_rehab_services_nstemi
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from biomarcadores_general_nstemi
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"::timestamp
,"ITEMID"
,"VALUE"::double precision
,"VALUEUOM"
from laboratorio_nutrition_nstemi
where "VALUE">0
order by "SUBJECT_ID";
--select * from notes_data;
\copy (SELECT * FROM notes_data_ns) to '/tmp/NOTEDATA_NSTEMI.csv' CSV HEADER;
