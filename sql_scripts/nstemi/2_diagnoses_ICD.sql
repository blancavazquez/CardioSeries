--- Script for extracting information of table of diagnoses_icd and procedures_icd
-- procedures_icd for NSTEMI patients

--**************************************************
-- Paso 1: Extract information from diagnoses_icd
drop view NSTEMI_diagnoses_icd;
create view NSTEMI_diagnoses_icd as
  select
    dx.subject_id
    ,dx.hadm_id
    ,dx.icd9_code
    ,dx.seq_num
  from diagnoses_icd dx
  inner join NSTEMI_patients st
  on dx.subject_id = st.subject_id
  where icd9_code is not null
  order by dx.subject_id;

--**************************************************
-- Paso 2: Extract information from procedures_icd
--- Selection of relevent procedures
drop view NSTEMI_procedures_icd;
create view NSTEMI_procedures_icd as
with temp_item as
(
select
   proc.subject_id
  ,proc.hadm_id
  ,proc.icd9_code
  ,proc.seq_num

from procedures_icd proc
inner join NSTEMI_patients st
on proc.subject_id=st.subject_id

where (proc.icd9_code='3615' or proc.icd9_code='3612' or proc.icd9_code='3613' or
    proc.icd9_code='3961' or proc.icd9_code='9962' or proc.icd9_code='3995' or
    proc.icd9_code='9920' or proc.icd9_code='3607' or proc.icd9_code='3606' or
    proc.icd9_code='45' or proc.icd9_code='9604' or
    proc.icd9_code='3722' or proc.icd9_code='9904' or proc.icd9_code='966' or
    proc.icd9_code='3761' or proc.icd9_code='3723' or proc.icd9_code='3778')
order by proc.subject_id
)

select
     proc.subject_id
    ,proc.hadm_id
    --,proc.icd9_code
    ,case
    when proc.icd9_code = '3615' then 'P3615'
    when proc.icd9_code = '3612' then 'P3612'
    when proc.icd9_code = '3613' then 'P3613'
    when proc.icd9_code = '3961' then 'P3961'
    when proc.icd9_code = '9962' then 'P9962'
    when proc.icd9_code = '3995' then 'P3995'
    when proc.icd9_code = '9920' then 'P9920'
    when proc.icd9_code = '3607' then 'P3607'
    when proc.icd9_code = '3606' then 'P3606'
    when proc.icd9_code = '45' then 'P45'
    when proc.icd9_code = '9604' then 'P9604'
    when proc.icd9_code = '3722' then 'P3722'
    when proc.icd9_code = '9904' then 'P9904'
    when proc.icd9_code = '966' then 'P966'
    when proc.icd9_code = '3761' then 'P3761'
    when proc.icd9_code = '3723' then 'P3723'
    when proc.icd9_code = '3778' then 'P3778'
    else null end as "final_itemid"
    ,proc.seq_num
from temp_item proc
order by proc.subject_id;

--**************************************************
-- Paso 3: Script for joining diagnoses + procedures for NSTEMI patients

drop view NSTEMI_dx_procedures;
create view NSTEMI_dx_procedures as

select
  subject_id
  ,hadm_id
  ,icd9_code
  ,seq_num
from NSTEMI_diagnoses_icd
UNION
select
  subject_id
  ,hadm_id
  ,"final_itemid"
  ,seq_num
from NSTEMI_procedures_icd
order by subject_id;

\copy (SELECT * FROM NSTEMI_dx_procedures) to '/tmp/DIAGNOSES_ICD.csv' CSV HEADER;
