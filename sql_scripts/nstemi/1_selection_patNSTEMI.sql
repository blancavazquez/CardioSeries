drop view NSTEMI_patients;
create view NSTEMI_patients as

with icd_codes as
(
  select
    dx.subject_id,
  CASE WHEN dx.icd9_code = '41070' then '1' else '0' END as "41070"
  ,CASE WHEN dx.icd9_code = '41071' then '1' else '0' END as "41071"
  ,CASE WHEN dx.icd9_code = '41072' then '1' else '0' END as "41072"
  from diagnoses_icd dx
  where dx.seq_num ='1'
  group by dx.subject_id, dx.icd9_code
  order by dx.subject_id
)
select distinct subject_id
  from icd_codes
  where
  "41070"='1' or
  "41071"='1' or
  "41072"='1'
  order by subject_id;

select count(distinct subject_id) from NSTEMI_patients;---1711 subject unicos
