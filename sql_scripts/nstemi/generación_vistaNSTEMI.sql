-- paso 2: Generación de vistas para cada uno de las sub-poblaciones

drop view NSTEMI_admissions;
create view NSTEMI_admissions as
  select
    ad.subject_id
    ,ad.hadm_id
    ,ad.admittime
    ,ad.dischtime
    ,ad.deathtime
    ,ad.admission_type
    ,ad.admission_location
    ,ad.discharge_location
    ,ad.hospital_expire_flag
  from admissions ad
  inner join NSTEMI_patients st
  on ad.subject_id = st.subject_id
  order by ad.subject_id;
\copy (SELECT * FROM NSTEMI_admissions) to '/tmp/NSTEMI_admissions.csv' CSV HEADER;

drop view NSTEMI_chartevents;
create view NSTEMI_chartevents as
  select
    ch.subject_id
    ,ch.hadm_id
    ,ch.icustay_id
    ,ch.itemid
    ,ch.charttime
    ,ch.storetime
    ,ch.value
    ,ch.valuenum
    ,ch.valueuom
    ,ch.warning
    ,ch.error
    ,ch.resultstatus
    ,ch.stopped
  from chartevents as ch
  inner join NSTEMI_patients  st
  on ch.subject_id=st.subject_id
  order by ch. subject_id;
\copy (SELECT * FROM NSTEMI_chartevents) to '/tmp/NSTEMI_chartevents.csv' CSV HEADER;

drop view NSTEMI_diagnoses_icd;
create view NSTEMI_diagnoses_icd as
  select
    dx.subject_id
    ,dx.hadm_id
    ,dx.seq_num
    ,dx.icd9_code
  from diagnoses_icd dx
  inner join NSTEMI_patients st
  on dx.subject_id = st.subject_id
  order by dx.subject_id;
\copy (SELECT * FROM NSTEMI_diagnoses_icd) to '/tmp/NSTEMI_diagnoses_icd.csv' CSV HEADER;

drop view NSTEMI_icustays;
create view NSTEMI_icustays as
  select
    icu.subject_id
    ,icu.hadm_id
    ,icu.icustay_id
    ,icu.first_careunit
    ,icu.last_careunit
    ,icu.intime
    ,icu.outtime
    ,icu.los
    from icustays icu
    inner join NSTEMI_patients st
    on icu.subject_id = st.subject_id
    order by icu.subject_id;
\copy (SELECT * FROM NSTEMI_icustays) to '/tmp/NSTEMI_icustays.csv' CSV HEADER;

drop view NSTEMI_labevents;
create view NSTEMI_labevents as
  select
  lab.subject_id
  ,lab.hadm_id
  ,lab.itemid
  ,lab.charttime
  ,lab.value
  ,lab.valuenum
  ,lab.valueuom
  from labevents lab
  inner join NSTEMI_patients st
  on lab.subject_id = st.subject_id
  order by lab.subject_id;
\copy (SELECT * FROM NSTEMI_labevents) to '/tmp/NSTEMI_labevents.csv' CSV HEADER;

drop view NSTEMI_noteevents;
create view NSTEMI_noteevents as
  select
  nt.subject_id
  ,nt.hadm_id
  ,nt.chartdate
  ,nt.category
  ,nt.description
  ,nt.text
  from noteevents nt
  inner join NSTEMI_patients st
  on nt.subject_id = st.subject_id
  where nt.iserror IS NULL
  order by nt.subject_id;
\copy (SELECT * FROM NSTEMI_noteevents) to '/tmp/NSTEMI_noteevents.csv' CSV HEADER;

drop view NSTEMI_outputevents;
create view NSTEMI_outputevents as
  select
  out.subject_id
  ,out.hadm_id
  ,out.icustay_id
  ,out.charttime
  ,out.itemid
  ,out.value
  ,out.valueuom
  ,out.stopped
  ,out.newbottle
  from outputevents out
  inner join NSTEMI_patients st
  on out.subject_id = st.subject_id
  order by out.subject_id;
\copy (SELECT * FROM NSTEMI_outputevents) to '/tmp/NSTEMI_outputevents.csv' CSV HEADER;

drop view NSTEMI_pat_table;
create view NSTEMI_pat_table as
  SELECT
  pt.subject_id
  ,pt.gender
  ,pt.dob
  ,pt.dod
  ,pt.dod_hosp
  ,pt.dod_ssn
  ,pt.expire_flag
  from patients pt
  inner join NSTEMI_patients st
  on pt.subject_id = st.subject_id
  order by pt.subject_id;
\copy (SELECT * FROM NSTEMI_pat_table) to '/tmp/NSTEMI_pat_table.csv' CSV HEADER;

drop view NSTEMI_prescriptions;
create view NSTEMI_prescriptions as
  select
  pre.subject_id
  ,pre.hadm_id
  ,pre.icustay_id
  ,pre.startdate
  ,pre.enddate
  ,pre.drug_type
  ,pre.drug
  ,pre.drug_name_poe
  ,pre.drug_name_generic
  ,pre.prod_strength
  ,pre.dose_val_rx
  ,pre.dose_unit_rx
  ,pre.form_val_disp
  ,pre.form_unit_disp
  from prescriptions pre
  inner join NSTEMI_patients pt
  on pre.subject_id = pt.subject_id
  order by pre.subject_id;
\copy (SELECT * FROM NSTEMI_prescriptions) to '/tmp/NSTEMI_prescriptions.csv' CSV HEADER;

drop view NSTEMI_procedures;
create view NSTEMI_procedures as
  select
  pro.subject_id
  ,pro.hadm_id
  ,pro.seq_num
  ,pro.icd9_code
  from procedures_icd pro
  inner join NSTEMI_patients st
  on pro.subject_id = st.subject_id
  order by pro.subject_id;
\copy (SELECT * FROM NSTEMI_procedures) to '/tmp/NSTEMI_procedures.csv' CSV HEADER;
