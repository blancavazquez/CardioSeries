--- Script for extracting information of table of admissions for NSTEMI patients
drop view NSTEMI_admissions;
create view NSTEMI_admissions as
  select
    ad.subject_id as "SUBJECT_ID"
    ,ad.hadm_id as "HADM_ID"
    ,ad.admittime as "ADMITTIME"
    ,ad.dischtime as "DISCHTIME"
    ,ad.deathtime as "DEATHTIME"
    ,ad.admission_type as "ADMISSION_TYPE"
    ,ad.admission_location as "ADMISSION_LOCATION"
    ,ad.discharge_location as "DISCHARGE_LOCATION"
    ,ad.diagnosis as "DIAGNOSIS"
    ,ad.hospital_expire_flag as "hospital_expire_flag"
  from admissions ad
  inner join NSTEMI_patients st
  on ad.subject_id = st.subject_id
  order by ad.subject_id;
\copy (SELECT * FROM NSTEMI_admissions) to '/tmp/ADMISSIONS.csv' CSV HEADER;
drop view NSTEMI_admissions;
