--- Script for extracting information of table of admissions for STEMI patients
drop view STEMI_admissions;
create view STEMI_admissions as
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
  inner join STEMI_patients st
  on ad.subject_id = st.subject_id
  order by ad.subject_id;
\copy (SELECT * FROM STEMI_admissions) to '/tmp/ADMISSIONS.csv' CSV HEADER;
drop view STEMI_admissions;
