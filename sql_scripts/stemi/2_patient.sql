--- Script for extracting information of table of patients for STEMI patients
drop view STEMI_pat_table;
create view STEMI_pat_table as
  SELECT
  pt.subject_id as "SUBJECT_ID"
  ,pt.gender as "GENDER"
  ,pt.dob as "DOB"
  ,pt.dod as "DOD"
  ,pt.dod_hosp as "DOD_HOSP"
  ,pt.dod_ssn as "DOD_SSN"
  ,pt.expire_flag as "EXPIRE_FLAG"
  from patients pt
  inner join STEMI_patients st
  on pt.subject_id = st.subject_id
  order by pt.subject_id;
\copy (SELECT * FROM STEMI_pat_table) to '/tmp/PATIENTS.csv' CSV HEADER;
drop view STEMI_pat_table;
