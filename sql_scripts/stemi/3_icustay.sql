--- Script for extracting information of table of icustay for STEMI patients
drop view STEMI_icustays;
create view STEMI_icustays as
  select
    icu.subject_id as "SUBJECT_ID"
    ,icu.hadm_id as "HADM_ID"
    ,icu.icustay_id as "ICUSTAY_ID"
    ,icu.dbsource as "DBSOURCE"
    ,icu.first_careunit as "FIRST_CAREUNIT"
    ,icu.last_careunit as "LAST_CAREUNIT"
    ,icu.first_wardid as "FIRST_WARDID"
    ,icu.last_wardid as "LAST_WARDID"
    ,icu.intime as "INTIME" 
    ,icu.outtime as "OUTTIME"
    ,icu.los as "LOS"
    from icustays icu
    inner join STEMI_patients st
    on icu.subject_id = st.subject_id
    order by icu.subject_id;
\copy (SELECT * FROM STEMI_icustays) to '/tmp/ICUSTAYS.csv' CSV HEADER;
