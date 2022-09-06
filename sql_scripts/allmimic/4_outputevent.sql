--- Script for extracting information of table of outputevents for STEMI patients

drop view all_outputevents;
create view all_outputevents as
  select
   out.subject_id as "SUBJECT_ID"
  ,out.hadm_id as "HADM_ID"
  ,out.icustay_id as "ICUSTAY_ID"
  ,out.charttime as "CHARTTIME"
  ,out.itemid as "ITEMID"
  ,out.value as "VALUE"
  ,out.valueuom as "VALUEUOM"
  ,out.iserror
  from outputevents out
  --inner join STEMI_patients st
  --on out.subject_id = st.subject_id
  where out.value IS NOT NULL 
        and out.value > 0 
  order by out.subject_id;
\copy (SELECT * FROM all_outputevents) to '/tmp/OUTPUTEVENTS.csv' CSV HEADER;
drop view all_outputevents;
