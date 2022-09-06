drop view STEMI_noteevents;
create view STEMI_noteevents as
  select
  nt.subject_id as "SUBJECT_ID"
  ,nt.hadm_id as "HADM_ID"
  ,nt.chartdate as "CHARTDATE"
  ,nt.category as "CATEGORY"
  ,nt.description as "DESCRIPTION"
  ,nt.text as "TEXT"
  from noteevents nt
  inner join STEMI_patients st
  on nt.subject_id = st.subject_id
  where nt.iserror IS NULL
  order by nt.subject_id;
\copy (SELECT * FROM STEMI_noteevents) to '/tmp/NOTEEVENTS.csv' CSV HEADER;