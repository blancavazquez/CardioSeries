--- Script for extracting information of table of inputevents_mv for NSTEMI patients

drop view NSTEMI_inputevents_mv;
create view NSTEMI_inputevents_mv as

with temp_itemid AS -- tabla temporal para seleccionar itemids
(  
  select
  int_mv.subject_id
  ,int_mv.hadm_id
  ,int_mv.icustay_id
  ,int_mv.starttime
  ,int_mv.itemid
  ,int_mv.amount
  ,int_mv.amountuom
  from inputevents_mv int_mv
  inner join NSTEMI_patients pt
  on int_mv.subject_id = pt.subject_id
  where (int_mv.itemid =30134 or int_mv.itemid =30110 or int_mv.itemid =30025 or
         int_mv.itemid =30185 or int_mv.itemid =30186 or int_mv.itemid =30209 or 
         int_mv.itemid =30315 or int_mv.itemid =30316 or int_mv.itemid =30321 or
         int_mv.itemid =30381 or int_mv.itemid =42423 or int_mv.itemid =42512 or
         int_mv.itemid =44609 or int_mv.itemid =44927 or int_mv.itemid =45939 or
         int_mv.itemid =46047 or int_mv.itemid =46056 or int_mv.itemid =46216 or
         int_mv.itemid =42647 or int_mv.itemid =41693 or int_mv.itemid =41694 or
         int_mv.itemid =46484 or int_mv.itemid =30115 or int_mv.itemid =42648 or
         int_mv.itemid =42763 or int_mv.itemid =30310 or int_mv.itemid =30045 or
         int_mv.itemid =30100 or int_mv.itemid =44354 or int_mv.itemid =44518 or
         int_mv.itemid =45186 or int_mv.itemid =45322 or int_mv.itemid =42342 or
         int_mv.itemid =30112 or int_mv.itemid =30306 or int_mv.itemid =30042 or
         int_mv.itemid =221261 or
         int_mv.itemid =225151 or int_mv.itemid =225157 or int_mv.itemid =225152 or --- ids correspondientes a metavision
         int_mv.itemid =225958 or int_mv.itemid =225975 or int_mv.itemid =225906 or
         int_mv.itemid =225908 or int_mv.itemid =225974 or int_mv.itemid =222318 or
         int_mv.itemid =221468 or int_mv.itemid =223257 or int_mv.itemid =223258 or
         int_mv.itemid =223259 or int_mv.itemid =223260 or int_mv.itemid =223261 or
         int_mv.itemid =223262 or int_mv.itemid =221347 or int_mv.itemid =228339 or 
         int_mv.itemid =221653)
  and (int_mv.amount IS NOT NULL and int_mv.amount > 0)
  and int_mv.statusdescription!='Rewritten'
  order by int_mv.subject_id
)
SELECT --- selección de dosis
   int_mv.subject_id as "SUBJECT_ID"
  ,int_mv.hadm_id as "HADM_ID"
  ,int_mv.icustay_id as "ICUSTAY_ID"
  ,int_mv.starttime as "CHARTTIME"
  ,int_mv.itemid as "ITEMID"
  ,round(cast(int_mv.amount as numeric),2) as "VALUE"
  ,int_mv.amountuom as "VALUEUOM" ---unidad de medida
FROM temp_itemid int_mv
where (itemid = 30134 and LOWER(amountuom) = 'mg') or
      (itemid = 30110 and LOWER(amountuom) = 'mg') or
      (itemid = 30115 and LOWER(amountuom) = 'mg') or
      (itemid = 42648 and LOWER(amountuom) = 'mg') or
      (itemid = 30112 and LOWER(amountuom) = 'mg') or
      (itemid = 30042 and LOWER(amountuom) = 'mg') or
      (itemid = 221261 and LOWER(amountuom) = 'mg') or
      (itemid = 225151 and LOWER(amountuom) = 'mg') or
      (itemid = 225157 and LOWER(amountuom) = 'mg') or
      (itemid = 225906 and LOWER(amountuom) = 'mg') or
      (itemid = 225908 and LOWER(amountuom) = 'mg') or
      (itemid = 225974 and LOWER(amountuom) = 'mg') or
      (itemid = 222318 and LOWER(amountuom) = 'mg') or
      (itemid = 221468 and LOWER(amountuom) = 'mg') or
      (itemid = 221347 and LOWER(amountuom) = 'mg') or
      (itemid = 228339 and LOWER(amountuom) = 'mg') or
      (itemid = 221653 and LOWER(amountuom) = 'mg') or
      (itemid = 30025 and LOWER(amountuom) = 'units') or
      (itemid = 30045 and LOWER(amountuom) = 'units') or
      (itemid = 30100 and LOWER(amountuom) = 'units') or
      (itemid = 225152 and LOWER(amountuom) = 'units') or
      (itemid = 223257 and LOWER(amountuom) = 'units') or
      (itemid = 223258 and LOWER(amountuom) = 'units') or
      (itemid = 223259 and LOWER(amountuom) = 'units') or
      (itemid = 223261 and LOWER(amountuom) = 'units') or
      (itemid = 223262 and LOWER(amountuom) = 'units') 
order by int_mv.subject_id;
\copy (SELECT * FROM NSTEMI_inputevents_mv) to '/tmp/NSTEMI_INPUTEVENTS_mv.csv' CSV HEADER;
drop view NSTEMI_inputevents_mv;
