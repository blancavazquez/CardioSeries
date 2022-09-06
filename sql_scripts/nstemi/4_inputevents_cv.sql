--- Script for extracting information of table of inputevents_cv for NSTEMI patients

drop view NSTEMI_inputevents_cv;
create view NSTEMI_inputevents_cv as

with temp_itemid AS -- tabla temporal para seleccionar itemids
(  
  select
  int_cv.subject_id
  ,int_cv.hadm_id
  ,int_cv.icustay_id
  ,int_cv.charttime
  ,int_cv.itemid
  ,int_cv.amount
  ,int_cv.amountuom
  from inputevents_cv int_cv
  inner join NSTEMI_patients pt
  on int_cv.subject_id = pt.subject_id
  where (int_cv.itemid =30134 or int_cv.itemid =30110 or int_cv.itemid =30025 or
         int_cv.itemid =30185 or int_cv.itemid =30186 or int_cv.itemid =30209 or 
         int_cv.itemid =30315 or int_cv.itemid =30316 or int_cv.itemid =30321 or
         int_cv.itemid =30381 or int_cv.itemid =42423 or int_cv.itemid =42512 or
         int_cv.itemid =44609 or int_cv.itemid =44927 or int_cv.itemid =45939 or
         int_cv.itemid =46047 or int_cv.itemid =46056 or int_cv.itemid =46216 or
         int_cv.itemid =42647 or int_cv.itemid =41693 or int_cv.itemid =41694 or
         int_cv.itemid =46484 or int_cv.itemid =30115 or int_cv.itemid =42648 or
         int_cv.itemid =42763 or int_cv.itemid =30310 or int_cv.itemid =30045 or
         int_cv.itemid =30100 or int_cv.itemid =44354 or int_cv.itemid =44518 or
         int_cv.itemid =45186 or int_cv.itemid =45322 or int_cv.itemid =42342 or
         int_cv.itemid =30112 or int_cv.itemid =30306 or int_cv.itemid =30042 or
         int_cv.itemid =221261 or
         int_cv.itemid =225151 or int_cv.itemid =225157 or int_cv.itemid =225152 or --- ids correspondientes a metavision
         int_cv.itemid =225958 or int_cv.itemid =225975 or int_cv.itemid =225906 or
         int_cv.itemid =225908 or int_cv.itemid =225974 or int_cv.itemid =222318 or
         int_cv.itemid =221468 or int_cv.itemid =223257 or int_cv.itemid =223258 or
         int_cv.itemid =223259 or int_cv.itemid =223260 or int_cv.itemid =223261 or
         int_cv.itemid =223262 or int_cv.itemid =221347 or int_cv.itemid =228339 or 
         int_cv.itemid =221653)
  and (int_cv.amount IS NOT NULL and int_cv.amount > 0)
  order by int_cv.subject_id
)
SELECT --- selección de dosis
   int_cv.subject_id as "SUBJECT_ID"
  ,int_cv.hadm_id as "HADM_ID"
  ,int_cv.icustay_id as "ICUSTAY_ID"
  ,int_cv.charttime as "CHARTTIME"
  ,int_cv.itemid as "ITEMID"
  ,round(cast(int_cv.amount as numeric),2) as "VALUE"
  ,int_cv.amountuom as "VALUEUOM" ---unidad de medida
FROM temp_itemid int_cv
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
      (itemid = 30025 and LOWER(amountuom) = 'u') or
      (itemid = 30045 and LOWER(amountuom) = 'u') or
      (itemid = 30100 and LOWER(amountuom) = 'u') or
      (itemid = 225152 and LOWER(amountuom) = 'u') or
      (itemid = 223257 and LOWER(amountuom) = 'u') or
      (itemid = 223258 and LOWER(amountuom) = 'u') or
      (itemid = 223259 and LOWER(amountuom) = 'u') or
      (itemid = 223261 and LOWER(amountuom) = 'u') or
      (itemid = 223262 and LOWER(amountuom) = 'u')
order by int_cv.subject_id;
\copy (SELECT * FROM NSTEMI_inputevents_cv) to '/tmp/NSTEMI_INPUTEVENTS_CV.csv' CSV HEADER;
drop view NSTEMI_inputevents_cv;
