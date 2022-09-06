-- paso 2: Generación de vistas para cada uno de las sub-poblaciones

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

drop view STEMI_chartevents;
create view STEMI_chartevents as
  select
    ch.subject_id as "SUBJECT_ID"
    ,ch.hadm_id as "HADM_ID"
    ,ch.icustay_id as "ICUSTAY_ID"
    ,ch.itemid as "ITEMID"
    ,ch.charttime as "CHARTTIME"
    ,ch.storetime as "STORETIME"
    ,ch.value as "VALUE"
    ,ch.valuenum as "VALUENUM"
    ,ch.valueuom as "VALUEUOM"
    ,ch.warning as "WARNING"
    ,ch.error as "ERROR" 
    ,ch.resultstatus as "RESULTSTATUS"  
    ,ch.stopped as "STOPPED"
  from chartevents as ch
  inner join STEMI_patients  st
  on ch.subject_id=st.subject_id
  where ch.valuenum IS NOT NULL and ch.valuenum > 0 -- lab values cannot be 0 and cannot be negative
  order by ch. subject_id;
\copy (SELECT * FROM STEMI_chartevents) to '/tmp/CHARTEVENTS.csv' CSV HEADER;

drop view STEMI_diagnoses_icd;
create view STEMI_diagnoses_icd as
  select
    dx.subject_id as "SUBJECT_ID"
    ,dx.hadm_id as "HADM_ID"
    ,dx.seq_num as "SEQ_NUM"
    ,dx.icd9_code as "ICD9_CODE"
  from diagnoses_icd dx
  inner join STEMI_patients st
  on dx.subject_id = st.subject_id
  order by dx.subject_id;
\copy (SELECT * FROM STEMI_diagnoses_icd) to '/tmp/DIAGNOSES_ICD.csv' CSV HEADER;

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

drop view STEMI_labevents;
create view STEMI_labevents as
  select
  lab.subject_id as "SUBJECT_ID"
  ,lab.hadm_id as "HADM_ID"
  ,lab.itemid as "ITEMID"
  ,lab.charttime as "CHARTTIME"
  ,lab.value as "VALUE"
  ,lab.valuenum as "VALUENUM"
  ,lab.valueuom as "VALUEUOM"
  from labevents lab
  inner join STEMI_patients st
  on lab.subject_id = st.subject_id
  where lab.valuenum IS NOT NULL and lab.valuenum > 0 -- lab values cannot be 0 and cannot be negative
  order by lab.subject_id;
\copy (SELECT * FROM STEMI_labevents) to '/tmp/LABEVENTS.csv' CSV HEADER;

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

drop view STEMI_outputevents;
create view STEMI_outputevents as
  select
  out.subject_id as "SUBJECT_ID"
  ,out.hadm_id as "HADM_ID"
  ,out.icustay_id as "ICUSTAY_ID"
  ,out.charttime as "CHARTTIME"
  ,out.itemid as "ITEMID"
  ,out.value as "VALUE"
  ,out.valueuom as "VALUEUOM"
  ,out.stopped as "STOPPED"
  ,out.newbottle as "NEWBOTTLE"
  from outputevents out
  inner join STEMI_patients st
  on out.subject_id = st.subject_id
  where out.value IS NOT NULL
  order by out.subject_id;
\copy (SELECT * FROM STEMI_outputevents) to '/tmp/OUTPUTEVENTS.csv' CSV HEADER;

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


drop view STEMI_procedures;
create view STEMI_procedures as
  select
  pro.subject_id as "SUBJECT_ID"
  ,pro.hadm_id as "HADM_ID"
  ,pro.seq_num as "SEQ_NUM"
  ,pro.icd9_code as "ICD9_CODE"
  from procedures_icd pro
  inner join STEMI_patients st
  on pro.subject_id = st.subject_id
  order by pro.subject_id;
\copy (SELECT * FROM STEMI_procedures) to '/tmp/PROCEDURES.csv' CSV HEADER;

drop view STEMI_inputevents_mv; 
create view STEMI_inputevents_mv as
  select
  int_mv.subject_id as "SUBJECT_ID"
  ,int_mv.hadm_id as "HADM_ID"
  ,int_mv.icustay_id as "ICUSTAY_ID"
  ,int_mv.starttime as "CHARTTIME"
  ,int_mv.itemid as "ITEMID"
  ,case 
    when int_mv.amountuom = 'units' and int_mv.itemid=225152 
    then int_mv.amount * 0.0071 -- (heparin) convert from U to mg

    when int_mv.amountuom = 'units' and (int_mv.itemid=223257 or int_mv.itemid=223258 or
                                        int_mv.itemid=223259 or int_mv.itemid=223260 or
                                      int_mv.itemid=223261 or int_mv.itemid=223262)
    then int_mv.amount * 0.0347 -- (insulin) convert from U to mg

    when int_mv.amountuom = 'mg' ----- or int_mv.amountuom = 'dose' -- mg y dose se mantiene la unidad y el valor
    then int_mv.amount
  else null end as "VALUE" -- dosis
  
  ,case
    when int_mv.amountuom = 'units' then 'mg'
    when int_mv.amountuom = 'mg' --------or  int_mv.amountuom = 'dose'  
        then int_mv.amountuom
  else null end as "VALUEUOM" -- unidad de medida

  ,icu.intime as "INTIME"
  from inputevents_mv int_mv
  inner join STEMI_patients pt
  on int_mv.subject_id = pt.subject_id
  inner join icustays icu
  on int_mv.icustay_id = icu.icustay_id
  where (int_mv.itemid =221261
          or int_mv.itemid =225151
          or int_mv.itemid =225157
          or int_mv.itemid =225152
          or int_mv.itemid =225958
          -- or int_mv.itemid =225975 heparina, , viene descrita en dosis! no se tomarán en cuenta
          -- or int_mv.itemid =225906 enoxaparina, viene descrita en dosis! no se tomarán en cuenta
          or int_mv.itemid =225908
          or int_mv.itemid =225974
          or int_mv.itemid =222318
          or int_mv.itemid =221468
          or int_mv.itemid =223257
          or int_mv.itemid =223258
          or int_mv.itemid =223259
          or int_mv.itemid =223260
          or int_mv.itemid =223261
          or int_mv.itemid =223262
          or int_mv.itemid =221347
          or int_mv.itemid =228339
          or int_mv.itemid =221653
          or int_mv.itemid =30134 --- ids correspondientes a carevision
          or int_mv.itemid =30110
          or int_mv.itemid =30025
          or int_mv.itemid =30185
          or int_mv.itemid =30186
          or int_mv.itemid =30209
          or int_mv.itemid =30315
          or int_mv.itemid =30316
          or int_mv.itemid =30321
          or int_mv.itemid =30381
          or int_mv.itemid =42423
          or int_mv.itemid =42512
          or int_mv.itemid =44609
          or int_mv.itemid =44927
          or int_mv.itemid =45939
          or int_mv.itemid =46047
          or int_mv.itemid =46056
          or int_mv.itemid =46216
          or int_mv.itemid =42647
          or int_mv.itemid =41693
          or int_mv.itemid =41694
          or int_mv.itemid =46484
          or int_mv.itemid =30115
          or int_mv.itemid =42648
          or int_mv.itemid =42763
          or int_mv.itemid =30310
          or int_mv.itemid =30045
          or int_mv.itemid =30100
          or int_mv.itemid =44354
          or int_mv.itemid =44518
          or int_mv.itemid =45186
          or int_mv.itemid =45322
          or int_mv.itemid =42342
          or int_mv.itemid =30112
          or int_mv.itemid =30306
          or int_mv.itemid =30042)
  and int_mv.statusdescription!='Rewritten'
  order by int_mv.subject_id;
\copy (SELECT * FROM STEMI_inputevents_mv where "VALUE"!=0 ) to '/tmp/INPUTEVENTS_MV.csv' CSV HEADER;

---**********************************************************************************
------------------ hacemos un paréntesis,
-- iré a buscar directamente en inputevents_mv cuando itemid= 225906 y 225975
-- para STEMIs
-- conclusion: las dosis, rate y amount todas dicen "1", no hay dato adicional
----------------------------------------------------que ayude en la conversión :(

drop view revisando_inputmv;
create view revisando_inputmv as
select
   inp.subject_id
  ,inp.hadm_id
  ,inp.icustay_id
  ,inp.starttime
  ,inp.endtime
  ,inp.itemid
  ,inp.amount
  ,inp.amountuom
  ,inp.rate
  ,inp.rateuom
  ,inp.storetime
  ,inp.orderid
  ,inp.linkorderid
  ,inp.ordercategoryname
  ,inp.secondaryordercategoryname
  ,inp.ordercomponenttypedescription
  ,inp.ordercategorydescription
  ,inp.patientweight
  ,inp.totalamount
  ,inp.totalamountuom
  ,inp.isopenbag
  ,inp.continueinnextdept
  ,inp.cancelreason
  ,inp.statusdescription
  ,inp.comments_editedby
  ,inp.comments_canceledby
  ,inp.comments_date
  ,inp.originalamount
  ,inp.originalrate
from inputevents_mv inp
inner join STEMI_patients pt
on inp.subject_id = pt.subject_id
where inp.itemid =225906 or inp.itemid = 225975
order by inp.subject_id;
\copy (SELECT * FROM revisando_inputmv) to '/tmp/revisando_inputmv.csv' CSV HEADER;