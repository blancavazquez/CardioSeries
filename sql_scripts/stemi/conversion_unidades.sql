--- Paso 6: conversión de unidades
drop view STEMI_prescriptions;
create view STEMI_prescriptions as
select
  pre.subject_id as "SUBJECT_ID"
  ,pre.hadm_id as "HADM_ID"
  ,pre.icustay_id as "ICUSTAY_ID"
  ,pre.startdate as "CHARTTIME"
  ,pre.enddate as "ENDDATE"
  ,pre.drug_type as "DRUG_TYPE"
  ,pre.itemid as "ITEMID"
  ,pre.drug as "DRUG" 
  ,pre.drug_name_poe as "DRUG_NAME_POE"
  ,pre.drug_name_generic as "DRUG_NAME_GENERIC"
  ,pre.prod_strength as "PROD_STRENGTH"

  ,case --- todos los medicamentos fueron convertidos a 'mg'
    when LOWER(pre.dose_unit_rx) = 'unit' and pre.itemid=30025 
    then pre.dosis * 0.0071 -- (heparin) convert from U to mg

    when LOWER(pre.dose_unit_rx) = 'unit' and (pre.itemid=4475 or pre.itemid = 223260)
    then pre.dosis * 0.0347 -- (insulin) convert from U to mg

    when LOWER(pre.dose_unit_rx) = 'ml' or LOWER(pre.dose_unit_rx) = 'mg'
    then pre.dosis --- se mantiene el valor de la dosis
  else null end as "VALUE" -- indica la dosis del meds administrado en mg.

  ,case
    when LOWER(pre.dose_unit_rx) = 'unit' then 'mg'
    when LOWER(pre.dose_unit_rx) = 'ml' then 'mg'
    when LOWER(pre.dose_unit_rx) = 'mg' then pre.dose_unit_rx
  else null end as "VALUEUOM" -- unidad de medida

  ,pre.form_val_disp as "FORM_VAL_DISP"
  ,pre.form_unit_disp as "FORM_UNIT_DISP"
  ,pre.intime as "INTIME"

  from STEMI_prescriptions_double pre
  order by pre.subject_id;
  \copy (SELECT * FROM STEMI_prescriptions where "VALUE"!=0) to '/tmp/PRESCRIPTIONS.csv' CSV HEADER;
