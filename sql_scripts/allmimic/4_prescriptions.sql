--- Script for extracting information of table of prescriptions for STEMI patients
--- particularly, extraction of selected treatments on specific measures (mg y units)

-- ******************************************************************
-- Paso 1: separar valores de la columna 'dose_val_rx': 1000-2000  (me quedaré con el último valor: 2000)
drop view all_prescriptions_preprocess cascade;
create view all_prescriptions_preprocess as
  select
  pre.subject_id
  ,pre.hadm_id
  ,pre.icustay_id
  ,pre.startdate
  ,pre.enddate
  ,pre.drug_type
  ,pre.drug
  ,pre.drug_name_poe
  ,pre.drug_name_generic
  ,pre.prod_strength
  ,pre.dose_val_rx
  ,pre.dose_unit_rx
  ,pre.form_val_disp
  ,pre.form_unit_disp
  ,icu.intime
  ,split_part(pre.dose_val_rx, '-', 2) as dosis  --- dado 5-10, me quedo con 10
  from prescriptions pre
  --inner join STEMI_patients pt
  --on pre.subject_id = pt.subject_id
  inner join icustays icu
  on pre.icustay_id = icu.icustay_id
  order by pre.subject_id;

-- ******************************************************************
-- Paso 2: Join two columns: dose_val_rx + dosis (new column)
drop view all_prescriptions_join;
create view all_prescriptions_join as
  select
  pre.subject_id
  ,pre.hadm_id
  ,pre.icustay_id
  ,pre.startdate
  ,pre.enddate
  ,pre.drug_type
  ,pre.drug
  ,pre.drug_name_poe
  ,pre.drug_name_generic
  ,pre.prod_strength
  ,pre.dose_val_rx
  ,pre.dose_unit_rx
  ,pre.form_val_disp
  ,pre.form_unit_disp
  ,pre.intime
  ,pre.dosis
  ,case
    when pre.dosis = '' then pre.dose_val_rx
    when pre.dosis != '' then pre.dosis
    else null end as final_dosis -- combino las columnas de dosis
  from all_prescriptions_preprocess pre
  order by pre.subject_id;

-- ******************************************************************
-- Paso 3: identificar tipo de unidad de medida por medicamento de interés
drop view all_prescriptions_unit;
create view all_prescriptions_unit as
  select
  pre.subject_id
  ,pre.drug
  ,pre.final_dosis
  ,pre.dose_unit_rx

  from all_prescriptions_join pre
  where pre.final_dosis IS NOT NULL  --- ¡no tenemos dosis vacías!
    and
        (
         LOWER(pre.drug) = 'acetylsalicylic acid' or
         LOWER(pre.drug) = 'salicylic acid' or
         LOWER(pre.drug) = 'aspirin' or
         LOWER(pre.drug) = 'aas' or
         LOWER(pre.drug) = 'as' or
         LOWER(pre.drug) = 'asa' or
         LOWER(pre.drug) = 'aspirin (buffered)' or
         LOWER(pre.drug) = 'aspirin (rectal)' or
         LOWER(pre.drug) = 'aspi' or
         LOWER(pre.drug) = 'aspir' or
         LOWER(pre.drug) = 'aspiri' or
         LOWER(pre.drug) = 'aspirin desens' or
         LOWER(pre.drug) = 'aspirin desensitization' or
         LOWER(pre.drug) = 'aspirin desensitization (aerd)' or
         LOWER(pre.drug) = 'aspirin desensitization (angioedema)' or
         LOWER(pre.drug) = 'aspirin ec' or
       
         LOWER(pre.drug) = 'plavix' or
         LOWER(pre.drug) = 'clopidogrel' or
         LOWER(pre.drug) = 'clopidogrel bisulfate' or
         LOWER(pre.drug) = 'clopidogrel desensitization' or

         LOWER(pre.drug) = 'prasugrel' or
         LOWER(pre.drug) = 'effient' or

         LOWER(pre.drug) = 'abciximab' or
         LOWER(pre.drug) = 'reopro' or

         LOWER(pre.drug) = 'eptifibatide' or
         LOWER(pre.drug) = 'integrilin' or

         LOWER(pre.drug) = 'tirofiban' or
         LOWER(pre.drug) = 'aggrastat' or

         LOWER(pre.drug) = 'unfractionated heparin' or
         LOWER(pre.drug) = 'low-molecular-weight' or
         LOWER(pre.drug) = 'clexane' or
         LOWER(pre.drug) = 'lovenox' or
         LOWER(pre.drug) = 'heparin' or
         LOWER(pre.drug) = 'heparin (crrt machine priming)' or
         LOWER(pre.drug) = 'heparin (iabp)' or
         LOWER(pre.drug) = 'heparin crrt' or
         LOWER(pre.drug) = 'heparin dwell (1000 units/ml)' or
         LOWER(pre.drug) = 'heparin flush (10 units/ml)' or
         LOWER(pre.drug) = 'heparin flush (100 units/ml)' or
         LOWER(pre.drug) = 'heparin flush (1000 units/ml)' or
         LOWER(pre.drug) = 'heparin flush (5000 units/ml)' or
         LOWER(pre.drug) = 'heparin flush crrt (5000 units/ml)' or
         LOWER(pre.drug) = 'heparin flush cvl  (100 units/ml)' or
         LOWER(pre.drug) = 'heparin flush hickman (100 units/ml)' or
         LOWER(pre.drug) = 'heparin flush midline (100 units/ml)' or
         LOWER(pre.drug) = 'heparin flush picc (100 units/ml)' or
         LOWER(pre.drug) = 'heparin flush port (10 units/ml)' or
         LOWER(pre.drug) = 'heparin flush port (10units/ml)' or
         LOWER(pre.drug) = 'heparin lock flush' or
         LOWER(pre.drug) = 'heparin dose' or
         LOWER(pre.drug) = 'heparin flush 10u/cc' or
         LOWER(pre.drug) = 'heparin flush' or
         LOWER(pre.drug) = 'heparin sodium' or
         LOWER(pre.drug) = 'heparin level' or
         LOWER(pre.drug) = 'heparin depend antby' or
         LOWER(pre.drug) = 'heparin/pic flush' or
         LOWER(pre.drug) = 'heparin lock flush' or
         LOWER(pre.drug) = '45ns + 1:1 heparin' or
         LOWER(pre.drug) = '.9ns + 0.5:1 heparin' or
         LOWER(pre.drug) = 'heparin via sheaths' or
         LOWER(pre.drug) = '1000ns/1000uheparin' or
         LOWER(pre.drug) = '.25 ns+0.5:1 heparin' or
         LOWER(pre.drug) = '.25 ns +1:1 heparin' or
         LOWER(pre.drug) = '.45ns + .5:1 heparin' or
         LOWER(pre.drug) = '.9ns + 1:1 heparin' or
         LOWER(pre.drug) = 'heparin(10 units/cc)' or
         LOWER(pre.drug) = 'pn d9.5 w/ heparin' or
         LOWER(pre.drug) = 'na acetate/heparin' or
         LOWER(pre.drug) = 'crrt heparin' or
         LOWER(pre.drug) = 'd10w with heparin' or
         LOWER(pre.drug) = 'tpnd9.5+heparin' or
         LOWER(pre.drug) = 'na acetate w/heparin' or
         LOWER(pre.drug) = 'heparin dose (per hour)' or
         LOWER(pre.drug) = 'heparin sodium (prophylaxis)' or
         LOWER(pre.drug) = 'heparin concentration (units/ml)' or
         LOWER(pre.drug) = 'heparin (hemodialysis)' or

         LOWER(pre.drug) = 'enoxaparin' or
         LOWER(pre.drug) = 'enoxaparin sodium' or
         LOWER(pre.drug) = 'clexane' or
         LOWER(pre.drug) = 'lovenox' or

         LOWER(pre.drug) = 'bivalirudin' or

         LOWER(pre.drug) = 'fondaparinux' or
         LOWER(pre.drug) = 'fondaparinux sodium' or

         LOWER(pre.drug) = 'metoprolol' or
         LOWER(pre.drug) = 'metoprolo' or
         LOWER(pre.drug) = 'betaloc' or
         LOWER(pre.drug) = 'spesicor' or
         LOWER(pre.drug) = 'lopressor' or
         LOWER(pre.drug) = 'toprol' or
         LOWER(pre.drug) = 'toprol xl' or
         LOWER(pre.drug) = 'metoprolol succinate xl' or
         LOWER(pre.drug) = 'metoprolol tartrate' or
         LOWER(pre.drug) = 'metoprolol xl' or
         LOWER(pre.drug) = 'metoprolol xl (toprol xl)' or

         LOWER(pre.drug) = 'bisoprolol' or
         LOWER(pre.drug) = 'concor' or
         LOWER(pre.drug) = 'ziac' or
         LOWER(pre.drug) = 'maxsoten' or

         LOWER(pre.drug) = 'atenolol' or
         LOWER(pre.drug) = 'tenormin' or

         LOWER(pre.drug) = 'carvedilol' or
         LOWER(pre.drug) = 'coreg' or
         LOWER(pre.drug) = 'coreg cr' or

         LOWER(pre.drug) = 'sotalol' or
         LOWER(pre.drug) = 'sotalol hcl' or

         LOWER(pre.drug) = 'verapamil' or
         LOWER(pre.drug) = 'verapamil hcl' or
         LOWER(pre.drug) = 'verapamil sr' or
         LOWER(pre.drug) = 'iproveratril' or
         LOWER(pre.drug) = 'calan' or
         LOWER(pre.drug) = 'calan sr' or
         LOWER(pre.drug) = 'overa hs' or
         LOWER(pre.drug) = 'isoptin' or
         LOWER(pre.drug) = 'verelan' or
         LOWER(pre.drug) = 'verelan pm' or
         LOWER(pre.drug) = 'verapamil drip' or

         LOWER(pre.drug) = 'diltiazem' or
         LOWER(pre.drug) = 'diltiazem extended-release' or
         LOWER(pre.drug) = 'cardizem' or
         LOWER(pre.drug) = 'cardizem cd' or
         LOWER(pre.drug) = 'cardizem la' or
         LOWER(pre.drug) = 'cardizem sr' or
         LOWER(pre.drug) = 'cartia xt' or
         LOWER(pre.drug) = 'dilacor xr' or
         LOWER(pre.drug) = 'dilt-cd' or
         LOWER(pre.drug) = 'dilt xr' or
         LOWER(pre.drug) = 'diltia xt' or
         LOWER(pre.drug) = 'taztia xt' or
         LOWER(pre.drug) = 'tiamate' or
         LOWER(pre.drug) = 'tiazac' or

         LOWER(pre.drug) = 'digoxin' or
         LOWER(pre.drug) = 'digoxin immune fab' or
         LOWER(pre.drug) = 'cardoxin' or
         LOWER(pre.drug) = 'digitek' or
         LOWER(pre.drug) = 'lanoxicaps' or
         LOWER(pre.drug) = 'lanoxin' or
         LOWER(pre.drug) = 'dilanacin' or

         LOWER(pre.drug) = 'captopril' or
         LOWER(pre.drug) = 'capoten' or
         LOWER(pre.drug) = 'kaplon' or

         LOWER(pre.drug) = 'enalapril' or
         LOWER(pre.drug) = 'vasotec' or
         LOWER(pre.drug) = 'enalapril maleate' or
         LOWER(pre.drug) = 'vasotec iv' or
         LOWER(pre.drug) = 'enalaprilat' or

         LOWER(pre.drug) = 'lisinopril' or

         LOWER(pre.drug) = 'atorvastatin' or
         LOWER(pre.drug) = 'atorvastatin study drug' or
         LOWER(pre.drug) = 'lipitor' or

         LOWER(pre.drug) = 'simvastatin' or
         LOWER(pre.drug) = 'simvastatin or placebo' or
         LOWER(pre.drug) = 'lovastatin' or
         LOWER(pre.drug) = 'velastatin' or
         LOWER(pre.drug) = 'synvinolin' or

         LOWER(pre.drug) = 'pravastatin' or
         LOWER(pre.drug) = 'pravachol' or

         LOWER(pre.drug) = 'rosuvastatin' or
         LOWER(pre.drug) = 'crestor' or
         LOWER(pre.drug) = 'ezallor' or
         LOWER(pre.drug) = 'rosuvastatin calcium' or

         LOWER(pre.drug) = 'isosorbide dinitrate' or
         LOWER(pre.drug) = 'isosorbide dinitrate sa' or
         LOWER(pre.drug) = 'isosorbide monon' or
         LOWER(pre.drug) = 'isosorbide mononi' or
         LOWER(pre.drug) = 'isosorbide mononitra' or
         LOWER(pre.drug) = 'isosorbide mononitrate' or
         LOWER(pre.drug) = 'isosorbide mononitrate (extended release)' or

         LOWER(pre.drug) = 'gemfibrozil' or
         LOWER(pre.drug) = 'lopid' or

         LOWER(pre.drug) = 'bezafibrate' or
         LOWER(pre.drug) = 'cedur' or

         LOWER(pre.drug) = 'fenofibrate' or
         LOWER(pre.drug) = 'fenofibrate micronized' or
         LOWER(pre.drug) = 'tricor' or
         LOWER(pre.drug) = 'fibricor' or
         LOWER(pre.drug) = 'lofibra' or

         LOWER(pre.drug) = 'insulin' or
         LOWER(pre.drug) = 'insulin pump' or
         LOWER(pre.drug) = 'insulin human regular' or
         LOWER(pre.drug) = 'humulin-r insulin' or
         LOWER(pre.drug) = 'humalog insulin' or
         LOWER(pre.drug) = 'insulin glargine' or
         LOWER(pre.drug) = 'insulin human nph' or
         LOWER(pre.drug) = 'insulin novolog' or
         LOWER(pre.drug) = 'insulin pump (self administering medication)' or
         LOWER(pre.drug) = 'insulin regular human (u-500)' or
         LOWER(pre.drug) = 'insulin regular pork (iletin ii)' or
         LOWER(pre.drug) = 'neo*iv*insulin (dilute)' or

         LOWER(pre.drug) = 'amiodarone' or
         LOWER(pre.drug) = 'cordarone' or
         LOWER(pre.drug) = 'pacerone' or
         LOWER(pre.drug) = 'amiodarone hcl' or

         LOWER(pre.drug) = 'dobutamine' or
         LOWER(pre.drug) = 'dobutrex' or
         LOWER(pre.drug) = 'dobutamine hcl' or

         LOWER(pre.drug) = 'dopamine' or
         LOWER(pre.drug) = 'dopamine hcl' 
         )
  order by pre.subject_id;

-- ******************************************************************
-- Paso 4: añadir itemid, selección de dosis (mg, unit) y replace de caracteres especiales en dosis
drop view all_prescriptions_itemid_dosis;
create view all_prescriptions_itemid_dosis as
with temp_itemid AS -- tabla temporal para añadir itemids
(
  select
  pre.subject_id
  ,pre.hadm_id
  ,pre.icustay_id
  ,pre.startdate
  ,pre.enddate
  ,pre.drug_type
  ,pre.drug

  ,case -- nota: los itemids tomados de UMLS, significa que no existen en la tabla de D_ITEMS
    --group: aspirin (mg)
    when LOWER(pre.drug) = 'aspirin' then 7325
    when LOWER(pre.drug) = 'aas' then 7325
    when LOWER(pre.drug) = 'as' then 7325
    when LOWER(pre.drug) = 'asa' then 7325
    when LOWER(pre.drug) = 'aspirin (buffered)' then 7325
    when LOWER(pre.drug) = 'aspirin (rectal)' then 7325
    when LOWER(pre.drug) = 'acetylsalicylic acid' then 7325
    when LOWER(pre.drug) = 'salicylic acid' then 7325
    when LOWER(pre.drug) = 'aspi' then 7325
    when LOWER(pre.drug) = 'aspir' then 7325
    when LOWER(pre.drug) = 'aspiri' then 7325
    when LOWER(pre.drug) = 'aspirin desens' then 7325
    when LOWER(pre.drug) = 'aspirin desensitization' then 7325
    when LOWER(pre.drug) = 'aspirin desensitization (aerd)' then 7325
    when LOWER(pre.drug) = 'aspirin desensitization (angioedema)' then 7325
    when LOWER(pre.drug) = 'aspirin ec' then 7325

    --group: clopidogrel (mg) --- el ITEMID lo saqué de UMLS
    when LOWER(pre.drug) = 'plavix' then 70166
    when LOWER(pre.drug) = 'clopidogrel' then 70166
    when LOWER(pre.drug) = 'clopidogrel bisulfate' then 70166
    when LOWER(pre.drug) = 'clopidogrel desensitization' then 70166

    --group: prasugrel (mg)--- el ITEMID lo saqué de UMLS
    when LOWER(pre.drug) = 'prasugrel' then 1620287
    when LOWER(pre.drug) = 'effient' then 1620287

    --group: abciximab (mg)
    when LOWER(pre.drug) = 'abciximab' then 221261
    when LOWER(pre.drug) = 'reopro' then 221261

    --group: eptifibatide (mg) 
    when LOWER(pre.drug) = 'eptifibatide' then 225151
    when LOWER(pre.drug) = 'integrilin' then 225151

    --group: tirofiban (mg) 
    when LOWER(pre.drug) = 'tirofiban' then 225157
    when LOWER(pre.drug) = 'aggrastat' then 225157
    
    --group: enoxaparin (mg)  
    when LOWER(pre.drug) = 'enoxaparin' then 225906
    when LOWER(pre.drug) = 'enoxaparin sodium' then 225906
    when LOWER(pre.drug) = 'clexane' then 225906
    when LOWER(pre.drug) = 'lovenox' then 225906
    
    --group: bivalirudin  (mg)
    when LOWER(pre.drug) = 'bivalirudin' then 1363
    when LOWER(pre.drug) = 'angiomax' then 1363
    when LOWER(pre.drug) = 'angiox' then 1363

    --group: fondaparinux (mg)  
    when LOWER(pre.drug) = 'fondaparinux' then 225908
    when LOWER(pre.drug) = 'fondaparinux sodium' then 225908

    --group: metoprolol (mg) 
    when LOWER(pre.drug) = 'metoprolol' then 225974
    when LOWER(pre.drug) = 'metoprolol succinate xl' then 225974
    when LOWER(pre.drug) = 'metoprolol tartrate' then 225974
    when LOWER(pre.drug) = 'metoprolol xl' then 225974
    when LOWER(pre.drug) = 'metoprolol xl (toprol xl)' then 225974
    when LOWER(pre.drug) = 'metoprolo' then 225974
    when LOWER(pre.drug) = 'betaloc' then 225974
    when LOWER(pre.drug) = 'spesicor' then 225974
    when LOWER(pre.drug) = 'lopressor' then 225974
    when LOWER(pre.drug) = 'toprol' then 225974
    when LOWER(pre.drug) = 'toprol xl' then 225974

    --group: atenolol (mg)  --- el ITEMID lo saqué de UMLS
    when LOWER(pre.drug) = 'atenolol' then 4147
    when LOWER(pre.drug) = 'tenormin' then 4147

    --group: carvedilol (mg)  --- el ITEMID lo saqué de UMLS
    when LOWER(pre.drug) = 'carvedilol' then 54836
    when LOWER(pre.drug) = 'coreg' then 54836
    when LOWER(pre.drug) = 'coreg cr' then 54836

    --group: sotalol (mg) --- el ITEMID lo saqué de UMLS
    when LOWER(pre.drug) = 'sotalol' then 37707
    when LOWER(pre.drug) = 'sotalol hcl' then 37707

    --group: verapamil (mg)
    when LOWER(pre.drug) = 'verapamil' then 1968
    when LOWER(pre.drug) = 'verapamil hcl' then 1968
    when LOWER(pre.drug) = 'verapamil sr' then 1968
    when LOWER(pre.drug) = 'iproveratril' then 1968
    when LOWER(pre.drug) = 'calan' then 1968
    when LOWER(pre.drug) = 'calan sr' then 1968
    when LOWER(pre.drug) = 'overa hs' then 1968
    when LOWER(pre.drug) = 'isoptin' then 1968
    when LOWER(pre.drug) = 'verelan' then 1968
    when LOWER(pre.drug) = 'verelan pm' then 1968
    when LOWER(pre.drug) = 'verapamil drip' then 1968

    --group: diltiazem (mg)
    when LOWER(pre.drug) = 'diltiazem' then 30115
    when LOWER(pre.drug) = 'diltiazem extended-release' then 30115
    when LOWER(pre.drug) = 'cardizem' then 30115
    when LOWER(pre.drug) = 'cardizem cd' then 30115
    when LOWER(pre.drug) = 'cardizem la' then 30115
    when LOWER(pre.drug) = 'cardizem sr' then 30115
    when LOWER(pre.drug) = 'cartia xt' then 30115
    when LOWER(pre.drug) = 'dilacor xr' then 30115
    when LOWER(pre.drug) = 'dilt-cd' then 30115
    when LOWER(pre.drug) = 'dilt xr' then 30115
    when LOWER(pre.drug) = 'diltia xt' then 30115
    when LOWER(pre.drug) = 'taztia xt' then 30115
    when LOWER(pre.drug) = 'tiamate' then 30115
    when LOWER(pre.drug) = 'tiazac' then 30115

    --group: digoxin (mg)
    when LOWER(pre.drug) = 'digoxin' then 227440
    when LOWER(pre.drug) = 'digoxin immune fab' then 227440
    when LOWER(pre.drug) = 'cardoxin' then 227440
    when LOWER(pre.drug) = 'digitek' then 227440
    when LOWER(pre.drug) = 'lanoxicaps' then 227440
    when LOWER(pre.drug) = 'lanoxin' then 227440
    when LOWER(pre.drug) = 'dilanacin' then 227440

    --group: captopril (mg)
    when LOWER(pre.drug) = 'captopril' then 3349
    when LOWER(pre.drug) = 'capoten' then 3349
    when LOWER(pre.drug) = 'kaplon' then 3349

    --group: enalapril (mg)
    when LOWER(pre.drug) = 'enalapril maleate' then 42648
    when LOWER(pre.drug) = 'enalapril' then 42648
    when LOWER(pre.drug) = 'vasotec' then 42648
    when LOWER(pre.drug) = 'vasotec iv' then 42648
    when LOWER(pre.drug) = 'enalaprilat' then 42648

    --group: lisinopril (mg)
    when LOWER(pre.drug) = 'lisinopril' then 65374    

    --group: atorvastatin (mg) --- el ITEMID lo saqué de UMLS
    when LOWER(pre.drug) = 'atorvastatin' then 286651
    when LOWER(pre.drug) = 'atorvastatin study drug' then 286651
    when LOWER(pre.drug) = 'lipitor' then 286651

    --group: simvastatin (mg) --- el ITEMID lo saqué de UMLS
    when LOWER(pre.drug) = 'simvastatin' then 74554
    when LOWER(pre.drug) = 'lovastatin' then 74554
    when LOWER(pre.drug) = 'simvastatin or placebo' then 74554
    when LOWER(pre.drug) = 'velastatin' then 74554
    when LOWER(pre.drug) = 'synvinolin' then 74554

    --group: pravastatin (mg) --- el ITEMID lo saqué de UMLS
    when LOWER(pre.drug) = 'pravastatin' then 85542
    when LOWER(pre.drug) = 'pravachol' then 85542

    --group: rosuvastatin (mg)--- el ITEMID lo saqué de UMLS
    when LOWER(pre.drug) = 'rosuvastatin' then '1101751'
    when LOWER(pre.drug) = 'rosuvastatin calcium' then '1101751'
    when LOWER(pre.drug) = 'crestor' then '1101751'
    when LOWER(pre.drug) = 'ezallor' then '1101751'

    --group: oral_nitrates (mg) --- el ITEMID lo saqué de UMLS
    when LOWER(pre.drug) = 'isosorbide dinitrate' then 22252
    when LOWER(pre.drug) = 'isosorbide dinitrate sa' then 22252
    when LOWER(pre.drug) = 'isosorbide monon' then 22252
    when LOWER(pre.drug) = 'isosorbide mononitrate' then 22252
    when LOWER(pre.drug) = 'isosorbide mononitrate (extended release)' then 22252
    when LOWER(pre.drug) = 'isosorbide mononi' then 22252
    when LOWER(pre.drug) = 'isosorbide mononitra' then 22252

    --group: gemfibrozil (mg)--- el ITEMID lo saqué de UMLS
    when LOWER(pre.drug) = 'gemfibrozil' then 17245
    when LOWER(pre.drug) = 'lopid' then 17245

    --group: fenofibrate (mg) --- el ITEMID lo saqué de UMLS
    when LOWER(pre.drug) = 'tricor' then 33228
    when LOWER(pre.drug) = 'fenofibrate' then 33228
    when LOWER(pre.drug) = 'fenofibrate micronized' then 33228
    when LOWER(pre.drug) = 'fibricor' then 33228
    when LOWER(pre.drug) = 'lofibra' then 33228

    --group: amiodarone (mg)
    when LOWER(pre.drug) = 'amiodarone' then 30112
    when LOWER(pre.drug) = 'amiodarone hcl' then 30112
    when LOWER(pre.drug) = 'cordarone' then 30112
    when LOWER(pre.drug) = 'pacerone' then 30112

    --group: dobutamine (mg)
    when LOWER(pre.drug) = 'dobutamine' then 221653
    when LOWER(pre.drug) = 'dobutamine hcl' then 221653
    when LOWER(pre.drug) = 'dobutrex' then 221653

    --group: dopamine (mg)
    when LOWER(pre.drug) = 'dopamine' then 221662
    when LOWER(pre.drug) = 'dopamine hcl' then 221662

    when LOWER(pre.drug) = 'unfractionated heparin' then 30025
    when LOWER(pre.drug) = 'heparin' then 30025
    when LOWER(pre.drug) = 'heparin (crrt machine priming)' then 30025
    when LOWER(pre.drug) = 'heparin (iabp)' then 30025
    when LOWER(pre.drug) = 'heparin crrt' then 30025
    when LOWER(pre.drug) = 'heparin dwell (1000 units/ml)' then 30025
    when LOWER(pre.drug) = 'heparin flush (10 units/ml)' then 30025
    when LOWER(pre.drug) = 'heparin flush (100 units/ml)' then 30025
    when LOWER(pre.drug) = 'heparin flush (1000 units/ml)' then 30025
    when LOWER(pre.drug) = 'heparin flush (5000 units/ml)' then 30025
    when LOWER(pre.drug) = 'heparin flush crrt (5000 units/ml)' then 30025
    when LOWER(pre.drug) = 'heparin flush cvl  (100 units/ml)' then 30025
    when LOWER(pre.drug) = 'heparin flush hickman (100 units/ml)' then 30025
    when LOWER(pre.drug) = 'heparin flush midline (100 units/ml)' then 30025
    when LOWER(pre.drug) = 'heparin flush picc (100 units/ml)' then 30025
    when LOWER(pre.drug) = 'heparin flush port (10 units/ml)' then 30025
    when LOWER(pre.drug) = 'heparin flush port (10units/ml)' then 30025
    when LOWER(pre.drug) = 'heparin lock flush' then 30025
    when LOWER(pre.drug) = 'hepatitis b vaccine' then 30025
    when LOWER(pre.drug) = 'heparin dose' then 30025
    when LOWER(pre.drug) = 'heparin flush 10u/cc' then 30025
    when LOWER(pre.drug) = 'heparin flush' then 30025
    when LOWER(pre.drug) = 'heparin sodium' then 30025
    when LOWER(pre.drug) = 'heparin level' then 30025
    when LOWER(pre.drug) = 'heparin depend antby' then 30025
    when LOWER(pre.drug) = 'heparin/pic flush' then 30025
    when LOWER(pre.drug) = 'heparin lock flush' then 30025
    when LOWER(pre.drug) = '45ns + 1:1 heparin' then 30025
    when LOWER(pre.drug) = '.9ns + 0.5:1 heparin' then 30025
    when LOWER(pre.drug) = 'heparin via sheaths' then 30025
    when LOWER(pre.drug) = '1000ns/1000uheparin' then 30025
    when LOWER(pre.drug) = '.25 ns+0.5:1 heparin' then 30025
    when LOWER(pre.drug) = '.25 ns +1:1 heparin' then 30025
    when LOWER(pre.drug) = '.45ns + .5:1 heparin' then 30025
    when LOWER(pre.drug) = '.9ns + 1:1 heparin' then 30025
    when LOWER(pre.drug) = 'heparin(10 units/cc)' then 30025
    when LOWER(pre.drug) = 'pn d9.5 w/ heparin' then 30025
    when LOWER(pre.drug) = 'na acetate/heparin' then 30025
    when LOWER(pre.drug) = 'crrt heparin' then 30025
    when LOWER(pre.drug) = 'd10w with heparin' then 30025
    when LOWER(pre.drug) = 'tpnd9.5+heparin' then 30025
    when LOWER(pre.drug) = 'na acetate w/heparin' then 30025
    when LOWER(pre.drug) = 'heparin dose (per hour)' then 30025
    when LOWER(pre.drug) = 'heparin sodium (prophylaxis)' then 30025
    when LOWER(pre.drug) = 'heparin concentration (units/ml)' then 30025
    when LOWER(pre.drug) = 'heparin (hemodialysis)' then 30025
    
    --group: insulin (units)
    when LOWER(pre.drug) = 'insulin' then 4475
    when LOWER(pre.drug) = 'insulin human regular' then 223258
    when LOWER(pre.drug) = 'humulin-r insulin' then 223258
    when LOWER(pre.drug) = 'insulin pump' then 4475
    when LOWER(pre.drug) = 'humalog insulin' then 4475
    when LOWER(pre.drug) = 'insulin human nph' then 4475
    when LOWER(pre.drug) = 'insulin novolog' then 4475
    when LOWER(pre.drug) = 'insulin pump (self administering medication)' then 4475
    when LOWER(pre.drug) = 'insulin regular human (u-500)' then 4475
    when LOWER(pre.drug) = 'insulin regular pork (iletin ii)' then 4475
    when LOWER(pre.drug) = 'neo*iv*insulin (dilute)' then 4475

    --group: insulin glargine (units)
    when LOWER(pre.drug) = 'insulin glargine' then 223260
    when LOWER(pre.drug) = 'lantus' then 223260
  else null end as itemid--- added itemid for treatment

  ,pre.drug_name_poe
  ,pre.drug_name_generic
  ,pre.prod_strength
  ,pre.final_dosis
  ,pre.dose_unit_rx
  ,pre.form_val_disp
  ,pre.form_unit_disp
  ,pre.intime

  from all_prescriptions_join pre
  where pre.final_dosis IS NOT NULL  --- ¡no tenemos dosis vacías!
  order by pre.subject_id
)

SELECT 
  pre.subject_id
  ,pre.hadm_id
  ,pre.icustay_id
  ,pre.startdate
  ,pre.enddate
  ,pre.drug_type
  ,pre.drug
  ,pre.itemid
  ,pre.drug_name_poe
  ,pre.drug_name_generic
  ,pre.prod_strength
  ,REPLACE(REPLACE(pre.final_dosis,',',''),'regular0','0') as doses_administered ---- eliminar caracteres especiales de final_dosis
  ,pre.dose_unit_rx
  ,pre.intime
FROM temp_itemid pre
where (itemid = 7325 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 70166 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 1620287 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 221261 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 225151 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 225157 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 225906 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 1363 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 225908 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 225974 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 4147 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 54836 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 37707 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 1968 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 30115 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 227440 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 3349 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 42648 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 65374 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 286651 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 74554 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 85542 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 1101751 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 22252 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 17245 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 33228 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 223260 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 30112 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 221653 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 221662 and LOWER(dose_unit_rx) = 'mg') or
      (itemid = 30025 and LOWER(dose_unit_rx) = 'unit') or
      (itemid = 4475 and LOWER(dose_unit_rx) = 'unit') or
      (itemid = 223258 and LOWER(dose_unit_rx) = 'unit') or
      (itemid = 223260 and LOWER(dose_unit_rx) = 'unit') and 
      pre.final_dosis IS NOT NULL
order by pre.subject_id;

-- ******************************************************************
-- Paso 5: convertir columna 'doses_administered' de text a 'double precision'
drop view all_prescriptions_double;
create view all_prescriptions_double as
  select
  pre.subject_id as "SUBJECT_ID"
  ,pre.hadm_id as "HADM_ID"
  ,pre.icustay_id as "ICUSTAY_ID"
  ,pre.startdate as "CHARTTIME"
  ,pre.enddate as "ENDDATE"
  ,pre.drug_type as "DRUG_TYPE"
  ,pre.itemid as "ITEMID"
  ,LOWER(pre.drug) as "DRUG" 
  ,pre.drug_name_poe as "DRUG_NAME_POE"
  ,pre.drug_name_generic as "DRUG_NAME_GENERIC"
  ,pre.prod_strength as "PROD_STRENGTH"

  ,CASE
      WHEN pre.doses_administered !='0' then CAST(pre.doses_administered AS DOUBLE PRECISION)
  else null end as "VALUE" -- indica la dosis del meds administrado
  ,pre.dose_unit_rx as "VALUEUOM" -- unidad de medida (mg o unit)
  ,pre.intime as "INTIME"
  from all_prescriptions_itemid_dosis pre
  order by pre.subject_id;
\copy (SELECT * FROM all_prescriptions_double) to '/tmp/PRESCRIPTIONS.csv' CSV HEADER;
