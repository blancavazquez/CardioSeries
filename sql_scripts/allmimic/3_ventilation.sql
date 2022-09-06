--- Script for identifying the presence of a mechanical ventilation using settings
--- particularly, we identified mech vent, oxygen therapy and extubated
--- in a binary form.


-- source: https://github.com/MIT-LCP/mimic-code/blob/master/concepts/durations/ventilation_classification.sql
drop view all_vent_class;
create view all_vent_class as
select
  icustay_id, subject_id, hadm_id, charttime
  -- case statement determining whether it is an instance of mech vent
  , max(
    case
      when itemid is null or value is null then 0 -- can't have null values
      when itemid = 720 and value != 'Other/Remarks' THEN 1  -- VentTypeRecorded
      when itemid = 223848 and value != 'Other' THEN 1
      when itemid = 223849 then 1 -- ventilator mode
      when itemid = 467 and value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
      when itemid in
        (
        445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
        , 639, 654, 681, 682, 683, 684,224685,224684,224686 -- tidal volume
        , 218,436,535,444,459,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
        , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
        , 543 -- PlateauPressure
        , 5865,5866,224707,224709,224705,224706 -- APRV pressure
        , 60,437,505,506,686,220339,224700 -- PEEP
        , 3459 -- high pressure relief
        , 501,502,503,224702 -- PCV
        , 223,667,668,669,670,671,672 -- TCPCV
        , 224701 -- PSVlevel
        )
        THEN 1
      else 0
    end
    ) as MechVent
    , max(
      case
        -- initiation of oxygen therapy indicates the ventilation has ended
        when itemid = 226732 and value in
        (
          'Nasal cannula', -- 153714 observations
          'Face tent', -- 24601 observations
          'Aerosol-cool', -- 24560 observations
          'Trach mask ', -- 16435 observations
          'High flow neb', -- 10785 observations
          'Non-rebreather', -- 5182 observations
          'Venti mask ', -- 1947 observations
          'Medium conc mask ', -- 1888 observations
          'T-piece', -- 1135 observations
          'High flow nasal cannula', -- 925 observations
          'Ultrasonic neb', -- 9 observations
          'Vapomist' -- 3 observations
        ) then 1
        when itemid = 467 and value in
        (
          'Cannula', -- 278252 observations
          'Nasal Cannula', -- 248299 observations
          -- 'None', -- 95498 observations
          'Face Tent', -- 35766 observations
          'Aerosol-Cool', -- 33919 observations
          'Trach Mask', -- 32655 observations
          'Hi Flow Neb', -- 14070 observations
          'Non-Rebreather', -- 10856 observations
          'Venti Mask', -- 4279 observations
          'Medium Conc Mask', -- 2114 observations
          'Vapotherm', -- 1655 observations
          'T-Piece', -- 779 observations
          'Hood', -- 670 observations
          'Hut', -- 150 observations
          'TranstrachealCat', -- 78 observations
          'Heated Neb', -- 37 observations
          'Ultrasonic Neb' -- 2 observations
        ) then 1
      else 0
      end
    ) as OxygenTherapy
    , max(
      case when itemid is null or value is null then 0
        -- extubated indicates ventilation event has ended
        when itemid = 640 and value = 'Extubated' then 1
        when itemid = 640 and value = 'Self Extubation' then 1
      else 0
      end
      )
      as Extubated
    , max(
      case when itemid is null or value is null then 0
        when itemid = 640 and value = 'Self Extubation' then 1
      else 0
      end
      )
      as SelfExtubated
from chartevents ce
--inner join STEMI_patients st
--on ce.subject_id = st.subject_id
where ce.value is not null
-- exclude rows marked as error
and (ce.error != 1 or ce.error IS NULL)
and itemid in
(
    -- the below are settings used to indicate ventilation
      720, 223849 -- vent mode
    , 223848 -- vent type
    , 445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
    , 639, 654, 681, 682, 683, 684,224685,224684,224686 -- tidal volume
    , 218,436,535,444,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean ("RespPressure")
    , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
    , 543 -- PlateauPressure
    , 5865,5866,224707,224709,224705,224706 -- APRV pressure
    , 60,437,505,506,686,220339,224700 -- PEEP
    , 3459 -- high pressure relief
    , 501,502,503,224702 -- PCV
    , 223,667,668,669,670,671,672 -- TCPCV
    , 224701 -- PSVlevel

    -- the below are settings used to indicate extubation
    , 640 -- extubated

    -- the below indicate oxygen/NIV, i.e. the end of a mechanical vent event
    , 468 -- O2 Delivery Device#2
    , 469 -- O2 Delivery Mode
    , 470 -- O2 Flow (lpm)
    , 471 -- O2 Flow (lpm) #2
    , 227287 -- O2 Flow (additional cannula)
    , 226732 -- O2 Delivery Device(s)
    , 223834 -- O2 Flow

    -- used in both oxygen + vent calculation
    , 467 -- O2 Delivery Device
)
group by icustay_id, subject_id, hadm_id, charttime
UNION DISTINCT
-- add in the extubation flags from procedureevents_mv
-- note that we only need the start time for the extubation
-- (extubation is always charted as ending 1 minute after it started)
select
  icustay_id, subject_id, hadm_id, starttime as charttime
  , 0 as MechVent
  , 0 as OxygenTherapy
  , 1 as Extubated
  , case when itemid = 225468 then 1 else 0 end as SelfExtubated
from procedureevents_mv
where itemid in
(
  227194 -- "Extubation"
, 225468 -- "Unplanned Extubation (patient-initiated)"
, 225477 -- "Unplanned Extubation (non-patient initiated)"
);
--\copy (SELECT * FROM all_vent_class) to '/tmp/all_vent_class.csv' CSV HEADER;

--***************************************
-- Paso2: added itemid, value and valueom for ventilation type, also
-- we did inner join with STEMI_icustays to add subject_id, hadm_id

--Self-extubation, defined as a deliberate action taken by the patient to remove the endotracheal tube
--source:https://pulmonarychronicles.com/index.php/pulmonarychronicles/article/view/169/392

drop view all_vent_itemid;
create view all_vent_itemid as
with temp_vent as
(
select 
   vt.icustay_id
   ,vt.subject_id
   ,vt.hadm_id
  ,vt.charttime
,case --- itemids taken from D_ITEMS
    when vt.mechVent = 1 then 226260 --- MechVent else 
    when vt.oxygentherapy = 1 then 226732 --- OxygenTherapy (O2 Delivery Device(s))
    when vt.extubated = 1 then 227194 --- Extubated 
else null end as itemid--- added itemid for ventilation type

from all_vent_class vt
order by vt.icustay_id
)
select 
   vt.subject_id as "SUBJECT_ID"
  ,vt.hadm_id "HADM_ID"
  ,vt.icustay_id as "ICUSTAY_ID"
  ,vt.charttime as "CHARTTIME"
  ,vt.itemid as "ITEMID"

  ,case
  when vt.itemid in (226260, 226732,227194) then 1
  else null end as "VALUE"--- added value for itemid

  ,case
  when vt.itemid in (226260, 226732,227194) then 'binary'
  else null end as "VALUEUOM"--- added value for itemid

from temp_vent vt
--inner join STEMI_icustays icu
--on vt.icustay_id = icu."ICUSTAY_ID"
where itemid IS NOT NULL 
order by vt.icustay_id;

\copy (SELECT * FROM all_vent_itemid) to '/tmp/VENTILATION.csv' CSV HEADER;
