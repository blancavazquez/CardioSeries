-- El objetivo de este script, es únicamente identificar los item_id
-- de los medicamentos, para posterior añadirlo al archivo de itemid_to_variable_map
-- Esta búsqueda de item_id debe estar alineada con los medicamentos
-- ya identificados en el archivo: treatment_list
--- source: https://opendata.stackexchange.com/questions/6723/mimic-iii-inputevents-mv-or-cv/6873

--Grupo:Aspirin
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%aspirin%';

--Grupo: Clopidogrel
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%clopidogrel%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%plavix%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%bisulfate%';

--Grupo: Prasugrel
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%prasugrel%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%effient%';

--Grupo: Ticagrelor
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%ticagrelor%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%brilinta%';

--Grupo: cangrelor
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%cangrelor%';

--Grupo: Abciximab
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%abciximab%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%reopro%';

--Grupo: Eptifibatide
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%eptifibatide%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%integrilin%';

--Grupo: Tirofiban
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%tirofiban%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%aggrastat%';

--Grupo: Unfractionated heparin
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%heparin%';

--Grupo:enoxaparin
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%enoxaparin%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%lovenox%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%clexane%';

--Grupo:bivalirudin
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%bivalirudin%';

--Grupo:fondaparinux
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%fondaparinux%';

--Grupo:metoprolol
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%metoprolol%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%betaloc%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%spesicor%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%lopressor%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%toprol%';


--Grupo:bisoprolol
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%bisoprolol%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%concor%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%ziac%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%maxsoten%';

--Grupo:atenolol
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%atenolol%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%tenormin%';

--Grupo:carvedilol
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%carvedilol%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%coreg%';

--Grupo:sotalol
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%sotalol%';

--Grupo:verapamil
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%verapamil%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%iproveratril%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%calan%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%covera%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%isoptin%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%verelan%';

--Grupo:diltiazem
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%diltiazem%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%cardizem%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%cartia%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%dilacor%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%dilt%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%taztia%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%tiamate%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%tiazac%';

--Grupo:digoxin
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%digoxin%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%cardoxin%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%digitek%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%lanoxicaps%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%dilanacin%';

--Grupo:captopril
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%captopril%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%capoten%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%kaplon%';

--Grupo:enalapril
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%enalapril%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%vasotec%';

--Grupo:atorvastatin
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%atorvastatin%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%lipitor%';

--Grupo:simvastatin
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%simvastatin%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%lovastatin%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%velastatin%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%synvinolin%';

--Grupo:pravastatin
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%pravachol%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%pravastatin%';

--Grupo:rosuvastatin
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%rosuvastatin%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%ezallor%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%crestor%';

--Grupo:oral_nitrates
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%sosorbide%';

--Grupo:gemfibrozil
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%gemfibrozil%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%lopid%';

--Grupo:bezafibrate
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%bezafibrate%';

--Grupo:fenofibrate
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%fenofibrate%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%cedur%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%tricor%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%fibricor%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%lofibra%';

--Grupo:insulin
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%insulin%';

--Grupo:amiodarone
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%amiodarone%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%cordarone%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%pacerone%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%amiodarone%';

--Grupo:dobutamine
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%dobutamine%';

select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%dobutrex%';

--Grupo:dopamine
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%dopamine%';

--Grupo:levosimendan
select itemid, label, dbsource, linksto
from d_items 
where lower(label) like '%levosimendan%';