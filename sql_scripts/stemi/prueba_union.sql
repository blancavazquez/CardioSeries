drop table prueba1;
create table prueba1 as
select nt."SUBJECT_ID"
      ,nt."HADM_ID"
      ,nt."CHARTDATE"
      ,nt."CHARTTIME"
      ,nt."ITEMID"
      ,nt."VALUE"
      ,nt."VALUEUOM"
from ck_discharge_temp2 nt
where nt."SUBJECT_ID"=24
order by nt."SUBJECT_ID";
select * from prueba1;

drop table prueba2;
create table prueba2 as
select nt."SUBJECT_ID"
      ,nt."HADM_ID"
      ,nt."CHARTDATE"
      ,nt."CHARTTIME"
      ,nt."ITEMID"
      ,nt."VALUE"
      ,nt."VALUEUOM"
from ck_discharge_temp2 nt
where nt."SUBJECT_ID"=24
order by nt."SUBJECT_ID";
select * from prueba2;

UPDATE prueba2 SET "VALUE"=100 WHERE "SUBJECT_ID"=24;

INSERT INTO prueba2("SUBJECT_ID", "HADM_ID", "CHARTDATE","CHARTTIME","ITEMID","VALUE","VALUEUOM")
VALUES (24, 161859, '2139-06-09 00:00:00', '2139-06-09 00:00:00',50910,200,'iu/l');

drop view union_prueba;
create view union_prueba as
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from prueba1
UNION
select
"SUBJECT_ID"
,"HADM_ID"
,"CHARTDATE"
,"CHARTTIME"
,"ITEMID"
,"VALUE"
,"VALUEUOM"
from prueba2
order by "SUBJECT_ID";
select * from union_prueba;