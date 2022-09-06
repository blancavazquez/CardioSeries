--**********************************************
--Extracción de troponin from Discharge notes
--estrategia: separar en varias vistas de acuerdo al número de regex usadas 
-- y luego unir las vistas

----artículo base:https://www.revespcardiol.org/es-marcadores-biologicos-necrosis-miocardica-articulo-13049653

--*****************regex 1
drop view extraccion_biomarcadores_discharge cascade;
create view extraccion_biomarcadores_discharge as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Discharge summary' --used for biomarkers of necrosis
),
temp_sin_espacios as(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,replace(replace(replace(texto, ' ', '*^'), '^*', ''), '*^', ' ') as texto_final
FROM temp_note_sin_saltos nt
order by nt.subject_id
)
--,
--temp_extraccion_value as (
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate::timestamp

      --Regex for biomarcadores de necrosis
      --ck-mb
     ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?[a-z]+? ?ck[-]?mb[-]? ?[>]?[0-9:]+)', 'g') as ckmb
     ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ck[(]?[c]?[p]?[k]?[)]?[-]?[[0-9:]+]?[*]? ?ck-mb[-]? ?[>]?[0-9:]+)', 'g') as ckmb1
     ,regexp_matches(LOWER(texto_final), '[,]? ?ck[-]?mb[-]?[>]? ?[a-z]*? ([0-9:]+)', 'g') as ckmb2

     --ck
     ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?ck[-]?[>]? ?[>]?[0-9:]+)', 'g') as ck
     ,regexp_matches(LOWER(texto_final), 'ck[-]?[>]? ?[a-z]*? ([0-9:]+)', 'g') as ck2

     --troponinT
     ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?[a-z]+? ?[a-z]+?[-]?[a-z]+?[-]? ?[>]?[0-9:]+? ?[c]?trop[a-z]?[a-z]?[-]? ?[0-9:]+.[0-9:]+)', 'g') as tropot
     ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?[c]?trop[a-z]?[a-z]?[-]? ?[>]?[0-9:]+.[0-9:]+)', 'g') as tropot2
     ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?[a-z]+? ?[a-z]+?[-]?[a-z]+?[-]? ?[a-z]+? ?[c]?trop[a-z]?[a-z]?[-]? ?[>]?[0-9:]+.[0-9:]+)', 'g') as tropot3
     ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?[a-z]+? ?[c]?[k]?[-]?[a-z]+?[-]? ?[0-9:]+?[*]? ?[a-z]+? ?[a-z]+?[-]?[0-9:]+?[.]?[0-9:]+?[*]? ?[c]?trop[a-z]?[a-z]?[-]? ?[>]?[0-9:]+.[0-9:]+)', 'g') as tropot4

      --mioglobina
     ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:]+[p]?[m]?[a]?[m]? ?[a-z]+? ?[a-z]+?[-]?[a-z]+?[-]?[0-9:]+?[*]? ?mb indx[-]?[0-9:]+.[0-9:]+)', 'g') as mb
     ,regexp_matches(LOWER(texto_final), '([\[\]0-9*-]+ ?[0-9:+]?[p]?[m]?[a]?[m]? ?ck[\(]?[a-z]+?[\)]?[-]?[0-9:]+?[*]? ?[a-z]+?[-]?[a-z]+?[-]?[0-9:]+?[*]? mb ?[a-z]+?[-]?[0-9:]+?[.]?[0-9:]+)', 'g') as mb2
     ,regexp_matches(LOWER(texto_final),'mb [a-z]*? ?([0-9:]+[.]?[0-9:]+)','g') as mb3

FROM temp_sin_espacios nt
order by nt.subject_id;

select * from extraccion_biomarcadores_discharge;
