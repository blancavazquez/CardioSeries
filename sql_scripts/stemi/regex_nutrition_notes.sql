--**********************************************
--Extracción de troponin from Nutrition notes
--estrategia: separar en varias vistas de acuerdo al número de regex usadas 
-- y luego unir las vistas

----artículo base:https://www.revespcardiol.org/es-marcadores-biologicos-necrosis-miocardica-articulo-13049653

--*****************regex 1
drop view extraccion_laboratory_nutrition cascade;
create view extraccion_laboratory_nutrition as
with temp_note_sin_saltos as
(
select nt.subject_id
      ,nt.hadm_id
      ,nt.chartdate
      ,text(replace(nt.text,chr(10),' ')) as texto
FROM noteevents nt
inner join STEMI_patients st
on nt.subject_id = st.subject_id
where category = 'Nutrition' ---for laboratory
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

     ----Regex for laboratory values
     ,regexp_matches(LOWER(texto_final), '(glucose ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as glucose
     ,regexp_matches(LOWER(texto_final), '(glucose finger stick ?[0-9:]+ ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as glucose_finger
     ,regexp_matches(LOWER(texto_final), '(bun ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as bun
     ,regexp_matches(LOWER(texto_final), '(creatinine ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as creatinine
     ,regexp_matches(LOWER(texto_final), '(sodium ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as sodium
     ,regexp_matches(LOWER(texto_final), '(potassium ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as potassium
     ,regexp_matches(LOWER(texto_final), '(chloride ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as chloride
     ,regexp_matches(LOWER(texto_final), '(tco2 ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as tco2
     ,regexp_matches(LOWER(texto_final), '(po2 \(arterial\) ?[0-9:]+ ?[a-z:]+? [a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as po2_arterial
     ,regexp_matches(LOWER(texto_final), '(pco2 \(arterial\) ?[0-9:]+ ?[a-z:]+? ?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as pco2_arterial
     ,regexp_matches(LOWER(texto_final), '(ph \(arterial\) ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as ph
     ,regexp_matches(LOWER(texto_final), '(co2 \(calc\) arterial ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as co2
     ,regexp_matches(LOWER(texto_final), '(albumin ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as albumin
     ,regexp_matches(LOWER(texto_final), '(calcium non-ionized ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as calcium
     ,regexp_matches(LOWER(texto_final), '(phosphorus ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as phosphorus
     ,regexp_matches(LOWER(texto_final), '(ionized calcium ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as ionized_calcium
     ,regexp_matches(LOWER(texto_final), '(magnesium ?[0-9:].[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as magnesium
     ,regexp_matches(LOWER(texto_final), '(alkaline phosphate ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as alkaline_phosphate
     ,regexp_matches(LOWER(texto_final), '(ast ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as ast
     ,regexp_matches(LOWER(texto_final), '(amylase ?[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as amylase
     ,regexp_matches(LOWER(texto_final), '(total bilirubin ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as total_bilirubin
     ,regexp_matches(LOWER(texto_final), '(wbc ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as wbc
     ,regexp_matches(LOWER(texto_final), '(hgb ?[0-9:]+.[0-9:]+ ?[a-z:]+?[/]?[a-z:]+? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as hgb
     ,regexp_matches(LOWER(texto_final), '(hematocrit ?[0-9:]+.[0-9:]+ [%]? ?[\[\]0-9*-]+ ?[0-9:]+ ?[p]?[m]?[a]?[m]?)', 'g') as hematocrit



FROM temp_sin_espacios nt
order by nt.subject_id;

select * from extraccion_laboratory_nutrition;
