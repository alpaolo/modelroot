"""
Neo4j Cypher queries used by the ModelRoot Streamlit app.
QUERY_MODULE_VERSION = 2 (model picker cyphers)
"""
QUERY_MODULE_VERSION = 2

LICENSE_GROUP_METADATA_CYPHER = """
MATCH (g:LicenseGroup)
RETURN g.id AS id, g.name AS name, g.compliance AS compliance
ORDER BY id
""".strip()

DISTINCT_TASKS_CYPHER = """
MATCH (m:Model)-[:PERFORMS]->(t:Task)
RETURN DISTINCT t.name AS task
ORDER BY task
""".strip()

DISTINCT_BRANDS_CYPHER = """
MATCH (m:Model)-[:PUBLISHED_BY]->(b:MainBrand)
RETURN DISTINCT b.name AS brand
ORDER BY brand
""".strip()

MODEL_CATALOG_CYPHER = """
MATCH (m:Model)
WHERE ($search_pattern IS NULL OR m.name =~ $search_pattern)
  AND (NOT $in_benchmark OR m.oll_rank IS NOT NULL)
  AND (size($tasks) = 0 OR EXISTS {
    MATCH (m)-[:PERFORMS]->(t:Task)
    WHERE t.name IN $tasks
  })
  AND (size($brands) = 0 OR EXISTS {
    MATCH (m)-[:PUBLISHED_BY]->(b:MainBrand)
    WHERE b.name IN $brands
  })
OPTIONAL MATCH (m)-[:UNDER_LICENSE]->(l:License)
OPTIONAL MATCH (l)-[:BELONGS_TO]->(g:LicenseGroup)
WITH m, l, g
WHERE size($license_groups) = 0 OR coalesce(g.id, 'UNKNOWN') IN $license_groups
OPTIONAL MATCH (m)-[:PERFORMS]->(t:Task)
WITH m, l, g, head(collect(DISTINCT coalesce(t.name, '—'))) AS task
OPTIONAL MATCH (m)-[:PUBLISHED_BY]->(b:MainBrand)
WITH m, l, g, task, head(collect(DISTINCT coalesce(b.name, '—'))) AS brand
RETURN m.name AS model,
       brand,
       task,
       coalesce(l.name, 'unknown') AS license,
       coalesce(g.id, 'UNKNOWN') AS license_group,
       coalesce(g.name, 'Unclassified') AS risk_level,
       coalesce(g.compliance, 'License not mapped to a risk group yet.') AS risk_guidance,
       coalesce(m.downloads, 0) AS downloads,
       CASE WHEN m.oll_rank IS NOT NULL THEN 'Yes' ELSE '—' END AS benchmark,
       m.hf_url AS hf_url,
       m.license_link AS license_link
ORDER BY downloads DESC
LIMIT $limit
""".strip()

MODEL_DETAIL_CYPHER = """
MATCH (m:Model {name: $model_name})
OPTIONAL MATCH (m)-[:UNDER_LICENSE]->(l:License)
OPTIONAL MATCH (l)-[:BELONGS_TO]->(g:LicenseGroup)
OPTIONAL MATCH (m)-[:PERFORMS]->(t:Task)
OPTIONAL MATCH (t)<-[:GROUPS]-(td:TechDomain)
OPTIONAL MATCH (m)-[:PUBLISHED_BY]->(b:MainBrand)
WITH m, l, g,
     head(collect(DISTINCT t.name)) AS task,
     head(collect(DISTINCT b.name)) AS brand,
     collect(DISTINCT td.name) AS tech_domain_names
RETURN m.name AS model,
       coalesce(l.name, 'unknown') AS license,
       coalesce(g.id, 'UNKNOWN') AS license_group,
       g.name AS license_group_name,
       g.compliance AS license_group_compliance,
       coalesce(task, '—') AS task,
       [name IN tech_domain_names WHERE name IS NOT NULL | name] AS tech_domain_names,
       coalesce(brand, '—') AS brand,
       coalesce(m.downloads, 0) AS downloads,
       m.hf_url AS hf_url,
       m.license_link AS license_link,
       m.oll_average AS oll_average,
       m.oll_rank AS oll_rank,
       m.oll_ifeval AS oll_ifeval,
       m.oll_bbh AS oll_bbh,
       m.oll_math_lvl5 AS oll_math_lvl5,
       m.oll_gpqa AS oll_gpqa,
       m.oll_musr AS oll_musr,
       m.oll_mmlu_pro AS oll_mmlu_pro,
       m.oll_params_b AS oll_params_b,
       m.oll_submission_date AS oll_submission_date
""".strip()

ALL_MODEL_NAMES_BY_DOWNLOADS_CYPHER = """
MATCH (m:Model)
RETURN m.name AS model
ORDER BY m.downloads DESC
""".strip()

MODEL_PICKER_BROWSE_CYPHER = """
MATCH (m:Model)
RETURN m.name AS model
ORDER BY coalesce(m.downloads, 0) DESC
LIMIT $limit
""".strip()

MODEL_PICKER_SEARCH_CYPHER = """
MATCH (m:Model)
WHERE m.name =~ $search_pattern
RETURN m.name AS model
ORDER BY CASE WHEN m.oll_rank IS NOT NULL THEN m.oll_rank ELSE 999999 END ASC,
         coalesce(m.downloads, 0) DESC
LIMIT $limit
""".strip()

DERIVED_MODELS_CYPHER = """
MATCH (derivative:Model)-[:DERIVED_FROM]->(base:Model {name: $model_name})
OPTIONAL MATCH (derivative)-[:UNDER_LICENSE]->(l:License)
RETURN derivative.name AS model,
       coalesce(derivative.downloads, 0) AS downloads,
       coalesce(l.name, 'unknown') AS license,
       derivative.hf_url AS hf_url
ORDER BY derivative.downloads DESC
LIMIT $limit
""".strip()

MODEL_NEIGHBORHOOD_CYPHER = """
MATCH (m:Model {name: $model_name})-[r]->(n)
RETURN type(r) AS relation,
       CASE
           WHEN labels(n)[0] = 'Paper' THEN n.id
           ELSE coalesce(n.name, n.id)
       END AS entity,
       labels(n)[0] AS entity_type,
       CASE
           WHEN labels(n)[0] = 'Paper'
           THEN coalesce(n.url, 'https://arxiv.org/abs/' + n.id)
           ELSE null
       END AS entity_url,
       m.name AS source,
       coalesce(n.name, n.id) AS target,
       'out' AS direction
UNION
MATCH (n)-[r]->(m:Model {name: $model_name})
RETURN type(r) AS relation,
       CASE
           WHEN labels(n)[0] = 'Paper' THEN n.id
           ELSE coalesce(n.name, n.id)
       END AS entity,
       labels(n)[0] AS entity_type,
       CASE
           WHEN labels(n)[0] = 'Paper'
           THEN coalesce(n.url, 'https://arxiv.org/abs/' + n.id)
           ELSE null
       END AS entity_url,
       coalesce(n.name, n.id) AS source,
       m.name AS target,
       'in' AS direction
LIMIT $limit
""".strip()

TOP_MODELS_BY_RISK_CYPHER = """
MATCH (m:Model)-[:UNDER_LICENSE]->(l:License)
OPTIONAL MATCH (l)-[:BELONGS_TO]->(g:LicenseGroup)
OPTIONAL MATCH (m)-[:PUBLISHED_BY]->(b:MainBrand)
RETURN coalesce(g.id, 'UNKNOWN') AS license_group,
       coalesce(g.name, 'Unclassified') AS risk_level,
       coalesce(g.compliance, 'License not mapped to a risk group yet.') AS risk_guidance,
       m.name AS model,
       coalesce(b.name, '—') AS brand,
       coalesce(l.name, 'unknown') AS license,
       coalesce(m.downloads, 0) AS downloads,
       m.hf_url AS hf_url
ORDER BY m.downloads DESC
LIMIT $limit
""".strip()

MODELS_BY_LICENSE_GROUP_CYPHER = """
MATCH (m:Model)-[:UNDER_LICENSE]->(l:License)
OPTIONAL MATCH (l)-[:BELONGS_TO]->(g:LicenseGroup)
RETURN coalesce(g.id, 'UNKNOWN') AS group_id, count(m) AS models
ORDER BY models DESC
""".strip()

TOP_LICENSES_CYPHER = """
MATCH (m:Model)-[:UNDER_LICENSE]->(l:License)
RETURN l.name AS license, count(m) AS models
ORDER BY models DESC
LIMIT $limit
""".strip()

TOP_DATASETS_BY_USAGE_CYPHER = """
MATCH (m:Model)-[:USED_DATASET]->(d:Dataset)
RETURN d.name AS dataset, count(m) AS uses
ORDER BY uses DESC
LIMIT $limit
""".strip()

MODELS_USING_DATASET_CYPHER = """
MATCH (m:Model)-[:USED_DATASET]->(d:Dataset {name: $dataset_name})
RETURN m.name AS model
ORDER BY m.downloads DESC
LIMIT $limit
""".strip()

RELATIONSHIP_TYPES_CYPHER = """
CALL db.relationshipTypes()
YIELD relationshipType
RETURN relationshipType AS rel
""".strip()

GRAPH_EXPLORER_SUBGRAPH_CYPHER = """
MATCH (m:Model {name: $center_model})-[r]->(target)
WHERE type(r) IN $rels
RETURN m.name AS model,
       coalesce(target.name, target.id) AS target_name,
       type(r) AS relation
UNION
MATCH (source)-[r]->(m:Model {name: $center_model})
WHERE type(r) IN $rels
RETURN m.name AS model,
       coalesce(source.name, source.id) AS target_name,
       type(r) AS relation
LIMIT $limit
""".strip()

DATABASE_SNAPSHOT_CYPHER = """
MATCH (m:Model) WITH count(m) AS models
MATCH (l:License) WITH models, count(l) AS licenses
MATCH (g:LicenseGroup) RETURN models, licenses, count(g) AS groups
""".strip()
