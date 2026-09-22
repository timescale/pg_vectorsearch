-- Install-time schema choice
--
-- This test manages its own CREATE/DROP EXTENSION cycles (including a
-- SCHEMA clause and an expected-to-fail ALTER EXTENSION), so it runs
-- against its own dedicated temp instance (see test/pg/meson.build)
-- instead of the shared --load-extension=pg_vectorsearch schedule, which
-- installs pg_vectorsearch once up front and depends on it staying
-- installed for every other regression script that follows.

-- Checked after every (re)install below. Plain views rather than a
-- literal in each place they're needed, so the list of mkt-fixed and
-- prism-fixed functions each only needs updating in one spot.
-- Independent of any particular install (to_regprocedure just returns
-- NULL for a signature that isn't there), so they can be created once,
-- up front, and survive every CREATE/DROP EXTENSION cycle in this file.
CREATE VIEW mkt_procs_present AS
SELECT count(*) = 3 AS present FROM (VALUES
    ('mkt.git_commit()'),
    ('mkt.extension_version()'),
    ('mkt.extension_name()')
) AS t(sig)
WHERE to_regprocedure(sig) IS NOT NULL;

CREATE VIEW prism_procs_present AS
SELECT count(*) = 9 AS present FROM (VALUES
    ('prism.rebalance(regclass,integer)'),
    ('prism.split_posting_list(regclass,bigint)'),
    ('prism.posting_pages(regclass)'),
    ('prism.tids_clusters(regclass,tid[])'),
    ('prism.index_settings(regclass)'),
    ('prism.convert_posting_to_fastscan(regclass,integer)'),
    ('prism.centroid_pages(regclass)'),
    ('prism.setup_pgvector_compat()'),
    ('prism.on_extension_create()')
) AS t(sig)
WHERE to_regprocedure(sig) IS NOT NULL;

-- =====================================================================
-- 1. mkt ownership guard
-- =====================================================================
-- The mkt schema must be owned by the extension's installer or a
-- superuser; a pre-existing mkt owned by an untrusted role must refuse
-- the install rather than silently adopt it.

CREATE ROLE reloc_untrusted NOSUPERUSER;
CREATE SCHEMA mkt AUTHORIZATION reloc_untrusted;

CREATE EXTENSION pg_vectorsearch;

SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_vectorsearch')
    AS pg_vectorsearch_installed_after_refusal;

DROP SCHEMA mkt;

-- A fresh mkt (created by this install, owned by the installer) is fine.
CREATE EXTENSION pg_vectorsearch;

SELECT (SELECT r.rolname FROM pg_namespace n JOIN pg_roles r
            ON r.oid = n.nspowner WHERE n.nspname = 'mkt') = current_user
    AS mkt_owned_by_installer;

DROP EXTENSION pg_vectorsearch CASCADE;
DROP ROLE reloc_untrusted;

-- =====================================================================
-- 1b. A trusted pre-existing mkt is used, but not made an extension member
-- =====================================================================
-- mkt is verified and installed into, but ALTER EXTENSION ... ADD SCHEMA
-- is never run on it -- matching how PostgreSQL itself treats a
-- pre-existing @extschema@ for any ordinary relocatable extension (only
-- objects the install script itself creates become members). So anything
-- already in a pre-existing mkt that this install did not create must
-- survive both install and DROP EXTENSION CASCADE.

CREATE SCHEMA mkt;
CREATE TABLE mkt.unrelated_dba_table (id int);
INSERT INTO mkt.unrelated_dba_table VALUES (1), (2);

CREATE EXTENSION pg_vectorsearch;

SELECT count(*) = 2 AS unrelated_table_survives_install
    FROM mkt.unrelated_dba_table;

DROP EXTENSION pg_vectorsearch CASCADE;

SELECT to_regnamespace('mkt') IS NOT NULL AS mkt_schema_survives_drop;
SELECT count(*) = 2 AS unrelated_table_survives_drop
    FROM mkt.unrelated_dba_table;
SELECT to_regprocedure('mkt.git_commit()') IS NULL
    AS mkt_functions_removed;

DROP SCHEMA mkt CASCADE;

-- =====================================================================
-- 1c. A trusted pre-existing prism is used, but not made an extension
--     member (same guarantee as 1b, for the index-specific schema)
-- =====================================================================

CREATE SCHEMA prism;
CREATE TABLE prism.unrelated_dba_table (id int);
INSERT INTO prism.unrelated_dba_table VALUES (1), (2);

CREATE EXTENSION pg_vectorsearch;

SELECT count(*) = 2 AS unrelated_prism_table_survives_install
    FROM prism.unrelated_dba_table;

DROP EXTENSION pg_vectorsearch CASCADE;

SELECT to_regnamespace('prism') IS NOT NULL AS prism_schema_survives_drop;
SELECT count(*) = 2 AS unrelated_prism_table_survives_drop
    FROM prism.unrelated_dba_table;
SELECT to_regprocedure('prism.rebalance(regclass,integer)') IS NULL
    AS prism_functions_removed;

DROP SCHEMA prism CASCADE;

-- =====================================================================
-- 2. Default install (no SCHEMA clause): types land on search_path,
--    typically public; mkt and prism are separate.
-- =====================================================================

CREATE EXTENSION pg_vectorsearch;

SELECT to_regtype('public.vec32') IS NOT NULL
   AND to_regtype('public.vec16') IS NOT NULL
   AND to_regtype('public.rabitq') IS NOT NULL AS types_land_in_public;

SELECT to_regnamespace('mkt') IS NOT NULL
   AND to_regtype('mkt.vec32') IS NULL AS mkt_is_a_separate_schema;
SELECT to_regnamespace('prism') IS NOT NULL
   AND to_regtype('prism.vec32') IS NULL AS prism_is_a_separate_schema;

SELECT present AS all_mkt_procedures_present FROM mkt_procs_present;
SELECT present AS all_prism_procedures_present FROM prism_procs_present;

SELECT to_regprocedure('public.prism_handler(internal)') IS NOT NULL
    AS prism_handler_lives_in_public;

-- A real index build + ANN query, to prove this is not just a catalog
-- placement exercise.
CREATE TABLE default_items (id serial, v vec32(4));
INSERT INTO default_items (v)
    SELECT ARRAY[(i % 3)::real, (i % 2)::real, 0, 0]::vec32(4)
    FROM generate_series(1, 12) i;
INSERT INTO default_items (v)
    SELECT ARRAY[10 + (i % 3)::real, 10 + (i % 2)::real, 10, 10]::vec32(4)
    FROM generate_series(1, 12) i;
CREATE INDEX default_items_i ON default_items USING prism (v)
    WITH (nlist = 1, centroid_fastscan = off);
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
    SELECT id FROM default_items ORDER BY v <-> '[0,0,0,0]'::vec32(4) LIMIT 3;
SELECT id FROM default_items ORDER BY v <-> '[0,0,0,0]'::vec32(4) LIMIT 3;
RESET enable_seqscan;
DROP TABLE default_items;

DROP EXTENSION pg_vectorsearch CASCADE;

SELECT to_regnamespace('mkt') IS NULL AS mkt_dropped_with_the_extension;
SELECT to_regnamespace('prism') IS NULL AS prism_dropped_with_the_extension;
SELECT to_regtype('public.vec32') IS NULL
    AS public_vec32_dropped_with_the_extension;

-- =====================================================================
-- 3. Custom schema at install time
-- =====================================================================

CREATE SCHEMA reloc_a;
CREATE EXTENSION pg_vectorsearch SCHEMA reloc_a;

SELECT to_regtype('reloc_a.vec32') IS NOT NULL
   AND to_regtype('reloc_a.vec16') IS NOT NULL
   AND to_regtype('reloc_a.rabitq') IS NOT NULL
   AND to_regtype('reloc_a.rabitq_params') IS NOT NULL
    AS types_land_in_reloc_a;

SELECT count(*) = 6 AS prism_opclasses_land_in_reloc_a
    FROM pg_opclass oc
    JOIN pg_namespace n ON n.oid = oc.opcnamespace
    JOIN pg_am am ON am.oid = oc.opcmethod
   WHERE am.amname = 'prism' AND n.nspname = 'reloc_a';

SELECT count(*) = 3 AS distance_operators_land_in_reloc_a
    FROM pg_operator o
    JOIN pg_namespace n ON n.oid = o.oprnamespace
   WHERE n.nspname = 'reloc_a' AND o.oprname IN ('<->', '<#>', '<=>')
     AND o.oprleft = 'reloc_a.vec32'::regtype;

SELECT to_regnamespace('mkt') IS NOT NULL
   AND to_regtype('mkt.vec32') IS NULL AS mkt_is_still_separate;
SELECT to_regnamespace('prism') IS NOT NULL
   AND to_regtype('prism.vec32') IS NULL AS prism_is_still_separate;

SELECT present AS all_mkt_procedures_present FROM mkt_procs_present;
SELECT present AS all_prism_procedures_present FROM prism_procs_present;

-- The fixed prism procedures must operate correctly on an index whose
-- column type lives in a non-default schema -- they take a plain
-- regclass, so nothing about them should care where the type ended up.
CREATE TABLE reloc_a.maint_items (id serial, v reloc_a.vec32(4));
INSERT INTO reloc_a.maint_items (v)
    SELECT ARRAY[(i % 3)::real, (i % 2)::real, 0, 0]::reloc_a.vec32(4)
    FROM generate_series(1, 12) i;
INSERT INTO reloc_a.maint_items (v)
    SELECT ARRAY[10 + (i % 3)::real, 10 + (i % 2)::real, 10, 10]::reloc_a.vec32(4)
    FROM generate_series(1, 12) i;
CREATE INDEX maint_items_i ON reloc_a.maint_items
    USING prism (v reloc_a.vec32_l2_ops)
    WITH (nlist = 1, centroid_fastscan = off);

SELECT count(*) > 0 AS index_settings_works_on_a_reloc_a_index
    FROM prism.index_settings('reloc_a.maint_items_i');
SELECT count(*) > 0 AS centroid_pages_works_on_a_reloc_a_index
    FROM prism.centroid_pages('reloc_a.maint_items_i');
CALL prism.rebalance('reloc_a.maint_items_i'::regclass);

-- Drop the table first so the schema drop below cascades to exactly one
-- thing (the extension itself, reported as a single unit rather than each
-- of its member objects) -- deterministic regardless of catalog scan order.
DROP TABLE reloc_a.maint_items;
DROP SCHEMA reloc_a CASCADE;

-- Confirm pg_vectorsearch went with it (it was installed inside reloc_a).
SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_vectorsearch')
    AS pg_vectorsearch_survived_reloc_a_drop;

-- =====================================================================
-- 4. ALTER EXTENSION ... SET SCHEMA is refused (relocatable = false)
-- =====================================================================
-- Not an oversight: ALTER EXTENSION SET SCHEMA moves every member object
-- together into one schema, which would either strand the mkt- and
-- prism-pinned procedures away from their fixed schemas or refuse for
-- unrelated reasons. Refusing it outright, cleanly, up front is the
-- documented behaviour -- see the control file. The extension must
-- remain fully usable afterward.

CREATE SCHEMA reloc_a;
CREATE EXTENSION pg_vectorsearch SCHEMA reloc_a;
CREATE SCHEMA reloc_target;

ALTER EXTENSION pg_vectorsearch SET SCHEMA reloc_target;

SELECT to_regtype('reloc_a.vec32') IS NOT NULL
    AS types_still_in_reloc_a_after_refused_alter;
SELECT '[1,2,3]'::reloc_a.vec32::text = '[1,2,3]'
    AS pg_vectorsearch_still_queryable_after_refused_alter;

DROP SCHEMA reloc_target;
DROP EXTENSION pg_vectorsearch CASCADE;
DROP SCHEMA reloc_a CASCADE;
