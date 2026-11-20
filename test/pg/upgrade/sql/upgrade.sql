-- Update a database from the previous release. scripts/ci/upgrade-test.sh
-- runs this against a server with both releases installed.
\getenv from_version PGVS_UPGRADE_FROM
\getenv to_version PGVS_UPGRADE_TO
\getenv own_library PGVS_UPGRADE_OWN_LIBRARY
\getenv abs_builddir PG_ABS_BUILDDIR

-- Rebuild a prism index if the current version cannot read it, which is
-- the case when an earlier release built it in an older on-disk format.
-- Silent either way, so a test's output does not depend on whether the
-- release under test changed the format. Upgrade tests that run after
-- this one can call it too.
--
-- Reading the index's settings is the probe. For an older format it
-- reports that the index is not a prism index, so the access method is
-- checked here to tell the two apart.
CREATE PROCEDURE reindex_if_needed(index regclass) LANGUAGE plpgsql AS $$
BEGIN
    PERFORM * FROM prism_index_settings(index);
EXCEPTION WHEN index_corrupted OR wrong_object_type THEN
    IF NOT EXISTS (SELECT FROM pg_class c JOIN pg_am a ON a.oid = c.relam
                   WHERE c.oid = index AND a.amname = 'prism') THEN
        RAISE;
    END IF;
    EXECUTE format('REINDEX INDEX %s', index);
END
$$;

-- A database on the previous release, with an index that release built.
SET client_min_messages = error;
CREATE EXTENSION pg_vectorsearch VERSION :'from_version';
RESET client_min_messages;

SELECT extversion = :'from_version' AS on_previous_release
FROM pg_extension WHERE extname = 'pg_vectorsearch';

CREATE TABLE items (id int PRIMARY KEY, v vec32(8));

INSERT INTO items
SELECT i, ('[' || repeat(i || ',', 7) || i || ']')::vec32
FROM generate_series(1, 500) i;

CREATE INDEX items_v_idx ON items USING prism (v);

SET enable_seqscan = off;

SELECT id FROM items ORDER BY v <-> '[5,5,5,5,5,5,5,5]' LIMIT 1;

-- Update in this session, which has used the previous release. When that
-- release has a library of its own, the new library refuses to load next
-- to it and the update fails. When one library serves both, the update
-- goes through.
SET upgrade_test.own_library = :'own_library';

DO $$
DECLARE
    refused boolean := false;
BEGIN
    BEGIN
        ALTER EXTENSION pg_vectorsearch UPDATE;
    EXCEPTION WHEN object_not_in_prerequisite_state THEN
        refused := true;
    END;

    IF refused <> current_setting('upgrade_test.own_library')::boolean THEN
        RAISE EXCEPTION 'update in a session that used the previous '
            'release: refused = %', refused;
    END IF;
END
$$;

SELECT extversion = CASE WHEN :'own_library'::boolean
                         THEN :'from_version' ELSE :'to_version' END
       AS expected_version
FROM pg_extension WHERE extname = 'pg_vectorsearch';

-- A new session, which has not used the extension, updates it. It has
-- nothing left to do when the update above went through.
\c
SET client_min_messages = warning;
ALTER EXTENSION pg_vectorsearch UPDATE;
RESET client_min_messages;

SELECT extversion = :'to_version' AS on_current_version
FROM pg_extension WHERE extname = 'pg_vectorsearch';

SELECT pg_vectorsearch_version() = :'to_version' AS current_library;

-- Functions still bound to a library other than the current one.
SELECT p.oid::regprocedure AS function, p.probin
FROM pg_proc p
JOIN pg_depend d
  ON d.classid = 'pg_proc'::regclass AND d.objid = p.oid
 AND d.refclassid = 'pg_extension'::regclass AND d.deptype = 'e'
JOIN pg_extension e ON e.oid = d.refobjid
WHERE e.extname = 'pg_vectorsearch'
  AND p.probin <> '$libdir/pg_vectorsearch'
ORDER BY 1;

-- The updated catalog is the same as a fresh install of the current
-- version: every member of the extension, with its comment, privileges
-- and definition. Object identifiers differ between two databases, so a
-- definition names what it refers to.
\set fresh_file :abs_builddir '/fresh.catalog'
\set updated_db :DBNAME

SELECT $q$
    SELECT pg_describe_object(m.classid, m.objid, 0) AS member,
           obj_description(m.objid, m.classid::regclass::text) AS comment,
           CASE m.classid
           WHEN 'pg_proc'::regclass THEN
               (SELECT p.proacl::text FROM pg_proc p WHERE p.oid = m.objid)
           WHEN 'pg_type'::regclass THEN
               (SELECT t.typacl::text FROM pg_type t WHERE t.oid = m.objid)
           END AS privileges,
           CASE m.classid
           WHEN 'pg_proc'::regclass THEN
               (SELECT pg_get_functiondef(p.oid) FROM pg_proc p
                WHERE p.oid = m.objid AND p.prokind <> 'a')
           WHEN 'pg_type'::regclass THEN
               (SELECT jsonb_build_object(
                           'len', t.typlen, 'byval', t.typbyval,
                           'align', t.typalign, 'storage', t.typstorage,
                           'category', t.typcategory,
                           'preferred', t.typispreferred,
                           'delim', t.typdelim,
                           'element', t.typelem::regtype,
                           'array', t.typarray::regtype,
                           'input', t.typinput, 'output', t.typoutput,
                           'receive', t.typreceive, 'send', t.typsend,
                           'modin', t.typmodin, 'modout', t.typmodout,
                           'analyze', t.typanalyze,
                           'subscript', t.typsubscript,
                           'collation', t.typcollation::regcollation)::text
                FROM pg_type t WHERE t.oid = m.objid)
           WHEN 'pg_cast'::regclass THEN
               (SELECT jsonb_build_object(
                           'function', c.castfunc::regprocedure,
                           'context', c.castcontext,
                           'method', c.castmethod)::text
                FROM pg_cast c WHERE c.oid = m.objid)
           WHEN 'pg_operator'::regclass THEN
               (SELECT jsonb_build_object(
                           'result', o.oprresult::regtype,
                           'function', o.oprcode::regprocedure,
                           'commutator', o.oprcom::regoperator,
                           'negator', o.oprnegate::regoperator,
                           'restrict', o.oprrest, 'join', o.oprjoin,
                           'merges', o.oprcanmerge,
                           'hashes', o.oprcanhash)::text
                FROM pg_operator o WHERE o.oid = m.objid)
           WHEN 'pg_opclass'::regclass THEN
               (SELECT jsonb_build_object(
                           'family', f.opfname,
                           'type', c.opcintype::regtype,
                           'default', c.opcdefault,
                           'keytype', c.opckeytype::regtype)::text
                FROM pg_opclass c
                JOIN pg_opfamily f ON f.oid = c.opcfamily
                WHERE c.oid = m.objid)
           WHEN 'pg_opfamily'::regclass THEN
               (SELECT jsonb_build_object(
                           'operators',
                           (SELECT jsonb_agg(jsonb_build_object(
                                       'strategy', o.amopstrategy,
                                       'purpose', o.amoppurpose,
                                       'operator', o.amopopr::regoperator,
                                       'sortfamily',
                                       (SELECT s.opfname FROM pg_opfamily s
                                        WHERE s.oid = o.amopsortfamily))
                                   ORDER BY o.amopstrategy,
                                            o.amopopr::regoperator::text)
                            FROM pg_amop o WHERE o.amopfamily = m.objid),
                           'functions',
                           (SELECT jsonb_agg(jsonb_build_object(
                                       'number', f.amprocnum,
                                       'left', f.amproclefttype::regtype,
                                       'right', f.amprocrighttype::regtype,
                                       'function', f.amproc::regprocedure)
                                   ORDER BY f.amprocnum,
                                            f.amproc::regprocedure::text)
                            FROM pg_amproc f
                            WHERE f.amprocfamily = m.objid))::text)
           WHEN 'pg_am'::regclass THEN
               (SELECT jsonb_build_object(
                           'handler', a.amhandler, 'type', a.amtype)::text
                FROM pg_am a WHERE a.oid = m.objid)
           WHEN 'pg_event_trigger'::regclass THEN
               (SELECT jsonb_build_object(
                           'event', t.evtevent,
                           'function', t.evtfoid::regproc,
                           'enabled', t.evtenabled,
                           'tags', t.evttags)::text
                FROM pg_event_trigger t WHERE t.oid = m.objid)
           END AS definition
    FROM pg_depend m
    JOIN pg_extension e
      ON m.refclassid = 'pg_extension'::regclass AND m.refobjid = e.oid
    WHERE e.extname = 'pg_vectorsearch' AND m.deptype = 'e'
$q$ AS members \gset

CREATE TABLE updated_catalog AS :members;

CREATE DATABASE upgrade_fresh;
\c upgrade_fresh
SET client_min_messages = error;
CREATE EXTENSION pg_vectorsearch;
COPY (:members) TO :'fresh_file';

\c :updated_db
CREATE TABLE fresh_catalog (LIKE updated_catalog);
COPY fresh_catalog FROM :'fresh_file';

SELECT count(*) > 0 AS has_members FROM fresh_catalog;

-- Members the comparison has no definition for. A new kind of member
-- needs a branch in the query above.
SELECT member FROM fresh_catalog WHERE definition IS NULL ORDER BY 1;

-- Members that differ, from either side.
SELECT 'updated' AS only_in, * FROM
    (TABLE updated_catalog EXCEPT TABLE fresh_catalog) u
UNION ALL
SELECT 'fresh', * FROM
    (TABLE fresh_catalog EXCEPT TABLE updated_catalog) f
ORDER BY 2, 1;

DROP TABLE updated_catalog, fresh_catalog;
DROP DATABASE upgrade_fresh;

-- The index the previous release built answers from the index, after a
-- rebuild if its format is older.
CALL reindex_if_needed('items_v_idx');

SET enable_seqscan = off;

SELECT id FROM items ORDER BY v <-> '[5,5,5,5,5,5,5,5]' LIMIT 1;

EXPLAIN (COSTS OFF)
SELECT id FROM items ORDER BY v <-> '[5,5,5,5,5,5,5,5]' LIMIT 1;
