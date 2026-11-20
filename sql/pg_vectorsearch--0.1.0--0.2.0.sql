/*
 * pg_vectorsearch 0.1.0 -> 0.2.0-dev
 *
 * 0.1.0 bound every C function to pg_vectorsearch-0.1.0.so. This script
 * rebinds them to the one library that serves every later version;
 * PostgreSQL resolves MODULE_PATHNAME from pg_vectorsearch.control.
 *
 * Run it in a session that has not used the extension. PostgreSQL loads
 * the new library to validate each function, and that library refuses to
 * load next to the 0.1.0 one.
 *
 * Indexes built by 0.1.0 have an older on-disk format. REINDEX each prism
 * index after the update.
 */

-- complain if script is sourced in psql, rather than via ALTER EXTENSION
\echo Use "ALTER EXTENSION pg_vectorsearch UPDATE" to load this file.\quit

-- =====================================================================
-- Rebind every C function to the unversioned library. The statements
-- are 0.1.0's own definitions with CREATE OR REPLACE, in the order
-- 0.1.0 created them.
-- =====================================================================

CREATE OR REPLACE FUNCTION pg_vectorsearch_git_commit() RETURNS text
    AS 'MODULE_PATHNAME', 'vs_git_commit'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION pg_vectorsearch_version() RETURNS text
    AS 'MODULE_PATHNAME', 'vs_extension_version'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_in(cstring, oid, integer) RETURNS vec32
    AS 'MODULE_PATHNAME', 'vs_vec32_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_out(vec32) RETURNS cstring
    AS 'MODULE_PATHNAME', 'vs_vec32_out'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_typmod_in(cstring[]) RETURNS integer
    AS 'MODULE_PATHNAME', 'vs_vec32_typmod_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION l2_distance(vec32, vec32) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_l2_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION inner_product(vec32, vec32) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION cosine_distance(vec32, vec32) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_cosine_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_dims(vec32) RETURNS integer
    AS 'MODULE_PATHNAME', 'vs_vec32_dims'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_norm(vec32) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_pg_vec32_norm'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_l2_squared_distance(vec32, vec32) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_vec32_l2_squared_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_negative_inner_product(vec32, vec32) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_vec32_negative_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_cmp(vec32, vec32) RETURNS int4
    AS 'MODULE_PATHNAME', 'vs_vec32_cmp'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_lt(vec32, vec32) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_vec32_lt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_le(vec32, vec32) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_vec32_le'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_eq(vec32, vec32) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_vec32_eq'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_ne(vec32, vec32) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_vec32_ne'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_ge(vec32, vec32) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_vec32_ge'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_gt(vec32, vec32) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_vec32_gt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32(@extschema@.vec32, integer, boolean) RETURNS vec32
    AS 'MODULE_PATHNAME', 'vs_vec32'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION array_to_vec32(real[], integer, boolean) RETURNS vec32
    AS 'MODULE_PATHNAME', 'vs_array_to_vec32'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION array_to_vec32(float8[], integer, boolean) RETURNS vec32
    AS 'MODULE_PATHNAME', 'vs_array_to_vec32'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_to_float4(@extschema@.vec32) RETURNS real[]
    AS 'MODULE_PATHNAME', 'vs_vec32_to_float4'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16_in(cstring, oid, integer) RETURNS vec16
    AS 'MODULE_PATHNAME', 'vs_vec16_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16_out(vec16) RETURNS cstring
    AS 'MODULE_PATHNAME', 'vs_vec16_out'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16_typmod_in(cstring[]) RETURNS integer
    AS 'MODULE_PATHNAME', 'vs_vec16_typmod_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION l2_distance(vec16, vec16) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_vec16_l2_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION inner_product(vec16, vec16) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_vec16_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION cosine_distance(vec16, vec16) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_vec16_cosine_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_dims(vec16) RETURNS integer
    AS 'MODULE_PATHNAME', 'vs_vec16_dims'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_norm(vec16) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_vec16_norm'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16_l2_squared_distance(vec16, vec16) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_vec16_l2_squared_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16_negative_inner_product(vec16, vec16) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_vec16_negative_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16_cmp(vec16, vec16) RETURNS int4
    AS 'MODULE_PATHNAME', 'vs_vec16_cmp'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16_lt(vec16, vec16) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_vec16_lt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16_le(vec16, vec16) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_vec16_le'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16_eq(vec16, vec16) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_vec16_eq'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16_ne(vec16, vec16) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_vec16_ne'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16_ge(vec16, vec16) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_vec16_ge'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16_gt(vec16, vec16) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_vec16_gt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16(@extschema@.vec16, integer, boolean) RETURNS vec16
    AS 'MODULE_PATHNAME', 'vs_vec16'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec16_to_vec32(@extschema@.vec16, integer, boolean)
    RETURNS vec32
    AS 'MODULE_PATHNAME', 'vs_vec16_to_vec32'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION vec32_to_vec16(@extschema@.vec32, integer, boolean)
    RETURNS vec16
    AS 'MODULE_PATHNAME', 'vs_vec32_to_vec16'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION array_to_vec16(real[], integer, boolean) RETURNS vec16
    AS 'MODULE_PATHNAME', 'vs_array_to_vec16'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION array_to_vec16(float8[], integer, boolean) RETURNS vec16
    AS 'MODULE_PATHNAME', 'vs_array_to_vec16'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_in(cstring, oid, integer) RETURNS rabitq
    AS 'MODULE_PATHNAME', 'vs_rabitq_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_out(rabitq) RETURNS cstring
    AS 'MODULE_PATHNAME', 'vs_rabitq_out'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_typmod_in(cstring[]) RETURNS integer
    AS 'MODULE_PATHNAME', 'vs_rabitq_typmod_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_dims(rabitq) RETURNS integer
    AS 'MODULE_PATHNAME', 'vs_rabitq_dims'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_f_add(rabitq) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_rabitq_f_add'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_f_rescale(rabitq) RETURNS float8
    AS 'MODULE_PATHNAME', 'vs_rabitq_f_rescale'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_cmp(rabitq, rabitq) RETURNS int4
    AS 'MODULE_PATHNAME', 'vs_rabitq_cmp'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_lt(rabitq, rabitq) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_rabitq_lt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_le(rabitq, rabitq) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_rabitq_le'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_eq(rabitq, rabitq) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_rabitq_eq'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_ne(rabitq, rabitq) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_rabitq_ne'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_ge(rabitq, rabitq) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_rabitq_ge'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_gt(rabitq, rabitq) RETURNS bool
    AS 'MODULE_PATHNAME', 'vs_rabitq_gt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq(@extschema@.rabitq, integer, boolean) RETURNS rabitq
    AS 'MODULE_PATHNAME', 'vs_rabitq'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_params_in(cstring, oid, integer)
    RETURNS rabitq_params
    AS 'MODULE_PATHNAME', 'vs_rabitq_params_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_params_out(rabitq_params) RETURNS cstring
    AS 'MODULE_PATHNAME', 'vs_rabitq_params_out'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_params_generate(dim integer, seed bigint)
    RETURNS rabitq_params
    AS 'MODULE_PATHNAME', 'vs_rabitq_params_generate_pg'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_params_dim(rabitq_params) RETURNS integer
    AS 'MODULE_PATHNAME', 'vs_rabitq_params_dim'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_params_seed(rabitq_params) RETURNS bigint
    AS 'MODULE_PATHNAME', 'vs_rabitq_params_seed'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION rabitq_encode(
    input vec32,
    centroid vec32,
    params rabitq_params
) RETURNS rabitq
    AS 'MODULE_PATHNAME', 'vs_rabitq_encode_pg'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION prism_handler(internal) RETURNS index_am_handler
    AS 'MODULE_PATHNAME', 'prism_handler' LANGUAGE C;

CREATE OR REPLACE FUNCTION prism_metric_l2(internal) RETURNS int4
    AS 'MODULE_PATHNAME', 'prism_metric_l2' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION prism_metric_ip(internal) RETURNS int4
    AS 'MODULE_PATHNAME', 'prism_metric_ip' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION prism_metric_cosine(internal) RETURNS int4
    AS 'MODULE_PATHNAME', 'prism_metric_cosine' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION prism_vec32_support(internal) RETURNS internal
    AS 'MODULE_PATHNAME', 'prism_vec32_support' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION prism_vec16_support(internal) RETURNS internal
    AS 'MODULE_PATHNAME', 'prism_vec16_support' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION prism_centroid_pages(regclass)
    RETURNS TABLE (
        blkno       integer,
        entry       smallint,
        level       smallint,
        format      text,
        child_blkno integer,
        child_count smallint,
        is_leaf     boolean
    )
    AS 'MODULE_PATHNAME', 'vs_centroid_pages'
    LANGUAGE C STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION prism_posting_pages(regclass)
    RETURNS TABLE (
        blkno       integer,
        cluster_id  integer,
        is_first    boolean,
        tombstoned  boolean,
        entry_count integer,
        dead_count  integer,
        max_entries integer,
        next_blkno  integer,
        chain_pos   integer,
        format      text
    )
    AS 'MODULE_PATHNAME', 'vs_posting_pages'
    LANGUAGE C STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION prism_tids_clusters(regclass, tid[])
    RETURNS TABLE (
        tid        tid,
        cluster_id integer
    )
    AS 'MODULE_PATHNAME', 'vs_tids_clusters'
    LANGUAGE C STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION prism_index_settings(regclass)
    RETURNS TABLE (
        name    text,
        setting text,
        source  text
    )
    AS 'MODULE_PATHNAME', 'vs_index_settings'
    LANGUAGE C STRICT PARALLEL SAFE;

CREATE OR REPLACE PROCEDURE prism_convert_posting_to_fastscan(
        index_oid regclass,
        cluster_id integer
    )
    AS 'MODULE_PATHNAME', 'vs_convert_posting_to_fastscan'
    LANGUAGE C;

CREATE OR REPLACE PROCEDURE prism_split_posting_list(
        index_oid regclass,
        head_blkno bigint
    )
    AS 'MODULE_PATHNAME', 'vs_split_posting_list'
    LANGUAGE C;

CREATE OR REPLACE PROCEDURE prism_rebalance(index_oid regclass, target_entries integer DEFAULT NULL)
    AS 'MODULE_PATHNAME', 'vs_rebalance'
    LANGUAGE C;

-- =====================================================================
-- Renamed functions. The norm functions take pgvector's name, and the
-- vec16 dims function its own type's prefix. A rename keeps the
-- function's OID, so a view, index or constraint that uses it keeps
-- working.
-- =====================================================================

ALTER FUNCTION vec32_norm(vec32) RENAME TO l2_norm;
ALTER FUNCTION vec32_norm(vec16) RENAME TO l2_norm;
ALTER FUNCTION vec32_dims(vec16) RENAME TO vec16_dims;

-- =====================================================================
-- New functions
-- =====================================================================

-- pgvector's name, so a query written for either type family works
CREATE FUNCTION vector_dims(vec32) RETURNS integer
    AS 'MODULE_PATHNAME', 'vs_vec32_dims'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_dims(vec16) RETURNS integer
    AS 'MODULE_PATHNAME', 'vs_vec16_dims'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
