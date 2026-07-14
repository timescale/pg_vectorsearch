/*
 * meerkat.sql - canonical extension install script
 *
 * The build copies this file verbatim to the version-named install
 * script (meerkat--<version>.sql); see src/pg/meson.build.
 * PostgreSQL resolves the placeholders when the script runs:
 *
 *   MODULE_PATHNAME  the version-named extension library, from the
 *                    version's own control file
 *                    (meerkat--<version>.control) — so every version
 *                    binds its own library, including each step of an
 *                    upgrade chain
 *   @extschema@      the extension schema, at CREATE EXTENSION time
 */

-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION meerkat" to load this file.\quit

-- =====================================================================
-- build identity
-- =====================================================================

CREATE FUNCTION git_commit() RETURNS text
    AS 'MODULE_PATHNAME', 'mkt_git_commit'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION extension_version() RETURNS text
    AS 'MODULE_PATHNAME', 'mkt_extension_version'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION extension_name() RETURNS text
    AS 'MODULE_PATHNAME', 'mkt_extension_name'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Prerelease install notice: warn at CREATE EXTENSION time when this
-- build is a prerelease (any -suffix version, e.g. -alpha1 or -dev).
-- A runtime check against extension_version(), so final releases
-- carry nothing to strip and the notice can never ship stale.
DO $$
BEGIN
    IF pg_catalog.strpos(@extschema@.extension_version(), '-')
        OPERATOR(pg_catalog.>) 0
    THEN
        RAISE WARNING '% % is a prerelease: upgrading to later '
            'versions might not be possible (reinstall instead) and '
            'its indexes may need rebuilding',
            @extschema@.extension_name(), @extschema@.extension_version();
    END IF;
END;
$$;

-- =====================================================================
-- vector type
-- =====================================================================

CREATE FUNCTION vector_in(cstring, oid, integer) RETURNS vector
    AS 'MODULE_PATHNAME', 'mkt_vector_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_out(vector) RETURNS cstring
    AS 'MODULE_PATHNAME', 'mkt_vector_out'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_typmod_in(cstring[]) RETURNS integer
    AS 'MODULE_PATHNAME', 'mkt_vector_typmod_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE vector (
    INPUT     = vector_in,
    OUTPUT    = vector_out,
    TYPMOD_IN = vector_typmod_in,
    INTERNALLENGTH = VARIABLE,
    STORAGE   = external,
    CATEGORY  = 'U',
    DELIMITER = ','
);

-- =====================================================================
-- vector distance functions
-- =====================================================================

CREATE FUNCTION l2_distance(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_l2_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION inner_product(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION cosine_distance(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_cosine_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- vector utility functions
-- =====================================================================

CREATE FUNCTION vector_dims(vector) RETURNS integer
    AS 'MODULE_PATHNAME', 'mkt_pg_vector_dims'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_norm(vector) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_pg_vector_norm'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- vector private functions (for operators and opclass)
-- =====================================================================

CREATE FUNCTION vector_l2_squared_distance(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_vector_l2_squared_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_negative_inner_product(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_vector_negative_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_cmp(vector, vector) RETURNS int4
    AS 'MODULE_PATHNAME', 'mkt_vector_cmp'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_lt(vector, vector) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vector_lt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_le(vector, vector) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vector_le'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_eq(vector, vector) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vector_eq'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_ne(vector, vector) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vector_ne'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_ge(vector, vector) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vector_ge'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_gt(vector, vector) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vector_gt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- vector cast functions
-- =====================================================================

CREATE FUNCTION vector(@extschema@.vector, integer, boolean) RETURNS vector
    AS 'MODULE_PATHNAME', 'mkt_vector'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_vector(real[], integer, boolean) RETURNS vector
    AS 'MODULE_PATHNAME', 'mkt_array_to_vector'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_vector(float8[], integer, boolean) RETURNS vector
    AS 'MODULE_PATHNAME', 'mkt_array_to_vector'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_to_float4(@extschema@.vector) RETURNS real[]
    AS 'MODULE_PATHNAME', 'mkt_vector_to_float4'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- vector casts
-- =====================================================================

CREATE CAST (@extschema@.vector AS @extschema@.vector)
    WITH FUNCTION vector(@extschema@.vector, integer, boolean) AS IMPLICIT;

CREATE CAST (real[] AS @extschema@.vector)
    WITH FUNCTION array_to_vector(real[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (float8[] AS @extschema@.vector)
    WITH FUNCTION array_to_vector(float8[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (@extschema@.vector AS real[])
    WITH FUNCTION vector_to_float4(@extschema@.vector);

-- =====================================================================
-- vector distance operators
-- =====================================================================

CREATE OPERATOR <-> (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = l2_distance,
    COMMUTATOR = '<->'
);

CREATE OPERATOR <#> (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = vector_negative_inner_product,
    COMMUTATOR = '<#>'
);

CREATE OPERATOR <=> (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = cosine_distance,
    COMMUTATOR = '<=>'
);

-- =====================================================================
-- vector comparison operators
-- =====================================================================

CREATE OPERATOR < (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = vector_lt,
    COMMUTATOR = '>', NEGATOR = '>=',
    RESTRICT = scalarltsel, JOIN = scalarltjoinsel
);

CREATE OPERATOR <= (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = vector_le,
    COMMUTATOR = '>=', NEGATOR = '>',
    RESTRICT = scalarlesel, JOIN = scalarlejoinsel
);

CREATE OPERATOR = (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = vector_eq,
    COMMUTATOR = '=', NEGATOR = '<>',
    RESTRICT = eqsel, JOIN = eqjoinsel,
    HASHES, MERGES
);

CREATE OPERATOR <> (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = vector_ne,
    COMMUTATOR = '<>', NEGATOR = '=',
    RESTRICT = neqsel, JOIN = neqjoinsel
);

CREATE OPERATOR >= (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = vector_ge,
    COMMUTATOR = '<=', NEGATOR = '<',
    RESTRICT = scalargesel, JOIN = scalargejoinsel
);

CREATE OPERATOR > (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = vector_gt,
    COMMUTATOR = '<', NEGATOR = '<=',
    RESTRICT = scalargtsel, JOIN = scalargtjoinsel
);

-- =====================================================================
-- vector btree opclass
-- =====================================================================

CREATE OPERATOR FAMILY vector_ops USING btree;

CREATE OPERATOR CLASS vector_ops DEFAULT FOR TYPE vector USING btree
    FAMILY vector_ops AS
    OPERATOR 1 <,
    OPERATOR 2 <=,
    OPERATOR 3 =,
    OPERATOR 4 >=,
    OPERATOR 5 >,
    FUNCTION 1 vector_cmp(vector, vector);

-- =====================================================================
-- halfvec type
-- =====================================================================

CREATE FUNCTION halfvec_in(cstring, oid, integer) RETURNS halfvec
    AS 'MODULE_PATHNAME', 'mkt_halfvec_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_out(halfvec) RETURNS cstring
    AS 'MODULE_PATHNAME', 'mkt_halfvec_out'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_typmod_in(cstring[]) RETURNS integer
    AS 'MODULE_PATHNAME', 'mkt_halfvec_typmod_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE halfvec (
    INPUT     = halfvec_in,
    OUTPUT    = halfvec_out,
    TYPMOD_IN = halfvec_typmod_in,
    INTERNALLENGTH = VARIABLE,
    STORAGE   = external,
    CATEGORY  = 'U',
    DELIMITER = ','
);

-- =====================================================================
-- halfvec distance functions
-- =====================================================================

CREATE FUNCTION l2_distance(halfvec, halfvec) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_halfvec_l2_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION inner_product(halfvec, halfvec) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_halfvec_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION cosine_distance(halfvec, halfvec) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_halfvec_cosine_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- halfvec utility functions
-- =====================================================================

CREATE FUNCTION vector_dims(halfvec) RETURNS integer
    AS 'MODULE_PATHNAME', 'mkt_halfvec_dims'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_norm(halfvec) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_halfvec_norm'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- halfvec private functions (for operators and opclass)
-- =====================================================================

CREATE FUNCTION halfvec_l2_squared_distance(halfvec, halfvec) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_halfvec_l2_squared_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_negative_inner_product(halfvec, halfvec) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_halfvec_negative_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_cmp(halfvec, halfvec) RETURNS int4
    AS 'MODULE_PATHNAME', 'mkt_halfvec_cmp'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_lt(halfvec, halfvec) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_halfvec_lt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_le(halfvec, halfvec) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_halfvec_le'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_eq(halfvec, halfvec) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_halfvec_eq'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_ne(halfvec, halfvec) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_halfvec_ne'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_ge(halfvec, halfvec) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_halfvec_ge'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_gt(halfvec, halfvec) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_halfvec_gt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- halfvec cast functions
-- =====================================================================

CREATE FUNCTION halfvec(@extschema@.halfvec, integer, boolean) RETURNS halfvec
    AS 'MODULE_PATHNAME', 'mkt_halfvec'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_to_vector(@extschema@.halfvec, integer, boolean)
    RETURNS vector
    AS 'MODULE_PATHNAME', 'mkt_halfvec_to_vector'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_to_halfvec(@extschema@.vector, integer, boolean)
    RETURNS halfvec
    AS 'MODULE_PATHNAME', 'mkt_vector_to_halfvec'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_halfvec(real[], integer, boolean) RETURNS halfvec
    AS 'MODULE_PATHNAME', 'mkt_array_to_halfvec'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_halfvec(float8[], integer, boolean) RETURNS halfvec
    AS 'MODULE_PATHNAME', 'mkt_array_to_halfvec'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- halfvec casts
-- =====================================================================

CREATE CAST (@extschema@.halfvec AS @extschema@.halfvec)
    WITH FUNCTION halfvec(@extschema@.halfvec, integer, boolean) AS IMPLICIT;

CREATE CAST (@extschema@.halfvec AS @extschema@.vector)
    WITH FUNCTION halfvec_to_vector(@extschema@.halfvec, integer, boolean)
    AS IMPLICIT;

CREATE CAST (@extschema@.vector AS @extschema@.halfvec)
    WITH FUNCTION vector_to_halfvec(@extschema@.vector, integer, boolean)
    AS ASSIGNMENT;

CREATE CAST (real[] AS @extschema@.halfvec)
    WITH FUNCTION array_to_halfvec(real[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (float8[] AS @extschema@.halfvec)
    WITH FUNCTION array_to_halfvec(float8[], integer, boolean) AS ASSIGNMENT;

-- =====================================================================
-- halfvec distance operators
-- =====================================================================

CREATE OPERATOR <-> (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = l2_distance,
    COMMUTATOR = '<->'
);

CREATE OPERATOR <#> (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = halfvec_negative_inner_product,
    COMMUTATOR = '<#>'
);

CREATE OPERATOR <=> (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = cosine_distance,
    COMMUTATOR = '<=>'
);

-- =====================================================================
-- halfvec comparison operators
-- =====================================================================

CREATE OPERATOR < (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = halfvec_lt,
    COMMUTATOR = '>', NEGATOR = '>=',
    RESTRICT = scalarltsel, JOIN = scalarltjoinsel
);

CREATE OPERATOR <= (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = halfvec_le,
    COMMUTATOR = '>=', NEGATOR = '>',
    RESTRICT = scalarlesel, JOIN = scalarlejoinsel
);

CREATE OPERATOR = (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = halfvec_eq,
    COMMUTATOR = '=', NEGATOR = '<>',
    RESTRICT = eqsel, JOIN = eqjoinsel,
    HASHES, MERGES
);

CREATE OPERATOR <> (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = halfvec_ne,
    COMMUTATOR = '<>', NEGATOR = '=',
    RESTRICT = neqsel, JOIN = neqjoinsel
);

CREATE OPERATOR >= (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = halfvec_ge,
    COMMUTATOR = '<=', NEGATOR = '<',
    RESTRICT = scalargesel, JOIN = scalargejoinsel
);

CREATE OPERATOR > (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = halfvec_gt,
    COMMUTATOR = '<', NEGATOR = '<=',
    RESTRICT = scalargtsel, JOIN = scalargtjoinsel
);

-- =====================================================================
-- halfvec btree opclass
-- =====================================================================

CREATE OPERATOR FAMILY halfvec_ops USING btree;

CREATE OPERATOR CLASS halfvec_ops DEFAULT FOR TYPE halfvec USING btree
    FAMILY halfvec_ops AS
    OPERATOR 1 <,
    OPERATOR 2 <=,
    OPERATOR 3 =,
    OPERATOR 4 >=,
    OPERATOR 5 >,
    FUNCTION 1 halfvec_cmp(halfvec, halfvec);

-- =====================================================================
-- rabitq type
-- =====================================================================

CREATE FUNCTION rabitq_in(cstring, oid, integer) RETURNS rabitq
    AS 'MODULE_PATHNAME', 'mkt_rabitq_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_out(rabitq) RETURNS cstring
    AS 'MODULE_PATHNAME', 'mkt_rabitq_out'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_typmod_in(cstring[]) RETURNS integer
    AS 'MODULE_PATHNAME', 'mkt_rabitq_typmod_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE rabitq (
    INPUT     = rabitq_in,
    OUTPUT    = rabitq_out,
    TYPMOD_IN = rabitq_typmod_in,
    INTERNALLENGTH = VARIABLE,
    STORAGE   = external,
    CATEGORY  = 'U',
    DELIMITER = ','
);

-- =====================================================================
-- rabitq accessor functions
-- =====================================================================

CREATE FUNCTION rabitq_dims(rabitq) RETURNS integer
    AS 'MODULE_PATHNAME', 'mkt_rabitq_dims'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_f_add(rabitq) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_rabitq_f_add'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_f_rescale(rabitq) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_rabitq_f_rescale'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- rabitq private functions (for operators and opclass)
-- =====================================================================

CREATE FUNCTION rabitq_cmp(rabitq, rabitq) RETURNS int4
    AS 'MODULE_PATHNAME', 'mkt_rabitq_cmp'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_lt(rabitq, rabitq) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_rabitq_lt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_le(rabitq, rabitq) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_rabitq_le'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_eq(rabitq, rabitq) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_rabitq_eq'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_ne(rabitq, rabitq) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_rabitq_ne'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_ge(rabitq, rabitq) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_rabitq_ge'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_gt(rabitq, rabitq) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_rabitq_gt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- rabitq cast functions
-- =====================================================================

CREATE FUNCTION rabitq(@extschema@.rabitq, integer, boolean) RETURNS rabitq
    AS 'MODULE_PATHNAME', 'mkt_rabitq'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- rabitq casts
-- =====================================================================

CREATE CAST (@extschema@.rabitq AS @extschema@.rabitq)
    WITH FUNCTION rabitq(@extschema@.rabitq, integer, boolean) AS IMPLICIT;

-- =====================================================================
-- rabitq comparison operators
-- =====================================================================

CREATE OPERATOR < (
    LEFTARG = rabitq, RIGHTARG = rabitq,
    FUNCTION = rabitq_lt,
    COMMUTATOR = '>', NEGATOR = '>=',
    RESTRICT = scalarltsel, JOIN = scalarltjoinsel
);

CREATE OPERATOR <= (
    LEFTARG = rabitq, RIGHTARG = rabitq,
    FUNCTION = rabitq_le,
    COMMUTATOR = '>=', NEGATOR = '>',
    RESTRICT = scalarlesel, JOIN = scalarlejoinsel
);

CREATE OPERATOR = (
    LEFTARG = rabitq, RIGHTARG = rabitq,
    FUNCTION = rabitq_eq,
    COMMUTATOR = '=', NEGATOR = '<>',
    RESTRICT = eqsel, JOIN = eqjoinsel,
    HASHES, MERGES
);

CREATE OPERATOR <> (
    LEFTARG = rabitq, RIGHTARG = rabitq,
    FUNCTION = rabitq_ne,
    COMMUTATOR = '<>', NEGATOR = '=',
    RESTRICT = neqsel, JOIN = neqjoinsel
);

CREATE OPERATOR >= (
    LEFTARG = rabitq, RIGHTARG = rabitq,
    FUNCTION = rabitq_ge,
    COMMUTATOR = '<=', NEGATOR = '<',
    RESTRICT = scalargesel, JOIN = scalargejoinsel
);

CREATE OPERATOR > (
    LEFTARG = rabitq, RIGHTARG = rabitq,
    FUNCTION = rabitq_gt,
    COMMUTATOR = '<', NEGATOR = '<=',
    RESTRICT = scalargtsel, JOIN = scalargtjoinsel
);

-- =====================================================================
-- rabitq btree opclass
-- =====================================================================

CREATE OPERATOR FAMILY rabitq_ops USING btree;

CREATE OPERATOR CLASS rabitq_ops DEFAULT FOR TYPE rabitq USING btree
    FAMILY rabitq_ops AS
    OPERATOR 1 <,
    OPERATOR 2 <=,
    OPERATOR 3 =,
    OPERATOR 4 >=,
    OPERATOR 5 >,
    FUNCTION 1 rabitq_cmp(rabitq, rabitq);

-- =====================================================================
-- rabitq_params type
-- =====================================================================

CREATE FUNCTION rabitq_params_in(cstring, oid, integer)
    RETURNS rabitq_params
    AS 'MODULE_PATHNAME', 'mkt_rabitq_params_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_params_out(rabitq_params) RETURNS cstring
    AS 'MODULE_PATHNAME', 'mkt_rabitq_params_out'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE rabitq_params (
    INPUT     = rabitq_params_in,
    OUTPUT    = rabitq_params_out,
    INTERNALLENGTH = VARIABLE,
    STORAGE   = extended,
    CATEGORY  = 'U'
);

-- =====================================================================
-- rabitq_params functions
-- =====================================================================

CREATE FUNCTION rabitq_params_generate(dim integer, seed bigint)
    RETURNS rabitq_params
    AS 'MODULE_PATHNAME', 'mkt_rabitq_params_generate_pg'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION rabitq_params_generate(integer, bigint) IS
'Generate RaBitQ quantization parameters (random orthogonal matrix) for a given dimension and seed.
The same dim+seed always produces the same matrix. Store the result for reuse across encode calls.';

CREATE FUNCTION rabitq_params_dim(rabitq_params) RETURNS integer
    AS 'MODULE_PATHNAME', 'mkt_rabitq_params_dim'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION rabitq_params_seed(rabitq_params) RETURNS bigint
    AS 'MODULE_PATHNAME', 'mkt_rabitq_params_seed'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- rabitq encoding function
-- =====================================================================

CREATE FUNCTION rabitq_encode(
    input vector,
    centroid vector,
    params rabitq_params
) RETURNS rabitq
    AS 'MODULE_PATHNAME', 'mkt_rabitq_encode_pg'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION rabitq_encode(vector, vector, rabitq_params) IS
'Encode a vector to RaBitQ binary quantization relative to a centroid.
Returns a rabitq value containing the quantized bits, f_add, and f_rescale.
The params argument provides the orthogonal transform matrix (see rabitq_params_generate).';

-- =====================================================================
-- mktann index access method
-- =====================================================================

CREATE FUNCTION mktann_handler(internal) RETURNS index_am_handler
    AS 'MODULE_PATHNAME' LANGUAGE C;

CREATE ACCESS METHOD mktann TYPE INDEX HANDLER mktann_handler;

COMMENT ON ACCESS METHOD mktann IS 'meerkat ANN index';

-- Metric identifier functions (FUNCTION 2 in opclass)
CREATE FUNCTION mktann_metric_l2(internal) RETURNS int4
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION mktann_metric_ip(internal) RETURNS int4
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION mktann_metric_cosine(internal) RETURNS int4
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Operator classes for vector type
CREATE OPERATOR CLASS vector_l2_ops
    DEFAULT FOR TYPE vector USING mktann AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_l2_squared_distance(vector, vector),
    FUNCTION 2 mktann_metric_l2(internal);

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING mktann AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_negative_inner_product(vector, vector),
    FUNCTION 2 mktann_metric_ip(internal);

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING mktann AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 cosine_distance(vector, vector),
    FUNCTION 2 mktann_metric_cosine(internal);

-- =====================================================================
-- index inspection functions
-- =====================================================================

CREATE FUNCTION centroid_pages(regclass)
    RETURNS TABLE (
        blkno       integer,
        entry       smallint,
        level       smallint,
        format      text,
        child_blkno integer,
        child_count smallint,
        is_leaf     boolean
    )
    AS 'MODULE_PATHNAME', 'mkt_centroid_pages'
    LANGUAGE C STRICT PARALLEL SAFE;

CREATE FUNCTION mkt.posting_pages(regclass)
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
    AS 'MODULE_PATHNAME', 'mkt_posting_pages'
    LANGUAGE C STRICT PARALLEL SAFE;

-- Map heap TIDs to the index cluster(s) that hold them (primary plus any
-- SOAR/boundary replica). For routing analysis: compare where a query's
-- true nearest neighbors live against which clusters the query scans.
-- Scans posting pages directly -- each already carries its cluster_id --
-- so it needs neither the centroid tree nor its format.
CREATE FUNCTION mkt.tids_clusters(regclass, tid[])
    RETURNS TABLE (
        tid        tid,
        cluster_id integer
    )
    AS 'MODULE_PATHNAME', 'mkt_tids_clusters'
    LANGUAGE C STRICT PARALLEL SAFE;

-- Convert one cluster's posting chain from AoS to fastscan format.
-- Updates centroid entries and metadata flag atomically.
-- Returns the new posting head block number.
CREATE FUNCTION mkt.convert_posting_to_fastscan(
        index_oid regclass,
        cluster_id integer
    )
    RETURNS integer
    AS 'MODULE_PATHNAME', 'mkt_convert_posting_to_fastscan'
    LANGUAGE C STRICT;

-- =====================================================================
-- pgvector binary cast support
-- =====================================================================
--
-- When pgvector (the vector extension) is installed, meerkat creates
-- zero-overhead binary casts between the two sets of types. The types
-- have identical binary layouts (varlena header + int16 dim + int16
-- unused + float[]), so WITHOUT FUNCTION casts produce RelabelType
-- nodes with no conversion overhead.
--
-- Cast directions:
--   pgvector -> meerkat: IMPLICIT (pgvector columns work transparently
--     with meerkat operators and indexes)
--   meerkat -> pgvector: ASSIGNMENT (avoids operator ambiguity when
--     both extensions define <->, <#>, <=>)
--
-- The casts are standalone objects (not owned by either extension).
-- PostgreSQL auto-drops them via type dependencies when the referenced
-- types are dropped, so DROP EXTENSION on either side removes the
-- casts without affecting the other extension.
--
-- Both install orderings are supported:
--   pgvector first, meerkat later: DO block below creates casts
--   meerkat first, pgvector later: event trigger creates casts

-- Helper: create binary casts between meerkat and pgvector types.
-- Uses exception handling for idempotency (CREATE CAST has no IF NOT
-- EXISTS clause).
CREATE FUNCTION create_pgvector_casts() RETURNS void
    LANGUAGE plpgsql AS $$
BEGIN
    BEGIN
        CREATE CAST (public.vector AS @extschema@.vector)
            WITHOUT FUNCTION AS IMPLICIT;
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
    BEGIN
        CREATE CAST (public.halfvec AS @extschema@.halfvec)
            WITHOUT FUNCTION AS IMPLICIT;
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
    BEGIN
        CREATE CAST (@extschema@.vector AS public.vector)
            WITHOUT FUNCTION AS ASSIGNMENT;
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
    BEGIN
        CREATE CAST (@extschema@.halfvec AS public.halfvec)
            WITHOUT FUNCTION AS ASSIGNMENT;
    EXCEPTION WHEN duplicate_object THEN NULL;
    END;
END;
$$;

-- Create casts now if pgvector is already installed.
-- During CREATE EXTENSION, objects created in DO blocks are auto-owned
-- by the extension. We immediately disassociate the casts so that
-- DROP EXTENSION meerkat does not cascade to (or through) pgvector.
-- The casts still get cleaned up via auto-dependencies on their
-- referenced types.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
    ) THEN
        PERFORM @extschema@.create_pgvector_casts();
        EXECUTE 'ALTER EXTENSION meerkat DROP CAST '
            '(public.vector AS @extschema@.vector)';
        EXECUTE 'ALTER EXTENSION meerkat DROP CAST '
            '(public.halfvec AS @extschema@.halfvec)';
        EXECUTE 'ALTER EXTENSION meerkat DROP CAST '
            '(@extschema@.vector AS public.vector)';
        EXECUTE 'ALTER EXTENSION meerkat DROP CAST '
            '(@extschema@.halfvec AS public.halfvec)';
    END IF;
END;
$$;

-- Event trigger: create casts when pgvector is installed after meerkat.
CREATE FUNCTION on_extension_create()
    RETURNS event_trigger LANGUAGE plpgsql AS $$
DECLARE
    obj record;
BEGIN
    FOR obj IN SELECT * FROM pg_event_trigger_ddl_commands()
               WHERE object_type = 'extension'
    LOOP
        IF obj.object_identity = 'vector' THEN
            PERFORM @extschema@.create_pgvector_casts();
        END IF;
    END LOOP;
END;
$$;

CREATE EVENT TRIGGER mkt_pgvector_cast_trigger
    ON ddl_command_end
    WHEN TAG IN ('CREATE EXTENSION')
    EXECUTE FUNCTION on_extension_create();
