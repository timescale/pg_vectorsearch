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
-- schema ownership guard
-- =====================================================================
-- Refuse to install into a pre-existing extension schema owned by an
-- untrusted role. meerkat is non-relocatable (schema = 'mkt') and its docs
-- have users put that schema on their search_path
-- (SET search_path = mkt, public) so meerkat's types and operators resolve
-- unqualified -- which makes mkt a schema users trust on their path. But
-- PostgreSQL does not check target-schema ownership at CREATE EXTENSION, so
-- a role with CREATE on the database can pre-create mkt and keep owning it
-- after install. The owner cannot touch meerkat's own objects (extension
-- membership protects them), but can add NEW objects to mkt -- e.g. an
-- operator on pgvector's type that shadows pgvector's own for anyone with
-- mkt ahead of public, running the attacker's code as that role. See the
-- security note above setup_pgvector_compat() for the alternatives weighed.
--
-- Allow only a schema the installer (current_user) or a superuser owns; a
-- fresh install, where CREATE EXTENSION creates the schema, is unaffected.
-- This assumes the fixed dedicated schema: if meerkat is ever made
-- relocatable (installable into public, owned by pg_database_owner), this
-- check must be revisited, or it would refuse a legitimate install there.
DO $$
DECLARE
    owner_name  name;
    owner_super boolean;
BEGIN
    SELECT r.rolname, r.rolsuper INTO owner_name, owner_super
      FROM pg_catalog.pg_namespace n
      JOIN pg_catalog.pg_roles r
        ON r.oid OPERATOR(pg_catalog.=) n.nspowner
     WHERE n.nspname OPERATOR(pg_catalog.=) '@extschema@';
    IF FOUND AND NOT (owner_super
                      OR owner_name OPERATOR(pg_catalog.=) current_user) THEN
        RAISE EXCEPTION
            'schema "@extschema@" already exists and is owned by "%", a role '
            'other than the installer or a superuser', owner_name
            USING HINT = 'meerkat refuses to install into a schema an '
                'untrusted role controls; drop or re-own the schema, or '
                'install as the role that owns it.';
    END IF;
END;
$$;

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
    RESTRICT = pg_catalog.scalarltsel, JOIN = pg_catalog.scalarltjoinsel
);

CREATE OPERATOR <= (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = vector_le,
    COMMUTATOR = '>=', NEGATOR = '>',
    RESTRICT = pg_catalog.scalarlesel, JOIN = pg_catalog.scalarlejoinsel
);

CREATE OPERATOR = (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = vector_eq,
    COMMUTATOR = '=', NEGATOR = '<>',
    RESTRICT = pg_catalog.eqsel, JOIN = pg_catalog.eqjoinsel,
    HASHES, MERGES
);

CREATE OPERATOR <> (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = vector_ne,
    COMMUTATOR = '<>', NEGATOR = '=',
    RESTRICT = pg_catalog.neqsel, JOIN = pg_catalog.neqjoinsel
);

CREATE OPERATOR >= (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = vector_ge,
    COMMUTATOR = '<=', NEGATOR = '<',
    RESTRICT = pg_catalog.scalargesel, JOIN = pg_catalog.scalargejoinsel
);

CREATE OPERATOR > (
    LEFTARG = vector, RIGHTARG = vector,
    FUNCTION = vector_gt,
    COMMUTATOR = '<', NEGATOR = '<=',
    RESTRICT = pg_catalog.scalargtsel, JOIN = pg_catalog.scalargtjoinsel
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
    RESTRICT = pg_catalog.scalarltsel, JOIN = pg_catalog.scalarltjoinsel
);

CREATE OPERATOR <= (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = halfvec_le,
    COMMUTATOR = '>=', NEGATOR = '>',
    RESTRICT = pg_catalog.scalarlesel, JOIN = pg_catalog.scalarlejoinsel
);

CREATE OPERATOR = (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = halfvec_eq,
    COMMUTATOR = '=', NEGATOR = '<>',
    RESTRICT = pg_catalog.eqsel, JOIN = pg_catalog.eqjoinsel,
    HASHES, MERGES
);

CREATE OPERATOR <> (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = halfvec_ne,
    COMMUTATOR = '<>', NEGATOR = '=',
    RESTRICT = pg_catalog.neqsel, JOIN = pg_catalog.neqjoinsel
);

CREATE OPERATOR >= (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = halfvec_ge,
    COMMUTATOR = '<=', NEGATOR = '<',
    RESTRICT = pg_catalog.scalargesel, JOIN = pg_catalog.scalargejoinsel
);

CREATE OPERATOR > (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    FUNCTION = halfvec_gt,
    COMMUTATOR = '<', NEGATOR = '<=',
    RESTRICT = pg_catalog.scalargtsel, JOIN = pg_catalog.scalargtjoinsel
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
    RESTRICT = pg_catalog.scalarltsel, JOIN = pg_catalog.scalarltjoinsel
);

CREATE OPERATOR <= (
    LEFTARG = rabitq, RIGHTARG = rabitq,
    FUNCTION = rabitq_le,
    COMMUTATOR = '>=', NEGATOR = '>',
    RESTRICT = pg_catalog.scalarlesel, JOIN = pg_catalog.scalarlejoinsel
);

CREATE OPERATOR = (
    LEFTARG = rabitq, RIGHTARG = rabitq,
    FUNCTION = rabitq_eq,
    COMMUTATOR = '=', NEGATOR = '<>',
    RESTRICT = pg_catalog.eqsel, JOIN = pg_catalog.eqjoinsel,
    HASHES, MERGES
);

CREATE OPERATOR <> (
    LEFTARG = rabitq, RIGHTARG = rabitq,
    FUNCTION = rabitq_ne,
    COMMUTATOR = '<>', NEGATOR = '=',
    RESTRICT = pg_catalog.neqsel, JOIN = pg_catalog.neqjoinsel
);

CREATE OPERATOR >= (
    LEFTARG = rabitq, RIGHTARG = rabitq,
    FUNCTION = rabitq_ge,
    COMMUTATOR = '<=', NEGATOR = '<',
    RESTRICT = pg_catalog.scalargesel, JOIN = pg_catalog.scalargejoinsel
);

CREATE OPERATOR > (
    LEFTARG = rabitq, RIGHTARG = rabitq,
    FUNCTION = rabitq_gt,
    COMMUTATOR = '<', NEGATOR = '<=',
    RESTRICT = pg_catalog.scalargtsel, JOIN = pg_catalog.scalargtjoinsel
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

-- Column type descriptors (support function 3). Optional: an opclass that
-- declares none indexes `vector`, which keeps the vector opclasses unchanged
-- and leaves an index built before this existed working. Returning the
-- descriptor from the opclass is what lets the access method agree with the
-- planner about a column's type without resolving a name or comparing an OID
-- -- see src/pg/mktann_typeinfo.h.
CREATE FUNCTION mktann_vector_support(internal) RETURNS internal
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION mktann_halfvec_support(internal) RETURNS internal
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

-- Operator classes for halfvec type
--
-- The index itself is unchanged: postings hold RaBitQ codes either way, and
-- the AM widens a halfvec tuple to float32 on read (the distance and encode
-- kernels are float32-only). What halfvec buys is the heap, which is what an
-- exact rerank reads -- at 768d a vector row is 3080 bytes and fits 2 to an
-- 8 kB page against halfvec's 1544 and 5. Centroids follow the column and are
-- stored half-precision too (MKT_CENTROID_FMT_HALF).
CREATE OPERATOR CLASS halfvec_l2_ops
    DEFAULT FOR TYPE halfvec USING mktann AS
    OPERATOR 1 <-> (halfvec, halfvec) FOR ORDER BY float_ops,
    FUNCTION 1 halfvec_l2_squared_distance(halfvec, halfvec),
    FUNCTION 2 mktann_metric_l2(internal),
    FUNCTION 3 mktann_halfvec_support(internal);

CREATE OPERATOR CLASS halfvec_ip_ops
    FOR TYPE halfvec USING mktann AS
    OPERATOR 1 <#> (halfvec, halfvec) FOR ORDER BY float_ops,
    FUNCTION 1 halfvec_negative_inner_product(halfvec, halfvec),
    FUNCTION 2 mktann_metric_ip(internal),
    FUNCTION 3 mktann_halfvec_support(internal);

CREATE OPERATOR CLASS halfvec_cosine_ops
    FOR TYPE halfvec USING mktann AS
    OPERATOR 1 <=> (halfvec, halfvec) FOR ORDER BY float_ops,
    FUNCTION 1 cosine_distance(halfvec, halfvec),
    FUNCTION 2 mktann_metric_cosine(internal),
    FUNCTION 3 mktann_halfvec_support(internal);

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

-- Effective index settings, one (name, setting, source) row per
-- setting, with automatic values resolved to what the index actually
-- uses: nlist/fan_out/nlevels and the page formats as the build chose
-- them (from the metadata page), build options read back from the
-- catalog with defaults filled in, and the session-effective nprobe
-- and distance_mode for this index. The source column tells where
-- each value came from: 'option' (explicit reloption), 'auto'
-- (resolved automatic default), 'default' (reloption default),
-- 'column'/'opclass' (index definition), 'derived' (computed from
-- other settings), or 'session' (GUC override).
CREATE FUNCTION mkt.index_settings(regclass)
    RETURNS TABLE (
        name    text,
        setting text,
        source  text
    )
    AS 'MODULE_PATHNAME', 'mkt_index_settings'
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
-- Incremental maintenance (posting-list split)
-- =====================================================================

-- Split one posting list (given its head block number) into two or more
-- balanced lists. A maintenance operation that mutates index state, so it is a
-- procedure (CALL) rather than a function: it returns no value and can manage
-- its own transactions. Reports the outcome via a NOTICE.
CREATE PROCEDURE mkt.split_posting_list(
        index_oid regclass,
        head_blkno bigint
    )
    AS 'MODULE_PATHNAME', 'mkt_split_posting_list'
    LANGUAGE C;

COMMENT ON PROCEDURE mkt.split_posting_list(regclass, bigint) IS
    'Split one posting list into two or more balanced lists. index_oid is the '
    'index; head_blkno is the block number of the list''s head page. '
    'Owner-only; reports the outcome via NOTICE. Use mkt.rebalance to split '
    'every oversized list in an index.';

-- Rebalance an index by splitting every posting list that has outgrown the
-- split trigger into lists of about target_entries entries each, and reclaiming
-- chains retired by earlier splits once no snapshot can still reach them.
-- target_entries is the size a list rests at, not a bound: a list is left alone
-- until it reaches twice the target, which leaves it room to absorb inserts and
-- deletes instead of re-splitting on the next row. NULL (the default) derives
-- the target from the table's row count, so a list rests at the size that
-- keeps a probe's cost flat however large the table grows; the nlist reloption
-- does not enter into it -- that is what the build was asked for, not a
-- maintenance policy. A mutating maintenance procedure (see
-- split_posting_list); reports the number of lists split via a NOTICE.
-- Splitting is the only rebalancing it performs, and it is driven by the
-- caller.
CREATE PROCEDURE mkt.rebalance(index_oid regclass, target_entries integer DEFAULT NULL)
    AS 'MODULE_PATHNAME', 'mkt_rebalance'
    LANGUAGE C;

COMMENT ON PROCEDURE mkt.rebalance(regclass, integer) IS
    'Split every posting list that has grown past twice target_entries into '
    'lists of about target_entries each. index_oid is the index; '
    'target_entries is the size a list rests at (NULL, the default, derives it '
    'from the row count). Owner-only; reports the number of lists split via '
    'NOTICE.';

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
-- Casts alone are not enough to make an mktann index reachable from a
-- query written against pgvector. An index is only considered for an
-- ORDER BY when the ordering operator belongs to the index's operator
-- family, and pgvector's <->, <#> and <=> belong to pgvector's families.
-- Without help, `ORDER BY v <-> $1` under a pgvector-first search_path
-- plans a sequential scan -- which returns correct rows, so it is easy to
-- mistake for a working index scan. setup_pgvector_compat() therefore adds
-- pgvector's three distance operators to meerkat's mktann families as ordering
-- members alongside the casts, so either spelling of the operator reaches the
-- index. Casts and operators are one function on purpose: they are a unit (the
-- operators rely on the casts' binary-coercibility), so a new install path can
-- never add one and forget the other.
--
-- The casts are standalone objects (not owned by either extension).
-- PostgreSQL auto-drops them via type dependencies when the referenced
-- types are dropped, so DROP EXTENSION on either side removes the
-- casts without affecting the other extension.
--
-- Both install orderings are supported:
--   pgvector first, meerkat later: DO block below sets up compat
--   meerkat first, pgvector later: event trigger sets up compat

-- Security note. This code runs at CREATE EXTENSION time (the install DO
-- block) and later from an event trigger on any CREATE EXTENSION, in both
-- cases as the invoking role. Three attack vectors were considered:
--
-- 1. search_path hijack of a builtin. An unqualified format() (or any
--    builtin) here could resolve to an attacker-planted overload on the
--    caller's path -- an exact-arity overload beats pg_catalog's VARIADIC one
--    from ANY path position, so pg_catalog being implicitly first does not
--    help. Closed by pinning SET search_path = pg_catalog, pg_temp on this
--    function and calling every builtin as pg_catalog.<fn>.
--
-- 2. Cast tampering. pgvector's type owner could pre-create a WITH FUNCTION
--    cast between the two extensions' types, whose function then runs as the
--    querying role. Closed by the loop below: it creates a cast only when
--    absent and RAISEs on any pre-existing cast that is not the expected
--    binary (WITHOUT FUNCTION) cast, instead of adopting it.
--
-- 3. Operator shadowing via the extension schema. If an untrusted role owns
--    mkt, it can add an operator on pgvector's type -- e.g.
--    mkt.<->(public.vector, public.vector) -- that shadows pgvector's own for
--    any role with mkt ahead of public, running attacker code as that role.
--    meerkat's OWN operators (on mkt.vector) are not shadowable: a
--    same-signature plant conflicts at install and installed objects are
--    membership-locked. Two fixes were weighed:
--      (a) Occupy the signatures -- have meerkat pre-create safe, delegating
--          versions of pgvector's operators AND functions in mkt so the
--          attacker cannot. Rejected: the surface is pgvector's whole public
--          API across all its types and it grows with pgvector versions, so a
--          newly added pgvector operator silently reopens the hole until
--          meerkat catches up -- a maintenance treadmill tied to another
--          project's API, and it still leaves non-pgvector shadows open.
--      (b) Ensure mkt is trusted-owned -- refuse to install into a mkt owned
--          by an untrusted role (the schema ownership guard at the top of
--          this script). Chosen: version-independent, comprehensive (nothing
--          hostile can live in mkt at all), ~10 lines. Cost: it assumes the
--          fixed dedicated schema and must be revisited if meerkat ever
--          becomes relocatable.
--
-- Set up pgvector interoperability in one step: the binary casts between the
-- two extensions' types, and the membership of pgvector's distance operators
-- in meerkat's mktann operator families. These belong together -- the casts
-- make pgvector's vector/halfvec binary-coercible to mkt.vector/mkt.halfvec,
-- which is exactly what lets pgvector's operators join a family whose opclass
-- is FOR TYPE mkt.<type>. Keeping them in a single function means a caller
-- cannot add the casts and forget the operators (which would silently downgrade
-- pgvector-operator queries to a sequential scan).
--
-- Casts first, then operators (the operators depend on the casts' coercibility).
-- Idempotent throughout via exception handling (neither CREATE CAST nor ALTER
-- OPERATOR FAMILY has an IF NOT EXISTS form).
CREATE FUNCTION setup_pgvector_compat() RETURNS void
    LANGUAGE plpgsql
    -- Reached at runtime from the event trigger under the DDL-runner's
    -- search_path, and from the install DO block. Pin the path so every
    -- unqualified name here resolves in pg_catalog, not in an
    -- attacker-controlled schema (see format() below).
    SET search_path = pg_catalog, pg_temp
    AS $$
DECLARE
    r record;
    existing_method  "char";
    existing_context "char";
    expected_context "char";
    pgv_ns text;   -- pgvector's schema (it is relocatable, so discovered)
    pgv text;      -- ...quote_ident'd, for building qualified type names
BEGIN
    -- pgvector is relocatable: its types and operators live in whatever
    -- schema it was installed into, not necessarily public. Discover it
    -- from the catalog rather than assuming public; extnamespace stays
    -- authoritative even after ALTER EXTENSION vector SET SCHEMA. The name
    -- is only ever interpolated through %I / quote_ident, so it stays
    -- injection-safe.
    SELECT n.nspname INTO pgv_ns
    FROM pg_catalog.pg_extension e
    JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
    WHERE e.extname = 'vector';

    IF pgv_ns IS NULL THEN
        RAISE EXCEPTION 'pgvector (extension "vector") is not installed';
    END IF;
    pgv := pg_catalog.quote_ident(pgv_ns);

    -- 1. Binary casts, both directions, for vector and halfvec.
    --
    -- Create each only if absent. A cast that already exists is accepted
    -- ONLY when it is the expected binary cast (WITHOUT FUNCTION,
    -- castmethod 'b') AND has the expected context (IMPLICIT vs ASSIGNMENT);
    -- anything else is rejected as tampering. pgvector as shipped is NOT a
    -- trusted extension (installing it needs superuser), and even a trusted
    -- install leaves its types owned by the bootstrap superuser -- so a
    -- WITH FUNCTION (or WITH INOUT) cast, whose function would run with the
    -- privileges of whatever role later triggers the coercion, can only be
    -- planted by a superuser or a role a superuser made the type's owner.
    -- This check is therefore defense in depth and loud tamper-evidence
    -- rather than protection against an unprivileged attacker; it costs
    -- nothing and catches a mistaken or malicious cast whoever made it.
    -- Validating the
    -- context too matters because the two directions differ deliberately
    -- (pgvector->meerkat IMPLICIT, meerkat->pgvector ASSIGNMENT): a binary
    -- cast planted with the wrong context still has method 'b' but changes
    -- coercion/operator-resolution behaviour. Silently adopting either (the
    -- old EXCEPTION WHEN duplicate_object THEN NULL) would hide it; we fail
    -- loudly instead.
    FOR r IN
        SELECT * FROM (VALUES
            (pgv || '.vector',      '@extschema@.vector',   'IMPLICIT'),
            (pgv || '.halfvec',     '@extschema@.halfvec',  'IMPLICIT'),
            ('@extschema@.vector',  pgv || '.vector',       'ASSIGNMENT'),
            ('@extschema@.halfvec', pgv || '.halfvec',      'ASSIGNMENT')
        ) AS t(src, tgt, ctx)
    LOOP
        -- pg_cast.castcontext code for the expected context: the first letter
        -- of the lowercased keyword (implicit->i, assignment->a).
        expected_context := pg_catalog.substr(pg_catalog.lower(r.ctx), 1, 1);

        SELECT castmethod, castcontext INTO existing_method, existing_context
        FROM pg_catalog.pg_cast
        WHERE castsource = r.src::pg_catalog.regtype
          AND casttarget = r.tgt::pg_catalog.regtype;

        IF FOUND THEN
            IF existing_method OPERATOR(pg_catalog.<>) 'b'
               OR existing_context OPERATOR(pg_catalog.<>) expected_context THEN
                RAISE EXCEPTION 'refusing pre-existing cast (% AS %): expected '
                    'a binary (WITHOUT FUNCTION) % cast but found castmethod=%, '
                    'castcontext=%; possible tampering',
                    r.src, r.tgt, r.ctx, existing_method, existing_context;
            END IF;
            -- Expected binary cast with the expected context already present.
        ELSE
            EXECUTE pg_catalog.format(
                'CREATE CAST (%s AS %s) WITHOUT FUNCTION AS %s',
                r.src, r.tgt, r.ctx);
        END IF;
    END LOOP;

    -- 2. pgvector's distance operators as ordering members of meerkat's mktann
    -- families. Strategy 1 and float_ops match the opclass declarations above;
    -- the operator's left type only has to be binary-coercible to the family's
    -- index type, which the casts above guarantee. Adding an operator family
    -- member requires superuser, so an attacker cannot pre-plant one; the
    -- duplicate_object catch here is pure idempotency for a legitimate re-run.
    FOR r IN
        SELECT * FROM (VALUES
            ('vector_l2_ops',      'vector',  '<->'),
            ('vector_ip_ops',      'vector',  '<#>'),
            ('vector_cosine_ops',  'vector',  '<=>'),
            ('halfvec_l2_ops',     'halfvec', '<->'),
            ('halfvec_ip_ops',     'halfvec', '<#>'),
            ('halfvec_cosine_ops', 'halfvec', '<=>')
        ) AS t(fam, typ, op)
    LOOP
        BEGIN
            EXECUTE pg_catalog.format(
                'ALTER OPERATOR FAMILY @extschema@.%I USING mktann '
                'ADD OPERATOR 1 %I.%s (%I.%I, %I.%I) '
                'FOR ORDER BY pg_catalog.float_ops',
                r.fam, pgv_ns, r.op, pgv_ns, r.typ, pgv_ns, r.typ);
        EXCEPTION WHEN duplicate_object THEN NULL;
        END;
    END LOOP;
END;
$$;

-- Create casts now if pgvector is already installed.
-- During CREATE EXTENSION, objects created in an anonymous DO ($$ ... $$)
-- block are auto-owned by the extension. We immediately disassociate the
-- casts so that
-- DROP EXTENSION meerkat does not cascade to (or through) pgvector.
-- The casts still get cleaned up via auto-dependencies on their
-- referenced types.
DO $$
DECLARE
    pgv_ns text;
BEGIN
    -- pgvector is relocatable; discover its schema (NULL if not installed).
    SELECT n.nspname INTO pgv_ns
    FROM pg_catalog.pg_extension e
    JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
    WHERE e.extname = 'vector';

    IF pgv_ns IS NOT NULL THEN
        PERFORM @extschema@.setup_pgvector_compat();
        EXECUTE pg_catalog.format('ALTER EXTENSION meerkat DROP CAST '
            '(%I.vector AS @extschema@.vector)', pgv_ns);
        EXECUTE pg_catalog.format('ALTER EXTENSION meerkat DROP CAST '
            '(%I.halfvec AS @extschema@.halfvec)', pgv_ns);
        EXECUTE pg_catalog.format('ALTER EXTENSION meerkat DROP CAST '
            '(@extschema@.vector AS %I.vector)', pgv_ns);
        EXECUTE pg_catalog.format('ALTER EXTENSION meerkat DROP CAST '
            '(@extschema@.halfvec AS %I.halfvec)', pgv_ns);
    END IF;
END;
$$;

-- Event trigger: create casts when pgvector is installed after meerkat.
CREATE FUNCTION on_extension_create()
    RETURNS event_trigger LANGUAGE plpgsql
    -- Runs later as an event trigger under the DDL-runner's own
    -- search_path. Pin it so unqualified names in this body (and the one
    -- it calls) resolve to pg_catalog, never an attacker-planted overload.
    SET search_path = pg_catalog, pg_temp
    AS $$
DECLARE
    obj record;
    is_super boolean;
BEGIN
    FOR obj IN SELECT * FROM pg_catalog.pg_event_trigger_ddl_commands()
               WHERE object_type = 'extension'
    LOOP
        IF obj.object_identity = 'vector' THEN
            -- setup_pgvector_compat() does superuser-only DDL (CREATE CAST
            -- WITHOUT FUNCTION, ALTER OPERATOR FAMILY). An event trigger runs
            -- as the role that ran CREATE EXTENSION, so if a NON-superuser
            -- installs pgvector -- possible where it is trusted, as some
            -- managed platforms allow -- this PERFORM would fail and roll back
            -- the whole pgvector install, making meerkat's presence break
            -- pgvector. Skip and warn instead; a superuser finishes the wiring
            -- later. (SECURITY DEFINER was rejected: it would run this
            -- superuser-only DDL for anyone who can create an extension.)
            SELECT r.rolsuper INTO is_super
              FROM pg_catalog.pg_roles r
             WHERE r.rolname OPERATOR(pg_catalog.=) current_user;
            IF is_super THEN
                PERFORM @extschema@.setup_pgvector_compat();
            ELSE
                RAISE WARNING 'meerkat did not set up pgvector compatibility: '
                    'it requires superuser privileges'
                    USING HINT = 'A superuser should run '
                        '@extschema@.setup_pgvector_compat() so pgvector-typed '
                        'columns can use meerkat indexes.';
            END IF;
        END IF;
    END LOOP;
END;
$$;

CREATE EVENT TRIGGER mkt_pgvector_cast_trigger
    ON ddl_command_end
    WHEN TAG IN ('CREATE EXTENSION')
    EXECUTE FUNCTION on_extension_create();
