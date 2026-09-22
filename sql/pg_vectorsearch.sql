/*
 * pg_vectorsearch.sql - canonical extension install script
 *
 * The build copies this file verbatim to the version-named install
 * script (pg_vectorsearch--<version>.sql); see src/pg/meson.build.
 * PostgreSQL resolves the placeholders when the script runs:
 *
 *   MODULE_PATHNAME  the version-named extension library, from the
 *                    version's own control file
 *                    (pg_vectorsearch--<version>.control) — so every
 *                    version binds its own library, including each
 *                    step of an upgrade chain
 */

-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION pg_vectorsearch" to load this file.\quit

-- =====================================================================
-- vectorsearch / prism: fixed schemas for maintenance, administration,
-- and inspection
-- =====================================================================
-- Everything else in this script installs into @extschema@. Maintenance,
-- administration, and inspection functions always live in one of two
-- fixed schemas instead, created here regardless of @extschema@ -- one
-- fixed, predictable, always-qualified path to them no matter which
-- schema holds the types:
--
--   vectorsearch  extension-wide identity (git_commit,
--                 extension_version, extension_name), not specific to
--                 any one index
--   prism         everything specific to the prism index access
--                 method: inspection, maintenance, and its pgvector
--                 operator-family wiring. Kept separate from
--                 vectorsearch because a second index sharing this
--                 extension would need its own equivalent of this
--                 schema, not a share of prism's
--
-- This is also why ALTER EXTENSION ... SET SCHEMA is refused (see the
-- control file): that command moves every member object into one schema,
-- and these functions are deliberately not in it.
--
-- Both schemas must be owned by the extension's installer or a
-- superuser. PostgreSQL does not check target-schema ownership at CREATE
-- EXTENSION, so an untrusted role could otherwise pre-create either one,
-- keep owning it, and plant lookalike objects there that a caller who has
-- not double-checked their tooling might mistake for the extension's own
-- (vectorsearch.<function> and prism.<function> calls are always
-- schema-qualified, never resolved via search_path). See the security
-- note above setup_pgvector_compat() for the analogous reasoning about
-- @extschema@ when pgvector is involved.
--
-- A pre-existing, trusted-owned vectorsearch or prism is used as-is, not
-- adopted into extension membership (no ALTER EXTENSION ... ADD SCHEMA):
-- matching how PostgreSQL treats a pre-existing @extschema@ for any
-- relocatable extension, only objects this script itself creates become
-- members. Otherwise DROP EXTENSION ... CASCADE could delete a schema --
-- and anything unrelated already in it -- that this extension never
-- created.
DO $$
DECLARE
    schema_name name;
    owner_name  name;
    owner_super boolean;
BEGIN
    FOREACH schema_name IN ARRAY ARRAY['vectorsearch', 'prism']
    LOOP
        SELECT r.rolname, r.rolsuper INTO owner_name, owner_super
          FROM pg_catalog.pg_namespace n
          JOIN pg_catalog.pg_roles r
            ON r.oid OPERATOR(pg_catalog.=) n.nspowner
         WHERE n.nspname OPERATOR(pg_catalog.=) schema_name;

        IF NOT FOUND THEN
            EXECUTE pg_catalog.format('CREATE SCHEMA %I', schema_name);
            CONTINUE;
        END IF;

        IF NOT (owner_super OR owner_name OPERATOR(pg_catalog.=) current_user) THEN
            RAISE EXCEPTION
                'schema "%" already exists and is owned by "%", a role '
                'other than the installer or a superuser',
                schema_name, owner_name
                USING HINT = 'pg_vectorsearch refuses to install into a '
                    'schema an untrusted role controls; drop or re-own the '
                    'schema, or install as the role that owns it.';
        END IF;
    END LOOP;
END;
$$;

-- =====================================================================
-- build identity (maintenance/administration/inspection: fixed in
-- vectorsearch because it is extension-wide rather than specific to
-- the prism index, regardless of @extschema@)
-- =====================================================================

CREATE FUNCTION vectorsearch.git_commit() RETURNS text
    AS 'MODULE_PATHNAME', 'mkt_git_commit'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vectorsearch.extension_version() RETURNS text
    AS 'MODULE_PATHNAME', 'mkt_extension_version'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vectorsearch.extension_name() RETURNS text
    AS 'MODULE_PATHNAME', 'mkt_extension_name'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Prerelease install notice: warn at CREATE EXTENSION time when this
-- build is a prerelease (any -suffix version, e.g. -alpha1 or -dev).
-- A runtime check against vectorsearch.extension_version(), so final
-- releases carry nothing to strip and the notice can never ship stale.
DO $$
BEGIN
    IF pg_catalog.strpos(vectorsearch.extension_version(), '-')
        OPERATOR(pg_catalog.>) 0
    THEN
        RAISE WARNING '% % is a prerelease: upgrading to later '
            'versions might not be possible (reinstall instead) and '
            'its indexes may need rebuilding',
            vectorsearch.extension_name(),
            vectorsearch.extension_version();
    END IF;
END;
$$;

-- =====================================================================
-- vec32 type
-- =====================================================================

CREATE FUNCTION vec32_in(cstring, oid, integer) RETURNS vec32
    AS 'MODULE_PATHNAME', 'mkt_vec32_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_out(vec32) RETURNS cstring
    AS 'MODULE_PATHNAME', 'mkt_vec32_out'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_typmod_in(cstring[]) RETURNS integer
    AS 'MODULE_PATHNAME', 'mkt_vec32_typmod_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE vec32 (
    INPUT     = vec32_in,
    OUTPUT    = vec32_out,
    TYPMOD_IN = vec32_typmod_in,
    INTERNALLENGTH = VARIABLE,
    STORAGE   = external,
    CATEGORY  = 'U',
    DELIMITER = ','
);

-- =====================================================================
-- vec32 distance functions
-- =====================================================================

CREATE FUNCTION l2_distance(vec32, vec32) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_l2_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION inner_product(vec32, vec32) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION cosine_distance(vec32, vec32) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_cosine_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- vec32 utility functions
-- =====================================================================

CREATE FUNCTION vec32_dims(vec32) RETURNS integer
    AS 'MODULE_PATHNAME', 'mkt_vec32_dims'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_norm(vec32) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_pg_vec32_norm'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- vec32 private functions (for operators and opclass)
-- =====================================================================

CREATE FUNCTION vec32_l2_squared_distance(vec32, vec32) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_vec32_l2_squared_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_negative_inner_product(vec32, vec32) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_vec32_negative_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_cmp(vec32, vec32) RETURNS int4
    AS 'MODULE_PATHNAME', 'mkt_vec32_cmp'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_lt(vec32, vec32) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vec32_lt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_le(vec32, vec32) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vec32_le'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_eq(vec32, vec32) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vec32_eq'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_ne(vec32, vec32) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vec32_ne'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_ge(vec32, vec32) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vec32_ge'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_gt(vec32, vec32) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vec32_gt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- vec32 cast functions
-- =====================================================================

CREATE FUNCTION vec32(@extschema@.vec32, integer, boolean) RETURNS vec32
    AS 'MODULE_PATHNAME', 'mkt_vec32'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_vec32(real[], integer, boolean) RETURNS vec32
    AS 'MODULE_PATHNAME', 'mkt_array_to_vec32'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_vec32(float8[], integer, boolean) RETURNS vec32
    AS 'MODULE_PATHNAME', 'mkt_array_to_vec32'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_to_float4(@extschema@.vec32) RETURNS real[]
    AS 'MODULE_PATHNAME', 'mkt_vec32_to_float4'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- vec32 casts
-- =====================================================================

CREATE CAST (@extschema@.vec32 AS @extschema@.vec32)
    WITH FUNCTION vec32(@extschema@.vec32, integer, boolean) AS IMPLICIT;

CREATE CAST (real[] AS @extschema@.vec32)
    WITH FUNCTION array_to_vec32(real[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (float8[] AS @extschema@.vec32)
    WITH FUNCTION array_to_vec32(float8[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (@extschema@.vec32 AS real[])
    WITH FUNCTION vec32_to_float4(@extschema@.vec32);

-- =====================================================================
-- vec32 distance operators
-- =====================================================================

CREATE OPERATOR <-> (
    LEFTARG = vec32, RIGHTARG = vec32,
    FUNCTION = l2_distance,
    COMMUTATOR = '<->'
);

CREATE OPERATOR <#> (
    LEFTARG = vec32, RIGHTARG = vec32,
    FUNCTION = vec32_negative_inner_product,
    COMMUTATOR = '<#>'
);

CREATE OPERATOR <=> (
    LEFTARG = vec32, RIGHTARG = vec32,
    FUNCTION = cosine_distance,
    COMMUTATOR = '<=>'
);

-- =====================================================================
-- vec32 comparison operators
-- =====================================================================

CREATE OPERATOR < (
    LEFTARG = vec32, RIGHTARG = vec32,
    FUNCTION = vec32_lt,
    COMMUTATOR = '>', NEGATOR = '>=',
    RESTRICT = pg_catalog.scalarltsel, JOIN = pg_catalog.scalarltjoinsel
);

CREATE OPERATOR <= (
    LEFTARG = vec32, RIGHTARG = vec32,
    FUNCTION = vec32_le,
    COMMUTATOR = '>=', NEGATOR = '>',
    RESTRICT = pg_catalog.scalarlesel, JOIN = pg_catalog.scalarlejoinsel
);

CREATE OPERATOR = (
    LEFTARG = vec32, RIGHTARG = vec32,
    FUNCTION = vec32_eq,
    COMMUTATOR = '=', NEGATOR = '<>',
    RESTRICT = pg_catalog.eqsel, JOIN = pg_catalog.eqjoinsel,
    HASHES, MERGES
);

CREATE OPERATOR <> (
    LEFTARG = vec32, RIGHTARG = vec32,
    FUNCTION = vec32_ne,
    COMMUTATOR = '<>', NEGATOR = '=',
    RESTRICT = pg_catalog.neqsel, JOIN = pg_catalog.neqjoinsel
);

CREATE OPERATOR >= (
    LEFTARG = vec32, RIGHTARG = vec32,
    FUNCTION = vec32_ge,
    COMMUTATOR = '<=', NEGATOR = '<',
    RESTRICT = pg_catalog.scalargesel, JOIN = pg_catalog.scalargejoinsel
);

CREATE OPERATOR > (
    LEFTARG = vec32, RIGHTARG = vec32,
    FUNCTION = vec32_gt,
    COMMUTATOR = '<', NEGATOR = '<=',
    RESTRICT = pg_catalog.scalargtsel, JOIN = pg_catalog.scalargtjoinsel
);

-- =====================================================================
-- vec32 btree opclass
-- =====================================================================

CREATE OPERATOR FAMILY vec32_ops USING btree;

CREATE OPERATOR CLASS vec32_ops DEFAULT FOR TYPE vec32 USING btree
    FAMILY vec32_ops AS
    OPERATOR 1 <,
    OPERATOR 2 <=,
    OPERATOR 3 =,
    OPERATOR 4 >=,
    OPERATOR 5 >,
    FUNCTION 1 vec32_cmp(vec32, vec32);

-- =====================================================================
-- vec16 type
-- =====================================================================

CREATE FUNCTION vec16_in(cstring, oid, integer) RETURNS vec16
    AS 'MODULE_PATHNAME', 'mkt_vec16_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec16_out(vec16) RETURNS cstring
    AS 'MODULE_PATHNAME', 'mkt_vec16_out'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec16_typmod_in(cstring[]) RETURNS integer
    AS 'MODULE_PATHNAME', 'mkt_vec16_typmod_in'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE vec16 (
    INPUT     = vec16_in,
    OUTPUT    = vec16_out,
    TYPMOD_IN = vec16_typmod_in,
    INTERNALLENGTH = VARIABLE,
    STORAGE   = external,
    CATEGORY  = 'U',
    DELIMITER = ','
);

-- =====================================================================
-- vec16 distance functions
-- =====================================================================

CREATE FUNCTION l2_distance(vec16, vec16) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_vec16_l2_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION inner_product(vec16, vec16) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_vec16_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION cosine_distance(vec16, vec16) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_vec16_cosine_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- vec16 utility functions
-- =====================================================================

CREATE FUNCTION vec32_dims(vec16) RETURNS integer
    AS 'MODULE_PATHNAME', 'mkt_vec16_dims'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_norm(vec16) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_vec16_norm'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- vec16 private functions (for operators and opclass)
-- =====================================================================

CREATE FUNCTION vec16_l2_squared_distance(vec16, vec16) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_vec16_l2_squared_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec16_negative_inner_product(vec16, vec16) RETURNS float8
    AS 'MODULE_PATHNAME', 'mkt_vec16_negative_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec16_cmp(vec16, vec16) RETURNS int4
    AS 'MODULE_PATHNAME', 'mkt_vec16_cmp'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec16_lt(vec16, vec16) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vec16_lt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec16_le(vec16, vec16) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vec16_le'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec16_eq(vec16, vec16) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vec16_eq'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec16_ne(vec16, vec16) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vec16_ne'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec16_ge(vec16, vec16) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vec16_ge'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec16_gt(vec16, vec16) RETURNS bool
    AS 'MODULE_PATHNAME', 'mkt_vec16_gt'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- vec16 cast functions
-- =====================================================================

CREATE FUNCTION vec16(@extschema@.vec16, integer, boolean) RETURNS vec16
    AS 'MODULE_PATHNAME', 'mkt_vec16'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec16_to_vec32(@extschema@.vec16, integer, boolean)
    RETURNS vec32
    AS 'MODULE_PATHNAME', 'mkt_vec16_to_vec32'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vec32_to_vec16(@extschema@.vec32, integer, boolean)
    RETURNS vec16
    AS 'MODULE_PATHNAME', 'mkt_vec32_to_vec16'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_vec16(real[], integer, boolean) RETURNS vec16
    AS 'MODULE_PATHNAME', 'mkt_array_to_vec16'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_vec16(float8[], integer, boolean) RETURNS vec16
    AS 'MODULE_PATHNAME', 'mkt_array_to_vec16'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- =====================================================================
-- vec16 casts
-- =====================================================================

CREATE CAST (@extschema@.vec16 AS @extschema@.vec16)
    WITH FUNCTION vec16(@extschema@.vec16, integer, boolean) AS IMPLICIT;

CREATE CAST (@extschema@.vec16 AS @extschema@.vec32)
    WITH FUNCTION vec16_to_vec32(@extschema@.vec16, integer, boolean)
    AS IMPLICIT;

CREATE CAST (@extschema@.vec32 AS @extschema@.vec16)
    WITH FUNCTION vec32_to_vec16(@extschema@.vec32, integer, boolean)
    AS ASSIGNMENT;

CREATE CAST (real[] AS @extschema@.vec16)
    WITH FUNCTION array_to_vec16(real[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (float8[] AS @extschema@.vec16)
    WITH FUNCTION array_to_vec16(float8[], integer, boolean) AS ASSIGNMENT;

-- =====================================================================
-- vec16 distance operators
-- =====================================================================

CREATE OPERATOR <-> (
    LEFTARG = vec16, RIGHTARG = vec16,
    FUNCTION = l2_distance,
    COMMUTATOR = '<->'
);

CREATE OPERATOR <#> (
    LEFTARG = vec16, RIGHTARG = vec16,
    FUNCTION = vec16_negative_inner_product,
    COMMUTATOR = '<#>'
);

CREATE OPERATOR <=> (
    LEFTARG = vec16, RIGHTARG = vec16,
    FUNCTION = cosine_distance,
    COMMUTATOR = '<=>'
);

-- =====================================================================
-- vec16 comparison operators
-- =====================================================================

CREATE OPERATOR < (
    LEFTARG = vec16, RIGHTARG = vec16,
    FUNCTION = vec16_lt,
    COMMUTATOR = '>', NEGATOR = '>=',
    RESTRICT = pg_catalog.scalarltsel, JOIN = pg_catalog.scalarltjoinsel
);

CREATE OPERATOR <= (
    LEFTARG = vec16, RIGHTARG = vec16,
    FUNCTION = vec16_le,
    COMMUTATOR = '>=', NEGATOR = '>',
    RESTRICT = pg_catalog.scalarlesel, JOIN = pg_catalog.scalarlejoinsel
);

CREATE OPERATOR = (
    LEFTARG = vec16, RIGHTARG = vec16,
    FUNCTION = vec16_eq,
    COMMUTATOR = '=', NEGATOR = '<>',
    RESTRICT = pg_catalog.eqsel, JOIN = pg_catalog.eqjoinsel,
    HASHES, MERGES
);

CREATE OPERATOR <> (
    LEFTARG = vec16, RIGHTARG = vec16,
    FUNCTION = vec16_ne,
    COMMUTATOR = '<>', NEGATOR = '=',
    RESTRICT = pg_catalog.neqsel, JOIN = pg_catalog.neqjoinsel
);

CREATE OPERATOR >= (
    LEFTARG = vec16, RIGHTARG = vec16,
    FUNCTION = vec16_ge,
    COMMUTATOR = '<=', NEGATOR = '<',
    RESTRICT = pg_catalog.scalargesel, JOIN = pg_catalog.scalargejoinsel
);

CREATE OPERATOR > (
    LEFTARG = vec16, RIGHTARG = vec16,
    FUNCTION = vec16_gt,
    COMMUTATOR = '<', NEGATOR = '<=',
    RESTRICT = pg_catalog.scalargtsel, JOIN = pg_catalog.scalargtjoinsel
);

-- =====================================================================
-- vec16 btree opclass
-- =====================================================================

CREATE OPERATOR FAMILY vec16_ops USING btree;

CREATE OPERATOR CLASS vec16_ops DEFAULT FOR TYPE vec16 USING btree
    FAMILY vec16_ops AS
    OPERATOR 1 <,
    OPERATOR 2 <=,
    OPERATOR 3 =,
    OPERATOR 4 >=,
    OPERATOR 5 >,
    FUNCTION 1 vec16_cmp(vec16, vec16);

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
    input vec32,
    centroid vec32,
    params rabitq_params
) RETURNS rabitq
    AS 'MODULE_PATHNAME', 'mkt_rabitq_encode_pg'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION rabitq_encode(vec32, vec32, rabitq_params) IS
'Encode a vec32 to RaBitQ binary quantization relative to a centroid.
Returns a rabitq value containing the quantized bits, f_add, and f_rescale.
The params argument provides the orthogonal transform matrix (see rabitq_params_generate).';

-- =====================================================================
-- prism index access method
-- =====================================================================
-- The SQL-visible names below (prism, prism_handler, prism_metric_*,
-- prism_vec32_support, prism_vec16_support) name the access method
-- itself, matching the C symbols they link to -- see src/pg/iam_handler.c.

CREATE FUNCTION prism_handler(internal) RETURNS index_am_handler
    AS 'MODULE_PATHNAME', 'prism_handler' LANGUAGE C;

CREATE ACCESS METHOD prism TYPE INDEX HANDLER prism_handler;

COMMENT ON ACCESS METHOD prism IS 'prism ANN index';

-- Metric identifier functions (FUNCTION 2 in opclass)
CREATE FUNCTION prism_metric_l2(internal) RETURNS int4
    AS 'MODULE_PATHNAME', 'prism_metric_l2' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION prism_metric_ip(internal) RETURNS int4
    AS 'MODULE_PATHNAME', 'prism_metric_ip' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION prism_metric_cosine(internal) RETURNS int4
    AS 'MODULE_PATHNAME', 'prism_metric_cosine' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Column type descriptors (support function 3). Optional: an opclass that
-- declares none indexes `vec32`, which keeps the vec32 opclasses unchanged
-- and leaves an index built before this existed working. Returning the
-- descriptor from the opclass is what lets the access method agree with the
-- planner about a column's type without resolving a name or comparing an OID
-- -- see src/pg/typeinfo.h.
CREATE FUNCTION prism_vec32_support(internal) RETURNS internal
    AS 'MODULE_PATHNAME', 'prism_vec32_support' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION prism_vec16_support(internal) RETURNS internal
    AS 'MODULE_PATHNAME', 'prism_vec16_support' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Operator classes for vec32 type
CREATE OPERATOR CLASS vec32_l2_ops
    DEFAULT FOR TYPE vec32 USING prism AS
    OPERATOR 1 <-> (vec32, vec32) FOR ORDER BY float_ops,
    FUNCTION 1 vec32_l2_squared_distance(vec32, vec32),
    FUNCTION 2 prism_metric_l2(internal);

CREATE OPERATOR CLASS vec32_ip_ops
    FOR TYPE vec32 USING prism AS
    OPERATOR 1 <#> (vec32, vec32) FOR ORDER BY float_ops,
    FUNCTION 1 vec32_negative_inner_product(vec32, vec32),
    FUNCTION 2 prism_metric_ip(internal);

CREATE OPERATOR CLASS vec32_cosine_ops
    FOR TYPE vec32 USING prism AS
    OPERATOR 1 <=> (vec32, vec32) FOR ORDER BY float_ops,
    FUNCTION 1 cosine_distance(vec32, vec32),
    FUNCTION 2 prism_metric_cosine(internal);

-- Operator classes for vec16 type
--
-- The index itself is unchanged: postings hold RaBitQ codes either way, and
-- the AM widens a vec16 tuple to float32 on read (the distance and encode
-- kernels are float32-only). What vec16 buys is the heap, which is what an
-- exact rerank reads -- at 768d a vec32 row is 3080 bytes and fits 2 to an
-- 8 kB page against vec16's 1544 and 5. Centroids follow the column and are
-- stored half-precision too (MKT_CENTROID_FMT_HALF).
CREATE OPERATOR CLASS vec16_l2_ops
    DEFAULT FOR TYPE vec16 USING prism AS
    OPERATOR 1 <-> (vec16, vec16) FOR ORDER BY float_ops,
    FUNCTION 1 vec16_l2_squared_distance(vec16, vec16),
    FUNCTION 2 prism_metric_l2(internal),
    FUNCTION 3 prism_vec16_support(internal);

CREATE OPERATOR CLASS vec16_ip_ops
    FOR TYPE vec16 USING prism AS
    OPERATOR 1 <#> (vec16, vec16) FOR ORDER BY float_ops,
    FUNCTION 1 vec16_negative_inner_product(vec16, vec16),
    FUNCTION 2 prism_metric_ip(internal),
    FUNCTION 3 prism_vec16_support(internal);

CREATE OPERATOR CLASS vec16_cosine_ops
    FOR TYPE vec16 USING prism AS
    OPERATOR 1 <=> (vec16, vec16) FOR ORDER BY float_ops,
    FUNCTION 1 cosine_distance(vec16, vec16),
    FUNCTION 2 prism_metric_cosine(internal),
    FUNCTION 3 prism_vec16_support(internal);

-- =====================================================================
-- index inspection functions
-- =====================================================================

CREATE FUNCTION prism.centroid_pages(regclass)
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

CREATE FUNCTION prism.posting_pages(regclass)
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
CREATE FUNCTION prism.tids_clusters(regclass, tid[])
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
CREATE FUNCTION prism.index_settings(regclass)
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
CREATE FUNCTION prism.convert_posting_to_fastscan(
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
CREATE PROCEDURE prism.split_posting_list(
        index_oid regclass,
        head_blkno bigint
    )
    AS 'MODULE_PATHNAME', 'mkt_split_posting_list'
    LANGUAGE C;

COMMENT ON PROCEDURE prism.split_posting_list(regclass, bigint) IS
    'Split one posting list into two or more balanced lists. index_oid is the '
    'index; head_blkno is the block number of the list''s head page. '
    'Owner-only; reports the outcome via NOTICE. Use prism.rebalance to split '
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
CREATE PROCEDURE prism.rebalance(index_oid regclass, target_entries integer DEFAULT NULL)
    AS 'MODULE_PATHNAME', 'mkt_rebalance'
    LANGUAGE C;

COMMENT ON PROCEDURE prism.rebalance(regclass, integer) IS
    'Split every posting list that has grown past twice target_entries into '
    'lists of about target_entries each. index_oid is the index; '
    'target_entries is the size a list rests at (NULL, the default, derives it '
    'from the row count). Owner-only; reports the number of lists split via '
    'NOTICE.';

-- =====================================================================
-- pgvector binary cast support
-- =====================================================================
--
-- When pgvector (the vec32 extension) is installed, pg_vectorsearch creates
-- zero-overhead binary casts between the two sets of types. The types
-- have identical binary layouts (varlena header + int16 dim + int16
-- unused + float[]), so WITHOUT FUNCTION casts produce RelabelType
-- nodes with no conversion overhead.
--
-- Cast directions:
--   pgvector -> pg_vectorsearch: IMPLICIT (pgvector columns work
--     transparently with pg_vectorsearch operators and indexes)
--   pg_vectorsearch -> pgvector: ASSIGNMENT (avoids operator ambiguity
--     when both extensions define <->, <#>, <=>)
--
-- Casts alone are not enough to make a prism index reachable from a
-- query written against pgvector. An index is only considered for an
-- ORDER BY when the ordering operator belongs to the index's operator
-- family, and pgvector's <->, <#> and <=> belong to pgvector's families.
-- Without help, `ORDER BY v <-> $1` under a pgvector-first search_path
-- plans a sequential scan -- which returns correct rows, so it is easy to
-- mistake for a working index scan. setup_pgvector_compat() therefore adds
-- pgvector's three distance operators to prism's operator families as
-- ordering members alongside the casts, so either spelling of the operator
-- reaches the index. Casts and operators are one function on purpose: they
-- are a unit (the operators rely on the casts' binary-coercibility), so a
-- new install path can never add one and forget the other.
--
-- The casts are standalone objects (not owned by either extension).
-- PostgreSQL auto-drops them via type dependencies when the referenced
-- types are dropped, so DROP EXTENSION on either side removes the
-- casts without affecting the other extension.
--
-- Both install orderings are supported:
--   pgvector first, pg_vectorsearch later: DO block below sets up compat
--   pg_vectorsearch first, pgvector later: event trigger sets up compat

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
-- 3. Operator shadowing via a schema the extension did not choose. If an
--    untrusted role owns a schema the extension's own objects end up in,
--    it can add an operator on pgvector's type there -- e.g.
--    that_schema.<->(public.vec32, public.vec32) -- that shadows pgvector's
--    own for any role with that_schema ahead of public, running attacker
--    code as that role. The extension's OWN operators are not shadowable:
--    a same-signature plant conflicts at install and installed objects are
--    membership-locked. Two fixes were weighed:
--      (a) Occupy the signatures -- pre-create safe, delegating versions of
--          pgvector's operators AND functions in every schema this
--          extension touches so the attacker cannot. Rejected: the surface
--          is pgvector's whole public API across all its types and it grows
--          with pgvector versions, so a newly added pgvector operator
--          silently reopens the hole until this extension catches up -- a
--          maintenance treadmill tied to another project's API, and it
--          still leaves non-pgvector shadows open.
--      (b) Ensure the schema is trusted-owned -- refuse to install into one
--          owned by an untrusted role (the schema ownership guard at the top
--          of this script). Chosen: version-independent, comprehensive
--          (nothing hostile can live there at all), ~10 lines. Applied only
--          to `vectorsearch` and `prism`: unlike @extschema@, which the
--          installer explicitly chose (or already had first on their own
--          search_path -- the same standing responsibility as installing
--          any relocatable extension), `vectorsearch` and `prism` must be
--          owned by the extension's installer or a superuser, since the
--          installer has no independent reason to have already vetted
--          their ownership.
--
-- Set up pgvector interoperability in one step: the binary casts between the
-- two extensions' types, and the membership of pgvector's distance operators
-- in prism's operator families. These belong together -- the casts
-- make pgvector's vector/halfvec binary-coercible to vec32/vec16, which is
-- exactly what lets pgvector's operators join a family whose opclass is
-- FOR TYPE <vec32/vec16>. Keeping them in a single function means a caller
-- cannot add the casts and forget the operators (which would silently
-- downgrade pgvector-operator queries to a sequential scan).
--
-- Casts first, then operators (the operators depend on the casts' coercibility).
-- Idempotent throughout via exception handling (neither CREATE CAST nor ALTER
-- OPERATOR FAMILY has an IF NOT EXISTS form).
CREATE FUNCTION prism.setup_pgvector_compat() RETURNS void
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
    ext_ns text;   -- The extension's own current schema, install-time
                   -- relocatable too -- discovered from
                   -- pg_extension.extnamespace rather than baked in via
                   -- @extschema@ substitution, the same reasoning applied
                   -- to pgv_ns above.
    ext text;      -- ...quote_ident'd, for building qualified type names
BEGIN
    -- pgvector is relocatable: its types and operators live in whatever
    -- schema it was installed into, not necessarily public. Discover it
    -- from the catalog rather than assuming public; extnamespace stays
    -- authoritative even after ALTER EXTENSION vec32 SET SCHEMA. The name
    -- is only ever interpolated through %I / quote_ident, so it stays
    -- injection-safe.
    SELECT n.nspname INTO pgv_ns
    FROM pg_catalog.pg_extension e
    JOIN pg_catalog.pg_namespace n
      ON n.oid OPERATOR(pg_catalog.=) e.extnamespace
    WHERE e.extname OPERATOR(pg_catalog.=) 'vector';

    IF pgv_ns IS NULL THEN
        RAISE EXCEPTION 'pgvector (extension "vector") is not installed';
    END IF;
    pgv := pg_catalog.quote_ident(pgv_ns);

    SELECT n.nspname INTO ext_ns
    FROM pg_catalog.pg_extension e
    JOIN pg_catalog.pg_namespace n
      ON n.oid OPERATOR(pg_catalog.=) e.extnamespace
    WHERE e.extname OPERATOR(pg_catalog.=) 'pg_vectorsearch';
    ext := pg_catalog.quote_ident(ext_ns);

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
    -- Validating the context too matters because the two directions differ
    -- deliberately (pgvector->pg_vectorsearch IMPLICIT,
    -- pg_vectorsearch->pgvector ASSIGNMENT): a binary cast planted with the
    -- wrong context still has method 'b' but changes
    -- coercion/operator-resolution behaviour. Silently adopting either (the
    -- old EXCEPTION WHEN duplicate_object THEN NULL) would hide it; we fail
    -- loudly instead.
    FOR r IN
        SELECT * FROM (VALUES
            (pgv OPERATOR(pg_catalog.||) '.vector',
                 ext OPERATOR(pg_catalog.||) '.vec32',   'IMPLICIT'),
            (pgv OPERATOR(pg_catalog.||) '.halfvec',
                 ext OPERATOR(pg_catalog.||) '.vec16',  'IMPLICIT'),
            (ext OPERATOR(pg_catalog.||) '.vec32',
                 pgv OPERATOR(pg_catalog.||) '.vector',  'ASSIGNMENT'),
            (ext OPERATOR(pg_catalog.||) '.vec16',
                 pgv OPERATOR(pg_catalog.||) '.halfvec', 'ASSIGNMENT')
        ) AS t(src, tgt, ctx)
    LOOP
        -- pg_cast.castcontext code for the expected context: the first letter
        -- of the lowercased keyword (implicit->i, assignment->a).
        expected_context := pg_catalog.substr(pg_catalog.lower(r.ctx), 1, 1);

        SELECT castmethod, castcontext INTO existing_method, existing_context
        FROM pg_catalog.pg_cast
        WHERE castsource OPERATOR(pg_catalog.=) r.src::pg_catalog.regtype
          AND casttarget OPERATOR(pg_catalog.=) r.tgt::pg_catalog.regtype;

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

    -- 2. pgvector's distance operators as ordering members of prism's
    -- operator families. Strategy 1 and float_ops match the opclass
    -- declarations above; the operator's left type only has to be
    -- binary-coercible to the family's index type, which the casts above
    -- guarantee. Adding an operator family
    -- member requires superuser, so an attacker cannot pre-plant one; the
    -- duplicate_object catch here is pure idempotency for a legitimate re-run.
    FOR r IN
        SELECT * FROM (VALUES
            ('vec32_l2_ops',      'vector',  '<->'),
            ('vec32_ip_ops',      'vector',  '<#>'),
            ('vec32_cosine_ops',  'vector',  '<=>'),
            ('vec16_l2_ops',     'halfvec', '<->'),
            ('vec16_ip_ops',     'halfvec', '<#>'),
            ('vec16_cosine_ops', 'halfvec', '<=>')
        ) AS t(fam, typ, op)
    LOOP
        BEGIN
            EXECUTE pg_catalog.format(
                'ALTER OPERATOR FAMILY %I.%I USING prism '
                'ADD OPERATOR 1 %I.%s (%I.%I, %I.%I) '
                'FOR ORDER BY pg_catalog.float_ops',
                ext_ns, r.fam, pgv_ns, r.op, pgv_ns, r.typ, pgv_ns, r.typ);
        EXCEPTION WHEN duplicate_object THEN NULL;
        END;
    END LOOP;
END;
$$;

-- Create casts now if pgvector is already installed.
-- During CREATE EXTENSION, objects created in an anonymous DO ($$ ... $$)
-- block are auto-owned by the extension. We immediately disassociate the
-- casts so that
-- DROP EXTENSION pg_vectorsearch does not cascade to (or through) pgvector.
-- The casts still get cleaned up via auto-dependencies on their
-- referenced types.
DO $$
DECLARE
    pgv_ns text;
    ext_ns text;   -- The extension's own current schema, discovered rather
                   -- than assumed via @extschema@ -- see the note on
                   -- ext_ns in setup_pgvector_compat() above.
BEGIN
    -- pgvector is relocatable; discover its schema (NULL if not installed).
    SELECT n.nspname INTO pgv_ns
    FROM pg_catalog.pg_extension e
    JOIN pg_catalog.pg_namespace n
      ON n.oid OPERATOR(pg_catalog.=) e.extnamespace
    WHERE e.extname OPERATOR(pg_catalog.=) 'vector';

    IF pgv_ns IS NOT NULL THEN
        SELECT n.nspname INTO ext_ns
        FROM pg_catalog.pg_extension e
        JOIN pg_catalog.pg_namespace n
          ON n.oid OPERATOR(pg_catalog.=) e.extnamespace
        WHERE e.extname OPERATOR(pg_catalog.=) 'pg_vectorsearch';

        PERFORM prism.setup_pgvector_compat();
        EXECUTE pg_catalog.format('ALTER EXTENSION pg_vectorsearch DROP CAST '
            '(%I.vector AS %I.vec32)', pgv_ns, ext_ns);
        EXECUTE pg_catalog.format('ALTER EXTENSION pg_vectorsearch DROP CAST '
            '(%I.halfvec AS %I.vec16)', pgv_ns, ext_ns);
        EXECUTE pg_catalog.format('ALTER EXTENSION pg_vectorsearch DROP CAST '
            '(%I.vec32 AS %I.vector)', ext_ns, pgv_ns);
        EXECUTE pg_catalog.format('ALTER EXTENSION pg_vectorsearch DROP CAST '
            '(%I.vec16 AS %I.halfvec)', ext_ns, pgv_ns);
    END IF;
END;
$$;

-- Event trigger: create casts when pgvector is installed after
-- pg_vectorsearch. Lives in prism alongside setup_pgvector_compat() -- it
-- is that function's automatic trigger, not part of the vec32/vec16 type
-- API, and prism is a fixed name it can reference directly (no schema
-- discovery needed for it).
CREATE FUNCTION prism.on_extension_create()
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
               WHERE object_type OPERATOR(pg_catalog.=) 'extension'
    LOOP
        IF obj.object_identity OPERATOR(pg_catalog.=) 'vector' THEN
            -- setup_pgvector_compat() does superuser-only DDL (CREATE CAST
            -- WITHOUT FUNCTION, ALTER OPERATOR FAMILY). An event trigger runs
            -- as the role that ran CREATE EXTENSION, so if a NON-superuser
            -- installs pgvector -- possible where it is trusted, as some
            -- managed platforms allow -- this PERFORM would fail and roll
            -- back the whole pgvector install, making pg_vectorsearch's
            -- presence break pgvector. Skip and warn instead; a superuser
            -- finishes the wiring later. (SECURITY DEFINER was rejected: it
            -- would run this superuser-only DDL for anyone who can create
            -- an extension.)
            SELECT r.rolsuper INTO is_super
              FROM pg_catalog.pg_roles r
             WHERE r.rolname OPERATOR(pg_catalog.=) current_user;

            IF is_super THEN
                PERFORM prism.setup_pgvector_compat();
            ELSE
                RAISE WARNING 'pg_vectorsearch did not set up pgvector '
                    'compatibility: it requires superuser privileges'
                    USING HINT = 'A superuser should run '
                        'prism.setup_pgvector_compat() so pgvector-typed '
                        'columns can use prism indexes.';
            END IF;
        END IF;
    END LOOP;
END;
$$;

CREATE EVENT TRIGGER prism_pgvector_cast_trigger
    ON ddl_command_end
    WHEN TAG IN ('CREATE EXTENSION')
    EXECUTE FUNCTION prism.on_extension_create();
