/* meerkat--0.1.0.sql */

-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION meerkat" to load this file.\quit

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
    STORAGE   = extended,
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
    STORAGE   = extended,
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
