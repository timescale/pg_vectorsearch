-- Run the same mutation scenarios with inline and spilled references.
\getenv abs_srcdir PG_ABS_SRCDIR
\set cases_sql :abs_srcdir '/sql/mutations_cases.sql'

CREATE FUNCTION mutvec(x real) RETURNS vec32 LANGUAGE plpgsql STABLE AS $$
BEGIN
    RETURN (ARRAY[x] || array_fill(0::real,
        ARRAY[current_setting('test.dim')::int - 1]))::vec32;
END
$$;

\set dim 3
\set run_fastscan true
SET test.dim = '3';
\i :cases_sql

-- A whole fastscan group does not fit a page at this dimension.
\set dim 4096
\set run_fastscan false
SET test.dim = '4096';
\i :cases_sql

DROP FUNCTION mutvec(real);
RESET test.dim;
