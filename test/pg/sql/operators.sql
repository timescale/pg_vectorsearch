-- Distance operators and ORDER BY

-- <-> operator (L2 distance)
SELECT '[0,0]'::vector <-> '[3,4]'::vector;

-- <#> operator (negative inner product)
SELECT '[1,2,3]'::vector <#> '[4,5,6]'::vector;

-- <=> operator (cosine distance)
SELECT round(('[1,0]'::vector <=> '[0,1]'::vector)::numeric, 5);

-- ORDER BY distance (nearest neighbor pattern)
CREATE TABLE t_vecs (id serial, v vector(3));
INSERT INTO t_vecs (v) VALUES
    ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]'),
    ('[1,1,0]'), ('[0,1,1]'), ('[1,0,1]');

SELECT id, v, v <-> '[1,1,1]'::vector AS dist
    FROM t_vecs ORDER BY v <-> '[1,1,1]'::vector LIMIT 3;

-- Halfvec operators
SELECT '[0,0]'::halfvec <-> '[3,4]'::halfvec;
SELECT '[1,2,3]'::halfvec <#> '[4,5,6]'::halfvec;
SELECT round(('[1,0]'::halfvec <=> '[0,1]'::halfvec)::numeric, 5);

-- Halfvec ORDER BY
CREATE TABLE t_hvecs (id serial, v halfvec(3));
INSERT INTO t_hvecs (v) VALUES
    ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]');

SELECT id, v, v <-> '[1,1,1]'::halfvec AS dist
    FROM t_hvecs ORDER BY v <-> '[1,1,1]'::halfvec;

DROP TABLE t_vecs;
DROP TABLE t_hvecs;
