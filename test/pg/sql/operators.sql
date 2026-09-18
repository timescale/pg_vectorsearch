-- Distance operators and ORDER BY

-- <-> operator (L2 distance)
SELECT '[0,0]'::vec32 <-> '[3,4]'::vec32;

-- <#> operator (negative inner product)
SELECT '[1,2,3]'::vec32 <#> '[4,5,6]'::vec32;

-- <=> operator (cosine distance)
SELECT round(('[1,0]'::vec32 <=> '[0,1]'::vec32)::numeric, 5);

-- ORDER BY distance (nearest neighbor pattern)
CREATE TABLE t_vecs (id serial, v vec32(3));
INSERT INTO t_vecs (v) VALUES
    ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]'),
    ('[1,1,0]'), ('[0,1,1]'), ('[1,0,1]');

SELECT id, v, v <-> '[1,1,1]'::vec32 AS dist
    FROM t_vecs ORDER BY v <-> '[1,1,1]'::vec32 LIMIT 3;

-- Halfvec operators
SELECT '[0,0]'::vec16 <-> '[3,4]'::vec16;
SELECT '[1,2,3]'::vec16 <#> '[4,5,6]'::vec16;
SELECT round(('[1,0]'::vec16 <=> '[0,1]'::vec16)::numeric, 5);

-- Halfvec ORDER BY
CREATE TABLE t_hvecs (id serial, v vec16(3));
INSERT INTO t_hvecs (v) VALUES
    ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]');

SELECT id, v, v <-> '[1,1,1]'::vec16 AS dist
    FROM t_hvecs ORDER BY v <-> '[1,1,1]'::vec16;

DROP TABLE t_vecs;
DROP TABLE t_hvecs;
