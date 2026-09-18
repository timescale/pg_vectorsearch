-- Vector type I/O, typmod, casts, comparison

-- Basic I/O
SELECT '[1,2,3]'::vec32;
SELECT '[1.5,-2.3,0]'::vec32;
SELECT '[0]'::vec32;

-- Typmod
SELECT '[1,2,3]'::vec32(3);

-- Typmod mismatch
SELECT '[1,2,3]'::vec32(2); -- error

-- Empty vec32
SELECT '[]'::vec32; -- error

-- NaN
SELECT '[1,NaN,3]'::vec32; -- error

-- Inf
SELECT '[1,Inf,3]'::vec32; -- error

-- Bad syntax
SELECT '[1,2,'::vec32; -- error
SELECT '1,2,3'::vec32; -- error
SELECT '[1 2]'::vec32; -- error: missing comma
SELECT '[1,2]abc'::vec32; -- error: trailing content

-- Leading/trailing whitespace
SELECT '  [1,2,3]  '::vec32;

-- Spaces between elements
SELECT '[ 1 , 2 , 3 ]'::vec32;

-- Utility functions
SELECT vec32_dims('[1,2,3]'::vec32);
SELECT vec32_norm('[3,4]'::vec32);

-- Table storage
CREATE TABLE t_vec (id serial, v vec32(3));
INSERT INTO t_vec (v) VALUES ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]');
SELECT * FROM t_vec ORDER BY id;

-- Array cast
SELECT ARRAY[1.0,2.0,3.0]::real[]::vec32;
SELECT ARRAY[1.0,2.0,3.0]::float8[]::vec32;

-- Vector to array
SELECT ('[1,2,3]'::vec32)::real[];

-- Array cast errors
SELECT ARRAY[1.0, NULL, 3.0]::real[]::vec32; -- error: null
SELECT '{}'::real[]::vec32; -- error: empty

-- Comparison operators
SELECT '[1,2,3]'::vec32 < '[1,2,4]'::vec32;
SELECT '[1,2,3]'::vec32 = '[1,2,3]'::vec32;
SELECT '[1,2,3]'::vec32 > '[1,2,2]'::vec32;
SELECT '[1,2,3]'::vec32 <> '[1,2,4]'::vec32;
SELECT '[1,2,3]'::vec32 <= '[1,2,3]'::vec32;
SELECT '[1,2,3]'::vec32 >= '[1,2,3]'::vec32;

-- ORDER BY (btree opclass)
SELECT v FROM t_vec ORDER BY v;

DROP TABLE t_vec;
