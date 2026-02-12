-- Vector type I/O, typmod, casts, comparison

-- Basic I/O
SELECT '[1,2,3]'::vector;
SELECT '[1.5,-2.3,0]'::vector;
SELECT '[0]'::vector;

-- Typmod
SELECT '[1,2,3]'::vector(3);

-- Typmod mismatch
SELECT '[1,2,3]'::vector(2); -- error

-- Empty vector
SELECT '[]'::vector; -- error

-- NaN
SELECT '[1,NaN,3]'::vector; -- error

-- Inf
SELECT '[1,Inf,3]'::vector; -- error

-- Bad syntax
SELECT '[1,2,'::vector; -- error
SELECT '1,2,3'::vector; -- error
SELECT '[1 2]'::vector; -- error: missing comma
SELECT '[1,2]abc'::vector; -- error: trailing content

-- Leading/trailing whitespace
SELECT '  [1,2,3]  '::vector;

-- Spaces between elements
SELECT '[ 1 , 2 , 3 ]'::vector;

-- Utility functions
SELECT vector_dims('[1,2,3]'::vector);
SELECT vector_norm('[3,4]'::vector);

-- Table storage
CREATE TABLE t_vec (id serial, v vector(3));
INSERT INTO t_vec (v) VALUES ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]');
SELECT * FROM t_vec ORDER BY id;

-- Array cast
SELECT ARRAY[1.0,2.0,3.0]::real[]::vector;
SELECT ARRAY[1.0,2.0,3.0]::float8[]::vector;

-- Vector to array
SELECT ('[1,2,3]'::vector)::real[];

-- Array cast errors
SELECT ARRAY[1.0, NULL, 3.0]::real[]::vector; -- error: null
SELECT '{}'::real[]::vector; -- error: empty

-- Comparison operators
SELECT '[1,2,3]'::vector < '[1,2,4]'::vector;
SELECT '[1,2,3]'::vector = '[1,2,3]'::vector;
SELECT '[1,2,3]'::vector > '[1,2,2]'::vector;
SELECT '[1,2,3]'::vector <> '[1,2,4]'::vector;
SELECT '[1,2,3]'::vector <= '[1,2,3]'::vector;
SELECT '[1,2,3]'::vector >= '[1,2,3]'::vector;

-- ORDER BY (btree opclass)
SELECT v FROM t_vec ORDER BY v;

DROP TABLE t_vec;
