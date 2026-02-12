-- Halfvec type I/O, casts, comparison

-- Basic I/O
SELECT '[1,2,3]'::halfvec;
SELECT '[1.5,-2.5,0]'::halfvec;
SELECT '[0]'::halfvec;

-- Typmod
SELECT '[1,2,3]'::halfvec(3);

-- Typmod mismatch
SELECT '[1,2,3]'::halfvec(2); -- error

-- Empty halfvec
SELECT '[]'::halfvec; -- error

-- Bad syntax
SELECT '1,2,3'::halfvec; -- error: missing bracket
SELECT '[1 2]'::halfvec; -- error: missing comma
SELECT '[1,2]xyz'::halfvec; -- error: trailing content
SELECT '[1,2,'::halfvec; -- error: unterminated

-- Leading/trailing whitespace
SELECT '  [1,2,3]  '::halfvec;

-- Spaces between elements
SELECT '[ 1 , 2 , 3 ]'::halfvec;

-- Table storage
CREATE TABLE t_hvec (id serial, v halfvec(3));
INSERT INTO t_hvec (v) VALUES ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]');
SELECT * FROM t_hvec ORDER BY id;

-- Halfvec to vector cast
SELECT ('[1,2,3]'::halfvec)::vector;

-- Vector to halfvec cast
SELECT ('[1,2,3]'::vector)::halfvec;

-- Array cast
SELECT ARRAY[1.0,2.0,3.0]::real[]::halfvec;

-- Array cast from float8
SELECT ARRAY[1.0,2.0,3.0]::float8[]::halfvec;

-- Array cast errors
SELECT ARRAY[1.0, NULL, 3.0]::real[]::halfvec; -- error: null
SELECT '{}'::real[]::halfvec; -- error: empty

-- Comparison operators (all six)
SELECT '[1,2,3]'::halfvec < '[1,2,4]'::halfvec;
SELECT '[1,2,3]'::halfvec <= '[1,2,3]'::halfvec;
SELECT '[1,2,3]'::halfvec = '[1,2,3]'::halfvec;
SELECT '[1,2,3]'::halfvec <> '[1,2,4]'::halfvec;
SELECT '[1,2,3]'::halfvec >= '[1,2,3]'::halfvec;
SELECT '[1,2,3]'::halfvec > '[1,2,2]'::halfvec;

-- ORDER BY (btree opclass)
SELECT v FROM t_hvec ORDER BY v;

-- Utility
SELECT vector_dims('[1,2,3]'::halfvec);
SELECT vector_norm('[3,4]'::halfvec);

DROP TABLE t_hvec;
