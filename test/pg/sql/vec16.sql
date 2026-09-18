-- Halfvec type I/O, casts, comparison

-- Basic I/O
SELECT '[1,2,3]'::vec16;
SELECT '[1.5,-2.5,0]'::vec16;
SELECT '[0]'::vec16;

-- Typmod
SELECT '[1,2,3]'::vec16(3);

-- Typmod mismatch
SELECT '[1,2,3]'::vec16(2); -- error

-- Empty vec16
SELECT '[]'::vec16; -- error

-- Bad syntax
SELECT '1,2,3'::vec16; -- error: missing bracket
SELECT '[1 2]'::vec16; -- error: missing comma
SELECT '[1,2]xyz'::vec16; -- error: trailing content
SELECT '[1,2,'::vec16; -- error: unterminated

-- Leading/trailing whitespace
SELECT '  [1,2,3]  '::vec16;

-- Spaces between elements
SELECT '[ 1 , 2 , 3 ]'::vec16;

-- Table storage
CREATE TABLE t_hvec (id serial, v vec16(3));
INSERT INTO t_hvec (v) VALUES ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]');
SELECT * FROM t_hvec ORDER BY id;

-- Halfvec to vec32 cast
SELECT ('[1,2,3]'::vec16)::vec32;

-- Vector to vec16 cast
SELECT ('[1,2,3]'::vec32)::vec16;

-- Array cast
SELECT ARRAY[1.0,2.0,3.0]::real[]::vec16;

-- Array cast from float8
SELECT ARRAY[1.0,2.0,3.0]::float8[]::vec16;

-- Array cast errors
SELECT ARRAY[1.0, NULL, 3.0]::real[]::vec16; -- error: null
SELECT '{}'::real[]::vec16; -- error: empty

-- Comparison operators (all six)
SELECT '[1,2,3]'::vec16 < '[1,2,4]'::vec16;
SELECT '[1,2,3]'::vec16 <= '[1,2,3]'::vec16;
SELECT '[1,2,3]'::vec16 = '[1,2,3]'::vec16;
SELECT '[1,2,3]'::vec16 <> '[1,2,4]'::vec16;
SELECT '[1,2,3]'::vec16 >= '[1,2,3]'::vec16;
SELECT '[1,2,3]'::vec16 > '[1,2,2]'::vec16;

-- ORDER BY (btree opclass)
SELECT v FROM t_hvec ORDER BY v;

-- Utility
SELECT vec32_dims('[1,2,3]'::vec16);
SELECT vec32_norm('[3,4]'::vec16);

DROP TABLE t_hvec;
