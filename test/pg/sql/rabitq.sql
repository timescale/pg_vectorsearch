-- RaBitQ type I/O, typmod, accessors, comparison, encoding

-- Basic I/O round-trip
SELECT '{10101010:1.5:2.3}'::rabitq;
SELECT '{0000:0:0}'::rabitq;
SELECT '{1111:100.5:200.75}'::rabitq;
SELECT '{1:1:1}'::rabitq;

-- Float precision round-trip (%.8g)
SELECT '{10101010:1.2345679:9.8765431}'::rabitq;

-- 16-bit (2 bytes)
SELECT '{1010101010101010:1.5:2.3}'::rabitq;

-- Typmod
SELECT '{10101010:1.5:2.3}'::rabitq(8);

-- Typmod mismatch
SELECT '{10101010:1.5:2.3}'::rabitq(4); -- error

-- Empty bits
SELECT '{:1:2}'::rabitq; -- error

-- No closing brace
SELECT '{10101010:1.5:2.3'::rabitq; -- error

-- No opening brace
SELECT '10101010:1.5:2.3}'::rabitq; -- error

-- Invalid bit character
SELECT '{10201010:1.5:2.3}'::rabitq; -- error

-- Missing f_add
SELECT '{10101010}'::rabitq; -- error

-- Missing f_rescale
SELECT '{10101010:1.5}'::rabitq; -- error

-- Trailing content
SELECT '{10101010:1.5:2.3}abc'::rabitq; -- error

-- NaN in f_add
SELECT '{10101010:NaN:2.3}'::rabitq; -- error

-- Inf in f_rescale
SELECT '{10101010:1.5:Inf}'::rabitq; -- error

-- Leading/trailing whitespace
SELECT '  {10101010:1.5:2.3}  '::rabitq;

-- Accessor functions
SELECT rabitq_dims('{10101010:1.5:2.3}'::rabitq);
SELECT rabitq_f_add('{10101010:1.5:2.3}'::rabitq);
SELECT rabitq_f_rescale('{10101010:1.5:2.3}'::rabitq);

-- 16-dim accessors
SELECT rabitq_dims('{1010101010101010:1.5:2.3}'::rabitq);

-- Comparison operators (bits are the quantized identity)
SELECT '{10101010:1.5:2.3}'::rabitq = '{10101010:1.5:2.3}'::rabitq;
SELECT '{10101010:1.5:2.3}'::rabitq = '{10101010:9.9:9.9}'::rabitq;
SELECT '{10101010:1.5:2.3}'::rabitq <> '{10101011:1.5:2.3}'::rabitq;
SELECT '{00000000:1.5:2.3}'::rabitq < '{10000000:1.5:2.3}'::rabitq;
SELECT '{10000000:1.5:2.3}'::rabitq > '{00000000:1.5:2.3}'::rabitq;
SELECT '{10101010:1.5:2.3}'::rabitq <= '{10101010:1.5:2.3}'::rabitq;
SELECT '{10101010:1.5:2.3}'::rabitq >= '{10101010:1.5:2.3}'::rabitq;

-- Different dimensions: shorter < longer
SELECT '{1010:1.5:2.3}'::rabitq < '{10101010:1.5:2.3}'::rabitq;

-- Table storage and ORDER BY
CREATE TABLE t_rq (id serial, q rabitq(8));
INSERT INTO t_rq (q) VALUES
    ('{10101010:1:1}'),
    ('{01010101:2:2}'),
    ('{11111111:3:3}'),
    ('{00000000:4:4}');
SELECT * FROM t_rq ORDER BY q;
SELECT * FROM t_rq ORDER BY q DESC;
DROP TABLE t_rq;

-- rabitq_params generation and accessors
SELECT rabitq_params_dim(rabitq_params_generate(8, 42));
SELECT rabitq_params_seed(rabitq_params_generate(8, 42));

-- rabitq_params text output
SELECT rabitq_params_generate(8, 42);

-- Oversized dimensions are refused: the transform matrix is O(dim^3) and the
-- function is callable by any role, so it is capped at the largest dimension
-- an mktann index can hold.
SELECT rabitq_params_generate(1969, 42); -- error

-- rabitq_params text input (not supported)
SELECT '{8:42}'::rabitq_params; -- error

-- Encoding: deterministic with known seed
-- Use zero centroid so residual = input
-- Check components separately with rounding (FP results vary across platforms)
SELECT rabitq_dims(q) AS dims,
       round(rabitq_f_add(q)::numeric, 0) AS f_add,
       round(rabitq_f_rescale(q)::numeric, 1) AS f_rescale
FROM (SELECT rabitq_encode(
    '[1,2,3,4,5,6,7,8]'::vector,
    '[0,0,0,0,0,0,0,0]'::vector,
    rabitq_params_generate(8, 42)
) AS q) t;

-- Encoding dimension mismatch
SELECT rabitq_encode(
    '[1,2,3]'::vector,
    '[0,0,0,0,0,0,0,0]'::vector,
    rabitq_params_generate(8, 42)
); -- error

-- Encoding: round-trip dims check
SELECT rabitq_dims(rabitq_encode(
    '[1,2,3,4,5,6,7,8]'::vector,
    '[0,0,0,0,0,0,0,0]'::vector,
    rabitq_params_generate(8, 42)
));
