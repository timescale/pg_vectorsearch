-- Distance functions for both vec32 and vec16

-- L2 distance: sqrt(9+9+9) = sqrt(27) ≈ 5.196
SELECT round(l2_distance('[1,2,3]'::vec32, '[4,5,6]'::vec32)::numeric, 3);

-- L2 known: sqrt(9+16) = 5.0
SELECT l2_distance('[0,0]'::vec32, '[3,4]'::vec32);

-- Inner product: 1*4 + 2*5 + 3*6 = 32
SELECT inner_product('[1,2,3]'::vec32, '[4,5,6]'::vec32);

-- Cosine: identical vectors = 0
SELECT cosine_distance('[1,1,1]'::vec32, '[1,1,1]'::vec32);

-- Cosine: orthogonal vectors = 1.0
SELECT round(cosine_distance('[1,0]'::vec32, '[0,1]'::vec32)::numeric, 5);

-- Dimension mismatch
SELECT l2_distance('[1,2]'::vec32, '[1,2,3]'::vec32); -- error

-- Halfvec L2
SELECT round(l2_distance('[1,2,3]'::vec16, '[4,5,6]'::vec16)::numeric, 2);

-- Halfvec inner product
SELECT round(inner_product('[1,2,3]'::vec16, '[4,5,6]'::vec16)::numeric, 0);

-- Halfvec cosine: identical = 0
SELECT round(cosine_distance('[1,1,1]'::vec16, '[1,1,1]'::vec16)::numeric, 5);

-- Halfvec dimension mismatch
SELECT l2_distance('[1,2]'::vec16, '[1,2,3]'::vec16); -- error

-- Vector cosine: parallel vectors = 0
SELECT round(cosine_distance('[2,0]'::vec32, '[4,0]'::vec32)::numeric, 5);

-- Halfvec cosine: orthogonal = 1.0
SELECT round(cosine_distance('[1,0]'::vec16, '[0,1]'::vec16)::numeric, 5);
