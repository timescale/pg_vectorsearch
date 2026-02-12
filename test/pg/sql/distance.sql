-- Distance functions for both vector and halfvec

-- L2 distance: sqrt(9+9+9) = sqrt(27) ≈ 5.196
SELECT round(l2_distance('[1,2,3]'::vector, '[4,5,6]'::vector)::numeric, 3);

-- L2 known: sqrt(9+16) = 5.0
SELECT l2_distance('[0,0]'::vector, '[3,4]'::vector);

-- Inner product: 1*4 + 2*5 + 3*6 = 32
SELECT inner_product('[1,2,3]'::vector, '[4,5,6]'::vector);

-- Cosine: identical vectors = 0
SELECT cosine_distance('[1,1,1]'::vector, '[1,1,1]'::vector);

-- Cosine: orthogonal vectors = 1.0
SELECT round(cosine_distance('[1,0]'::vector, '[0,1]'::vector)::numeric, 5);

-- Dimension mismatch
SELECT l2_distance('[1,2]'::vector, '[1,2,3]'::vector); -- error

-- Halfvec L2
SELECT round(l2_distance('[1,2,3]'::halfvec, '[4,5,6]'::halfvec)::numeric, 2);

-- Halfvec inner product
SELECT round(inner_product('[1,2,3]'::halfvec, '[4,5,6]'::halfvec)::numeric, 0);

-- Halfvec cosine: identical = 0
SELECT round(cosine_distance('[1,1,1]'::halfvec, '[1,1,1]'::halfvec)::numeric, 5);

-- Halfvec dimension mismatch
SELECT l2_distance('[1,2]'::halfvec, '[1,2,3]'::halfvec); -- error

-- Vector cosine: parallel vectors = 0
SELECT round(cosine_distance('[2,0]'::vector, '[4,0]'::vector)::numeric, 5);

-- Halfvec cosine: orthogonal = 1.0
SELECT round(cosine_distance('[1,0]'::halfvec, '[0,1]'::halfvec)::numeric, 5);
