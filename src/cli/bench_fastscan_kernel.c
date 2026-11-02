/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * vectorsearch bench fastscan-kernel
 *
 * Benchmark VPSHUFB fastscan kernel components:
 * - LUT construction (uint8 and uint16 hacc)
 * - Code packing
 * - Accumulate kernel (the hot inner loop)
 * - Split accumulate (a group scored in one call vs across a page break)
 * - Full distance batch (LUT + accumulate + distance formula)
 * - Posting scan (built posting lists, fastscan vs AoS)
 */

#include "vs_config.h"

#include <getopt.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench.h"
#include "cmd.h"
#include "core/memory.h"
#include "core/platform.h"
#include "core/types.h"
#include "index/posting_build.h"
#include "index/posting_page.h"
#include "index/posting_scan.h"
#include "kv.h"
#include "quant/fastscan.h"
#include "quant/rabitq.h"

#define DEFAULT_DIM	  768
#define DEFAULT_COUNT 10000
#define DEFAULT_RUNS  5
#define DEFAULT_LIST  256

typedef struct
{
	Dimension	dim;
	uint32_t	count;
	uint32_t	runs;
	uint32_t	list; /* vectors per posting list */
	const char *simd;
	const char *simd_level;
	bool		help;
	Kv		   *metrics;
} BenchConfig;

static void
add_fs_metric(
		const BenchConfig *c, const char *what, const char *unit, double v)
{
	if (c->simd_level != NULL)
		kv_arr_add_metric(c->metrics, unit, v, "%s/%s", c->simd_level, what);
	else
		kv_arr_add_metric(c->metrics, unit, v, "%s", what);
}

static void
fill_random(void *buf, size_t bytes, uint32_t seed)
{
	uint32_t rng = seed;
	uint8_t *p	 = buf;
	for (size_t i = 0; i < bytes; i++)
		p[i] = (uint8_t)(cmd_rand(&rng) & 0xFF);
}

static void
fill_random_floats(float *buf, uint32_t n, uint32_t seed)
{
	uint32_t rng = seed;
	for (uint32_t i = 0; i < n; i++)
		buf[i] = (float)((int)(cmd_rand(&rng) % 10000) - 5000) / 5000.0f;
}

/* ----------------------------------------------------------------
 * LUT construction benchmark
 * ---------------------------------------------------------------- */

static void
benchmark_lut_build(const BenchConfig *config)
{
	Dimension dim	  = config->dim;
	uint32_t  queries = 1000;

	printf("LUT construction (dim=%u, %u queries):\n", dim, queries);

	float *transformed = vs_alloc(dim * sizeof(float));

	uint32_t lut_bytes = VS_FASTSCAN_LUT_BYTES(dim);
	uint8_t *lut	   = vs_alloc_aligned(lut_bytes, 64);
	float	 scale, bias;

	uint32_t lut_hacc_bytes = VS_FASTSCAN_LUT_HACC_BYTES(dim);
	uint8_t *lut_hacc		= vs_alloc_aligned(lut_hacc_bytes, 64);

	/* Benchmark uint8 LUT */
	double best_vps = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = bench_now_ns();
		for (uint32_t q = 0; q < queries; q++)
		{
			fill_random_floats(transformed, dim, 42 + q);
			vs_fastscan_build_lut(transformed, dim, lut, &scale, &bias);
		}
		uint64_t end = bench_now_ns();
		double	 vps = cmd_ratio(
				  (double)queries, bench_ns_to_ms(end - start) / 1000.0);
		if (vps > best_vps)
			best_vps = vps;
	}
	printf("  uint8 LUT:   %7.0fK builds/s  (%u nsq, %u bytes)\n",
		   best_vps / 1000.0,
		   VS_FASTSCAN_NSQ(dim),
		   lut_bytes);
	add_fs_metric(config, "lut_uint8", "1/s", best_vps);

	/* Benchmark uint16 hacc LUT */
	best_vps = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = bench_now_ns();
		for (uint32_t q = 0; q < queries; q++)
		{
			fill_random_floats(transformed, dim, 42 + q);
			vs_fastscan_build_lut_hacc(
					transformed, dim, lut_hacc, &scale, &bias);
		}
		uint64_t end = bench_now_ns();
		double	 vps = cmd_ratio(
				  (double)queries, bench_ns_to_ms(end - start) / 1000.0);
		if (vps > best_vps)
			best_vps = vps;
	}
	printf("  uint16 LUT:  %7.0fK builds/s  (%u nsq, %u bytes)\n",
		   best_vps / 1000.0,
		   VS_FASTSCAN_NSQ(dim),
		   lut_hacc_bytes);
	add_fs_metric(config, "lut_uint16", "1/s", best_vps);

	vs_free_aligned(lut_hacc);
	vs_free_aligned(lut);
	vs_free(transformed);
}

/* ----------------------------------------------------------------
 * Accumulate kernel benchmark
 * ---------------------------------------------------------------- */

static void
benchmark_accumulate(const BenchConfig *config)
{
	Dimension dim		   = config->dim;
	uint32_t  count		   = config->count;
	uint32_t  packed_bytes = (dim + 7) / 8;
	uint32_t  ngroups = (count + VS_FASTSCAN_GROUP - 1) / VS_FASTSCAN_GROUP;
	uint32_t  group_bytes = VS_FASTSCAN_GROUP_BYTES(dim);

	printf("\nAccumulate kernel (dim=%u, %u vectors, %u groups):\n",
		   dim,
		   count,
		   ngroups);

	float *transformed = vs_alloc(dim * sizeof(float));
	fill_random_floats(transformed, dim, 42);

	uint8_t *bits = vs_alloc((size_t)count * packed_bytes);
	fill_random(bits, (size_t)count * packed_bytes, 99);

	/* Build LUT */
	uint32_t lut_bytes = VS_FASTSCAN_LUT_BYTES(dim);
	uint8_t *lut	   = vs_alloc_aligned(lut_bytes, 64);
	float	 scale, bias;
	vs_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

	/* Pack codes */
	uint32_t codes_size = vs_fastscan_codes_size(count, dim);
	uint8_t *codes		= vs_alloc_aligned(codes_size, 64);
	vs_fastscan_pack_codes(bits, count, dim, codes);

	uint16_t accum[VS_FASTSCAN_GROUP];

	/* Warmup */
	for (int w = 0; w < 3; w++)
		for (uint32_t g = 0; g < ngroups; g++)
			vs_fastscan_accumulate(codes + g * group_bytes, lut, accum, dim);

	/* Benchmark uint8 accumulate */
	double best_vps = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = bench_now_ns();
		for (uint32_t g = 0; g < ngroups; g++)
			vs_fastscan_accumulate(codes + g * group_bytes, lut, accum, dim);
		uint64_t end = bench_now_ns();
		double	 vps =
				cmd_ratio((double)count, bench_ns_to_ms(end - start) / 1000.0);
		if (vps > best_vps)
			best_vps = vps;
	}
	printf("  uint8:   %7.1fM vec/s  (%5.2f ms/%uk)\n",
		   best_vps / 1e6,
		   cmd_ratio((double)count, best_vps) * 1000.0,
		   count / 1000);
	add_fs_metric(config, "accum_uint8", "vec/s", best_vps);

	/* Benchmark uint16 hacc accumulate */
	uint32_t lut_hacc_bytes = VS_FASTSCAN_LUT_HACC_BYTES(dim);
	uint8_t *lut_hacc		= vs_alloc_aligned(lut_hacc_bytes, 64);
	vs_fastscan_build_lut_hacc(transformed, dim, lut_hacc, &scale, &bias);
	int32_t accum_hacc[VS_FASTSCAN_GROUP];

	best_vps = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = bench_now_ns();
		for (uint32_t g = 0; g < ngroups; g++)
			vs_fastscan_accumulate_hacc(
					codes + g * group_bytes, lut_hacc, accum_hacc, dim);
		uint64_t end = bench_now_ns();
		double	 vps =
				cmd_ratio((double)count, bench_ns_to_ms(end - start) / 1000.0);
		if (vps > best_vps)
			best_vps = vps;
	}
	printf("  uint16:  %7.1fM vec/s  (%5.2f ms/%uk)\n",
		   best_vps / 1e6,
		   cmd_ratio((double)count, best_vps) * 1000.0,
		   count / 1000);
	add_fs_metric(config, "accum_uint16", "vec/s", best_vps);

	vs_free_aligned(lut_hacc);
	vs_free_aligned(codes);
	vs_free_aligned(lut);
	vs_free(bits);
	vs_free(transformed);
}

/* ----------------------------------------------------------------
 * Split accumulate: a group scored in one call, or in two calls split
 * at a 64-byte chunk boundary, as when a page boundary cuts the group.
 * The difference is the per-split cost of the packed page layout.
 * ---------------------------------------------------------------- */

static void
benchmark_accumulate_split(const BenchConfig *config)
{
	Dimension dim		   = config->dim;
	uint32_t  count		   = config->count;
	uint32_t  packed_bytes = (dim + 7) / 8;
	uint32_t  ngroups = (count + VS_FASTSCAN_GROUP - 1) / VS_FASTSCAN_GROUP;
	uint32_t  group_bytes = VS_FASTSCAN_GROUP_BYTES(dim);
	uint32_t  col_bytes	  = VS_FASTSCAN_GROUP_BYTES(8);
	uint32_t  ncols		  = group_bytes / col_bytes;
	uint32_t  split		  = (ncols / 2) & ~1u; /* a chunk boundary */

	if (split == 0)
		return;

	printf("\nSplit accumulate (dim=%u, %u groups, split at column %u of "
		   "%u):\n",
		   dim,
		   ngroups,
		   split,
		   ncols);

	float *transformed = vs_alloc(dim * sizeof(float));
	fill_random_floats(transformed, dim, 42);
	uint8_t *bits = vs_alloc((size_t)count * packed_bytes);
	fill_random(bits, (size_t)count * packed_bytes, 99);
	uint8_t *codes = vs_alloc_aligned(vs_fastscan_codes_size(count, dim), 64);
	vs_fastscan_pack_codes(bits, count, dim, codes);

	VsFastscanAccumulateHaccFn hacc_fn = vs_fastscan_get_accumulate_hacc();
	int32_t					   accum[VS_FASTSCAN_GROUP];
	volatile int32_t		   sink = 0;

	for (int hacc = 0; hacc < 2; hacc++)
	{
		uint32_t lut_bytes = hacc ? VS_FASTSCAN_LUT_HACC_BYTES(dim)
								  : VS_FASTSCAN_LUT_BYTES(dim);
		uint8_t *lut	   = vs_alloc_aligned(lut_bytes, 64);
		float	 scale, bias;
		if (hacc)
			vs_fastscan_build_lut_hacc(transformed, dim, lut, &scale, &bias);
		else
			vs_fastscan_build_lut(transformed, dim, lut, &scale, &bias);

		double best[2] = {0.0, 0.0}; /* whole, split */
		for (uint32_t run = 0; run < config->runs + 1; run++)
		{
			for (int split_run = 0; split_run < 2; split_run++)
			{
				uint64_t start = bench_now_ns();
				for (uint32_t g = 0; g < ngroups; g++)
				{
					const uint8_t *gc = codes + (size_t)g * group_bytes;
					uint32_t	   n1 = split_run ? split : ncols;

					memset(accum, 0, sizeof(accum));
					if (hacc)
						vs_fastscan_accumulate_hacc_range(
								hacc_fn, gc, lut, 0, n1, accum);
					else
						vs_fastscan_accumulate_range(gc, lut, 0, n1, accum);
					if (split_run && hacc)
						vs_fastscan_accumulate_hacc_range(
								hacc_fn,
								gc + (size_t)split * col_bytes,
								lut,
								split,
								ncols - split,
								accum);
					else if (split_run)
						vs_fastscan_accumulate_range(
								gc + (size_t)split * col_bytes,
								lut,
								split,
								ncols - split,
								accum);
					sink += accum[0];
				}
				uint64_t end = bench_now_ns();
				double	 vps = cmd_ratio(
						  (double)count, bench_ns_to_ms(end - start) / 1000.0);
				/* The first round is warmup. */
				if (run > 0 && vps > best[split_run])
					best[split_run] = vps;
			}
		}

		/* What one page break costs a group. */
		double split_ns = (cmd_ratio((double)count, best[1]) -
						   cmd_ratio((double)count, best[0])) *
						  1e9 / ngroups;
		printf("  %-7s whole: %7.1fM vec/s   split: %7.1fM vec/s   "
			   "(%+.1f ns per split)\n",
			   hacc ? "uint16" : "uint8",
			   best[0] / 1e6,
			   best[1] / 1e6,
			   split_ns);
		add_fs_metric(
				config,
				hacc ? "accum_uint16_whole" : "accum_uint8_whole",
				"vec/s",
				best[0]);
		add_fs_metric(
				config,
				hacc ? "accum_uint16_split" : "accum_uint8_split",
				"vec/s",
				best[1]);
		add_fs_metric(
				config,
				hacc ? "split_cost_uint16" : "split_cost_uint8",
				"ns",
				split_ns);
		vs_free_aligned(lut);
	}

	(void)sink;
	vs_free_aligned(codes);
	vs_free(bits);
	vs_free(transformed);
}

/* ----------------------------------------------------------------
 * Posting scan: posting lists written by the real builder into memory
 * pages, scanned by the real scan path. Measures the page layout as a
 * whole: pages per list, bytes per vector, and scan throughput, for
 * fastscan and AoS lists of the same vectors.
 * ---------------------------------------------------------------- */

typedef struct BenchPageStorage
{
	VsStorage base;
	char	 *pages;
	uint32_t  npages;
	uint32_t  cap;
} BenchPageStorage;

static Page
bench_page_at(VsStorage *self, BlockNumber blkno)
{
	return ((BenchPageStorage *)self)->pages + (size_t)blkno * BLCKSZ;
}

static void
bench_page_noop(VsStorage *self, BlockNumber blkno)
{
	(void)self;
	(void)blkno;
}

static Page
bench_page_new(VsStorage *self, BlockNumber *blkno_out)
{
	BenchPageStorage *s = (BenchPageStorage *)self;
	if (s->npages >= s->cap)
	{
		fprintf(stderr, "bench: posting page estimate too small\n");
		exit(1);
	}
	*blkno_out = s->npages++;
	return bench_page_at(self, *blkno_out);
}

static const VsStorageOps bench_page_ops = {
		.read_page	  = bench_page_at,
		.release_page = bench_page_noop,
		.write_page	  = bench_page_at,
		.new_page	  = bench_page_new,
		.commit_page  = bench_page_noop,
};

/* Write nlists lists of `list` random pre-encoded vectors each. */
static void
build_bench_lists(
		BenchPageStorage *st,
		Dimension		  dim,
		uint32_t		  nlists,
		uint32_t		  list,
		bool			  fastscan,
		const float		 *pt_centroid,
		BlockNumber		 *heads)
{
	uint32_t packed_bytes = VS_RABITQ_BYTES(dim);
	uint8_t *bits		  = vs_alloc(packed_bytes);
	uint32_t rng		  = 1234;

	for (uint32_t c = 0; c < nlists; c++)
	{
		PrismPostingBuilder b;
		prism_posting_builder_init_fmt(
				&b, &st->base, NULL, dim, c, NULL, pt_centroid, fastscan);
		for (uint32_t i = 0; i < list; i++)
		{
			ItemPointerData tid;
			ItemPointerSet(&tid, c * list + i + 1, 1);
			fill_random(bits, packed_bytes, cmd_rand(&rng));
			float f_add		= (float)(cmd_rand(&rng) % 1000) / 500.0f;
			float f_rescale = 0.5f + (float)(cmd_rand(&rng) % 1000) / 1000.0f;
			prism_posting_builder_add_encoded(
					&b, tid, f_add, f_rescale, 0.01f, bits);
		}
		heads[c] = prism_posting_builder_finish(&b);
		prism_posting_builder_cleanup(&b);
	}
	vs_free(bits);
}

static void
benchmark_posting_scan(const BenchConfig *config)
{
	Dimension dim	 = config->dim;
	uint32_t  list	 = config->list;
	uint32_t  nlists = config->count / list > 0 ? config->count / list : 1;
	uint32_t  total	 = nlists * list;

	/* Generous for either layout: a page holds at least half its bytes of
	 * entries, plus the head and one page of slack per list. */
	uint32_t vec_bytes = PRISM_POSTING_ENTRY_SIZE(dim);
	uint32_t fs_bytes  = prism_fastscan_group_section_bytes(dim) /
								VS_FASTSCAN_GROUP +
						1;
	if (fs_bytes > vec_bytes)
		vec_bytes = fs_bytes;
	uint64_t per_list = 3 + ((uint64_t)list * vec_bytes + 32 * fs_bytes) /
									(prism_posting_page_usable() / 2);

	printf("\nPosting scan (dim=%u, %u lists of %u vectors, k=10):\n",
		   dim,
		   nlists,
		   list);

	RaBitQParams *params = vs_rabitq_create(dim, 42);
	float		 *zero	 = vs_alloc0(dim * sizeof(float));
	float		 *query	 = vs_alloc(dim * sizeof(float));
	fill_random_floats(query, dim, 7);
	Vec32Ref		  qref	 = {.data = query, .dim = dim};
	Vec32Ref		  cref	 = {.data = zero, .dim = dim};
	RaBitQQueryState *qstate = vs_rabitq_prepare_query(params, qref, cref);
	BlockNumber		 *heads	 = vs_alloc(nlists * sizeof(BlockNumber));

	for (int fs = 1; fs >= 0; fs--)
	{
		BenchPageStorage st = {
				.base = {.ops = &bench_page_ops},
				.cap  = (uint32_t)(nlists * per_list),
		};
		st.pages = vs_alloc_aligned((size_t)st.cap * BLCKSZ, 4096);
		build_bench_lists(&st, dim, nlists, list, fs != 0, zero, heads);

		PrismPostingScan scan;
		prism_posting_scan_init(
				&scan,
				&st.base,
				NULL,
				params,
				dim,
				prism_posting_max_entries(dim));
		if (fs)
			prism_posting_scan_enable_fastscan(&scan, 16);

		VsTopK topk;
		vs_topk_init(&topk, 10);
		double best = 0.0;
		for (uint32_t run = 0; run < config->runs + 1; run++)
		{
			vs_topk_reset(&topk);
			uint64_t start = bench_now_ns();
			for (uint32_t c = 0; c < nlists; c++)
			{
				prism_posting_scan_begin_cluster(&scan, qstate, heads[c]);
				if (fs)
					prism_posting_scan_cluster_fastscan(&scan, &topk);
				else
					prism_posting_scan_cluster(&scan, &topk);
				prism_posting_scan_end_cluster(&scan);
			}
			uint64_t end = bench_now_ns();
			double	 vps = cmd_ratio(
					  (double)total, bench_ns_to_ms(end - start) / 1000.0);
			/* The first round is warmup. */
			if (run > 0 && vps > best)
				best = vps;
		}

		double pages = cmd_ratio((double)st.npages, (double)nlists);
		printf("  %-9s %6.2f pages/list  %6.1f B/vector  %7.1fM vec/s  "
			   "(%5.2f us/list)\n",
			   fs ? "fastscan:" : "aos:",
			   pages,
			   cmd_ratio((double)st.npages * BLCKSZ, (double)total),
			   best / 1e6,
			   cmd_ratio((double)list, best) * 1e6);
		add_fs_metric(
				config, fs ? "scan_fastscan" : "scan_aos", "vec/s", best);
		add_fs_metric(
				config,
				fs ? "pages_per_list_fastscan" : "pages_per_list_aos",
				"pages",
				pages);

		vs_topk_cleanup(&topk);
		prism_posting_scan_cleanup(&scan);
		vs_free_aligned(st.pages);
	}

	vs_free(heads);
	vs_rabitq_free_query(qstate);
	vs_free(query);
	vs_free(zero);
	vs_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Distance throughput: fastscan vs rabitq asymmetric
 *
 * Uses properly encoded RaBitQ vectors so both kernels operate
 * on the same data. Output matches bench rabitq-kernel format.
 * ---------------------------------------------------------------- */

static void
benchmark_distance_throughput(const BenchConfig *config)
{
	Dimension dim		   = config->dim;
	uint32_t  count		   = config->count;
	uint32_t  packed_bytes = (dim + 7) / 8;
	uint32_t  ngroups = (count + VS_FASTSCAN_GROUP - 1) / VS_FASTSCAN_GROUP;

	printf("\nDistance throughput (dim=%u, count=%u):\n", dim, count);

	/* Encode vectors with RaBitQ */
	RaBitQParams *params   = vs_rabitq_create(dim, 42);
	float		 *centroid = vs_alloc0(dim * sizeof(float));
	Vec32Ref	  cent_ref = {.data = centroid, .dim = dim};

	float *vectors = vs_alloc((size_t)count * dim * sizeof(float));
	for (uint32_t i = 0; i < count; i++)
		fill_random_floats(vectors + (size_t)i * dim, dim, i);

	float	*f_add	   = vs_alloc(count * sizeof(float));
	float	*f_rescale = vs_alloc(count * sizeof(float));
	uint8_t *bits	   = vs_alloc((size_t)count * packed_bytes);

	vs_rabitq_encode_batch(
			params,
			vectors,
			VS_VEC_F32,
			cent_ref,
			f_add,
			f_rescale,
			bits,
			(uint16_t)(count > 65535 ? 65535 : count));

	float *query = vs_alloc(dim * sizeof(float));
	fill_random_floats(query, dim, 99999);
	Vec32Ref query_ref = {.data = query, .dim = dim};

	RaBitQQueryState *state =
			vs_rabitq_prepare_query(params, query_ref, cent_ref);

	float *distances = vs_alloc(count * sizeof(float));

	/* --- RaBitQ per-vector asymmetric --- */
	size_t	 data_size = VS_RABITQ_DATA_SIZE(dim);
	uint8_t *data_buf  = vs_alloc(data_size);

	RaBitQData *temp_data = (RaBitQData *)data_buf;
	for (int w = 0; w < 3; w++)
		for (uint32_t i = 0; i < count; i++)
		{
			temp_data->f_add	 = f_add[i];
			temp_data->f_rescale = f_rescale[i];
			memcpy(temp_data->bits,
				   bits + (size_t)i * packed_bytes,
				   packed_bytes);
			distances[i] = vs_rabitq_distance(state, temp_data, dim);
		}

	double rabitq_best = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = bench_now_ns();
		for (uint32_t i = 0; i < count; i++)
		{
			temp_data->f_add	 = f_add[i];
			temp_data->f_rescale = f_rescale[i];
			memcpy(temp_data->bits,
				   bits + (size_t)i * packed_bytes,
				   packed_bytes);
			distances[i] = vs_rabitq_distance(state, temp_data, dim);
		}
		uint64_t end = bench_now_ns();
		double	 vps =
				cmd_ratio((double)count, bench_ns_to_ms(end - start) / 1000.0);
		if (vps > rabitq_best)
			rabitq_best = vps;
	}

	/* --- Fastscan uint8 --- */
	uint32_t codes_size = vs_fastscan_codes_size(count, dim);
	uint8_t *codes		= vs_alloc_aligned(codes_size, 64);
	vs_fastscan_pack_codes(bits, count, dim, codes);

	uint8_t	 *lut_buf = vs_alloc_aligned(VS_FASTSCAN_LUT_BYTES(dim), 64);
	uint16_t *accum_buf =
			vs_alloc_aligned(VS_FASTSCAN_GROUP * sizeof(uint16_t), 64);

	for (int w = 0; w < 3; w++)
		vs_fastscan_distance_batch(
				state,
				f_add,
				f_rescale,
				codes,
				ngroups,
				count,
				dim,
				distances,
				lut_buf,
				accum_buf);

	double fs8_best = 0.0;
	for (uint32_t run = 0; run < config->runs; run++)
	{
		uint64_t start = bench_now_ns();
		vs_fastscan_distance_batch(
				state,
				f_add,
				f_rescale,
				codes,
				ngroups,
				count,
				dim,
				distances,
				lut_buf,
				accum_buf);
		uint64_t end = bench_now_ns();
		double	 vps =
				cmd_ratio((double)count, bench_ns_to_ms(end - start) / 1000.0);
		if (vps > fs8_best)
			fs8_best = vps;
	}

	printf("  asymmetric: %7.1fK vec/s  (%5.2f ms/%uk)\n",
		   rabitq_best / 1000.0,
		   cmd_ratio((double)count, rabitq_best / 1000.0),
		   count / 1000);
	printf("  fastscan:   %7.1fK vec/s  (%5.2f ms/%uk)\n",
		   fs8_best / 1000.0,
		   cmd_ratio((double)count, fs8_best / 1000.0),
		   count / 1000);
	printf("  speedup:    %5.1fx\n", cmd_ratio(fs8_best, rabitq_best));
	add_fs_metric(config, "asymmetric", "vec/s", rabitq_best);
	add_fs_metric(config, "fastscan", "vec/s", fs8_best);

	vs_free_aligned(accum_buf);
	vs_free_aligned(lut_buf);
	vs_free_aligned(codes);
	vs_free(data_buf);
	vs_free(distances);
	vs_rabitq_free_query(state);
	vs_free(query);
	vs_free(bits);
	vs_free(f_rescale);
	vs_free(f_add);
	vs_free(vectors);
	vs_free(centroid);
	vs_rabitq_destroy(params);
}

/* ----------------------------------------------------------------
 * Entry point
 * ---------------------------------------------------------------- */

static void
print_usage(CmdContext *ctx)
{
	CMD_USAGE_HEADER(ctx, "bench fastscan-kernel");
	printf("Benchmark VPSHUFB fastscan kernel components.\n\n");
	printf("Options:\n");
	printf("  --dim <int>     Vector dimension (default: %d)\n", DEFAULT_DIM);
	printf("  --count <int>   Number of vectors (default: %d)\n",
		   DEFAULT_COUNT);
	printf("  --runs <int>    Benchmark iterations (default: %d)\n",
		   DEFAULT_RUNS);
	printf("  --list <int>    Vectors per posting list in the posting scan "
		   "(default: %d)\n",
		   DEFAULT_LIST);
	printf("  --simd <str>    SIMD level: avx512, avx2, scalar "
		   "(default: all)\n");
	printf("  --help          Show this help message\n");
	CMD_USAGE_JSON();
	printf("\n");
	printf("Examples:\n");
	CMD_USAGE_EXAMPLE(
			ctx, "bench fastscan-kernel", "--dim 768 --count 100000");
	CMD_USAGE_EXAMPLE(ctx, "bench fastscan-kernel", "--simd avx2 --runs 10");
	CMD_USAGE_EXAMPLE(ctx, "bench fastscan-kernel", "--dim 1536 --list 512");
}

int
cmd_bench_fastscan_kernel(CmdContext *ctx)
{
	BenchConfig config = {
			.dim   = DEFAULT_DIM,
			.count = DEFAULT_COUNT,
			.runs  = DEFAULT_RUNS,
			.list  = DEFAULT_LIST,
			.help  = false,
	};

	static struct option long_options[] = {
			{"dim", required_argument, 0, 'd'},
			{"count", required_argument, 0, 'c'},
			{"runs", required_argument, 0, 'r'},
			{"list", required_argument, 0, 'l'},
			{"simd", required_argument, 0, 's'},
			{"help", no_argument, 0, 'h'},
			{0, 0, 0, 0},
	};

	optind = 1;
	int opt;
	while ((opt = getopt_long(
					ctx->argc,
					ctx->argv,
					"d:c:r:l:s:h",
					long_options,
					NULL)) != -1)
	{
		switch (opt)
		{
		case 'd':
			config.dim = (Dimension)atoi(optarg);
			break;
		case 'c':
			config.count = (uint32_t)atoi(optarg);
			break;
		case 'r':
			config.runs = (uint32_t)atoi(optarg);
			break;
		case 'l':
			config.list = (uint32_t)atoi(optarg);
			break;
		case 's':
			config.simd = optarg;
			break;
		case 'h':
			config.help = true;
			break;
		default:
			print_usage(ctx);
			return 1;
		}
	}

	if (config.help)
	{
		print_usage(ctx);
		return 0;
	}
	if (config.list == 0)
	{
		fprintf(stderr, "bench fastscan-kernel: --list must be positive\n");
		return 1;
	}

	printf("Fastscan kernel benchmark\n");
	printf("=========================\n\n");

	KvBuild *kb	  = NULL;
	Kv		*root = NULL;
	if (ctx->json_path != NULL)
	{
		kb	 = kv_build_open();
		root = kv_obj();
		kv_obj_put(root, "bench", kv_str("fastscan-kernel"));
		kv_obj_put(root, "dim", kv_int(config.dim));
		kv_obj_put(root, "count", kv_int(config.count));
		kv_obj_put(root, "runs", kv_int(config.runs));
		kv_obj_put(root, "list", kv_int(config.list));
		config.metrics = kv_arr();
		kv_obj_put(root, "metrics", config.metrics);
	}

	SimdCapability caps = vs_detect_simd();

#ifdef VS_SIMD_FULL
	/* Benchmark each available SIMD level (widest first) */
	typedef struct
	{
		uint32_t	cap;
		const char *name;
		bool		run_distance;
	} SimdLevel;

#if defined(__x86_64__) || defined(_M_X64)
	SimdLevel levels[] = {
			{VS_SIMD_AVX512_BW, "avx512", true},
			{SIMD_AVX2, "avx2", false},
			{0, "scalar", false},
	};
#elif defined(__aarch64__) || defined(_M_ARM64)
	SimdLevel levels[] = {
			{SIMD_NEON, "neon", true},
			{0, "scalar", false},
	};
#else
	SimdLevel levels[] = {
			{0, "scalar", true},
	};
#endif
	uint32_t nlevels = sizeof(levels) / sizeof(levels[0]);

	for (uint32_t l = 0; l < nlevels; l++)
	{
		if (levels[l].cap != 0 && !(caps & levels[l].cap))
			continue;
		if (config.simd != NULL && strcmp(config.simd, levels[l].name) != 0)
			continue;

		vs_simd_set_override(levels[l].cap ? levels[l].cap : SIMD_NONE);
		vs_fastscan_reset_simd();
		vs_fastscan_init_simd();

		printf("--- %s ---\n\n", levels[l].name);
		config.simd_level = levels[l].name;
		benchmark_lut_build(&config);
		benchmark_accumulate(&config);
		benchmark_accumulate_split(&config);
		if (levels[l].run_distance)
		{
			benchmark_distance_throughput(&config);
			benchmark_posting_scan(&config);
		}
		printf("\n");
	}

	vs_simd_set_override(0xFFFFFFFF);
	vs_fastscan_reset_simd();
	vs_fastscan_init_simd();
#else
	(void)caps;
	vs_fastscan_init_simd();
	benchmark_lut_build(&config);
	benchmark_accumulate(&config);
	benchmark_accumulate_split(&config);
	benchmark_distance_throughput(&config);
	benchmark_posting_scan(&config);
	printf("\n");
#endif

	int rc = cmd_kv_print(root, ctx->json_path);
	kv_build_close(kb);
	return rc;
}
