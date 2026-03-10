/*
 * mkt verify rabitq
 *
 * Compare meerkat's RaBitQ encoding and distance computation against FAISS
 * using real datasets in HDF5 format (e.g. ann-benchmarks glove-100-angular).
 *
 * Two verification phases:
 *   1. Encoding: bit-for-bit comparison of quantized codes
 *   2. Distance: compare estimated distances against true L2
 *
 * Requires both FAISS and HDF5 dependencies (-Dfaiss=true -Dhdf5=true).
 */

#include "mkt_config.h"

#if defined(MKT_HAVE_FAISS) && defined(MKT_HAVE_HDF5)

#include <RaBitQuantizer_c.h>
#include <error_c.h>
#include <getopt.h>
#include <hdf5.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cmd.h"
#include "core/memory.h"
#include "mkt_types.h"
#include "quant/matrix.h"
#include "quant/rabitq.h"

/* Defaults */
#define DEFAULT_SEED 42

/* Distance mismatch threshold for non-verbose output */
#define DIST_MISMATCH_REL_THRESHOLD 0.01f /* 1% relative difference */

/*
 * Print a progress bar on stderr (overwriting current line).
 * Only emits when the percentage changes to avoid excessive I/O.
 * Width: [========>               ] 42%  (label)
 */
static void
print_progress(
		const char *label,
		uint64_t	current,
		uint64_t	total,
		uint32_t   *last_pct)
{
	if (total == 0)
		return;
	uint32_t pct = (uint32_t)(current * 100 / total);
	if (last_pct && pct == *last_pct && current != total)
		return;
	if (last_pct)
		*last_pct = pct;

	int bar_width = 30;
	int filled	  = (int)(pct * bar_width / 100);

	fprintf(stderr, "\r  %s [", label);
	for (int i = 0; i < bar_width; i++)
	{
		if (i < filled)
			fputc('=', stderr);
		else if (i == filled)
			fputc('>', stderr);
		else
			fputc(' ', stderr);
	}
	fprintf(stderr, "] %3u%%", pct);

	if (current == total)
		fputc('\n', stderr);
}

/* Clear a progress line (when interleaving mismatch output) */
static void
clear_progress(void)
{
	fprintf(stderr, "\r%*s\r", 60, "");
}

/*
 * Load a 2D float dataset from an HDF5 file.
 * Returns heap-allocated array (caller frees). Sets *rows and *cols.
 */
static float *
load_hdf5_float(
		const char *path, const char *dataset, hsize_t *rows, hsize_t *cols)
{
	hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0)
	{
		fprintf(stderr, "Error: cannot open HDF5 file '%s'\n", path);
		return NULL;
	}

	hid_t dset = H5Dopen2(file, dataset, H5P_DEFAULT);
	if (dset < 0)
	{
		fprintf(stderr,
				"Error: dataset '%s' not found in '%s'\n",
				dataset,
				path);
		H5Fclose(file);
		return NULL;
	}

	hid_t	space = H5Dget_space(dset);
	hsize_t dims[2];
	int		ndims = H5Sget_simple_extent_ndims(space);
	if (ndims != 2)
	{
		fprintf(stderr,
				"Error: dataset '%s' has %d dims, expected 2\n",
				dataset,
				ndims);
		H5Sclose(space);
		H5Dclose(dset);
		H5Fclose(file);
		return NULL;
	}

	H5Sget_simple_extent_dims(space, dims, NULL);
	*rows = dims[0];
	*cols = dims[1];

	float *data = malloc(*rows * *cols * sizeof(float));
	if (data == NULL)
	{
		fprintf(stderr,
				"Error: cannot allocate %.1f MB for '%s'\n",
				(double)(*rows * *cols * sizeof(float)) / (1024 * 1024),
				dataset);
		H5Sclose(space);
		H5Dclose(dset);
		H5Fclose(file);
		return NULL;
	}

	H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

	H5Sclose(space);
	H5Dclose(dset);
	H5Fclose(file);
	return data;
}

/*
 * Compute L2 squared distance between two vectors.
 */
static float
l2_distance_sq(const float *a, const float *b, Dimension dim)
{
	float sum = 0.0f;
	for (Dimension i = 0; i < dim; i++)
	{
		float diff = a[i] - b[i];
		sum += diff * diff;
	}
	return sum;
}

/*
 * Compute centroid (mean) of a set of vectors.
 */
static void
compute_centroid(
		const float *vectors, uint32_t count, Dimension dim, float *centroid)
{
	memset(centroid, 0, dim * sizeof(float));
	for (uint32_t i = 0; i < count; i++)
	{
		const float *v = vectors + (size_t)i * dim;
		for (Dimension j = 0; j < dim; j++)
			centroid[j] += v[j];
	}
	float inv = 1.0f / count;
	for (Dimension j = 0; j < dim; j++)
		centroid[j] *= inv;
}

/* Removed: TopKEntry, topk_select, compute_recall, verify_recall
 * -- linear scan recall is too slow for large datasets and the ground
 * truth from HDF5 assumes the full dataset is loaded. */

/*
 * Phase 1: Encoding verification
 *
 * Encode each vector with both meerkat and FAISS, compare bit patterns.
 * Default: only print mismatches with progress bar.
 * Verbose: print every vector result.
 */
static void
verify_encoding(
		RaBitQParams *params,
		const float	 *vectors,
		const float	 *centroid,
		uint32_t	  count,
		Dimension	  dim,
		bool		  verbose,
		bool		  use_progress,
		const char	 *dump_dir)
{
	printf("=== Phase 1: Encoding Verification ===\n\n");

	size_t nbytes = MKT_RABITQ_BYTES(dim);

	/* Pre-transform all vectors for FAISS: P^T * (v - centroid) */
	float *transformed = malloc((size_t)count * dim * sizeof(float));
	float *residual	   = malloc(dim * sizeof(float));
	if (transformed == NULL || residual == NULL)
	{
		fprintf(stderr, "Error: allocation failed\n");
		free(transformed);
		free(residual);
		return;
	}

	for (uint32_t i = 0; i < count; i++)
	{
		const float *v = vectors + (size_t)i * dim;
		for (Dimension j = 0; j < dim; j++)
			residual[j] = v[j] - centroid[j];
		mkt_matrix_transpose_vector_mul(
				params->P, residual, transformed + (size_t)i * dim, dim);
	}
	free(residual);

	/* Create FAISS quantizer */
	FaissRaBitQuantizer *faiss_rq = NULL;
	if (faiss_RaBitQuantizer_new_with(&faiss_rq, dim, METRIC_L2, 1) != 0)
	{
		fprintf(stderr,
				"Error: FAISS quantizer creation failed: %s\n",
				faiss_get_last_error());
		free(transformed);
		return;
	}

	/* FAISS encoding */
	size_t	 faiss_code_size = faiss_RaBitQuantizer_code_size(faiss_rq);
	uint8_t *faiss_codes	 = malloc(count * faiss_code_size);
	if (faiss_codes == NULL)
	{
		fprintf(stderr, "Error: allocation failed for FAISS codes\n");
		faiss_RaBitQuantizer_free(faiss_rq);
		free(transformed);
		return;
	}
	faiss_RaBitQuantizer_compute_codes(
			faiss_rq, transformed, faiss_codes, count);

	VectorRef cent_ref = {.data = centroid, .dim = dim};

	/* Allocate buffer for mkt codes (for dump and comparison) */
	uint8_t *mkt_codes = dump_dir ? malloc((size_t)count * nbytes) : NULL;
	if (dump_dir && mkt_codes == NULL)
	{
		fprintf(stderr, "Error: allocation failed for mkt codes\n");
		faiss_RaBitQuantizer_free(faiss_rq);
		free(faiss_codes);
		free(transformed);
		return;
	}

	uint32_t matches  = 0;
	uint32_t last_pct = UINT32_MAX;
	for (uint32_t i = 0; i < count; i++)
	{
		if (use_progress && !verbose)
			print_progress("Encoding", i, count, &last_pct);

		VectorRef	vec_ref = {.data = vectors + (size_t)i * dim, .dim = dim};
		RaBitQData *mkt_enc = mkt_rabitq_encode(params, vec_ref, cent_ref);

		if (mkt_codes)
			memcpy(mkt_codes + (size_t)i * nbytes, mkt_enc->bits, nbytes);

		if (memcmp(mkt_enc->bits, faiss_codes + i * faiss_code_size, nbytes) ==
			0)
		{
			if (verbose)
				printf("ENC  v=%-8u MATCH\n", i);
			matches++;
		}
		else
		{
			/* Find first mismatching byte */
			for (size_t b = 0; b < nbytes; b++)
			{
				if (mkt_enc->bits[b] != faiss_codes[i * faiss_code_size + b])
				{
					if (use_progress && !verbose)
						clear_progress();
					printf("ENC  v=%-8u MISMATCH  byte=%zu "
						   "mkt=0x%02X faiss=0x%02X\n",
						   i,
						   b,
						   mkt_enc->bits[b],
						   faiss_codes[i * faiss_code_size + b]);
					break;
				}
			}
		}
		mkt_free(mkt_enc);
	}

	if (use_progress && !verbose)
		print_progress("Encoding", count, count, &last_pct);

	printf("\nENCODING SUMMARY: %u/%u match (%.2f%%)\n\n",
		   matches,
		   count,
		   100.0f * matches / count);

	/* Dump encoded bits to files if requested */
	if (dump_dir)
	{
		char path[512];

		snprintf(path, sizeof(path), "%s/mkt_codes.bin", dump_dir);
		FILE *f = fopen(path, "wb");
		if (f)
		{
			fwrite(mkt_codes, nbytes, count, f);
			fclose(f);
		}
		else
			fprintf(stderr, "Error: cannot write %s\n", path);

		snprintf(path, sizeof(path), "%s/faiss_codes.bin", dump_dir);
		f = fopen(path, "wb");
		if (f)
		{
			for (uint32_t i = 0; i < count; i++)
				fwrite(faiss_codes + i * faiss_code_size, 1, nbytes, f);
			fclose(f);
		}
		else
			fprintf(stderr, "Error: cannot write %s\n", path);

		printf("Wrote mkt_codes.bin and faiss_codes.bin to %s "
			   "(%u vectors, %zu bytes each)\n\n",
			   dump_dir,
			   count,
			   nbytes);
	}

	faiss_RaBitQuantizer_free(faiss_rq);
	free(faiss_codes);
	free(mkt_codes);
	free(transformed);
}

/*
 * Phase 2: Distance verification
 *
 * For each query, batch-compute meerkat and FAISS estimated distances,
 * then compare. True L2 is only computed for a subsample (summary stats)
 * and on-demand for mismatches (verbose/mismatch output).
 *
 * Performance: the inner loop uses mkt_rabitq_distance_batch() (SIMD)
 * instead of per-vector calls, and avoids the O(nqueries*ntrain*dim)
 * true L2 computation by sampling.
 */

/* Number of random pairs to sample for true-L2 error statistics */
#define TRUE_L2_SAMPLE_SIZE 10000

static void
verify_distances(
		RaBitQParams *params,
		const float	 *train,
		uint32_t	  ntrain,
		const float	 *queries,
		uint32_t	  nqueries,
		const float	 *centroid,
		Dimension	  dim,
		bool		  verbose,
		bool		  use_progress)
{
	printf("=== Phase 2: Distance Verification ===\n\n");

	/* Pre-transform all train vectors for FAISS */
	float *train_transformed = malloc((size_t)ntrain * dim * sizeof(float));
	float *residual			 = malloc(dim * sizeof(float));
	if (train_transformed == NULL || residual == NULL)
	{
		fprintf(stderr, "Error: allocation failed\n");
		free(train_transformed);
		free(residual);
		return;
	}

	for (uint32_t i = 0; i < ntrain; i++)
	{
		const float *v = train + (size_t)i * dim;
		for (Dimension j = 0; j < dim; j++)
			residual[j] = v[j] - centroid[j];
		mkt_matrix_transpose_vector_mul(
				params->P, residual, train_transformed + (size_t)i * dim, dim);
	}
	free(residual);

	/* Create FAISS quantizer and encode */
	FaissRaBitQuantizer *faiss_rq = NULL;
	faiss_RaBitQuantizer_new_with(&faiss_rq, dim, METRIC_L2, 1);

	size_t	 faiss_code_size = faiss_RaBitQuantizer_code_size(faiss_rq);
	uint8_t *faiss_codes	 = malloc(ntrain * faiss_code_size);
	faiss_RaBitQuantizer_compute_codes(
			faiss_rq, train_transformed, faiss_codes, ntrain);

	/* Encode all train vectors with meerkat into SoA arrays for
	 * batch distance computation */
	VectorRef cent_ref		= {.data = centroid, .dim = dim};
	uint32_t  pbytes		= MKT_RABITQ_BYTES(dim);
	float	 *enc_f_add		= malloc(ntrain * sizeof(float));
	float	 *enc_f_rescale = malloc(ntrain * sizeof(float));
	uint8_t	 *enc_bits		= malloc((size_t)ntrain * pbytes);
	if (!enc_f_add || !enc_f_rescale || !enc_bits)
	{
		fprintf(stderr, "Error: allocation failed for mkt codes\n");
		faiss_RaBitQuantizer_free(faiss_rq);
		free(faiss_codes);
		free(train_transformed);
		free(enc_f_add);
		free(enc_f_rescale);
		free(enc_bits);
		return;
	}

	mkt_rabitq_encode_batch(
			params,
			train,
			MKT_VEC_F32,
			cent_ref,
			enc_f_add,
			enc_f_rescale,
			enc_bits,
			(uint16_t)(ntrain < UINT16_MAX ? ntrain : UINT16_MAX));

	/* Handle ntrain > UINT16_MAX by encoding remaining in chunks */
	for (uint32_t offset = UINT16_MAX; offset < ntrain;)
	{
		uint32_t  remaining	 = ntrain - offset;
		uint16_t  chunk		 = remaining < UINT16_MAX ? (uint16_t)remaining
													  : UINT16_MAX;
		VectorRef chunk_cent = {.data = centroid, .dim = dim};
		mkt_rabitq_encode_batch(
				params,
				train + (size_t)offset * dim,
				MKT_VEC_F32,
				chunk_cent,
				enc_f_add + offset,
				enc_f_rescale + offset,
				enc_bits + (size_t)offset * pbytes,
				chunk);
		offset += chunk;
	}

	/* Pre-allocate per-query distance buffers */
	float *mkt_dists   = malloc(ntrain * sizeof(float));
	float *faiss_dists = malloc(ntrain * sizeof(float));
	float *q_buf	   = malloc(dim * sizeof(float));
	float *qt_buf	   = malloc(dim * sizeof(float));
	if (!mkt_dists || !faiss_dists || !q_buf || !qt_buf)
	{
		fprintf(stderr, "Error: allocation failed for buffers\n");
		goto dist_cleanup;
	}

	/* Statistics accumulators */
	float	 mkt_faiss_max_abs = 0.0f, mkt_faiss_sum_abs = 0.0f;
	uint64_t total_pairs	 = 0;
	uint32_t dist_mismatches = 0;
	uint32_t last_pct		 = UINT32_MAX;

	/* True L2 error stats via inline sampling.
	 * Sample vectors per query using a stride so the sample spans
	 * all queries and naturally includes mismatched vectors. */
	float	 mkt_true_max_rel = 0.0f, mkt_true_sum_rel = 0.0f;
	float	 faiss_true_max_rel = 0.0f, faiss_true_sum_rel = 0.0f;
	uint32_t n_sampled		   = 0;
	uint32_t samples_per_query = TRUE_L2_SAMPLE_SIZE /
								 (nqueries ? nqueries : 1);
	if (samples_per_query < 1)
		samples_per_query = 1;
	if (samples_per_query > ntrain)
		samples_per_query = ntrain;
	uint32_t sample_stride = ntrain / samples_per_query;
	if (sample_stride < 1)
		sample_stride = 1;

	for (uint32_t q = 0; q < nqueries; q++)
	{
		const float *query	   = queries + (size_t)q * dim;
		VectorRef	 query_ref = {.data = query, .dim = dim};

		/* Batch meerkat distances (SIMD) */
		RaBitQQueryState *mkt_state =
				mkt_rabitq_prepare_query(params, query_ref, cent_ref);
		mkt_rabitq_distance_batch(
				mkt_state,
				enc_f_add,
				enc_f_rescale,
				enc_bits,
				ntrain,
				dim,
				mkt_dists);
		mkt_rabitq_free_query(mkt_state);

		/* FAISS distances */
		for (Dimension j = 0; j < dim; j++)
			q_buf[j] = query[j] - centroid[j];
		mkt_matrix_transpose_vector_mul(params->P, q_buf, qt_buf, dim);

		FaissRaBitQDistanceComputer *faiss_dc = NULL;
		faiss_RaBitQuantizer_get_distance_computer(
				faiss_rq, &faiss_dc, 0, NULL, 0);
		faiss_RaBitQDistanceComputer_set_query(faiss_dc, qt_buf);

		for (uint32_t v = 0; v < ntrain; v++)
		{
			faiss_RaBitQDistanceComputer_distance_to_code(
					faiss_dc,
					faiss_codes + v * faiss_code_size,
					&faiss_dists[v]);
		}
		faiss_RaBitQDistanceComputer_free(faiss_dc);

		/* Compare mkt vs faiss distances and sample true L2 */
		for (uint32_t v = 0; v < ntrain; v++)
		{
			float abs_diff	  = fabsf(mkt_dists[v] - faiss_dists[v]);
			float max_est	  = mkt_dists[v] > faiss_dists[v] ? mkt_dists[v]
															  : faiss_dists[v];
			float rel_diff	  = max_est > 1e-6f ? abs_diff / max_est : 0.0f;
			bool  is_mismatch = rel_diff > DIST_MISMATCH_REL_THRESHOLD;

			if (verbose || is_mismatch)
			{
				/* Compute true L2 on-demand for output */
				float true_dist =
						l2_distance_sq(query, train + (size_t)v * dim, dim);
				float mkt_err	= 0.0f;
				float faiss_err = 0.0f;
				if (true_dist > 1e-6f)
				{
					mkt_err	  = fabsf(mkt_dists[v] - true_dist) / true_dist;
					faiss_err = fabsf(faiss_dists[v] - true_dist) / true_dist;
				}

				if (use_progress && !verbose)
					clear_progress();
				printf("DIST q=%-4u v=%-8u mkt=%-10.4f "
					   "faiss=%-10.4f true=%-10.4f "
					   "mkt_err=%.2f%%  faiss_err=%.2f%%\n",
					   q,
					   v,
					   mkt_dists[v],
					   faiss_dists[v],
					   true_dist,
					   mkt_err * 100.0f,
					   faiss_err * 100.0f);
			}

			if (is_mismatch)
				dist_mismatches++;

			if (abs_diff > mkt_faiss_max_abs)
				mkt_faiss_max_abs = abs_diff;
			mkt_faiss_sum_abs += abs_diff;

			/* Inline true L2 sampling: stride-based per query */
			if (v % sample_stride == 0)
			{
				float td = l2_distance_sq(query, train + (size_t)v * dim, dim);
				if (td > 1e-6f)
				{
					float me = fabsf(mkt_dists[v] - td) / td;
					float fe = fabsf(faiss_dists[v] - td) / td;
					if (me > mkt_true_max_rel)
						mkt_true_max_rel = me;
					mkt_true_sum_rel += me;
					if (fe > faiss_true_max_rel)
						faiss_true_max_rel = fe;
					faiss_true_sum_rel += fe;
				}
				n_sampled++;
			}
		}

		total_pairs += ntrain;

		if (use_progress && !verbose)
			print_progress("Distance", q + 1, nqueries, &last_pct);
	}

	if (!verbose && dist_mismatches > 0)
	{
		printf("  %u/%lu pairs exceeded %.0f%% relative "
			   "difference threshold\n",
			   dist_mismatches,
			   (unsigned long)total_pairs,
			   (double)DIST_MISMATCH_REL_THRESHOLD * 100.0);
	}

	printf("\nDISTANCE SUMMARY:\n");
	printf("  mkt  vs faiss: max_abs_diff=%.4f  "
		   "mean_abs_diff=%.4f\n",
		   mkt_faiss_max_abs,
		   total_pairs ? (double)mkt_faiss_sum_abs / total_pairs : 0.0);
	printf("  mkt  vs true:  max_rel_err=%.1f%%  "
		   "mean_rel_err=%.1f%%  (sampled %u pairs)\n",
		   mkt_true_max_rel * 100.0f,
		   n_sampled ? (mkt_true_sum_rel / n_sampled) * 100.0f : 0.0f,
		   n_sampled);
	printf("  faiss vs true: max_rel_err=%.1f%%  "
		   "mean_rel_err=%.1f%%  (sampled %u pairs)\n\n",
		   faiss_true_max_rel * 100.0f,
		   n_sampled ? (faiss_true_sum_rel / n_sampled) * 100.0f : 0.0f,
		   n_sampled);

dist_cleanup:
	faiss_RaBitQuantizer_free(faiss_rq);
	free(faiss_codes);
	free(enc_f_add);
	free(enc_f_rescale);
	free(enc_bits);
	free(train_transformed);
	free(mkt_dists);
	free(faiss_dists);
	free(q_buf);
	free(qt_buf);
}

/*
 * Print usage for verify rabitq command.
 */
static void
print_usage(const CmdContext *ctx)
{
	printf("Usage: %s verify rabitq --dataset <path.hdf5> "
		   "[options]\n\n",
		   ctx->prog_name);
	printf("Compare meerkat RaBitQ against FAISS using "
		   "ann-benchmarks HDF5 datasets.\n\n");
	printf("Options:\n");
	printf("  --dataset <path>   Path to HDF5 file (required)\n");
	printf("  --count <N>        Number of train vectors "
		   "(default: all)\n");
	printf("  --queries <N>      Number of query vectors "
		   "(default: all)\n");
	printf("  --seed <S>         RNG seed (default: %d)\n", DEFAULT_SEED);
	printf("  --dump <dir>       Dump encoded bits to "
		   "directory after Phase 1\n");
	printf("  -v, --verbose      Print every result "
		   "(default: mismatches only)\n");
	printf("  -h, --help         Show this help\n");
	printf("\nExamples:\n");
	CMD_USAGE_EXAMPLE(
			ctx, "verify rabitq", "--dataset glove-100-angular.hdf5");
	CMD_USAGE_EXAMPLE(
			ctx,
			"verify rabitq",
			"--dataset glove.hdf5 --count 1000 "
			"--queries 10");
}

/*
 * cmd_verify_rabitq - Entry point for "mkt verify rabitq"
 */
int
cmd_verify_rabitq(CmdContext *ctx)
{
	const char *dataset_path = NULL;
	const char *dump_dir	 = NULL;
	uint32_t	count		 = 0; /* 0 = use all */
	uint32_t	nqueries	 = 0; /* 0 = use all */
	uint64_t	seed		 = DEFAULT_SEED;
	bool		verbose		 = false;

	static struct option long_options[] = {
			{"dataset", required_argument, NULL, 'd'},
			{"count", required_argument, NULL, 'c'},
			{"queries", required_argument, NULL, 'q'},
			{"seed", required_argument, NULL, 's'},
			{"dump", required_argument, NULL, 'D'},
			{"verbose", no_argument, NULL, 'v'},
			{"help", no_argument, NULL, 'h'},
			{NULL, 0, NULL, 0},
	};

	optind = 1; /* Reset getopt */
	int opt;
	while ((opt = getopt_long(
					ctx->argc,
					ctx->argv,
					"hvd:c:q:s:D:",
					long_options,
					NULL)) != -1)
	{
		switch (opt)
		{
		case 'd':
			dataset_path = optarg;
			break;
		case 'c':
			count = (uint32_t)atoi(optarg);
			break;
		case 'q':
			nqueries = (uint32_t)atoi(optarg);
			break;
		case 's':
			seed = (uint64_t)strtoull(optarg, NULL, 10);
			break;
		case 'D':
			dump_dir = optarg;
			break;
		case 'v':
			verbose = true;
			break;
		case 'h':
			print_usage(ctx);
			return 0;
		default:
			print_usage(ctx);
			return 1;
		}
	}

	if (dataset_path == NULL)
	{
		fprintf(stderr, "Error: --dataset is required\n\n");
		print_usage(ctx);
		return 1;
	}

	/* Load HDF5 datasets */
	printf("Loading dataset: %s\n", dataset_path);

	hsize_t train_rows, train_cols;
	float  *train =
			load_hdf5_float(dataset_path, "train", &train_rows, &train_cols);
	if (train == NULL)
		return 1;

	hsize_t test_rows, test_cols;
	float  *test =
			load_hdf5_float(dataset_path, "test", &test_rows, &test_cols);
	if (test == NULL)
	{
		free(train);
		return 1;
	}

	Dimension dim = (Dimension)train_cols;
	if (test_cols != train_cols)
	{
		fprintf(stderr,
				"Error: dimension mismatch train=%llu test=%llu\n",
				(unsigned long long)train_cols,
				(unsigned long long)test_cols);
		free(train);
		free(test);
		return 1;
	}

	/* Apply count limits */
	uint32_t ntrain = (uint32_t)train_rows;
	if (count > 0 && count < ntrain)
		ntrain = count;

	uint32_t nq = (uint32_t)test_rows;
	if (nqueries > 0 && nqueries < nq)
		nq = nqueries;

	printf("  train: %u vectors, dim=%u (of %llu available)\n",
		   ntrain,
		   dim,
		   (unsigned long long)train_rows);
	printf("  test:  %u queries (of %llu available)\n",
		   nq,
		   (unsigned long long)test_rows);
	printf("  seed: %lu\n\n", (unsigned long)seed);

	/* Compute centroid */
	float *centroid = malloc(dim * sizeof(float));
	if (centroid == NULL)
	{
		fprintf(stderr, "Error: allocation failed for centroid\n");
		free(train);
		free(test);
		return 1;
	}
	printf("Computing centroid from %u vectors...\n", ntrain);
	compute_centroid(train, ntrain, dim, centroid);

	/* Initialize RaBitQ parameters */
	printf("Initializing RaBitQ (dim=%u, seed=%lu)...\n\n",
		   dim,
		   (unsigned long)seed);
	RaBitQParams *params = mkt_rabitq_create(dim, seed);
	if (params == NULL)
	{
		fprintf(stderr, "Error: RaBitQ parameter creation failed\n");
		free(centroid);
		free(train);
		free(test);
		return 1;
	}

	/* Show progress bars on stderr when not in verbose mode and
	 * stderr is a terminal (not piped/redirected). */
	bool use_progress = !verbose && isatty(STDERR_FILENO);

	/* Run verification phases */
	verify_encoding(
			params,
			train,
			centroid,
			ntrain,
			dim,
			verbose,
			use_progress,
			dump_dir);
	verify_distances(
			params,
			train,
			ntrain,
			test,
			nq,
			centroid,
			dim,
			verbose,
			use_progress);
	/* Cleanup */
	mkt_rabitq_destroy(params);
	free(centroid);
	free(train);
	free(test);

	return 0;
}

#endif /* MKT_HAVE_FAISS && MKT_HAVE_HDF5 */
