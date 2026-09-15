/*
 * cost.h - Cost model for mktann index scans
 *
 * See docs/cost-model-design.md. The model derives work from pages rather
 * than from the row count, because pages already carry every effect that
 * makes a probe more expensive than the rows suggest: SOAR and boundary
 * replicas, dead entries a fastscan page cannot flag individually, AoS
 * pages appended by inserts, and split chains not yet reclaimed.
 *
 * The constants below are per-dimension fits to the measurements in §3 of
 * that document, expressed in cpu_operator_cost units against the 50 ns
 * anchor of §5. They are constants rather than settings on purpose: a knob
 * that scaled the estimate would hide a miscalibration instead of fixing
 * it, and the planner's own page-cost and cpu_*_cost settings already
 * describe the machine.
 */

#ifndef MKT_PG_COST_H
#define MKT_PG_COST_H

#include <postgres.h>

#include <nodes/pathnodes.h>

/*
 * Per-entry scoring.
 *
 * Measured by scanning the same rows through two indexes with different
 * nlist at the same nprobe, which separates the per-entry cost from the
 * per-list one -- they are collinear in any sweep that varies only nprobe,
 * because entries per list stays roughly constant.
 *
 *   128 dimensions   4.6 ns/entry
 *   768 dimensions  15.5 ns/entry
 *
 * So it does grow with the dimension, sublinearly, as the kernel reads more
 * of each code: 2.4 + 0.017*dim ns, which is 0.05 + 0.00034*dim cop at the
 * 50 ns anchor. The two page formats are not separated here; the AoS
 * constants below are the design document's original estimates.
 */
#define MKT_COST_ENTRY_BASE	 0.05
#define MKT_COST_ENTRY_SLOPE 0.00034

/*
 * Per-probed-list setup: building the fastscan lookup table for a cluster.
 *
 * Flat in the dimension, which is the opposite of what this model first
 * assumed:
 *
 *   128 dimensions  5.34 us/list
 *   768 dimensions  4.84 us/list
 *
 * The table is indexed by the query's quantized code, so its size follows
 * the code width rather than the vector, and the build is dominated by
 * fixed setup. The original constant extrapolated linearly from a single
 * 128-dimension figure and reached 100 us per list at 768 dimensions -- 20
 * times the measurement, and enough on its own to lose plans at high
 * nprobe. 5 us is 100 cop.
 */
/*
 * Per-entry scoring on AoS pages, measured the same way: 15.7 ns at 128
 * dimensions and 34.2 ns at 768. Two to three times the fastscan cost at
 * both, since the kernel reads each code separately rather than a packed
 * group.
 */
#define MKT_COST_AOS_ENTRY_BASE	 0.24
#define MKT_COST_AOS_ENTRY_SLOPE 0.00058

/*
 * Per-probed-list overhead, paid by both page formats.
 *
 *   fastscan  5.34 us at 128d, 4.84 us at 768d
 *   AoS       3.32 us at 128d, 4.20 us at 768d
 *
 * Flat in the dimension and mostly independent of format, so it is not the
 * lookup table: it is the cost of opening a cluster at all -- reading the
 * head page, taking its full-precision pt_centroid, and scoring the exact
 * centroid distance that phase A ranks the probe set by. Both formats pay
 * it, which is why it is kept separate from the fastscan-only surcharge
 * below: folding the two together charges AoS nothing for work it does.
 *
 * The lookup table is the residue: fastscan measures about 1.3 us per list
 * above AoS at both dimensions.
 */
#define MKT_COST_LIST_OPEN	 76.0
#define MKT_COST_CLUSTER_LUT 26.0

/*
 * Per-candidate rerank: read the heap tuple and compute one exact distance.
 *
 * Measured at 0.93 us for a 128-dimension column and 0.87 us for a
 * 768-dimension one -- flat, not proportional to the dimension, because the
 * cost is dominated by reaching the tuple rather than by the arithmetic
 * over it. 0.9 us is 18 cop at the 50 ns anchor.
 *
 * What does move it, by a factor of thirty, is whether the value is stored
 * out of line: the same 768-dimension column measured 25.7 us per candidate
 * under TOASTed external storage and 0.87 us under STORAGE PLAIN, because
 * every candidate then costs a fetch and a decompress from the toast
 * relation. This is the one term no competing estimator needs -- HNSW and
 * IVFFlat keep their vectors inside index pages, which cannot be toasted,
 * so they materialize nothing from the heap and pay this zero times.
 */
#define MKT_COST_FETCH		   18.0
#define MKT_COST_FETCH_DETOAST 30.0

/*
 * Top-k maintenance grows with k: the posting scan measured three times
 * longer at k = 1000 than at k = 10, everything else equal, so the heap
 * work per entry rises by about 0.3 per doubling.
 */
#define MKT_COST_TOPK_LOG_COEFF 0.30

/*
 * Centroid descent, per beam slot per level, and per routed cluster whose
 * head page phase A re-ranks exactly. The least-calibrated constants here:
 * §3 measures the whole descent at 0.04-0.2 ms, small enough at every size
 * measured that the fit is loose. They matter most on a deep tree.
 */
#define MKT_COST_CENTROID_SLOT	  100.0
#define MKT_COST_PROBE_HEAD_BASE  1.0
#define MKT_COST_PROBE_HEAD_SLOPE 0.01

void mktann_cost_estimate(
		PlannerInfo *root,
		IndexPath	*path,
		double		 loop_count,
		Cost		*startup_cost,
		Cost		*total_cost,
		Selectivity *selectivity,
		double		*correlation,
		double		*index_pages);

#endif /* MKT_PG_COST_H */
