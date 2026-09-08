/*
 * explain.h - EXPLAIN ANALYZE hook for mktann
 *
 * Registers an explain_per_node_hook that emits mktann scan stats
 * (phase timing, filter effectiveness, I/O counts) when EXPLAIN
 * (ANALYZE, VERBOSE) is used on a query with an mktann index scan.
 */

#ifndef MKT_EXPLAIN_H
#define MKT_EXPLAIN_H

void mktann_explain_init(void);

#endif /* MKT_EXPLAIN_H */
