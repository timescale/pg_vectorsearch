/*
 * explain.h - EXPLAIN ANALYZE hook for prism
 *
 * Registers an explain_per_node_hook that emits prism scan stats
 * (phase timing, filter effectiveness, I/O counts) when EXPLAIN
 * (ANALYZE, VERBOSE) is used on a query with a prism index scan.
 */

#ifndef MKT_EXPLAIN_H
#define MKT_EXPLAIN_H

void prism_explain_init(void);

#endif /* MKT_EXPLAIN_H */
