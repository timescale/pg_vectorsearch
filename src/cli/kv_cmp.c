/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * kv_cmp.c - Noise-aware metric compare
 *
 * Fail only if the drop is outside combined SE and past the unit-class
 * bar. Improvements never fail.
 */

#include <math.h>
#include <string.h>

#include "kv_cmp.h"

#define K_SE	 2.0
#define WARN_REL 0.15
#define FAIL_REL 0.30
#define WARN_ABS 0.005
#define FAIL_ABS 0.02
/* Slack under each bar, so binary roundoff cannot split equal drops
 * (0.94 - 0.92 lands a hair below 0.02; 0.99 - 0.97 a hair above). */
#define BAR_EPS 1e-9

/* +1 when a larger value is better, -1 when smaller is (time, and
 * the slots a lookup walks). */
static int
better_sign(const char *unit)
{
	if (unit == NULL)
		return 1;
	if (strcmp(unit, "us") == 0 || strcmp(unit, "ns") == 0 ||
		strcmp(unit, "s") == 0 || strcmp(unit, "ms") == 0 ||
		strcmp(unit, "slots") == 0)
		return -1;
	return 1;
}

static bool
is_abs(const char *unit)
{
	return unit != NULL && strcmp(unit, "ratio") == 0;
}

/* A diagnostic tally, like topk's buffered survivors, has no better
 * direction: report the change, never gate on it. */
static bool
is_info(const char *unit)
{
	return unit != NULL && strcmp(unit, "count") == 0;
}

/* Standard error of the sample's mean; 0 when stats are missing. */
static double
se_mean(KvSample s)
{
	if (s.n <= 0.0 || s.stddev <= 0.0)
		return 0.0;
	return s.stddev / sqrt(s.n);
}

/*
 * Classify head against base. A change for the worse is noise while
 * it sits inside k·SE of the difference, and warns or fails once it
 * passes the unit-class bars -- except that noise only softens a
 * fail-grade drop to a warning, never to silence, so a wild stddev
 * in a result file cannot blank the gate. Ratio units compare
 * absolute deltas; everything else compares relative to base.
 */
void
kv_cmp_values(const char *unit, KvSample base, KvSample head, KvCmp *out)
{
	double den;
	double delta;
	double se;
	double warn_bar;
	double fail_bar;
	bool   absu;

	memset(out, 0, sizeof(*out));
	/* A non-finite side is a failed measurement; NaN would also slip
	 * through every later comparison as not-worse. */
	if (!isfinite(base.value) || !isfinite(head.value))
	{
		out->kind  = KV_CMP_FAIL;
		out->worse = true;
		return;
	}
	den		 = fabs(base.value);
	delta	 = head.value - base.value;
	out->rel = den > 0.0 ? delta / den : 0.0;
	absu	 = is_abs(unit);
	out->mag = absu ? fabs(delta) : fabs(out->rel);
	if (is_info(unit))
		return;
	out->worse	  = (double)better_sign(unit) * delta < 0.0;
	se			  = hypot(se_mean(base), se_mean(head));
	out->se_delta = se;

	/*
	 * A zero baseline leaves a relative change undefined, so judge the
	 * raw delta against the SE band alone: a worsening the band does
	 * not cover fails outright rather than hiding behind rel == 0.
	 */
	if (!absu && den == 0.0 && delta != 0.0)
	{
		out->inside_noise = se > 0.0 && fabs(delta) < K_SE * se;
		if (out->worse && !out->inside_noise)
			out->kind = KV_CMP_FAIL;
		return;
	}

	if (se > 0.0)
	{
		double band = absu ? K_SE * se : (den > 0.0 ? K_SE * se / den : 0.0);

		out->inside_noise = out->mag < band;
	}

	warn_bar = absu ? WARN_ABS : WARN_REL;
	fail_bar = absu ? FAIL_ABS : FAIL_REL;

	if (!out->worse)
		out->kind = KV_CMP_OK;
	else if (out->mag >= fail_bar - BAR_EPS)
		out->kind = out->inside_noise ? KV_CMP_WARN : KV_CMP_FAIL;
	else if (out->inside_noise || out->mag < warn_bar - BAR_EPS)
		out->kind = KV_CMP_OK;
	else
		out->kind = KV_CMP_WARN;
}
