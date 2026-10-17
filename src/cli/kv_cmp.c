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
#define IGN_REL	 0.03
#define IGN_ABS	 0.002

static int
better_sign(const char *unit)
{
	if (unit == NULL)
		return 1;
	if (strcmp(unit, "us") == 0 || strcmp(unit, "ns") == 0 ||
		strcmp(unit, "s") == 0 || strcmp(unit, "ms") == 0)
		return -1;
	return 1;
}

static bool
is_abs(const char *unit)
{
	return unit != NULL && strcmp(unit, "ratio") == 0;
}

static double
se_mean(KvSample s)
{
	if (s.n <= 0.0 || s.stddev <= 0.0)
		return 0.0;
	return s.stddev / sqrt(s.n);
}

void
kv_cmp_values(const char *unit, KvSample base, KvSample head, KvCmp *out)
{
	double den;
	double delta;
	double se;
	double noise;
	double floor;
	double ignore;
	double warn_bar;
	double fail_bar;
	bool   absu;

	memset(out, 0, sizeof(*out));
	den			  = fabs(base.value);
	delta		  = head.value - base.value;
	out->rel	  = den > 0.0 ? delta / den : 0.0;
	absu		  = is_abs(unit);
	out->mag	  = absu ? fabs(delta) : fabs(out->rel);
	out->worse	  = (double)better_sign(unit) * delta < 0.0;
	se			  = hypot(se_mean(base), se_mean(head));
	out->se_delta = se;
	if (se > 0.0)
	{
		noise = absu ? K_SE * se : (den > 0.0 ? K_SE * se / den : 0.0);
		out->inside_noise = out->mag < noise;
	}
	else
		noise = 0.0;

	floor	 = absu ? IGN_ABS : IGN_REL;
	ignore	 = noise > floor ? noise : floor;
	warn_bar = absu ? WARN_ABS : WARN_REL;
	fail_bar = absu ? FAIL_ABS : FAIL_REL;

	if (!out->worse || out->mag < ignore)
		out->kind = KV_CMP_OK;
	else if (out->mag >= fail_bar)
		out->kind = KV_CMP_FAIL;
	else if (out->mag >= warn_bar)
		out->kind = KV_CMP_WARN;
	else
		out->kind = KV_CMP_OK;
}

static KvSample
sample_of(const KvMetric *m)
{
	return (KvSample){m->value, m->stddev, m->n};
}

void
kv_cmp_metrics(const Kv *base, const Kv *head, KvCmp *out)
{
	KvMetric	b;
	KvMetric	h;
	const char *unit;

	kv_metric_load(base, &b);
	kv_metric_load(head, &h);
	unit = h.unit != NULL ? h.unit : b.unit;
	kv_cmp_values(unit, sample_of(&b), sample_of(&h), out);
}
