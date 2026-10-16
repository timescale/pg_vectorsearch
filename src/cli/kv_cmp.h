/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * kv_cmp.h - Compare two metric samples
 *
 * Noise comes from each sample's stddev and n (Welch SE of the
 * difference). Unit-class floors still apply when stats are missing.
 */

#ifndef VS_CLI_KV_CMP_H
#define VS_CLI_KV_CMP_H

#include "kv.h"

typedef struct KvSample
{
	double value;
	double stddev; /* 0 if unknown */
	double n;	   /* 0 if unknown */
} KvSample;

typedef enum KvCmpKind
{
	KV_CMP_OK = 0,
	KV_CMP_WARN,
	KV_CMP_FAIL,
} KvCmpKind;

typedef struct KvCmp
{
	KvCmpKind kind;
	double	  rel;			/* (head - base) / |base| */
	double	  mag;			/* |Δ| in the unit's compare space */
	double	  se_delta;		/* combined SE of (head - base) */
	bool	  inside_noise; /* |Δ| < k · SE */
	bool	  worse;
} KvCmp;

void kv_cmp_values(const char *unit, KvSample base, KvSample head, KvCmp *out);

void kv_cmp_metrics(const Kv *base, const Kv *head, KvCmp *out);

#endif /* VS_CLI_KV_CMP_H */
