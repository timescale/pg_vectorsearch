/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * test_vs_platform.c - Platform abstraction tests
 */

#include <stdint.h>
#include <stdio.h>

#include "core/platform.h"
#include "vs_test.h"

TEST_GROUP(Platform);

TEST(simd_detect_returns_valid)
{
	SimdCapability caps = vs_detect_simd();

	/* Result should be one of the valid combinations */
	/* On x86, we should have at least SSE2 on any modern CPU */
	/* On ARM64, we should have NEON */
	/* SIMD_NONE is valid for unknown architectures */

	ASSERT_TRUE(caps >= SIMD_NONE, "capabilities should be non-negative");
}

TEST(simd_detect_is_cached)
{
	/* Calling multiple times should return same result */
	SimdCapability caps1 = vs_detect_simd();
	SimdCapability caps2 = vs_detect_simd();
	SimdCapability caps3 = vs_detect_simd();

	ASSERT_EQ(caps1, caps2, "cached result should be consistent");
	ASSERT_EQ(caps2, caps3, "cached result should be consistent");
}

TEST(simd_hierarchy_x86)
{
	SimdCapability caps = vs_detect_simd();

	/* On x86, capabilities form a hierarchy */
	/* AVX512F implies AVX2 which implies SSE4.1 which implies SSE2 */

	if (caps & SIMD_AVX512F)
	{
		ASSERT_TRUE(caps & SIMD_AVX2, "AVX512F should imply AVX2");
	}

	if (caps & SIMD_AVX2)
	{
		ASSERT_TRUE(caps & SIMD_SSE4_1, "AVX2 should imply SSE4.1");
	}

	if (caps & SIMD_SSE4_1)
	{
		ASSERT_TRUE(caps & SIMD_SSE2, "SSE4.1 should imply SSE2");
	}
}

TEST(vs_has_simd_helper)
{
	SimdCapability caps = vs_detect_simd();

	/* vs_has_simd should match direct flag check */
	ASSERT_EQ(
			(caps & SIMD_SSE2) != 0,
			vs_has_simd(SIMD_SSE2),
			"vs_has_simd should match direct check");
	ASSERT_EQ(
			(caps & SIMD_AVX2) != 0,
			vs_has_simd(SIMD_AVX2),
			"vs_has_simd should match direct check");
	ASSERT_EQ(
			(caps & SIMD_NEON) != 0,
			vs_has_simd(SIMD_NEON),
			"vs_has_simd should match direct check");
}

TEST(alignment_macro_power_of_two)
{
	/* Test VS_ALIGN with various inputs */
	ASSERT_EQ(0, VS_ALIGN(0, 16), "align 0 to 16");
	ASSERT_EQ(16, VS_ALIGN(1, 16), "align 1 to 16");
	ASSERT_EQ(16, VS_ALIGN(15, 16), "align 15 to 16");
	ASSERT_EQ(16, VS_ALIGN(16, 16), "align 16 to 16");
	ASSERT_EQ(32, VS_ALIGN(17, 16), "align 17 to 16");

	ASSERT_EQ(64, VS_ALIGN(33, 64), "align 33 to 64");
	ASSERT_EQ(64, VS_ALIGN(64, 64), "align 64 to 64");
	ASSERT_EQ(128, VS_ALIGN(65, 64), "align 65 to 64");
}

TEST(is_aligned_macro)
{
	uintptr_t addr = 0x1000; /* 4096, aligned to many powers of 2 */

	ASSERT_TRUE(VS_IS_ALIGNED(addr, 16), "0x1000 aligned to 16");
	ASSERT_TRUE(VS_IS_ALIGNED(addr, 64), "0x1000 aligned to 64");
	ASSERT_TRUE(VS_IS_ALIGNED(addr, 256), "0x1000 aligned to 256");
	ASSERT_TRUE(VS_IS_ALIGNED(addr, 4096), "0x1000 aligned to 4096");

	ASSERT_TRUE(!VS_IS_ALIGNED(addr + 1, 16), "0x1001 not aligned to 16");
	ASSERT_TRUE(!VS_IS_ALIGNED(addr + 8, 16), "0x1008 not aligned to 16");
	ASSERT_TRUE(VS_IS_ALIGNED(addr + 16, 16), "0x1010 aligned to 16");
}

TEST(prefetch_compiles)
{
	/*
	 * Just verify prefetch macros compile without error.
	 * Actual prefetch effect is not testable.
	 */
	int data[64];
	vs_prefetch_read(&data[0]);
	vs_prefetch_write(&data[32]);

	/* Prevent optimizer from removing the array */
	data[0] = 1;

	/* If we got here, macros compiled successfully */
	ASSERT_TRUE(1, "prefetch macros compile");
}

TEST(branch_hints_compile)
{
	/*
	 * Verify branch hint macros compile and work correctly.
	 * The hints don't change semantics, just compiler optimization.
	 */
	int x = 42;

	if (vs_likely(x > 0))
	{
		x++;
	}

	if (vs_unlikely(x < 0))
	{
		x--;
	}

	ASSERT_EQ(43, x, "branch hints should not change logic");
}

TEST(compiler_barrier_compiles)
{
	/*
	 * Verify compiler barrier macro compiles.
	 * Effect is not directly testable without inspecting assembly.
	 */
	volatile int x = 1;
	vs_compiler_barrier();
	x = 2;
	vs_compiler_barrier();

	/* Read once, here: ASSERT_EQ names its operands twice, and a
	 * volatile read is a side effect to repeat. */
	int observed = x;
	ASSERT_EQ(2, observed, "compiler barrier should not change values");
}

TEST(print_detected_capabilities)
{
	/*
	 * Not a real test - just prints detected capabilities for debugging.
	 * Always passes.
	 */
	SimdCapability caps = vs_detect_simd();

	TEST_PRINT("Detected SIMD capabilities: 0x%x\n", caps);
	TEST_PRINT("  SSE2:    %s\n", (caps & SIMD_SSE2) ? "yes" : "no");
	TEST_PRINT("  SSE4.1:  %s\n", (caps & SIMD_SSE4_1) ? "yes" : "no");
	TEST_PRINT("  AVX2:    %s\n", (caps & SIMD_AVX2) ? "yes" : "no");
	TEST_PRINT("  AVX512F: %s\n", (caps & SIMD_AVX512F) ? "yes" : "no");
	TEST_PRINT("  NEON:    %s\n", (caps & SIMD_NEON) ? "yes" : "no");

	ASSERT_TRUE(1, "capability print");
}

/*
 * Architecture-specific tests.
 *
 * These tests verify that SIMD detection works correctly on each platform.
 * They are conditionally compiled so they only run on the appropriate
 * architecture, ensuring CI catches detection failures.
 */

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
		defined(_M_IX86)

TEST(x86_has_sse2)
{
	/*
	 * All x86-64 CPUs support SSE2 (it's part of the x86-64 spec).
	 * 32-bit x86 may not, but any CPU from ~2003 onward has it.
	 */
	SimdCapability caps = vs_detect_simd();
	ASSERT_TRUE(caps & SIMD_SSE2, "x86 should have SSE2");
}

TEST(x86_no_neon)
{
	/* NEON is ARM-only, should never be detected on x86 */
	SimdCapability caps = vs_detect_simd();
	ASSERT_TRUE(!(caps & SIMD_NEON), "x86 should not have NEON");
}

TEST(x86_avx2_implies_sse)
{
	/*
	 * If AVX2 is present, all lower capabilities must also be present.
	 * This validates the detection logic sets all implied flags.
	 */
	SimdCapability caps = vs_detect_simd();

	if (caps & SIMD_AVX2)
	{
		ASSERT_TRUE(caps & SIMD_SSE2, "AVX2 requires SSE2");
		ASSERT_TRUE(caps & SIMD_SSE4_1, "AVX2 requires SSE4.1");
	}
	else
	{
		/* Skip test if AVX2 not available */
		ASSERT_TRUE(1, "AVX2 not available, skipping hierarchy check");
	}
}

TEST(x86_avx512_implies_avx2)
{
	/*
	 * If AVX-512 is present, AVX2 must also be present.
	 */
	SimdCapability caps = vs_detect_simd();

	if (caps & SIMD_AVX512F)
	{
		ASSERT_TRUE(caps & SIMD_AVX2, "AVX512F requires AVX2");
		ASSERT_TRUE(caps & SIMD_SSE4_1, "AVX512F requires SSE4.1");
		ASSERT_TRUE(caps & SIMD_SSE2, "AVX512F requires SSE2");
	}
	else
	{
		ASSERT_TRUE(1, "AVX512F not available, skipping hierarchy check");
	}
}

#endif /* x86 */

#if defined(__aarch64__) || defined(_M_ARM64)

TEST(arm64_has_neon)
{
	/*
	 * NEON is mandatory on AArch64 - all ARM64 CPUs have it.
	 * This test will fail if detection is broken on ARM.
	 */
	SimdCapability caps = vs_detect_simd();
	ASSERT_TRUE(caps & SIMD_NEON, "ARM64 must have NEON");
}

TEST(arm64_no_x86_features)
{
	/* x86 features should never be detected on ARM */
	SimdCapability caps = vs_detect_simd();
	ASSERT_TRUE(!(caps & SIMD_SSE2), "ARM64 should not have SSE2");
	ASSERT_TRUE(!(caps & SIMD_SSE4_1), "ARM64 should not have SSE4.1");
	ASSERT_TRUE(!(caps & SIMD_AVX2), "ARM64 should not have AVX2");
	ASSERT_TRUE(!(caps & SIMD_AVX512F), "ARM64 should not have AVX512F");
}

#endif /* ARM64 */

#if defined(__arm__) || defined(_M_ARM)

TEST(arm32_no_x86_features)
{
	/* x86 features should never be detected on ARM */
	SimdCapability caps = vs_detect_simd();
	ASSERT_TRUE(!(caps & SIMD_SSE2), "ARM32 should not have SSE2");
	ASSERT_TRUE(!(caps & SIMD_AVX2), "ARM32 should not have AVX2");
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
TEST(arm32_has_neon_when_compiled_with_neon)
{
	/* If compiled with NEON support, detection should find it */
	SimdCapability caps = vs_detect_simd();
	ASSERT_TRUE(caps & SIMD_NEON, "ARM32 with NEON should detect NEON");
}
#endif

#endif /* ARM32 */

/*
 * Cross-validation tests against OS-provided sources of truth.
 *
 * These tests verify our SIMD detection matches what the OS reports,
 * catching bugs in CPUID interpretation or OS support checks.
 */

#ifdef __linux__

#include <string.h>

/*
 * Helper: check if /proc/cpuinfo flags contain a specific flag.
 * Returns 1 if found, 0 if not found, -1 on error.
 */
static int
cpuinfo_has_flag(const char *flag)
{
	FILE *f = fopen("/proc/cpuinfo", "r");
	if (!f)
		return -1;

	char line[4096];
	int	 found = 0;

	while (fgets(line, sizeof(line), f))
	{
		/* Look for "flags" line (x86) or "Features" line (ARM) */
		if (strncmp(line, "flags", 5) == 0 ||
			strncmp(line, "Features", 8) == 0)
		{
			/* Check if flag appears as a whole word */
			char *p = strstr(line, flag);
			if (p)
			{
				/* Verify it's a whole word (space or end before/after) */
				char before = (p > line) ? p[-1] : ' ';
				char after	= p[strlen(flag)];
				if ((before == ' ' || before == ':' || before == '\t') &&
					(after == ' ' || after == '\n' || after == '\0'))
				{
					found = 1;
					break;
				}
			}
		}
	}

	fclose(f);
	return found;
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
		defined(_M_IX86)

TEST(x86_detection_matches_procfs)
{
	/*
	 * Verify our CPUID-based detection matches /proc/cpuinfo.
	 * This catches bugs in CPUID interpretation.
	 */
	SimdCapability caps = vs_detect_simd();

	int os_sse2	   = cpuinfo_has_flag("sse2");
	int os_sse4_1  = cpuinfo_has_flag("sse4_1");
	int os_avx2	   = cpuinfo_has_flag("avx2");
	int os_avx512f = cpuinfo_has_flag("avx512f");

	/* Skip if we couldn't read cpuinfo */
	if (os_sse2 < 0)
	{
		TEST_PRINT("(skipping: could not read /proc/cpuinfo)\n");
		ASSERT_TRUE(1, "skipped");
		return;
	}

	TEST_PRINT(
			"/proc/cpuinfo: sse2=%d sse4_1=%d avx2=%d avx512f=%d\n",
			os_sse2,
			os_sse4_1,
			os_avx2,
			os_avx512f);
	TEST_PRINT(
			"vs_detect:    sse2=%d sse4_1=%d avx2=%d avx512f=%d\n",
			!!(caps & SIMD_SSE2),
			!!(caps & SIMD_SSE4_1),
			!!(caps & SIMD_AVX2),
			!!(caps & SIMD_AVX512F));

	/*
	 * Our detection should match OS for features we claim to have.
	 * We might detect fewer features if OS doesn't support saving state.
	 */
	if (os_sse2)
		ASSERT_TRUE(caps & SIMD_SSE2, "SSE2 in cpuinfo but not detected");
	if (os_sse4_1)
		ASSERT_TRUE(caps & SIMD_SSE4_1, "SSE4.1 in cpuinfo but not detected");

	/*
	 * For AVX/AVX2/AVX512, OS might report CPU capability but we correctly
	 * detect that OS doesn't support saving state. So we only check that
	 * if we detect it, OS also has it.
	 */
	if (caps & SIMD_AVX2)
		ASSERT_TRUE(os_avx2, "AVX2 detected but not in cpuinfo");
	if (caps & SIMD_AVX512F)
		ASSERT_TRUE(os_avx512f, "AVX512F detected but not in cpuinfo");
}

#endif /* x86 linux */

#if defined(__aarch64__) || defined(_M_ARM64)

#include <sys/auxv.h>

#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD (1 << 1)
#endif

TEST(arm64_detection_matches_auxval)
{
	/*
	 * Verify our detection matches getauxval(AT_HWCAP).
	 * On ARM64, ASIMD (NEON) is indicated by HWCAP_ASIMD.
	 */
	SimdCapability caps	 = vs_detect_simd();
	unsigned long  hwcap = getauxval(AT_HWCAP);

	TEST_PRINT(
			"getauxval(AT_HWCAP): 0x%lx, HWCAP_ASIMD=%d\n",
			hwcap,
			!!(hwcap & HWCAP_ASIMD));
	TEST_PRINT("vs_detect: NEON=%d\n", !!(caps & SIMD_NEON));

	if (hwcap & HWCAP_ASIMD)
		ASSERT_TRUE(caps & SIMD_NEON, "ASIMD in hwcap but NEON not detected");

	if (caps & SIMD_NEON)
		ASSERT_TRUE(hwcap & HWCAP_ASIMD, "NEON detected but not in hwcap");
}

TEST(arm64_detection_matches_procfs)
{
	/*
	 * Also verify against /proc/cpuinfo Features line.
	 */
	SimdCapability caps = vs_detect_simd();

	int os_asimd = cpuinfo_has_flag("asimd");

	if (os_asimd < 0)
	{
		TEST_PRINT("(skipping: could not read /proc/cpuinfo)\n");
		ASSERT_TRUE(1, "skipped");
		return;
	}

	TEST_PRINT("/proc/cpuinfo: asimd=%d\n", os_asimd);
	TEST_PRINT("vs_detect:    NEON=%d\n", !!(caps & SIMD_NEON));

	if (os_asimd)
		ASSERT_TRUE(
				caps & SIMD_NEON, "asimd in cpuinfo but NEON not detected");
}

#endif /* ARM64 linux */

#endif /* __linux__ */

#ifdef __APPLE__

#include <sys/sysctl.h>

/*
 * Helper: check sysctl boolean value.
 * Returns 1 if true, 0 if false or not found.
 */
static int
sysctl_has_feature(const char *name)
{
	int	   value = 0;
	size_t size	 = sizeof(value);
	if (sysctlbyname(name, &value, &size, NULL, 0) == 0)
		return value;
	return 0;
}

#if defined(__x86_64__)

TEST(x86_macos_detection_matches_sysctl)
{
	/*
	 * Verify our detection matches macOS sysctl hw.optional.* values.
	 */
	SimdCapability caps = vs_detect_simd();

	int os_sse2	   = 1; /* Always present on x86_64 macOS */
	int os_sse4_1  = sysctl_has_feature("hw.optional.sse4_1");
	int os_avx2	   = sysctl_has_feature("hw.optional.avx2_0");
	int os_avx512f = sysctl_has_feature("hw.optional.avx512f");

	TEST_PRINT(
			"sysctl:     sse4_1=%d avx2=%d avx512f=%d\n",
			os_sse4_1,
			os_avx2,
			os_avx512f);
	TEST_PRINT(
			"vs_detect: sse2=%d sse4_1=%d avx2=%d avx512f=%d\n",
			!!(caps & SIMD_SSE2),
			!!(caps & SIMD_SSE4_1),
			!!(caps & SIMD_AVX2),
			!!(caps & SIMD_AVX512F));

	ASSERT_TRUE(caps & SIMD_SSE2, "x86_64 macOS must have SSE2");

	if (os_sse4_1)
		ASSERT_TRUE(caps & SIMD_SSE4_1, "SSE4.1 in sysctl but not detected");
	if (caps & SIMD_AVX2)
		ASSERT_TRUE(os_avx2, "AVX2 detected but not in sysctl");
	if (caps & SIMD_AVX512F)
		ASSERT_TRUE(os_avx512f, "AVX512F detected but not in sysctl");
}

#endif /* x86_64 macOS */

#if defined(__aarch64__) || defined(__arm64__)

TEST(arm64_macos_detection_matches_sysctl)
{
	/*
	 * Verify NEON detection on Apple Silicon.
	 * NEON (ASIMD) is mandatory on ARM64, so this should always pass.
	 */
	SimdCapability caps = vs_detect_simd();

	/* hw.optional.neon exists on Apple Silicon */
	int os_neon = sysctl_has_feature("hw.optional.neon");

	/* Also check via AdvSIMD which is the ARM64 name */
	int os_advsimd = sysctl_has_feature("hw.optional.AdvSIMD");

	TEST_PRINT("sysctl:     neon=%d advsimd=%d\n", os_neon, os_advsimd);
	TEST_PRINT("vs_detect: NEON=%d\n", !!(caps & SIMD_NEON));

	/* Apple Silicon always has NEON */
	ASSERT_TRUE(caps & SIMD_NEON, "ARM64 macOS must have NEON");

	/* Verify no false x86 detections */
	ASSERT_TRUE(!(caps & SIMD_SSE2), "ARM64 macOS should not have SSE2");
	ASSERT_TRUE(!(caps & SIMD_AVX2), "ARM64 macOS should not have AVX2");
}

#endif /* ARM64 macOS */

#endif /* __APPLE__ */
