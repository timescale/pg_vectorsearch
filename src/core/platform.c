/*
 * platform.c - Platform abstraction implementation
 */

#include <stdbool.h>

#include "platform.h"

/* Cached SIMD capabilities (computed once) */
static SimdCapability g_simd_caps	  = SIMD_NONE;
static bool			  g_simd_detected = false;

/* Override mask for testing/benchmarking (0xFFFFFFFF = no override) */
static uint32_t g_simd_override = 0xFFFFFFFF;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
		defined(_M_IX86)

#define MKT_X86 1

/*
 * CPUID wrapper for x86/x64.
 *
 * GCC/Clang provide __get_cpuid, but we use inline asm for portability
 * and to handle extended leaves properly.
 */
static void
cpuid(uint32_t	leaf,
	  uint32_t	subleaf,
	  uint32_t *eax,
	  uint32_t *ebx,
	  uint32_t *ecx,
	  uint32_t *edx)
{
#if defined(__GNUC__) || defined(__clang__)
	__asm__ __volatile__("cpuid"
						 : "=a"(*eax), "=b"(*ebx), "=c"(*ecx), "=d"(*edx)
						 : "a"(leaf), "c"(subleaf));
#elif defined(_MSC_VER)
	int regs[4];
	__cpuidex(regs, (int)leaf, (int)subleaf);
	*eax = regs[0];
	*ebx = regs[1];
	*ecx = regs[2];
	*edx = regs[3];
#else
	*eax = *ebx = *ecx = *edx = 0;
#endif
}

/*
 * Check if OS supports AVX state saving (required for AVX/AVX2/AVX-512).
 *
 * Even if CPU supports AVX, the OS must save/restore the wider registers
 * on context switch. We check this via XGETBV.
 */
static bool
os_supports_avx(void)
{
	uint32_t eax, ebx, ecx, edx;

	/* Check OSXSAVE bit in CPUID.1:ECX */
	cpuid(1, 0, &eax, &ebx, &ecx, &edx);
	if (!(ecx & (1 << 27)))
		return false;

	/* Check XCR0 via XGETBV - bits 1 and 2 must be set for AVX */
#if defined(__GNUC__) || defined(__clang__)
	uint32_t xcr0_lo, xcr0_hi;
	__asm__ __volatile__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
	return (xcr0_lo & 0x6) == 0x6;
#elif defined(_MSC_VER)
	uint64_t xcr0 = _xgetbv(0);
	return (xcr0 & 0x6) == 0x6;
#else
	return false;
#endif
}

/*
 * Check if OS supports AVX-512 state saving.
 *
 * AVX-512 requires additional state beyond AVX (opmask, ZMM upper bits).
 */
static bool
os_supports_avx512(void)
{
	if (!os_supports_avx())
		return false;

#if defined(__GNUC__) || defined(__clang__)
	uint32_t xcr0_lo, xcr0_hi;
	__asm__ __volatile__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
	/* Bits 5, 6, 7 for opmask, ZMM_Hi256, Hi16_ZMM */
	return (xcr0_lo & 0xE6) == 0xE6;
#elif defined(_MSC_VER)
	uint64_t xcr0 = _xgetbv(0);
	return (xcr0 & 0xE6) == 0xE6;
#else
	return false;
#endif
}

static SimdCapability
detect_simd_x86(void)
{
	SimdCapability caps = SIMD_NONE;
	uint32_t	   eax, ebx, ecx, edx;

	/* Get max supported CPUID leaf */
	cpuid(0, 0, &eax, &ebx, &ecx, &edx);
	uint32_t max_leaf = eax;

	if (max_leaf < 1)
		return caps;

	/* CPUID leaf 1: basic feature flags */
	cpuid(1, 0, &eax, &ebx, &ecx, &edx);

	/* SSE2: EDX bit 26 */
	if (edx & (1 << 26))
		caps |= SIMD_SSE2;

	/* SSE4.1: ECX bit 19 */
	if (ecx & (1 << 19))
		caps |= SIMD_SSE4_1;

	/* AVX requires OS support */
	if ((ecx & (1 << 28)) && os_supports_avx())
	{
		/* Check AVX2 in extended features (leaf 7) */
		if (max_leaf >= 7)
		{
			cpuid(7, 0, &eax, &ebx, &ecx, &edx);

			/* AVX2: EBX bit 5 */
			if (ebx & (1 << 5))
				caps |= SIMD_AVX2;

			/* AVX-512F: EBX bit 16 (requires OS support) */
			if ((ebx & (1 << 16)) && os_supports_avx512())
				caps |= SIMD_AVX512F;
		}
	}

	return caps;
}

#elif defined(__aarch64__) || defined(_M_ARM64)

#define MKT_ARM64 1

static SimdCapability
detect_simd_arm64(void)
{
	/* NEON is mandatory on AArch64 */
	return SIMD_NEON;
}

#elif defined(__arm__) || defined(_M_ARM)

#define MKT_ARM32 1

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
static SimdCapability
detect_simd_arm32(void)
{
	return SIMD_NEON;
}
#else
static SimdCapability
detect_simd_arm32(void)
{
	return SIMD_NONE;
}
#endif

#endif /* architecture detection */

SimdCapability
mkt_detect_simd(void)
{
	if (g_simd_detected)
		return g_simd_caps;

	SimdCapability detected;

#if defined(MKT_X86)
	detected = detect_simd_x86();
#elif defined(MKT_ARM64)
	detected = detect_simd_arm64();
#elif defined(MKT_ARM32)
	detected = detect_simd_arm32();
#else
	detected = SIMD_NONE;
#endif

	/* Apply override mask if set */
	if (g_simd_override != 0xFFFFFFFF)
		g_simd_caps = detected & g_simd_override;
	else
		g_simd_caps = detected;

	g_simd_detected = true;
	return g_simd_caps;
}

void
mkt_simd_set_override(uint32_t mask)
{
	g_simd_override = mask;
	g_simd_detected = false; /* Force re-detection */
}

void
mkt_simd_reset_cache(void)
{
	g_simd_detected = false;
	g_simd_caps		= SIMD_NONE;
}
