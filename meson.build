project('meerkat', 'c',
  version: '0.1.0',
  default_options: [
    'c_std=c2x',  # C23 (use c2x for older meson/gcc compatibility)
    'warning_level=2',
    'werror=true',
    'b_ndebug=if-release',
  ],
)

cc = meson.get_compiler('c')

# Get build options
simd_mode = get_option('simd')
use_native = get_option('native')

# Common flags (MKT_STANDALONE added per-target, not globally)
add_project_arguments(
  '-D_GNU_SOURCE',
  language: 'c',
)

# Generate mkt_config.h with SIMD mode configuration
# Use set() conditionally - undefined macros appear as /* #undef X */ comments
# Source code uses #ifdef checks (standard autoconf-style pattern)
config_data = configuration_data()
if simd_mode == 'full'
  config_data.set('MKT_SIMD_FULL', 1)
endif
if simd_mode == 'none'
  config_data.set('MKT_SIMD_NONE', 1)
endif
config_data.set_quoted('MKT_SIMD_MODE', simd_mode)

# Disable target_clones in coverage builds - each clone is a separate function
# and only the one matching the CPU gets called, leaving others uncovered
if get_option('b_coverage')
  config_data.set('MKT_COVERAGE', 1)
endif

# Disable ifuncs in sanitizer builds - the ifunc resolver runs before the
# sanitizer runtime is initialized, causing crashes on instrumented globals
sanitize_opt = get_option('b_sanitize')
if sanitize_opt != 'none'
  config_data.set('MKT_SANITIZER', 1)
endif

# CBLAS library (optional, for optimized matrix operations)
# Detected early so MKT_HAVE_CBLAS can be set in config header
# Detection order: vendor-specific -> OpenBLAS -> generic CBLAS
blas_opt = get_option('blas')
cblas_dep = []
have_cblas = false
blas_name = ''

if blas_opt != 'disabled'
  # 1. Intel MKL (via pkg-config)
  if not have_cblas
    mkl_dep = dependency('mkl-dynamic-lp64-iomp', required: false)
    if mkl_dep.found()
      cblas_dep = mkl_dep
      have_cblas = true
      blas_name = 'Intel MKL'
    endif
  endif

  # 2. macOS Accelerate framework
  if not have_cblas and host_machine.system() == 'darwin'
    accelerate_dep = dependency('Accelerate', required: false)
    if accelerate_dep.found()
      cblas_dep = accelerate_dep
      have_cblas = true
      blas_name = 'Apple Accelerate'
      # Use new Accelerate API to avoid deprecation warnings (macOS 13.3+)
      add_project_arguments('-DACCELERATE_NEW_LAPACK', language: 'c')
    endif
  endif

  # 3. ARM Performance Libraries (via environment module or find_library)
  if not have_cblas and host_machine.cpu_family() in ['aarch64', 'arm']
    armpl_dir = run_command('sh', '-c', 'echo $ARMPL_DIR', check: false).stdout().strip()
    if armpl_dir != ''
      armpl_lib = cc.find_library('armpl_lp64',
        dirs: [armpl_dir / 'lib'],
        required: false
      )
      if armpl_lib.found() and cc.has_header('armpl.h', args: '-I' + armpl_dir / 'include_lp64')
        cblas_dep = declare_dependency(
          dependencies: armpl_lib,
          compile_args: ['-I' + armpl_dir / 'include_lp64'],
        )
        have_cblas = true
        blas_name = 'ARM Performance Libraries'
      endif
    endif
    if not have_cblas
      armpl_lib = cc.find_library('armpl_lp64', required: false)
      if armpl_lib.found() and cc.has_header('cblas.h')
        cblas_dep = armpl_lib
        have_cblas = true
        blas_name = 'ARM Performance Libraries'
      endif
    endif
  endif

  # 4. OpenBLAS (via pkg-config or libopenblas)
  if not have_cblas
    openblas_dep = dependency('openblas', required: false)
    if openblas_dep.found()
      cblas_dep = openblas_dep
      have_cblas = true
      blas_name = 'OpenBLAS'
    endif
  endif
  if not have_cblas
    openblas_lib = cc.find_library('openblas', required: false)
    if openblas_lib.found() and cc.has_header('cblas.h')
      cblas_dep = openblas_lib
      have_cblas = true
      blas_name = 'OpenBLAS'
    endif
  endif

  # 5. Generic CBLAS (AMD AOCL, reference BLAS, or other implementations)
  # These provide standard libblas.so/libcblas.so with cblas.h
  if not have_cblas
    cblas_lib = cc.find_library('cblas', required: false)
    blas_lib = cc.find_library('blas', required: false)
    if cblas_lib.found() and blas_lib.found() and cc.has_header('cblas.h')
      cblas_dep = [cblas_lib, blas_lib]
      have_cblas = true
      blas_name = 'CBLAS'
    elif blas_lib.found() and cc.has_header('cblas.h')
      cblas_dep = blas_lib
      have_cblas = true
      blas_name = 'CBLAS'
    endif
  endif

  # Summary message
  if have_cblas
    config_data.set('MKT_HAVE_CBLAS', 1)
    message('BLAS: ' + blas_name + ' (optimized matrix operations enabled)')
  elif blas_opt == 'enabled'
    error('BLAS required but not found')
  else
    message('BLAS: not found (using built-in matrix operations)')
    message('  For better RaBitQ encoding performance, install a BLAS library:')
    if host_machine.cpu_family() in ['x86', 'x86_64']
      message('    OpenBLAS: blas-openblas (Arch), libopenblas-dev (Debian/Ubuntu)')
    elif host_machine.cpu_family() in ['aarch64', 'arm']
      message('    OpenBLAS: libopenblas-dev (Debian/Ubuntu)')
    else
      message('    OpenBLAS is recommended')
    endif
  endif
else
  message('BLAS: disabled by configuration')
endif


# FAISS library (optional, for benchmark comparison)
# Expects FAISS built in ../faiss with the C API patch from third-party/
use_faiss = get_option('faiss')
faiss_dep = []

if use_faiss
  faiss_root = meson.project_source_root() / '..' / 'faiss'
  faiss_build = faiss_root / 'build'
  faiss_lib_dir = faiss_build / 'faiss'
  faiss_capi_dir = faiss_build / 'c_api'
  omp_dep = dependency('openmp', required: false)

  faiss_dep = declare_dependency(
    compile_args: [
      '-I' + faiss_root / 'c_api',
      '-I' + faiss_root,
    ],
    link_args: [
      '-L' + faiss_capi_dir,
      '-L' + faiss_lib_dir,
      '-lfaiss_c',
      '-lfaiss',
      '-lstdc++',
    ],
    dependencies: [omp_dep] + (have_cblas ? [cblas_dep] : []),
  )
  config_data.set('MKT_HAVE_FAISS', 1)
  message('FAISS integration enabled (using ' + faiss_root + ')')
endif

# HDF5 library (optional, for loading ann-benchmarks datasets)
hdf5_opt = get_option('hdf5')
hdf5_dep = dependency('hdf5', language: 'c', required: hdf5_opt)
use_hdf5 = hdf5_dep.found()

if use_hdf5
  config_data.set('MKT_HAVE_HDF5', 1)
  message('HDF5 integration enabled')
else
  message('HDF5 not found: bench search --hdf5 and verify rabitq disabled')
  hdf5_dep = []
endif

# F16C: hardware half-precision ↔ float conversion (available on all AVX2
# CPUs, i.e. Haswell 2013+). Enables _cvtsh_ss / _cvtss_sh scalar intrinsics
# globally so that inline half↔float conversion in mkt_halfvec.h compiles to
# single vcvtsh2ss/vcvtss2sh instructions instead of software fallback.
if simd_mode != 'none' and cc.compiles(
  '#include <immintrin.h>\nfloat f(unsigned short h) { return _cvtsh_ss(h); }',
  args: ['-mf16c'],
  name: 'F16C half-precision conversion',
)
  add_project_arguments('-mf16c', language: 'c')
  config_data.set('MKT_HAVE_F16C', 1)
endif

mkt_config_h = configure_file(
  input: 'src/mkt_config.h.in',
  output: 'mkt_config.h',
  configuration: config_data,
)

# Embed the git commit hash at build time. vcs_tag re-runs when HEAD
# moves, so a plain `ninja` picks up a fresh commit without a full
# reconfigure. Falls back to "unknown" outside of a git checkout.
git_commit_h = vcs_tag(
  input: 'src/git_commit.h.in',
  output: 'git_commit.h',
  fallback: 'unknown',
  command: ['git', 'rev-parse', 'HEAD'],
  replace_string: '@VCS_TAG@',
)

# Generate .clang-format-ignore to exclude build directory from formatting
configure_file(
  input: 'clang-format-ignore.in',
  output: '.clang-format-ignore',
  copy: true,
)

# Native build: -march=native for local optimization (non-portable)
if use_native
  add_project_arguments('-march=native', language: 'c')
endif

# Compiler vectorization flags (for simd=full and simd=compiler, NOT for simd=none)
# These enable effective auto-vectorization with GCC/Clang for target_clones baseline
if simd_mode != 'none'
  add_project_arguments(
    '-ftree-vectorize',        # Explicit auto-vectorization (GCC)
    '-fassociative-math',      # Allow reassociation for better vectorization
    '-fno-signed-zeros',       # IEEE 754 relaxation for performance
    '-fno-trapping-math',      # No FP exceptions for performance
    # GCC defaults to -ffp-contract=off in ISO C modes (-std=c23, -std=c2x)
    # because the C standard leaves FP contraction implementation-defined.
    # GNU modes (-std=gnu23) default to -ffp-contract=fast. Without this,
    # GCC generates separate vmulps + vaddps instead of vfmadd231ps, which
    # is ~3x slower for dot-product loops (no FMA, plus horizontal reduction
    # per vector chunk instead of accumulate-and-reduce-once).
    # See: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
    '-ffp-contract=fast',
    language: 'c',
  )

endif

# Math library
m_dep = cc.find_library('m', required: true)

# Source files for core library (standalone, no PostgreSQL)
mkt_core_sources = files(
  'src/core/memory_standalone.c',
  'src/core/platform.c',
  'src/standalone/thread_pool.c',
  'src/mkt_halfvec.c',
  'src/mkt_vector.c',
  'src/algo/distance.c',
  'src/algo/kmeans.c',
  'src/algo/kmeans_elkan.c',
  'src/algo/kmeans_hamerly.c',
  'src/algo/kmeans_lloyd.c',
  'src/algo/vecops.c',
  'src/quant/rabitq.c',
  'src/quant/fastscan.c',
  'src/quant/matrix.c',
  'src/algo/topk.c',
  'src/algo/hkmeans.c',
  'src/index/centroid_build.c',
  'src/index/centroid_page.c',
  'src/index/centroid_search.c',
  'src/index/posting_build_parallel.c',
  'src/index/index_build.c',
  'src/index/posting_page.c',
  'src/index/posting_build.c',
  'src/index/posting_scan.c',
  'src/index/posting_convert.c',
  'src/index/query_scan.c',
  'src/standalone/vector_source.c',
  'src/standalone/index.c',
  'src/standalone/query.c',
  'src/standalone/api.c',
)

# Include directories
# Include build directory for generated mkt_config.h
src_inc = include_directories('src', '.')

# SIMD source files and libraries
# With file-level target attributes, SIMD files can be compiled as regular sources
# Exception: 32-bit ARM needs -mfpu=neon flag, kept as separate library
simd_sources = []
simd_libs = []

if simd_mode == 'full'
  # Hand-optimized SIMD implementations (simd=full only)
  if host_machine.cpu_family() in ['x86', 'x86_64']
    # AVX2 and AVX-512 use file-level target pragmas, compile as regular sources
    simd_sources += files(
      'src/algo/distance_avx2.c',
      'src/algo/distance_avx512.c',
      'src/quant/rabitq_avx2.c',
      'src/quant/rabitq_avx512.c',
      'src/quant/fastscan_avx2.c',
      'src/quant/fastscan_avx512.c',
    )
  elif host_machine.cpu_family() == 'aarch64'
    # AArch64: NEON is mandatory, no special flags needed
    simd_sources += files(
      'src/algo/distance_neon.c',
      'src/quant/rabitq_neon.c',
      'src/quant/fastscan_neon.c',
    )
  elif host_machine.cpu_family() == 'arm'
    # 32-bit ARM: needs -mfpu=neon flag, keep as separate library
    mkt_neon_lib = static_library('mkt_neon',
      [
        'src/algo/distance_neon.c',
        'src/quant/rabitq_neon.c',
        'src/quant/fastscan_neon.c',
      ],
      c_args: ['-mfpu=neon'],
      include_directories: src_inc,
    )
    simd_libs += mkt_neon_lib
  endif
endif

# Combine all sources
mkt_all_sources = mkt_core_sources + simd_sources

thread_dep = dependency('threads')

# Core dependency: compile sources directly instead of static library
# This avoids the 5-10% overhead from static library compilation
mkt_core_dep = declare_dependency(
  sources: mkt_all_sources,
  compile_args: ['-DMKT_STANDALONE'],
  link_with: simd_libs,
  include_directories: src_inc,
  dependencies: [m_dep, thread_dep] + (have_cblas ? [cblas_dep] : []),
)

# CLI tool
mkt_cli_sources = files(
  'src/cli/main.c',
  'src/cli/bench_distance.c',
  'src/cli/bench_quantize.c',
  'src/cli/bench_cluster.c',
  'src/cli/bench_search.c',
  'src/cli/bench_rabitq_kernel.c',
  'src/cli/bench_page_score.c',
  'src/cli/bench_fastscan_kernel.c',
  'src/cli/distance_pgvector.c',
  'src/cli/kmeans_pgvector.c',
)

if use_hdf5
  mkt_cli_sources += files('src/standalone/hdf5_source.c')
endif

if use_faiss and use_hdf5
  mkt_cli_sources += files('src/cli/verify_rabitq.c')
endif

mkt_exe = executable('mkt',
  mkt_cli_sources,
  dependencies: [mkt_core_dep]
    + (use_faiss ? [faiss_dep] : [])
    + (use_hdf5 ? [hdf5_dep] : []),
  include_directories: src_inc,
  install: false,
)

# Shared library for Python bindings (ann-benchmarks, etc.)
# Skip in sanitizer builds — the shared library can't link the
# sanitizer runtime (only executables get it automatically).
if sanitize_opt == 'none'
  mkt_bindings_lib = shared_library('mkt',
    files('src/standalone/api.c'),
    dependencies: [mkt_core_dep],
    include_directories: src_inc,
    install: false,
  )
endif

# Create symlink in ./bin for easy local execution
bin_dir = meson.project_source_root() / 'bin'
mkt_symlink = custom_target('mkt-symlink',
  output: 'mkt-link-stamp',
  command: [
    'sh', '-c',
    'mkdir -p @0@ && ln -sf @1@ @2@ && touch @OUTPUT@'.format(
      bin_dir,
      meson.project_build_root() / 'mkt',
      bin_dir / 'mkt'
    )
  ],
  build_by_default: true,
  depends: mkt_exe,
)

# Unit tests
subdir('test/unit')

# ============================================================================
# PostgreSQL extension
# ============================================================================

opt_postgresql = get_option('postgresql')
pg_required = opt_postgresql.enabled()

build_pg_extension = false

if not opt_postgresql.disabled()
  pg_config_path = get_option('pg_config')
  if pg_config_path != ''
    pg_config = find_program(pg_config_path, required: true)
  else
    pg_config = find_program('pg_config', required: pg_required)
  endif

  if pg_config.found()
    build_pg_extension = true
  elif not pg_required
    warning('pg_config not found. PostgreSQL extension will not be built.')
    warning('Install PostgreSQL or set -Dpg_config=/path/to/pg_config')
  endif
endif

# Verify compiler family matches pg_config --cc
# Run the PG compiler with --version to identify it (gcc vs clang),
# rather than comparing binary names which vary across systems
# (cc, gcc, gcc-13, x86_64-linux-gnu-gcc, etc.)
if build_pg_extension
  pg_cc = run_command(pg_config, '--cc', check: true).stdout().strip()
  pg_cc_version = run_command('sh', '-c', pg_cc + ' --version 2>&1 || true',
    check: true).stdout().to_lower()

  if pg_cc_version.contains('clang')
    pg_compiler_id = 'clang'
  else
    pg_compiler_id = 'gcc'
  endif

  if pg_compiler_id != cc.get_id()
    if pg_required or pg_config_path != ''
      error('Compiler mismatch: PostgreSQL was built with ' + pg_cc +
            ' (' + pg_compiler_id + ') but meson is using ' + cc.get_id() + '.\n' +
            'Reconfigure with: CC=' + pg_cc + ' meson setup builddir --reconfigure')
    else
      warning('Compiler mismatch: PostgreSQL was built with ' + pg_cc +
              ' (' + pg_compiler_id + ') but meson is using ' + cc.get_id() +
              '. Skipping extension.')
      build_pg_extension = false
    endif
  endif
endif

if build_pg_extension
  # Get PostgreSQL paths
  pg_includedir_server = run_command(pg_config, '--includedir-server', check: true).stdout().strip()
  pg_pkglibdir = run_command(pg_config, '--pkglibdir', check: true).stdout().strip()
  pg_sharedir = run_command(pg_config, '--sharedir', check: true).stdout().strip()
  pg_libdir = run_command(pg_config, '--libdir', check: true).stdout().strip()
  pg_bindir = run_command(pg_config, '--bindir', check: true).stdout().strip()

  # Parse CFLAGS and CPPFLAGS, skipping C90-era restriction.
  # CPPFLAGS is needed for macOS Homebrew where gettext (libintl.h) is
  # keg-only and the include path is only in --cppflags.
  pg_cflags_raw = run_command(pg_config, '--cflags', check: true).stdout().strip().split()
  pg_cppflags_raw = run_command(pg_config, '--cppflags', check: true).stdout().strip().split()
  pg_cflags = []
  foreach flag : pg_cflags_raw + pg_cppflags_raw
    if flag == '-Wdeclaration-after-statement'
      continue
    endif
    if cc.has_argument(flag) or flag.startswith('-I') or flag.startswith('-D')
      pg_cflags += [flag]
    endif
  endforeach

  # Check for PostgreSQL server development headers before compiling anything
  if not cc.has_header('postgres.h', args: ['-I' + pg_includedir_server] + pg_cflags)
    if pg_required
      error('PostgreSQL server headers not found at: ' + pg_includedir_server +
            '. Install postgresql-server-dev.')
    endif
    warning('PostgreSQL server headers not found at: ' + pg_includedir_server)
    warning('Install postgresql-server-dev or set -Dpostgresql=disabled')
    build_pg_extension = false
  endif

  # Require PostgreSQL 18+
  if build_pg_extension
    pg_version = run_command(pg_config, '--version', check: true).stdout().strip()
    # pg_config --version returns e.g. "PostgreSQL 18.1" or "PostgreSQL 18devel"
    pg_version_num = pg_version.split(' ')[1].split('.')[0].split('d')[0].split('r')[0].to_int()
    if pg_version_num < 18
      if pg_required or pg_config_path != ''
        error('Meerkat requires PostgreSQL 18+, found: ' + pg_version)
      else
        warning('Meerkat requires PostgreSQL 18+, found: ' + pg_version + '. Skipping extension.')
        build_pg_extension = false
      endif
    endif
  endif
endif

if build_pg_extension
  # Parse LDFLAGS, extracting rpath separately
  pg_ldflags_raw = run_command(pg_config, '--ldflags', check: true).stdout().strip()
  pg_ldflags = []
  pg_rpath = []
  foreach flag : pg_ldflags_raw.split()
    if flag.startswith('-Wl,-rpath,')
      rpath_val = flag.split(',')[2].strip('\'\"')
      pg_rpath += [rpath_val]
    elif flag != '-Wl,--enable-new-dtags'
      pg_ldflags += [flag]
    endif
  endforeach

  # Core sources compiled for PG (without MKT_STANDALONE, without
  # memory_standalone.c)
  mkt_pg_core_sources = files(
    'src/core/platform.c',
    'src/mkt_vector.c',
    'src/mkt_halfvec.c',
    'src/algo/distance.c',
    'src/algo/kmeans.c',
    'src/algo/kmeans_elkan.c',
    'src/algo/kmeans_hamerly.c',
    'src/algo/kmeans_lloyd.c',
    'src/algo/vecops.c',
    'src/quant/rabitq.c',
    'src/quant/fastscan.c',
    'src/quant/matrix.c',
    'src/algo/hkmeans.c',
    'src/index/centroid_build.c',
    'src/index/centroid_page.c',
    'src/index/centroid_search.c',
    'src/index/posting_build_parallel.c',
    'src/index/index_build.c',
    'src/algo/topk.c',
    'src/index/posting_page.c',
    'src/index/posting_build.c',
    'src/index/posting_scan.c',
    'src/index/posting_convert.c',
    'src/index/query_scan.c',
  )

  # SIMD sources for PG build (same as standalone)
  mkt_pg_simd_sources = []
  if simd_mode == 'full'
    if host_machine.cpu_family() in ['x86', 'x86_64']
      mkt_pg_simd_sources += files(
        'src/algo/distance_avx2.c',
        'src/algo/distance_avx512.c',
        'src/quant/rabitq_avx2.c',
        'src/quant/rabitq_avx512.c',
        'src/quant/fastscan_avx2.c',
        'src/quant/fastscan_avx512.c',
      )
    elif host_machine.cpu_family() == 'aarch64'
      mkt_pg_simd_sources += files(
        'src/algo/distance_neon.c',
        'src/quant/rabitq_neon.c',
        'src/quant/fastscan_neon.c',
      )
    endif
  endif

  mkt_pg_cflags = ['-I' + pg_includedir_server] + pg_cflags

  mkt_pg_lib = static_library('mkt_pg',
    mkt_pg_core_sources + mkt_pg_simd_sources,
    c_args: mkt_pg_cflags,
    include_directories: src_inc,
    dependencies: [m_dep] + (have_cblas ? [cblas_dep] : []),
    override_options: ['werror=false'],
  )

  subdir('src/pg')
  subdir('test/pg')
endif

# Header files
mkt_core_headers = files(
  'src/core/memory.h',
  'src/core/memory_standalone.h',
  'src/core/memory_pg.h',
  'src/core/platform.h',
  'src/mkt_halfvec.h',
  'src/mkt_types.h',
  'src/mkt_vector.h',
)

# Format target (uses pre-commit for consistency)
pre_commit = find_program('pre-commit', required: false)
if pre_commit.found()
  run_target('format',
    command: [pre_commit, 'run', 'clang-format', '--all-files'],
  )
endif

# Coverage (built-in meson support)
# Usage:
#   meson setup builddir-cov -Db_coverage=true
#   meson test -C builddir-cov
#   ninja -C builddir-cov coverage-html
#
# Meson auto-detects compiler (GCC/Clang) and coverage tools (gcovr, lcov, llvm-cov)

# Static analysis with clang-tidy
# Requires compile_commands.json (auto-generated by meson in builddir)
clang_tidy = find_program('clang-tidy', required: false)
if clang_tidy.found()
  run_target('tidy',
    command: [
      clang_tidy,
      '-p', meson.project_build_root(),
      mkt_core_sources,
    ],
  )
endif

# Static analysis with scan-build (Clang Static Analyzer)
# Usage: scan-build meson compile -C builddir
# Or with report: scan-build -o reports meson compile -C builddir

# Pre-commit hooks
pre_commit = find_program('pre-commit', required: false)
if pre_commit.found()
  # Install hooks automatically during setup (may fail in sandbox, that's OK)
  install_result = run_command(pre_commit, 'install', '--install-hooks',
    check: false, capture: true)

  if install_result.returncode() == 0
    message('Pre-commit hooks installed successfully')
  else
    warning('Failed to install pre-commit hooks. Run: pre-commit install')
  endif

  # Manual targets
  run_target('pre-commit',
    command: [pre_commit, 'run', '--all-files'],
  )

  run_target('pre-commit-install',
    command: [pre_commit, 'install', '--install-hooks'],
  )
else
  warning('pre-commit not found. Install with: pip install pre-commit')
endif
