# Unit tests

# Check for GSL (optional - for reference validation tests)
gsl_dep = dependency('gsl', required: false)
if gsl_dep.found()
  message('GSL found: Reference validation tests will be enabled')
else
  message('GSL not found: Reference validation tests will be skipped')
  message('  Install GSL to enable: sudo pacman -S gsl (Arch) or sudo apt install libgsl-dev (Debian/Ubuntu)')
endif

# Generate test config header
test_config = configuration_data()
test_config.set('HAVE_GSL', gsl_dep.found() ? 1 : 0)

test_config_h = configure_file(
  output: 'test_config.h',
  configuration: test_config
)

unit_test_sources = files(
  'mkt_test.c',
  'run_tests.c',
  'test_barrier.c',
  'test_shm_toc.c',
  'test_centroid_page.c',
  'test_centroid_search.c',
  'test_fastscan.c',
  'test_posting_page.c',
  'test_distance.c',
  'test_fixtures.c',
  'test_halfvec.c',
  'test_hkmeans.c',
  'test_kmeans.c',
  'test_memory.c',
  'test_platform.c',
  'test_rabitq.c',
  'test_secondary_batch.c',
  'test_standalone.c',
  'test_thread_pool.c',
  'test_topk.c',
  'test_vecops.c',
  'test_vector.c',
)

test_dependencies = [mkt_core_dep]
if gsl_dep.found()
  test_dependencies += gsl_dep
endif

# Create include directory for builddir (where test_config.h is generated)
test_inc = include_directories('.')

test_runner = executable('run_tests',
  [unit_test_sources, test_config_h],
  dependencies: test_dependencies,
  include_directories: test_inc,
  install: false,
)

# Run all tests as one suite (timeout must accommodate sanitizer builds
# which run ~5-10x slower than normal)
test('unit_tests', test_runner,
  args: ['--tap'],
  protocol: 'tap',
  workdir: meson.project_source_root(),
  timeout: 300,
  suite: 'all',
)

# Auto-discover test groups from source files using Python
# Python filters for test_*.c files and extracts TEST_GROUP() declarations
python = find_program('python3', required: true)
test_groups_result = run_command(
  python,
  '-c',
  '''import re, sys, os
groups = set()
for f in sys.argv[1:]:
    if os.path.basename(f).startswith("test_"):
        for m in re.finditer(r"TEST_GROUP\((\w+)\)", open(f).read()):
            groups.add(m.group(1))
for g in sorted(groups):
    print(g)
''',
  unit_test_sources,
  check: true,
)

# Split output into lines and register a test for each group
test_groups = test_groups_result.stdout().strip().split('\n')
foreach group : test_groups
  if group != ''
    test(group, test_runner,
      args: ['--tap', '--group', group],
      protocol: 'tap',
      workdir: meson.project_source_root(),
      timeout: 120,
      suite: 'unit',
    )
  endif
endforeach
