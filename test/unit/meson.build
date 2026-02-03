# Unit tests

unit_test_sources = files(
  'mkt_test.c',
  'run_tests.c',
  'test_mkt_memory.c',
  'test_mkt_platform.c',
  'test_mkt_vector.c',
)

test_runner = executable('run_tests',
  unit_test_sources,
  dependencies: [mkt_core_dep],
  include_directories: include_directories('.'),
  install: false,
)

test('unit_tests', test_runner,
  workdir: meson.project_source_root(),
  timeout: 60,
)
