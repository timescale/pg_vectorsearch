# Unit tests

unit_test_sources = files(
  'mkt_test.c',
  'run_tests.c',
  'test_fixtures.c',
  'test_memory.c',
  'test_platform.c',
  'test_vector.c',
)

test_runner = executable('run_tests',
  unit_test_sources,
  dependencies: [mkt_core_dep],
  include_directories: include_directories('.'),
  install: false,
)

# Run all tests as one suite
test('unit_tests', test_runner,
  args: ['--tap'],
  protocol: 'tap',
  workdir: meson.project_source_root(),
  timeout: 60,
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
      timeout: 30,
      suite: 'unit',
    )
  endif
endforeach
