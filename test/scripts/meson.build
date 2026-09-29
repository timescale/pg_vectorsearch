# Tests for the scripts in scripts/ and scripts/ci/.
#
# Shell rather than C, and outside build_tools: what needs proving is
# the git and gh behaviour of scripts CI runs, which no build artifact
# is involved in. The framework and fixtures are in lib.sh; see its
# header for how a fixture stands in for a remote.
#
# One test per test_*.sh, discovered rather than listed, so adding a
# file needs no edit here -- the same approach test/unit takes for its
# test groups.

bash_prog = find_program('bash', required: false)
git_prog = find_program('git', required: false)
python_prog = find_program('python3', required: false)

if bash_prog.found() and git_prog.found() and python_prog.found()
  script_tests = run_command(
    python_prog,
    '-c',
    '''import glob, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "test_*.sh"))):
    print(os.path.basename(f)[:-3])
''',
    meson.current_source_dir(),
    check: true,
  ).stdout().strip().split('\n')

  foreach script_test : script_tests
    if script_test != ''
      test(script_test,
        bash_prog,
        args: [meson.current_source_dir() / (script_test + '.sh')],
        protocol: 'tap',
        workdir: meson.project_source_root(),
        timeout: 120,
        suite: 'scripts',
      )
    endif
  endforeach
else
  message('bash, git or python3 missing: script tests disabled')
endif
