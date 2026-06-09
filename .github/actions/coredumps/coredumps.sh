#!/usr/bin/env bash
# Reusable CI core-dump capture and analysis (Linux only).
#
#   enable   route cores to /tmp/cores and ensure gdb is installed
#   analyze  print a gdb backtrace for every core left in /tmp/cores
#
# Run the tests themselves with `ulimit -c unlimited` (the shared CI scripts
# do) so a crash actually writes a core.
set -euo pipefail

[ "${RUNNER_OS:-Linux}" = "Linux" ] || exit 0

case "${1:-}" in
enable)
  command -v gdb >/dev/null 2>&1 || sudo apt-get install -y gdb
  sudo mkdir -p /tmp/cores
  sudo chmod 1777 /tmp/cores
  # Override the runner default (systemd-coredump/apport pipe) so cores land
  # as plain files gdb can read.
  echo '/tmp/cores/core.%e.%p' | sudo tee /proc/sys/kernel/core_pattern
  ;;
analyze)
  shopt -s nullglob
  cores=(/tmp/cores/core.*)
  if [ ${#cores[@]} -eq 0 ]; then
    echo "No core dumps in /tmp/cores."
    exit 0
  fi
  for core in "${cores[@]}"; do
    # Recover the crashing executable's path from the core itself, so this
    # works for any process (postgres backend, unit test runner, the CLI).
    exe=$(file "$core" | sed -n "s/.*execfn: '\([^']*\)'.*/\1/p")
    echo "=== backtrace: $core (exe: ${exe:-unknown}) ==="
    if [ -n "$exe" ] && [ -f "$exe" ]; then
      gdb -q -batch -ex 'set pagination off' \
        -ex 'thread apply all bt full' "$exe" --core="$core" 2>&1 | head -300
    else
      gdb -q -batch -ex 'set pagination off' \
        -ex 'thread apply all bt full' --core="$core" 2>&1 | head -300
    fi
  done
  ;;
*)
  echo "usage: coredumps.sh enable|analyze" >&2
  exit 2
  ;;
esac
