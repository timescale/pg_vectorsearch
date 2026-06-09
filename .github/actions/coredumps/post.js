// Runs automatically at job end: print a gdb backtrace for any core dump a
// crash left behind (no-op when there are none).
const { execFileSync } = require('child_process');
const { join } = require('path');

execFileSync('bash', [join(__dirname, 'coredumps.sh'), 'analyze'], {
  stdio: 'inherit',
});
