// Enable core dumps for the job (see coredumps.sh). The matching post step
// (post.js) prints backtraces at job end.
const { execFileSync } = require('child_process');
const { join } = require('path');

execFileSync('bash', [join(__dirname, 'coredumps.sh'), 'enable'], {
  stdio: 'inherit',
});
