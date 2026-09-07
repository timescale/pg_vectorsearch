# A crash inside the split's flip window.
#
# The split builds fresh chains and only then repoints the centroid leaf at
# them, in one page write. That ordering is the crash-safety argument: before
# the flip the old list is authoritative, after it the new ones are, and there
# is no in-between state on disk. A crash in the window should therefore lose
# nothing and corrupt nothing -- it can only leak the new pages, which no
# centroid entry points at.
#
# The cluster ids the new lists were going to use are minted from the leaf
# count and written back to the metapage before the leaves exist. Whether that
# reservation is still there after the crash depends on whether its WAL had
# reached disk, so the invariant to check is one-sided: nlist may run ahead of
# the leaf count, and must never fall behind it. Running ahead only wastes
# ids; falling behind would hand the same id to two lists.
#
# The window is entered with an injection point rather than by timing, and the
# cluster is killed with an immediate shutdown so the restart has to replay
# WAL.

use strict;
use warnings;

use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $rows   = 600;
my $target = 60;
my $k      = 10;

my $node = PostgreSQL::Test::Cluster->new('crash');
$node->init;
$node->start;

if ($node->psql('postgres', 'CREATE EXTENSION injection_points') != 0)
{
	plan skip_all => 'injection_points extension not available';
}

$node->safe_psql('postgres', 'CREATE EXTENSION meerkat');
$node->safe_psql('postgres',
	'ALTER DATABASE postgres SET search_path = mkt, public');

# Points spread along one axis: deterministic, and with enough structure that
# clustering has something to find.
$node->safe_psql('postgres', <<"SQL");
CREATE TABLE c (id int, v vector(3));
INSERT INTO c SELECT g, format('[%s,0,0]', g)::vector
    FROM generate_series(1, $rows) g;
CREATE INDEX c_idx ON c USING mktann (v)
    WITH (nlist = 1, centroid_fastscan = off);
SQL

# What the index must still answer after the crash. Taken by brute force so
# the expectation does not come from the structure under test.
my $truth = $node->safe_psql('postgres', <<"SQL");
SET enable_indexscan = off;
SELECT string_agg(id::text, ',' ORDER BY id) FROM (
    SELECT id FROM c ORDER BY v <-> '[1,0,0]'::vector LIMIT $k) s;
SQL

my $reader = <<"SQL";
SET enable_seqscan = off;
SET mkt.nprobe = 10000;
SELECT string_agg(id::text, ',' ORDER BY id) FROM (
    SELECT id FROM c ORDER BY v <-> '[1,0,0]'::vector LIMIT $k) s;
SQL

my $heads = "SELECT count(*) FROM posting_pages('c_idx'::regclass) "
          . "WHERE is_first";
my $ids   = "SELECT count(DISTINCT cluster_id) "
          . "FROM posting_pages('c_idx'::regclass) WHERE is_first";
my $nlist = "SELECT setting::int FROM index_settings('c_idx'::regclass) "
          . "WHERE name = 'nlist'";

my $heads_before = $node->safe_psql('postgres', $heads);
is($node->safe_psql('postgres', $reader), $truth,
	'before the crash: index agrees with brute force');

# Enter the window and stay there. The attachment is left global rather than
# session-local: nothing else in this test splits, so the only session that
# can reach the point is the one running the split.
$node->safe_psql('postgres',
	"SELECT injection_points_attach('mktann-split-before-flip', 'wait')");

# Fire the split and do not wait for it -- an empty pattern returns as soon
# as the query has been sent, leaving it parked on the injection point.
my $bg = $node->background_psql('postgres', on_error_stop => 0);
$bg->query_until(qr//, "CALL rebalance('c_idx', $target);\n");

my $parked = $node->poll_query_until('postgres', <<'SQL');
SELECT count(*) > 0 FROM pg_stat_activity
 WHERE wait_event_type = 'InjectionPoint'
   AND wait_event = 'mktann-split-before-flip'
SQL
if (!$parked)
{
	diag('sessions: ' . $node->safe_psql('postgres', <<'SQL'));
SELECT string_agg(pid || ' ' || state || ' ' ||
                  coalesce(wait_event_type, '-') || '/' ||
                  coalesce(wait_event, '-') || ' ' || left(query, 60),
                  E'\n')
  FROM pg_stat_activity WHERE backend_type = 'client backend'
SQL
	die 'split never reached the flip window';
}

# The chains are written and committed at this point; the tree still points at
# the old head. Kill the cluster hard so the restart has to recover.
$node->stop('immediate');
$node->start;

# Nothing was lost: the old list is still the authoritative one and answers
# exactly as it did before.
is($node->safe_psql('postgres', $reader), $truth,
	'after the crash: index still agrees with brute force');

my $heads_after = $node->safe_psql('postgres', $heads);
is($heads_after, $heads_before,
	"after the crash: the flip did not happen ($heads_after heads)");

is($node->safe_psql('postgres', $ids), $heads_after,
	'after the crash: every reachable list has its own cluster id');

my $nlist_after = $node->safe_psql('postgres', $nlist);
cmp_ok($nlist_after, '>=', $heads_after,
	"after the crash: nlist is not behind the leaf count "
  . "($nlist_after >= $heads_after)");

# And the index is not wedged: a fresh pass splits the list it failed to split
# before, and the answer is still right afterwards.
$node->safe_psql('postgres', <<"SQL");
SET client_min_messages = warning;
CALL rebalance('c_idx', $target);
SQL

my $heads_retry = $node->safe_psql('postgres', $heads);
cmp_ok($heads_retry, '>', $heads_before,
	"after the crash: a later split succeeds ($heads_before -> $heads_retry)");
is($node->safe_psql('postgres', $ids), $heads_retry,
	'after the retry: cluster ids are still distinct');
is($node->safe_psql('postgres', $reader), $truth,
	'after the retry: index agrees with brute force');

$node->stop;
done_testing();
