# A known race between a split and an insert that rolls back.
#
# A split keeps rows whose insert has not committed, since they may still
# commit, and reads their vectors from TOAST. If the insert rolls back between
# the split fetching the row and reading its vector, on-access pruning can
# remove the vector's chunks first, and the split fails with "missing chunk
# number". It is accepted rather than fixed: the window is narrow, and a
# retry succeeds. This test reproduces it, so a change in either direction --
# the race fixed, or failing some other way -- shows up.
#
# Pruning only removes the chunks if the TOAST page is nearly full and already
# marked prunable. The fixture arranges that: every vector is one ~2 KB chunk,
# four to a page; 24 indexed rows fill pages 0-5, and three rows with only a
# padding value put three chunks on page 6, one of them deleted. The
# uncommitted insert then takes the last slot on page 6.

use strict;
use warnings;

use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $node = PostgreSQL::Test::Cluster->new('toast_race');
$node->init;
# Autovacuum on the TOAST table would clean page 6 before the race.
$node->append_conf('postgresql.conf', 'autovacuum = off');
$node->start;

if ($node->psql('postgres', 'CREATE EXTENSION injection_points') != 0)
{
	plan skip_all => 'injection_points extension not available';
}

$node->safe_psql('postgres', 'CREATE EXTENSION pg_vectorsearch');
$node->safe_psql('postgres',
	'ALTER DATABASE postgres SET search_path = public');

# Each statement separately: the deleted row has to be deleted by its own,
# committed transaction for its chunk to be prunable.
$node->safe_psql('postgres', $_) for (
	q{CREATE TABLE tp (id int, v vec32(498), pad text,
	                   note text DEFAULT repeat('n', 16))
	      WITH (toast_tuple_target = 128)},
	q{ALTER TABLE tp ALTER COLUMN v SET STORAGE EXTERNAL},
	q{ALTER TABLE tp ALTER COLUMN pad SET STORAGE EXTERNAL},
	q{INSERT INTO tp (id, v)
	      SELECT i, (SELECT array_agg(((i % 2) * 10 + sin(i * 0.1 + j * 0.7))::real)
	                 FROM generate_series(0, 497) j)::vec32(498)
	      FROM generate_series(1, 24) i},
	q{INSERT INTO tp (id, pad)
	      SELECT 100 + i, repeat(chr(65 + i), 1996) FROM generate_series(1, 3) i},
	q{CREATE INDEX tp_idx ON tp USING prism (v)
	      WITH (nlist = 1, centroid_fastscan = off)},
	q{DELETE FROM tp WHERE id = 101});

my $ins = $node->background_psql('postgres');
$ins->query_safe(q{
	BEGIN;
	INSERT INTO tp (id, v)
	    SELECT 999, (SELECT array_agg((10 + sin(j * 0.7))::real)
	                 FROM generate_series(0, 497) j)::vec32(498);
});

my $split = $node->background_psql('postgres', on_error_stop => 0);
$split->query_safe(q{
	SELECT injection_points_set_local();
	SELECT injection_points_attach('prism-split-fetched-uncommitted', 'wait');
});
$split->query_until(qr//, "CALL prism_rebalance('tp_idx', 10);\n");

ok( $node->poll_query_until(
		'postgres', q{
	SELECT EXISTS (SELECT FROM pg_stat_activity
	    WHERE wait_event_type = 'InjectionPoint'
	      AND wait_event = 'prism-split-fetched-uncommitted')}),
	'the split paused after fetching the uncommitted row');

$ins->query_safe('ROLLBACK');
$node->safe_psql('postgres',
	"SELECT injection_points_wakeup('prism-split-fetched-uncommitted')");

# The CALL's error, if any, arrives with the next query's output. Only the
# ERROR line counts: a split that succeeds reports itself as a NOTICE, which
# psql also writes to stderr.
$split->query('SELECT 1');
my ($err) = $split->{stderr} =~ /(ERROR:.*)/;
$err //= '';

ok($err eq '' || $err =~ /missing chunk number \d+ for toast value/,
	'the split fails only the known way, if at all')
  or diag($err);

TODO:
{
	local $TODO = 'accepted race: pruning can remove a rolled-back '
	  . "insert's chunks mid-read";
	is($err, '', 'the split survives an insert rolling back mid-read');
}

$split->quit;
$ins->quit;

# The old list is still in place, so a retry splits it and every row stays
# reachable through the index.
$node->safe_psql('postgres', "CALL prism_rebalance('tp_idx', 10)");
cmp_ok(
	$node->safe_psql(
		'postgres',
		q{SELECT count(*) FROM prism_posting_pages('tp_idx'::regclass)
		  WHERE is_first}),
	'>', 1,
	'a retry splits the list');
is( $node->safe_psql(
		'postgres', q{
	SET enable_seqscan = off;
	SET prism.nprobe = 10000;
	SELECT count(*) FROM (
	    SELECT id FROM tp ORDER BY v <-> (SELECT v FROM tp WHERE id = 1)
	    LIMIT 100) t}),
	'24',
	'every indexed row is reachable after the retry');

$node->stop;
done_testing();
