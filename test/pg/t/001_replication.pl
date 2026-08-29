# Streaming replication of mktann indexes, across page formats and the
# operations that write them.
#
# meerkat pages are not standard-layout: entries live in the content area
# between pd_lower and pd_upper, which PostgreSQL otherwise treats as free
# space and omits from a standard full-page image. Anything that WAL-logs
# such a page as standard leaves a standby with the content zeroed while the
# page opaque still reports it present -- wrong answers on the standby, with
# nothing wrong on the primary, which still has its pages on disk.
#
# Every index here is built AFTER the standby is streaming, so the standby's
# only source for it is the WAL. Building before the base backup would copy
# the pages across as files and prove nothing.
#
# The invariant checked is that the same index, query and settings give the
# same answer on both nodes. Comparing the two index scans rather than the
# standby against brute force keeps it exact: an ANN index is approximate by
# design, so a recall comparison would be fuzzy, while two reads of the same
# index data cannot legitimately differ.

use strict;
use warnings;

use PostgreSQL::Test::Cluster;
use PostgreSQL::Test::Utils;
use Test::More;

my $dim  = 32;
my $rows = 2000;
my $k    = 10;

# Query with a row's own vector rather than the origin: this data is
# generated from sin(), so every point sits at roughly the same distance from
# the origin and "nearest" would barely mean anything. Against a real row
# there is an unambiguous nearest neighbour -- the row itself, at distance
# zero.
my $qvec = '(SELECT v FROM %s WHERE id = 1)';

my $primary = PostgreSQL::Test::Cluster->new('primary');
$primary->init(allows_streaming => 1);
$primary->start;
$primary->safe_psql('postgres', 'CREATE EXTENSION meerkat');

# Set the search path once, at database level, so every session on both nodes
# resolves meerkat's types and distance operators the same way. Note this also
# means new tables land in the mkt schema, since it is first.
$primary->safe_psql('postgres',
	'ALTER DATABASE postgres SET search_path = mkt, public');

# Two tables so both vector types are covered; halfvec drives the
# half-precision centroid format, which no vector column can reach.
my %vtype = (tv => "vector($dim)", th => "halfvec($dim)");

for my $name (sort keys %vtype)
{
	my $type = $vtype{$name};
	$primary->safe_psql('postgres', <<"SQL");
CREATE TABLE $name (id int, v $type);
INSERT INTO $name
    SELECT g, ('[' || (SELECT string_agg((sin(g * 0.7 + j))::text, ',')
                       FROM generate_series(1, $dim) j) || ']')::$type
    FROM generate_series(1, $rows) g;
ANALYZE $name;
SQL
}

# Base backup and standby, both taken before any index exists.
$primary->backup('bkp');
my $standby = PostgreSQL::Test::Cluster->new('standby');
$standby->init_from_backup($primary, 'bkp', has_streaming => 1);
$standby->start;

# Run the same read on both nodes and require identical results.
sub agrees
{
	my ($tbl, $label) = @_;
	my $q = sprintf($qvec, $tbl);

	my $sql = <<"SQL";
SET enable_seqscan = off;
SET mkt.nprobe = 10000;
SELECT string_agg(id::text, ',' ORDER BY id) FROM (
    SELECT id FROM $tbl ORDER BY v <-> $q LIMIT $k) s;
SQL

	$primary->wait_for_catchup($standby);
	my $on_primary = $primary->safe_psql('postgres', $sql);
	my $on_standby = $standby->safe_psql('postgres', $sql);

	like($on_primary, qr/^\d+(,\d+)*$/, "$label: primary returns neighbours");
	is($on_standby, $on_primary, "$label: standby agrees with primary");
}

# Assert the case exercised the page formats it claims, so a silent fallback
# cannot leave a format uncovered while the test still passes.
sub formats_are
{
	my ($idx, $posting, $centroid, $label) = @_;

	my $got = $primary->safe_psql('postgres', <<"SQL");
SELECT (SELECT string_agg(DISTINCT format, '+' ORDER BY format)
          FROM posting_pages('$idx'::regclass) WHERE entry_count > 0)
    || ' / ' ||
       (SELECT string_agg(DISTINCT format, '+' ORDER BY format)
          FROM centroid_pages('$idx'::regclass));
SQL
	is($got, "$posting / $centroid", "$label: page formats as expected");
}

# The matrix. Each entry names the posting and centroid formats it should
# produce; those are asserted, not assumed.
my @cases = (
	{
		label    => 'aos posting, float centroids',
		table    => 'tv',
		opts     => 'fastscan = off, centroid_compression = off, '
		          . 'centroid_fastscan = off',
		posting  => 'aos',
		centroid => 'float',
	},
	{
		label    => 'aos posting, rabitq centroids',
		table    => 'tv',
		opts     => 'fastscan = off, centroid_compression = true, '
		          . 'centroid_fastscan = off',
		posting  => 'aos',
		centroid => 'rabitq',
	},
	{
		label    => 'fastscan posting, fastscan centroids',
		table    => 'tv',
		opts     => 'fastscan = on, centroid_compression = true, '
		          . 'centroid_fastscan = on',
		posting  => 'fastscan',
		centroid => 'fastscan',
	},
	{
		label    => 'halfvec column, half centroids',
		table    => 'th',
		opts     => 'fastscan = off, centroid_compression = off, '
		          . 'centroid_fastscan = off',
		posting  => 'aos',
		centroid => 'half',
	},
);

for my $c (@cases)
{
	my ($tbl, $label) = ($c->{table}, $c->{label});
	my $idx = 'idx_repl';

	# Build: pages reach the standby only through log_newpage_range().
	$primary->safe_psql('postgres',
		"CREATE INDEX $idx ON $tbl USING mktann (v) WITH ($c->{opts})");
	formats_are($idx, $c->{posting}, $c->{centroid}, $label);
	agrees($tbl, "$label, after build");

	# Insert: pages reach the standby through the aminsert GenericXLog path,
	# which covers the hole itself rather than relying on page_std.
	my $type = $vtype{$tbl};
	$primary->safe_psql('postgres', <<"SQL");
INSERT INTO $tbl
    SELECT g, ('[' || (SELECT string_agg((sin(g * 0.31 + j))::text, ',')
                       FROM generate_series(1, $dim) j) || ']')::$type
    FROM generate_series($rows + 1, $rows + 200) g;
SQL
	agrees($tbl, "$label, after insert");

	# Delete then vacuum: tombstoning and bulkdelete write pages too.
	$primary->safe_psql('postgres',
		"DELETE FROM $tbl WHERE id % 7 = 0 AND id > 100");
	$primary->safe_psql('postgres', "VACUUM $tbl");
	agrees($tbl, "$label, after delete and vacuum");

	# Reset for the next case.
	$primary->safe_psql('postgres', "DROP INDEX mkt.$idx");
	$primary->safe_psql('postgres', "DELETE FROM $tbl WHERE id > $rows");
}

# Parallel build. The workers write centroid and posting pages but WAL-log
# nothing themselves; the leader's single log_newpage_range() covers every
# block, so a parallel build must replicate exactly like a serial one. Force
# workers with the table reloption plus the maintenance-worker GUCs and enough
# rows that a build actually launches them.
{
	my $idx = 'idx_par';
	$primary->safe_psql('postgres', <<"SQL");
INSERT INTO tv
    SELECT g, ('[' || (SELECT string_agg((sin(g * 0.53 + j))::text, ',')
                       FROM generate_series(1, $dim) j) || ']')::vector($dim)
    FROM generate_series($rows + 1, $rows + 12000) g;
ALTER TABLE tv SET (parallel_workers = 3);
ANALYZE tv;
SQL
	$primary->safe_psql('postgres', <<"SQL");
SET max_parallel_maintenance_workers = 3;
SET min_parallel_table_scan_size = 0;
CREATE INDEX $idx ON tv USING mktann (v);
SQL
	agrees('tv', 'parallel build');

	# Reset for what follows.
	$primary->safe_psql('postgres', "DROP INDEX mkt.$idx");
	$primary->safe_psql('postgres', 'ALTER TABLE tv RESET (parallel_workers)');
	$primary->safe_psql('postgres', "DELETE FROM tv WHERE id > $rows");
}

# Complement: pin the layout property that forces page_std = false. This
# cannot detect the bug -- the flag changes what is logged, not the page --
# but it fails if the layout ever moves entries out of the hole, which is the
# assumption the flag rests on.
my $has_pageinspect =
  $primary->psql('postgres', 'CREATE EXTENSION pageinspect') == 0;
SKIP:
{
	skip 'pageinspect not available', 1 unless $has_pageinspect;

	$primary->safe_psql('postgres',
		'CREATE INDEX idx_hole ON tv USING mktann (v)');

	my $all_holed = $primary->safe_psql('postgres', <<'SQL');
SELECT count(*) = 0
  FROM posting_pages('idx_hole'::regclass) p,
       LATERAL page_header(get_raw_page('idx_hole', p.blkno)) h
 WHERE p.entry_count > 0
   AND NOT (h.lower = 24 AND h.upper = h.special);
SQL
	is($all_holed, 't',
		'posting entries live between pd_lower and pd_upper, so page_std must be false');
}

$standby->stop;
$primary->stop;
done_testing();
