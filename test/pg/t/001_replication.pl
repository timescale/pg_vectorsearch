# Streaming replication of prism indexes, across page formats and the
# operations that write them.
#
# prism pages are not standard-layout: entries live in the content area
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

# Query with a row's own vec32 rather than the origin: this data is
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
	'ALTER DATABASE postgres SET search_path = mkt, prism, public');

# Two tables so both vec32 types are covered; vec16 drives the
# half-precision centroid format, which no vec32 column can reach.
my %vtype = (tv => "vec32($dim)", th => "vec16($dim)");

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
SET prism.nprobe = 10000;
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
		label    => 'vec16 column, half centroids',
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
		"CREATE INDEX $idx ON $tbl USING prism (v) WITH ($c->{opts})");
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
                       FROM generate_series(1, $dim) j) || ']')::vec32($dim)
    FROM generate_series($rows + 1, $rows + 12000) g;
ALTER TABLE tv SET (parallel_workers = 3);
ANALYZE tv;
SQL
	$primary->safe_psql('postgres', <<"SQL");
SET max_parallel_maintenance_workers = 3;
SET min_parallel_table_scan_size = 0;
CREATE INDEX $idx ON tv USING prism (v);
SQL
	agrees('tv', 'parallel build');

	# Reset for what follows.
	$primary->safe_psql('postgres', "DROP INDEX mkt.$idx");
	$primary->safe_psql('postgres', 'ALTER TABLE tv RESET (parallel_workers)');
	$primary->safe_psql('postgres', "DELETE FROM tv WHERE id > $rows");
}

# convert_posting_to_fastscan: a runtime AoS -> fastscan rewrite. It writes NEW
# posting pages by extending the relation and must WAL-log them itself -- there
# is no closing log_newpage_range() for a runtime operation, so it must not run
# in build_mode. The index is built to exercise all three failure modes at once:
#
#   - centroid_fastscan = on: the leaf's child block lives in the packed
#     per-group array, at a different offset than the AoS meta array, so the
#     repoint must be format-aware.
#   - fan_out = 4: forces a multi-level centroid tree, so the leaf-traversal
#     order diverges from cluster_id -- a positional leaves[cluster_id] lookup
#     then converts the wrong cluster. (A flat tree hides this: order == id.)
#   - converting every cluster and comparing the standby: without per-page WAL
#     the new fastscan chains are dirtied but never shipped, while the
#     WAL-logged centroid repoint points at them.
{
	my $idx = 'idx_conv';
	$primary->safe_psql('postgres',
		"CREATE INDEX $idx ON tv USING prism (v) "
	  . "WITH (fastscan = off, centroid_compression = true, "
	  . "centroid_fastscan = on, fan_out = 4)");
	formats_are($idx, 'aos', 'fastscan', 'convert: pre-convert formats');

	# Convert every cluster (not just cluster 0 -- that alone would miss the
	# repoint bug, which only shows up once the packed-array offset is used).
	my $nconv = $primary->safe_psql('postgres', <<"SQL");
SELECT count(convert_posting_to_fastscan('$idx'::regclass, cluster_id))
FROM posting_pages('$idx'::regclass) WHERE is_first;
SQL
	my $nclusters = $primary->safe_psql('postgres',
		"SELECT count(*) FROM posting_pages('$idx'::regclass) WHERE is_first");
	is($nconv, $nclusters, "convert: converted all $nclusters clusters");

	my $all_fs = $primary->safe_psql('postgres', <<"SQL");
SELECT bool_and(format = 'fastscan')
FROM posting_pages('$idx'::regclass) WHERE is_first;
SQL
	is($all_fs, 't', 'convert: every posting head is now fastscan');

	# The new pages must have reached the standby: posting_pages walks the
	# centroid-reachable chains, so it follows each repointed head into the new
	# fastscan pages -- absent on the standby without the WAL fix. Compare the
	# full layout across all clusters, not just one.
	my $layout = <<"SQL";
SELECT string_agg(cluster_id || ':' || format || ':' || entry_count,
                  ',' ORDER BY blkno)
FROM posting_pages('$idx'::regclass)
SQL
	$primary->wait_for_catchup($standby);
	my $on_primary = $primary->safe_psql('postgres', $layout);
	my $on_standby = $standby->safe_psql('postgres', $layout);
	like($on_primary, qr/fastscan/, 'convert: primary posting layout has fastscan');
	is($on_standby, $on_primary,
		'convert: standby posting layout matches primary');

	# End-to-end: the query still agrees across nodes.
	agrees('tv', 'convert: after fastscan conversion');
	$primary->safe_psql('postgres', "DROP INDEX mkt.$idx");
}

# Incremental split: prism.rebalance rewrites an oversized posting list into
# several fresh chains, repoints the centroid leaf at them and raises nlist in
# the metapage. All of it is runtime work outside build_mode, so every page has
# to be WAL-logged as it is written -- there is no closing
# log_newpage_range() to cover for a missed one. Three distinct writes have to
# arrive for the standby to agree:
#
#   - the new posting chains, extended and logged page by page;
#   - the centroid leaf flip, which repoints one entry and appends the rest;
#   - the metapage's nlist, raised before the leaves are written so a crash
#     leaves the count high rather than reusing a cluster id.
#
# The split needs a flat tree with RaBitQ centroid pages, which is what the
# reloptions below ask for; anything else is refused outright.
{
	my $idx = 'idx_split';
	$primary->safe_psql('postgres',
		"CREATE INDEX $idx ON tv USING prism (v) "
	  . "WITH (fastscan = off, centroid_compression = true, "
	  . "centroid_fastscan = off, nlist = 8)");
	formats_are($idx, 'aos', 'rabitq', 'split: pre-split formats');

	my $heads = "SELECT count(*) FROM posting_pages('$idx'::regclass) "
	          . "WHERE is_first";
	my $before = $primary->safe_psql('postgres', $heads);

	# rebalance commits on its own behalf, so it cannot run inside a
	# caller's transaction; sent this way each statement is its own.
	$primary->safe_psql('postgres', <<"SQL");
SET client_min_messages = warning;
CALL rebalance('$idx', 60);
SQL

	my $after = $primary->safe_psql('postgres', $heads);
	ok($after > $before,
		"split: rebalance split lists ($before -> $after heads)");

	# The metapage write: nlist is raised past the built-in count, and the
	# standby has to see the same number or its scans probe the wrong range.
	my $nlist = "SELECT setting FROM index_settings('$idx'::regclass) "
	          . "WHERE name = 'nlist'";
	$primary->wait_for_catchup($standby);
	my $nlist_p = $primary->safe_psql('postgres', $nlist);
	my $nlist_s = $standby->safe_psql('postgres', $nlist);
	ok($nlist_p > 8, "split: nlist raised past the built count ($nlist_p)");
	is($nlist_s, $nlist_p, 'split: standby metapage agrees on nlist');

	# The pages themselves. posting_pages walks the centroid-reachable
	# chains, so it only reaches the new heads if both the flip and the
	# chains replicated; centroid_pages pins the leaf entries that do the
	# pointing.
	my $players = <<"SQL";
SELECT string_agg(cluster_id || ':' || format || ':' || entry_count,
                  ',' ORDER BY blkno)
FROM posting_pages('$idx'::regclass)
SQL
	my $clayout = <<"SQL";
SELECT string_agg(blkno || '/' || entry || ':' || child_blkno || ':'
                  || is_leaf, ',' ORDER BY blkno, entry)
FROM centroid_pages('$idx'::regclass)
SQL
	for my $probe (['posting', $players], ['centroid', $clayout])
	{
		my ($what, $sql) = @$probe;
		my $on_primary = $primary->safe_psql('postgres', $sql);
		my $on_standby = $standby->safe_psql('postgres', $sql);
		like($on_primary, qr/\d/, "split: primary has a $what layout");
		is($on_standby, $on_primary,
			"split: standby $what layout matches primary");
	}

	agrees('tv', 'split: after rebalance');

	# Retired chains are reclaimed by a later pass, which tombstones and
	# frees pages the tree no longer points at. That is another set of
	# runtime page writes, and the layouts have to stay in step across it.
	$primary->safe_psql('postgres', <<"SQL");
SET client_min_messages = warning;
CALL rebalance('$idx', 60);
SQL
	$primary->wait_for_catchup($standby);
	is($standby->safe_psql('postgres', $players),
		$primary->safe_psql('postgres', $players),
		'split: standby posting layout matches after reclaim');
	agrees('tv', 'split: after reclaiming retired chains');

	$primary->safe_psql('postgres', "DROP INDEX mkt.$idx");
}

# A split that has to grow the centroid level, rather than fitting its new
# leaves on the page the old one was on. That is the only path that extends the
# centroid chain at runtime: it initialises a fresh page, writes the leaf
# there, and then reopens the previous tail to link it. A standby that missed
# either write has a chain that stops short, so the leaves past the break
# simply are not there.
#
# Reaching it needs entries wide enough that a page holds few of them, so this
# uses a high-dimension table with few rows rather than the shared one.
{
	my $idx  = 'idx_split_wide';
	my $wdim = 1968;
	$primary->safe_psql('postgres', <<"SQL");
CREATE TABLE tw (id int, v vec32($wdim));
INSERT INTO tw
    SELECT g, ('[' || (SELECT string_agg((sin(g * 0.7 + j))::text, ',')
                       FROM generate_series(1, $wdim) j) || ']')::vec32($wdim)
    FROM generate_series(1, 400) g;
ANALYZE tw;
SQL
	$primary->safe_psql('postgres',
		"CREATE INDEX $idx ON tw USING prism (v) "
	  . "WITH (fastscan = off, centroid_compression = true, "
	  . "centroid_fastscan = off, nlist = 32)");

	my $cpages = "SELECT count(DISTINCT blkno) "
	           . "FROM centroid_pages('$idx'::regclass)";
	my $before = $primary->safe_psql('postgres', $cpages);

	$primary->safe_psql('postgres', <<"SQL");
SET client_min_messages = warning;
CALL rebalance('$idx', 4);
SQL

	my $after = $primary->safe_psql('postgres', $cpages);
	ok($after > $before,
		"split: centroid chain grew ($before -> $after pages)");

	# Walk the whole chain on both nodes: a link that never arrived shows up
	# as a shorter chain here, and as missing leaves in the layout.
	my $clayout = <<"SQL";
SELECT count(DISTINCT blkno) || '/' || count(*) || ' ' ||
       string_agg(child_blkno::text, ',' ORDER BY blkno, entry)
FROM centroid_pages('$idx'::regclass)
SQL
	$primary->wait_for_catchup($standby);
	is($standby->safe_psql('postgres', $clayout),
		$primary->safe_psql('postgres', $clayout),
		'split: standby centroid chain matches after it grew');
	agrees('tw', 'split: after a chain-growing split');

	$primary->safe_psql('postgres', "DROP INDEX mkt.$idx");
	$primary->safe_psql('postgres', 'DROP TABLE tw');
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
		'CREATE INDEX idx_hole ON tv USING prism (v)');

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
