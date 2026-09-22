-- Build-introspection log output.
--
-- prism.log_build_stats gates the per-phase resource lines; the planned-allocation
-- line and the final per-phase summary are always logged. The log content
-- carries timings/byte counts (non-deterministic), so this asserts only the
-- presence/absence of each line by pattern, reading the server log via
-- pg_current_logfile() (logging_collector is on in the test config).
--
-- After each build the final summary line is the last thing the build logs, so
-- the test polls the new log segment until that line appears; once it is visible
-- everything emitted earlier in the build (planned-allocation, per-phase lines)
-- has been flushed too, so the counts never race the logging collector.
--
-- The search patterns are assembled from fragments at run time so the literal
-- strings never appear in this script's own text — otherwise the script (which
-- the server may log) would self-match.

CREATE TABLE lg (v vec32(8)) WITH (parallel_workers = 2);
INSERT INTO lg
    SELECT format('[%s,%s,%s,%s,%s,%s,%s,%s]',
                  i % 19, i % 7, i % 5, i % 3, i % 11, i % 13, i % 2, i % 17)
               ::vec32
    FROM generate_series(1, 5000) i;
ANALYZE lg;

CREATE TABLE logres (label text, ok boolean);

DO $$
DECLARE
    fn		  text := pg_current_logfile();
    pos		  int;
    seg		  text;
    pat_plan  text := 'prism build: ' || 'planned allocations';
    pat_phase text := 'prism build: ' || 'phase "';
    pat_sum	  text := 'build: ' || 'sample ';
    n_planned int;
    n_phase	  int;
    n_summary int;
    tries	  int;
BEGIN
    PERFORM set_config('max_parallel_maintenance_workers', '2', false);
    PERFORM set_config('min_parallel_table_scan_size', '0', false);

    -- ---- Build 1: parallel, prism.log_build_stats = on ----
    pos := length(pg_read_file(fn));
    PERFORM set_config('prism.log_build_stats', 'on', false);
    EXECUTE 'CREATE INDEX lg_on ON lg USING prism (v) WITH (nlist = 64)';
    tries := 0;
    LOOP
        seg := substr(pg_read_file(fn), pos + 1);
        EXIT WHEN seg ~ pat_sum OR tries > 200;
        PERFORM pg_sleep(0.05);
        tries := tries + 1;
    END LOOP;
    n_planned := (SELECT count(*) FROM regexp_matches(seg, pat_plan, 'g'));
    n_phase	  := (SELECT count(*) FROM regexp_matches(seg, pat_phase, 'g'));
    n_summary := (SELECT count(*) FROM regexp_matches(seg, pat_sum, 'g'));
    INSERT INTO logres VALUES
        ('on_has_planned', n_planned >= 1),
        ('on_has_phases', n_phase > 0),
        ('on_has_summary', n_summary >= 1);

    -- ---- Build 2: parallel, prism.log_build_stats = off ----
    pos := length(pg_read_file(fn));
    PERFORM set_config('prism.log_build_stats', 'off', false);
    EXECUTE 'CREATE INDEX lg_off ON lg USING prism (v) WITH (nlist = 64)';
    tries := 0;
    LOOP
        seg := substr(pg_read_file(fn), pos + 1);
        EXIT WHEN seg ~ pat_sum OR tries > 200;
        PERFORM pg_sleep(0.05);
        tries := tries + 1;
    END LOOP;
    n_planned := (SELECT count(*) FROM regexp_matches(seg, pat_plan, 'g'));
    n_phase	  := (SELECT count(*) FROM regexp_matches(seg, pat_phase, 'g'));
    n_summary := (SELECT count(*) FROM regexp_matches(seg, pat_sum, 'g'));
    -- Per-phase lines are gated off; the always-on lines remain.
    INSERT INTO logres VALUES
        ('off_has_planned', n_planned >= 1),
        ('off_no_phases', n_phase = 0),
        ('off_has_summary', n_summary >= 1);
END $$;

SELECT label, ok FROM logres ORDER BY label;

DROP TABLE logres;
DROP TABLE lg;
