#!/usr/bin/env python3
"""
Benchmark meerkat (prism), pgvector (ivfflat/hnsw) on an
ann-benchmarks HDF5 dataset.

Loads the HDF5 dataset from ann-benchmarks, builds an ANN index, runs
queries with recall measurement against ground truth, and supports a
profile mode for perf sampling.

Requirements: pip install psycopg numpy h5py

Usage:
    # Full run: load data, build prism index, measure recall
    python3 scripts/bench_ann.py --dsn "host=..." \
        --dataset glove-100-angular --index-type prism

    # Cohere (768-dim, 1M), skip load, sweep nprobe on an existing index
    python3 scripts/bench_ann.py --dsn "..." --skip-load --skip-index \
        --dataset cohere-wikipedia-22-12-1M-angular --nprobe 20

    # Profile mode (tight query loop for perf sampling)
    python3 scripts/bench_ann.py --dsn "..." --skip-load --skip-index \
        --dataset glove-100-angular \
        --profile --profile-queries 5000 --nprobe 10
"""

import argparse
import math
import os
import re
import signal
import subprocess
import sys
import time

import h5py
import numpy as np
import psycopg
from pgvector.psycopg import register_vector


ANN_BENCHMARKS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "..",
    "ann-benchmarks",
    "data",
)

PROGRESS_EVERY = 50_000


def default_table_name(dataset: str) -> str:
    """Turn 'glove-100-angular' into 'glove_100_angular'."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", dataset).lower()


def load_data(conn, hdf5_path: str, table: str):
    """Load train vectors from HDF5 into PostgreSQL via binary COPY."""
    with h5py.File(hdf5_path, "r") as f:
        train = f["train"][:]

    n, dim = train.shape
    train = train.astype(np.float32)

    print(f"\nCreating table {table} (dim={dim})...")
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        cur.execute(f"""
            CREATE TABLE {table} (
                id bigint GENERATED ALWAYS AS IDENTITY,
                v vector({dim}) STORAGE PLAIN
            )
        """)
    conn.commit()

    print(f"Streaming {n:,} vectors via binary COPY...")
    t0 = time.monotonic()
    loaded = 0
    next_mark = PROGRESS_EVERY

    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} (v) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["vector"])
            for row in train:
                copy.write_row([row])
                loaded += 1

                if loaded >= next_mark or loaded == n:
                    elapsed = time.monotonic() - t0
                    rate = loaded / elapsed if elapsed > 0 else 0
                    print(f"  {loaded:,}/{n:,} ({rate:,.0f} vec/s, {elapsed:.0f}s)")
                    next_mark += PROGRESS_EVERY

    conn.commit()
    elapsed = time.monotonic() - t0
    rate = loaded / elapsed if elapsed > 0 else 0
    print(f"\nLoaded {loaded:,} vectors in {elapsed:.1f}s ({rate:,.0f} vectors/s)")


def build_index(conn, table: str, index_type: str, nlist: int | None, **kwargs):
    """Build an ANN index on the table."""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT indexname FROM pg_indexes
            WHERE tablename = '{table}'
              AND (indexdef LIKE '%prism%'
                OR indexdef LIKE '%ivfflat%'
                OR indexdef LIKE '%hnsw%'
                OR indexdef LIKE '%scann%')
        """)
        for (idx,) in cur.fetchall():
            print(f"Dropping {idx}")
            cur.execute(f"DROP INDEX IF EXISTS {idx}")
    conn.commit()

    fan_out = kwargs.get("fan_out")
    kmeans_nredo = kwargs.get("kmeans_nredo")
    centroid_compression = kwargs.get("centroid_compression", False)

    with conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '2GB'")

        if index_type == "prism":
            idx_name = f"idx_{table}_mkt"
            with_parts = []
            if nlist is not None:
                with_parts.append(f"nlist = {nlist}")
            if fan_out is not None:
                with_parts.append(f"fan_out = {fan_out}")
            if kmeans_nredo is not None:
                with_parts.append(f"kmeans_nredo = {kmeans_nredo}")
            if centroid_compression:
                with_parts.append("centroid_compression = true")
            with_clause = ""
            if with_parts:
                with_clause = f" WITH ({', '.join(with_parts)})"
            sql = (
                f"CREATE INDEX {idx_name} ON {table} "
                f"USING prism (v mkt.vec32_cosine_ops)"
                f"{with_clause}"
            )
        elif index_type == "hnsw":
            idx_name = f"idx_{table}_hnsw"
            m = kwargs.get("m", 16)
            ef_construction = kwargs.get("ef_construction", 200)
            sql = (
                f"CREATE INDEX {idx_name} ON {table} "
                f"USING hnsw (v vector_cosine_ops) "
                f"WITH (m = {m}, "
                f"ef_construction = {ef_construction})"
            )
        elif index_type == "scann":
            idx_name = f"idx_{table}_scann"
            if nlist is None:
                cur.execute(f"SELECT count(*) FROM {table}")
                count = cur.fetchone()[0]
                nlist = int(math.sqrt(count))
            sql = (
                f"CREATE INDEX {idx_name} ON {table} "
                f"USING scann (v cosine) "
                f"WITH (mode = 'MANUAL', num_leaves = {nlist})"
            )
        else:
            idx_name = f"idx_{table}_ivf"
            if nlist is None:
                cur.execute(f"SELECT count(*) FROM {table}")
                count = cur.fetchone()[0]
                nlist = int(math.sqrt(count))
            sql = (
                f"CREATE INDEX {idx_name} ON {table} "
                f"USING ivfflat (v vector_cosine_ops) "
                f"WITH (lists = {nlist})"
            )

        print(f"\nBuilding {index_type} index ({idx_name})...")
        if nlist is not None:
            print(f"  nlist/lists = {nlist}")
        if fan_out is not None:
            print(f"  fan_out = {fan_out}")
        if kmeans_nredo is not None:
            print(f"  kmeans_nredo = {kmeans_nredo}")
        if centroid_compression:
            print("  centroid_compression = true")

        t0 = time.monotonic()
        cur.execute(sql)
        elapsed = time.monotonic() - t0

        cur.execute(f"""
            SELECT pg_size_pretty(
                       pg_relation_size('{idx_name}'::regclass)),
                   relpages
            FROM pg_class WHERE relname = '{idx_name}'
        """)
        size, pages = cur.fetchone()
        print(f"  Built in {elapsed:.1f}s — {size} ({pages} pages)")

    conn.commit()


def query_sql(table: str, index_type: str, k: int) -> str:
    """Return the SELECT used to dispatch to the right operator class.

    prism uses the mkt.vec32_cosine_ops opclass, which is keyed on
    mkt.<=>; the query must cast the literal to mkt.vec32 so the
    planner can match the index. pgvector ivfflat/hnsw use the
    built-in <=> operator.
    """
    if index_type == "prism":
        return (
            f"SELECT id FROM {table} "
            f"ORDER BY v OPERATOR(mkt.<=>) %s::vector::mkt.vec32 "
            f"LIMIT {k}"
        )
    return f"SELECT id FROM {table} ORDER BY v <=> %s LIMIT {k}"


def set_search_params(
    cur,
    index_type: str,
    nprobe: int,
    ef_search: int,
    topk: int | None,
    rerank: bool | None = None,
):
    if index_type == "prism":
        cur.execute(f"SET prism.nprobe = {nprobe}")
        if topk is not None:
            cur.execute(f"SET mkt.topk = {topk}")
        if rerank is not None:
            cur.execute(f"SET prism.rerank = {'on' if rerank else 'off'}")
    elif index_type == "hnsw":
        cur.execute(f"SET hnsw.ef_search = {ef_search}")
    elif index_type == "scann":
        cur.execute(f"SET scann.num_leaves_to_search = {nprobe}")
    else:
        cur.execute(f"SET ivfflat.probes = {nprobe}")


def run_recall(
    conn,
    hdf5_path: str,
    table: str,
    index_type: str,
    k: int,
    num_queries: int,
    nprobe: int,
    ef_search: int = 100,
    topk: int | None = None,
    rerank: bool | None = None,
    warmup: int = 20,
    warmup_only: bool = False,
):
    """Run queries and measure recall against ground truth."""
    with h5py.File(hdf5_path, "r") as f:
        test = f["test"][:].astype(np.float32)
        neighbors = f["neighbors"][:]  # 0-indexed ground truth

    sql = query_sql(table, index_type, k)

    with conn.cursor() as cur:
        set_search_params(cur, index_type, nprobe, ef_search, topk, rerank)
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET max_parallel_workers_per_gather = 0")

        if warmup > 0:
            n_warmup = min(warmup, len(test))
            print(f"\nWarming up ({n_warmup} queries at nprobe={nprobe})...")
            for qi in range(n_warmup):
                cur.execute(sql, (test[qi],), prepare=True, binary=True)
                cur.fetchall()
            if warmup_only:
                print("Warm.")
                return

    n_test = min(num_queries, len(test))
    if index_type == "hnsw":
        print(
            f"\nRunning {n_test} queries, k={k}, "
            f"ef_search={ef_search}, index_type={index_type}"
        )
    else:
        print(
            f"\nRunning {n_test} queries, k={k}, nprobe={nprobe}, "
            f"index_type={index_type}"
        )

    with conn.cursor() as cur:
        set_search_params(cur, index_type, nprobe, ef_search, topk, rerank)
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET max_parallel_workers_per_gather = 0")

        recall_sum = 0.0
        latency_sum = 0.0
        latencies = []

        for qi in range(n_test):
            gt = set(int(x) + 1 for x in neighbors[qi, :k])

            t0 = time.monotonic()
            cur.execute(sql, (test[qi],), prepare=True, binary=True)
            result_ids = set(r[0] for r in cur.fetchall())
            elapsed = time.monotonic() - t0

            recall = len(gt & result_ids) / k
            recall_sum += recall
            latency_sum += elapsed
            latencies.append(elapsed)

            if qi < 5 or (qi + 1) % 500 == 0:
                avg_r = recall_sum / (qi + 1)
                avg_ms = latency_sum / (qi + 1) * 1000
                print(
                    f"  query {qi + 1:>5}: recall@{k}={recall:.2f}  "
                    f"avg_recall={avg_r:.3f}  "
                    f"avg_lat={avg_ms:.2f}ms"
                )

        cur.execute("RESET enable_seqscan")

    avg_recall = recall_sum / n_test
    avg_lat_ms = latency_sum / n_test * 1000
    latencies.sort()
    p50 = latencies[len(latencies) // 2] * 1000
    p99 = latencies[int(len(latencies) * 0.99)] * 1000

    print()
    print("=" * 60)
    print(f"  Table:         {table}")
    print(f"  Index type:    {index_type}")
    print(f"  Queries:       {n_test}")
    print(f"  k:             {k}")
    if index_type == "hnsw":
        print(f"  ef_search:     {ef_search}")
    else:
        print(f"  nprobe:        {nprobe}")
    print(f"  Recall@{k}:    {avg_recall:.4f}")
    print(f"  Avg latency:   {avg_lat_ms:.2f} ms")
    print(f"  P50 latency:   {p50:.2f} ms")
    print(f"  P99 latency:   {p99:.2f} ms")
    print(f"  Throughput:    {n_test / latency_sum:.0f} qps")
    print("=" * 60)


def run_profile(
    conn,
    hdf5_path: str,
    table: str,
    index_type: str,
    k: int,
    nprobe: int,
    num_queries: int,
    ef_search: int = 100,
    topk: int | None = None,
    rerank: bool | None = None,
    warmup: int = 50,
    profile_output: str = "profiles/perf.data",
    perf_freq: int = 4999,
):
    """Run queries in a tight loop with automatic perf profiling.

    Warmup queries run first (outside perf), then perf attaches to the
    backend for the measured queries.  A flamegraph SVG is generated if
    stackcollapse-perf.pl and flamegraph.pl are on PATH.
    """
    with h5py.File(hdf5_path, "r") as f:
        test = f["test"][:].astype(np.float32)

    n_test = min(num_queries, len(test))
    sql = query_sql(table, index_type, k)

    with conn.cursor() as cur:
        cur.execute("SELECT pg_backend_pid()")
        pid = cur.fetchone()[0]

    print(f"\nBackend PID: {pid}")
    print(f"Running {n_test} queries (k={k}, nprobe={nprobe})")

    with conn.cursor() as cur:
        set_search_params(cur, index_type, nprobe, ef_search, topk, rerank)
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET max_parallel_workers_per_gather = 0")

        # --- warmup (no perf) ---
        n_warmup = min(warmup, n_test)
        print(f"Warming up ({n_warmup} queries)...")
        for qi in range(n_warmup):
            cur.execute(sql, (test[qi % len(test)],), prepare=True, binary=True)
            cur.fetchall()

        # --- start perf ---
        os.makedirs(os.path.dirname(profile_output) or ".", exist_ok=True)
        perf_cmd = [
            "perf", "record",
            "-F", str(perf_freq),
            "-g",
            "-p", str(pid),
            "-o", profile_output,
        ]
        print(f"Starting perf: {' '.join(perf_cmd)}")
        perf_proc = subprocess.Popen(
            perf_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        time.sleep(0.3)

        # --- measured queries ---
        t0 = time.monotonic()
        for qi in range(n_test):
            cur.execute(sql, (test[qi % len(test)],), prepare=True, binary=True)
            cur.fetchall()

            if (qi + 1) % 1000 == 0:
                elapsed = time.monotonic() - t0
                qps = (qi + 1) / elapsed
                print(f"  {qi + 1}/{n_test} ({qps:.0f} qps)")

        cur.execute("RESET enable_seqscan")
        cur.execute("RESET max_parallel_workers_per_gather")

    elapsed = time.monotonic() - t0
    qps = n_test / elapsed
    print(f"\nDone: {n_test} queries in {elapsed:.1f}s ({qps:.0f} qps)")

    # --- stop perf ---
    perf_proc.send_signal(signal.SIGINT)
    perf_stderr = perf_proc.communicate(timeout=10)[1].decode()
    print(perf_stderr.strip())

    if not os.path.isfile(profile_output):
        print(f"ERROR: perf did not produce {profile_output}")
        return

    # --- generate report ---
    report_path = profile_output.replace(".data", "-report.txt")
    subprocess.run(
        [
            "perf", "report",
            "-i", profile_output,
            "--stdio", "--no-children",
            "-g", "none",
            "--percent-limit", "0.3",
        ],
        stdout=open(report_path, "w"),
        stderr=subprocess.DEVNULL,
    )
    print(f"\nPerf report: {report_path}")

    # Print top functions
    with open(report_path) as f:
        lines = f.readlines()
    func_lines = [l for l in lines if re.match(r"\s+\d+\.\d+%", l)]
    print(f"\nTop functions ({len(func_lines)} above 0.3%):")
    for line in func_lines[:30]:
        print(line, end="")

    # --- flamegraph ---
    try:
        script_out = subprocess.run(
            ["perf", "script", "-i", profile_output],
            capture_output=True, timeout=30,
        )
        folded = subprocess.run(
            ["stackcollapse-perf.pl"],
            input=script_out.stdout, capture_output=True, timeout=30,
        )
        svg_path = profile_output.replace(".data", ".svg")
        with open(svg_path, "wb") as svg:
            subprocess.run(
                ["flamegraph.pl"],
                input=folded.stdout, stdout=svg, timeout=30,
            )
        print(f"Flamegraph: {svg_path}")
    except FileNotFoundError:
        print("(stackcollapse-perf.pl / flamegraph.pl not found, skipping flamegraph)")
    except subprocess.TimeoutExpired:
        print("(flamegraph generation timed out)")


def resolve_hdf5_path(dataset: str, override: str | None) -> str:
    if override:
        return os.path.realpath(override)
    return os.path.realpath(os.path.join(ANN_BENCHMARKS_DIR, f"{dataset}.hdf5"))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark prism/ivfflat/hnsw on an ann-benchmarks HDF5 dataset"
    )
    parser.add_argument(
        "-d", "--dsn", type=str, required=True, help="PostgreSQL connection string"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="glove-100-angular",
        help="Dataset name under ann-benchmarks/data/ (default: glove-100-angular)",
    )
    parser.add_argument(
        "--hdf5",
        type=str,
        default=None,
        help="Explicit HDF5 path; overrides --dataset lookup",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default=None,
        help="PostgreSQL table name (default: derived from --dataset)",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip loading data (table must already exist)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip building index (index must already exist)",
    )
    parser.add_argument(
        "--index-type",
        choices=["prism", "ivfflat", "hnsw", "scann"],
        default="prism",
        help="Index type to benchmark (default: prism)",
    )
    parser.add_argument(
        "-k", type=int, default=10, help="Top-k neighbors (default: 10)"
    )
    parser.add_argument(
        "-n",
        "--num-queries",
        type=int,
        default=1000,
        help="Number of queries to run (default: 1000)",
    )
    parser.add_argument(
        "--nprobe", type=int, default=10, help="Probes for index scan (default: 10)"
    )
    parser.add_argument(
        "--nprobe-sweep",
        type=str,
        default=None,
        help="Comma-separated nprobe values to sweep in one session (e.g., '5,10,20,40,80')",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=None,
        help="Number of lists/clusters (default: auto)",
    )
    parser.add_argument(
        "--nlist-sweep",
        type=str,
        default=None,
        help="Comma-separated nlist values; rebuilds index for each, "
        "then runs --nprobe-sweep (e.g., '100,1000,10000')",
    )
    parser.add_argument(
        "--fan-out",
        type=int,
        default=None,
        help="prism tree branching factor (default: auto from nlist)",
    )
    parser.add_argument(
        "--kmeans-nredo",
        type=int,
        default=None,
        help="K-means restarts for cluster quality (default: 1)",
    )
    parser.add_argument(
        "--centroid-compression",
        action="store_true",
        help="Use RaBitQ compression for centroid pages",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="prism rerank candidates (default: server default)",
    )
    parser.add_argument(
        "--rerank",
        choices=["on", "off"],
        default=None,
        help="prism rerank toggle: sets prism.rerank (default: server default)",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=100,
        help="HNSW ef_search parameter (default: 100)",
    )
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="HNSW ef_construction parameter (default: 200)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=16,
        help="HNSW m (max connections) parameter (default: 16)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup queries before measuring (default: 20)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile mode: tight query loop for perf sampling",
    )
    parser.add_argument(
        "--profile-queries",
        type=int,
        default=1000,
        help="Queries to run in profile mode (default: 1000)",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default="profiles/perf.data",
        help="Output path for perf data (default: profiles/perf.data)",
    )
    parser.add_argument(
        "--perf-freq",
        type=int,
        default=4999,
        help="Perf sampling frequency in Hz (default: 4999)",
    )
    args = parser.parse_args()

    hdf5_path = resolve_hdf5_path(args.dataset, args.hdf5)
    if not os.path.isfile(hdf5_path):
        print(f"Error: HDF5 file not found: {hdf5_path}")
        sys.exit(1)

    table = args.table_name or default_table_name(args.dataset)

    conn = psycopg.connect(args.dsn, autocommit=True)
    register_vector(conn)
    print(f"Connected (server {conn.info.server_version})")
    print(f"Dataset: {args.dataset}  Table: {table}")

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if args.index_type == "prism":
            cur.execute("CREATE EXTENSION IF NOT EXISTS meerkat")
    print("Extensions ready")

    if not args.skip_load:
        load_data(conn, hdf5_path, table)
    else:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}")
            count = cur.fetchone()[0]
        print(f"Table {table}: {count:,} rows")

    if not args.skip_index and not args.nlist_sweep:
        build_index(
            conn,
            table,
            args.index_type,
            args.nlist,
            m=args.m,
            ef_construction=args.ef_construction,
            fan_out=args.fan_out,
            kmeans_nredo=args.kmeans_nredo,
            centroid_compression=args.centroid_compression,
        )

    rerank = None
    if args.rerank is not None:
        rerank = args.rerank == "on"

    nprobe_values = None
    if args.nprobe_sweep:
        nprobe_values = [int(x) for x in args.nprobe_sweep.split(",")]

    if args.nlist_sweep:
        if not nprobe_values:
            nprobe_values = [args.nprobe]
        for nlist in [int(x) for x in args.nlist_sweep.split(",")]:
            build_index(
                conn, table, args.index_type, nlist,
                m=args.m, ef_construction=args.ef_construction,
                fan_out=args.fan_out,
                kmeans_nredo=args.kmeans_nredo,
                centroid_compression=args.centroid_compression,
            )
            max_nprobe = max(nprobe_values)
            run_recall(
                conn, hdf5_path, table, args.index_type, args.k,
                args.num_queries, max_nprobe, ef_search=args.ef_search,
                topk=args.topk, rerank=rerank, warmup=args.warmup,
                warmup_only=True,
            )
            for nprobe in nprobe_values:
                run_recall(
                    conn, hdf5_path, table, args.index_type, args.k,
                    args.num_queries, nprobe, ef_search=args.ef_search,
                    topk=args.topk, rerank=rerank, warmup=0,
                )
    elif nprobe_values:
        max_nprobe = max(nprobe_values)
        run_recall(
            conn, hdf5_path, table, args.index_type, args.k,
            args.num_queries, max_nprobe, ef_search=args.ef_search,
            topk=args.topk, rerank=rerank, warmup=args.warmup,
            warmup_only=True,
        )
        for nprobe in nprobe_values:
            run_recall(
                conn, hdf5_path, table, args.index_type, args.k,
                args.num_queries, nprobe, ef_search=args.ef_search,
                topk=args.topk, rerank=rerank, warmup=0,
            )
    elif args.profile:
        run_profile(
            conn,
            hdf5_path,
            table,
            args.index_type,
            args.k,
            args.nprobe,
            args.profile_queries,
            ef_search=args.ef_search,
            topk=args.topk,
            rerank=rerank,
            warmup=args.warmup,
            profile_output=args.profile_output,
            perf_freq=args.perf_freq,
        )
    else:
        run_recall(
            conn,
            hdf5_path,
            table,
            args.index_type,
            args.k,
            args.num_queries,
            args.nprobe,
            ef_search=args.ef_search,
            topk=args.topk,
            rerank=rerank,
            warmup=args.warmup,
        )

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
