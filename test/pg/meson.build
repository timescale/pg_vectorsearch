# PostgreSQL regression tests using pg_regress

if not build_pg_extension
  message('Skipping regression tests (PostgreSQL headers not available)')
else
  pg_regress = find_program(
    pg_pkglibdir / 'pgxs' / 'src' / 'test' / 'regress' / 'pg_regress',
    'pg_regress',
    required: false,
  )

  if pg_regress.found()
    # Generate postgresql.conf snippet for temp instance
    pg_test_conf = configure_file(
      output: 'meerkat_test.conf',
      command: ['sh', '-c',
        'echo "search_path = \'mkt, public\'" > @OUTPUT@'],
    )

    # Same, plus shared_preload_libraries for the centroid-cache tests: the
    # cache's control struct + LWLock + DSA only exist when meerkat is
    # preloaded, so the cache is exercised only under this config.
    pg_cache_conf = configure_file(
      output: 'meerkat_cache.conf',
      command: ['sh', '-c',
        'printf "search_path = \'mkt, public\'\n' +
        'shared_preload_libraries = \'meerkat\'\n" > @OUTPUT@'],
    )

    test_names = [
      'vector',
      'halfvec',
      'distance',
      'operators',
      'rabitq',
      'mktann',
      'mutations',
      'rerank',
      'inspect',
      'parallel',
      # cache must be unavailable when NOT preloaded (this instance is not)
      'centroid_cache_disabled',
    ]

    test('regress',
      pg_regress,
      args: [
        '--inputdir=' + meson.current_source_dir(),
        '--outputdir=' + meson.current_build_dir(),
        '--bindir=' + pg_bindir,
        '--temp-instance=' + meson.project_build_root() / 'testdb',
        '--temp-config=' + meson.current_build_dir() / 'meerkat_test.conf',
        '--load-extension=meerkat',
      ] + test_names,
      depends: [meerkat_ext_so],
      timeout: 300,
      is_parallel: false,
      protocol: 'tap',
    )

    # Centroid-cache regression tests, run against a PRELOADED temp instance so
    # the shared cache is active. Separate instance/config from `regress`.
    cache_test_names = [
      'centroid_cache',
      'centroid_cache_lru',
      'centroid_cache_drop',
    ]

    test('regress_cache',
      pg_regress,
      args: [
        '--inputdir=' + meson.current_source_dir(),
        '--outputdir=' + meson.current_build_dir(),
        '--bindir=' + pg_bindir,
        '--temp-instance=' + meson.project_build_root() / 'cache_testdb',
        '--temp-config=' + meson.current_build_dir() / 'meerkat_cache.conf',
        '--load-extension=meerkat',
      ] + cache_test_names,
      depends: [meerkat_ext_so],
      timeout: 300,
      is_parallel: false,
      protocol: 'tap',
    )

    # Isolation tests for incremental writes (MVCC visibility + rebuild vs an
    # open insert). Unlike the build-progress spec below, these need no
    # injection points.
    pg_isolation_regress_mut = find_program(
      pg_pkglibdir / 'pgxs' / 'src' / 'test' / 'isolation' / 'pg_isolation_regress',
      'pg_isolation_regress',
      required: false,
    )
    if pg_isolation_regress_mut.found()
      test('isolation_mutations',
        pg_isolation_regress_mut,
        args: [
          '--inputdir=' + meson.current_source_dir(),
          '--outputdir=' + meson.current_build_dir(),
          '--bindir=' + pg_bindir,
          '--temp-instance=' + meson.project_build_root() / 'mut_isolation_testdb',
          '--temp-config=' + meson.current_build_dir() / 'meerkat_test.conf',
          '--load-extension=meerkat',
          'mutations_concurrent',
        ],
        depends: [meerkat_ext_so],
        timeout: 120,
        is_parallel: false,
        protocol: 'tap',
      )

      # Centroid-cache concurrency (REINDEX vs cache-using readers), against a
      # PRELOADED temp instance so the cache is active.
      test('isolation_cache',
        pg_isolation_regress_mut,
        args: [
          '--inputdir=' + meson.current_source_dir(),
          '--outputdir=' + meson.current_build_dir(),
          '--bindir=' + pg_bindir,
          '--temp-instance=' + meson.project_build_root() / 'cache_isolation_testdb',
          '--temp-config=' + meson.current_build_dir() / 'meerkat_cache.conf',
          '--load-extension=meerkat',
          'centroid_cache_concurrency',
        ],
        depends: [meerkat_ext_so],
        timeout: 120,
        is_parallel: false,
        protocol: 'tap',
      )
    endif
  else
    message('pg_regress not found, skipping PostgreSQL regression tests')
  endif

  # ------------------------------------------------------------------
  # Injection-point isolation tests
  #
  # These pause an in-progress index build via an injection point so a
  # second session can observe locking / progress state. They are skipped
  # (not failed) unless both hold:
  #   1. PostgreSQL was built with injection points (USE_INJECTION_POINTS).
  #   2. The injection_points test extension is installed in the prefix.
  #      PostgreSQL's meson build does not install test modules, so install
  #      it against the target PG with, e.g.:
  #        make -C <pg-srcdir>/src/test/modules/injection_points \
  #             install USE_PGXS=1 PG_CONFIG=<prefix>/bin/pg_config
  # ------------------------------------------------------------------
  fs = import('fs')
  have_injection_points = cc.has_header_symbol(
    'pg_config.h',
    'USE_INJECTION_POINTS',
    args: ['-I' + pg_includedir_server] + pg_cflags,
  )
  injection_points_installed = fs.is_file(
    pg_sharedir / 'extension' / 'injection_points.control')

  if not have_injection_points
    message(
      'Skipping isolation tests (PostgreSQL not built with injection points)')
  elif not injection_points_installed
    message(
      'Skipping isolation tests (injection_points extension not installed; ' +
      'build it against this PostgreSQL with make ... install USE_PGXS=1)')
  else
    pg_isolation_regress = find_program(
      pg_pkglibdir / 'pgxs' / 'src' / 'test' / 'isolation' / 'pg_isolation_regress',
      'pg_isolation_regress',
      required: false,
    )

    if not pg_isolation_regress.found()
      message('pg_isolation_regress not found, skipping isolation tests')
    else
      # The injection_points extension is installed in the standard location,
      # so the temp instance only needs the search_path (same as regress).
      iso_conf = configure_file(
        output: 'meerkat_isolation.conf',
        command: ['sh', '-c',
          'echo "search_path = \'mkt, public\'" > @OUTPUT@'],
      )

      test('isolation',
        pg_isolation_regress,
        args: [
          '--inputdir=' + meson.current_source_dir(),
          '--outputdir=' + meson.current_build_dir(),
          '--bindir=' + pg_bindir,
          '--temp-instance=' + meson.project_build_root() / 'isolation_testdb',
          '--temp-config=' + meson.current_build_dir() / 'meerkat_isolation.conf',
          '--load-extension=meerkat',
          'build_progress',
        ],
        depends: [meerkat_ext_so],
        timeout: 120,
        is_parallel: false,
        protocol: 'tap',
      )

      # Centroid-cache BUILDING-state race: needs both injection points AND a
      # preloaded instance (so the cache is active), hence its own config.
      cache_iso_conf = configure_file(
        output: 'meerkat_cache_isolation.conf',
        command: ['sh', '-c',
          'printf "search_path = \'mkt, public\'\n' +
          'shared_preload_libraries = \'meerkat\'\n" > @OUTPUT@'],
      )

      test('isolation_cache_building',
        pg_isolation_regress,
        args: [
          '--inputdir=' + meson.current_source_dir(),
          '--outputdir=' + meson.current_build_dir(),
          '--bindir=' + pg_bindir,
          '--temp-instance=' + meson.project_build_root() / 'cache_building_testdb',
          '--temp-config=' + meson.current_build_dir() / 'meerkat_cache_isolation.conf',
          '--load-extension=meerkat',
          'centroid_cache_building',
        ],
        depends: [meerkat_ext_so],
        timeout: 120,
        is_parallel: false,
        protocol: 'tap',
      )
    endif
  endif
endif
