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

    test_names = [
      'vector',
      'halfvec',
      'distance',
      'operators',
      'rabitq',
      'mktann',
      'rerank',
      'inspect',
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
  else
    message('pg_regress not found, skipping PostgreSQL regression tests')
  endif
endif
