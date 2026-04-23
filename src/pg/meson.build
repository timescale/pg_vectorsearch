# meerkat PostgreSQL extension

extension_sources = files(
  'mkt_pg.c',
  'mktann_inspect.c',
  'mkt_pg_vector.c',
  'mkt_pg_halfvec.c',
  'mkt_pg_rabitq.c',
  'mkt_pg_rabitq_params.c',
  'mktann.c',
  'mktann_build.c',
  'mktann_cache.c',
  'mktann_scan.c',
  'mktann_storage.c',
)

pg_ext_cflags = [
  '-D_GNU_SOURCE',
  '-Wno-unused-parameter',
  '-Wno-sign-compare',
  # PG headers use GNU extensions (__int128, statement expressions, etc.)
  '-Wno-pedantic',
]

# Suppress clang-specific PG header warnings
if cc.get_id() == 'clang'
  pg_ext_cflags += [
    '-Wno-gnu-statement-expression-from-macro-expansion',
    '-Wno-flexible-array-extensions',
  ]
endif

pg_ext_cflags += ['-I' + pg_includedir_server]
pg_ext_cflags += pg_cflags

# Add PostgreSQL source tree headers for IDE navigation.
# Use -isystem so installed headers (above) take precedence and
# warnings from the source tree are suppressed.
pg_srcdir_path = get_option('pg_srcdir')
if pg_srcdir_path != ''
  fs = import('fs')
  if fs.is_dir(pg_srcdir_path)
    pg_ext_cflags += ['-isystem', pg_srcdir_path / 'src/include']
  endif
endif

# Build the extension as a shared library
meerkat_ext_so = shared_library('meerkat',
  extension_sources,
  c_args: pg_ext_cflags,
  link_args: pg_ldflags,
  link_with: mkt_pg_lib,
  include_directories: src_inc,
  name_prefix: '',
  build_rpath: pg_rpath.length() > 0 ? pg_rpath[0] : '',
  install_rpath: pg_libdir,
  install: true,
  install_dir: pg_pkglibdir,
  override_options: ['b_lundef=false', 'warning_level=1', 'werror=false'],
)

# Install extension control and SQL files
install_data(
  meson.project_source_root() / 'meerkat.control',
  install_dir: pg_sharedir / 'extension',
)
install_data(
  meson.project_source_root() / 'sql' / 'meerkat--0.1.0.sql',
  install_dir: pg_sharedir / 'extension',
)
