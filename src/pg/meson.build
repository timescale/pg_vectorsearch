# meerkat PostgreSQL extension

extension_sources = files(
  'support_pg.c',
  'typeinfo.c',
  'mktann_inspect.c',
  'vector_pg.c',
  'halfvec_pg.c',
  'rabitq_pg.c',
  'rabitq_params_pg.c',
  'mktann.c',
  'mktann_build.c',
  'parallel_backend.c',
  'build_progress.c',
  '../index/parallel_build_leader.c',
  '../index/parallel_build_worker.c',
  'mktann_cache.c',
  'mktann_explain.c',
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

# -isystem: see mkt_pg_cflags in the top-level meson.build.
pg_ext_cflags += ['-isystem' + pg_includedir_server]
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

# The extension shared library is version-named so multiple extension
# versions can be installed side by side. Every install script pins its
# own library (see the SQL generation below), letting databases in one
# cluster run different versions and upgrade independently.
ext_version = meson.project_version()
ext_module_name = extension_name + '-' + ext_version
ext_module_pathname = '$libdir' / ext_module_name

# Build the extension as a shared library. The vcs_tag-generated
# git_commit.h is listed as a source so ninja produces it before
# compiling any TU that #includes it.
meerkat_ext_so = shared_module(ext_module_name,
  extension_sources,
  git_commit_h,
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

# Generate and install the extension control and SQL files. The
# version comes from meson.build (single source of truth); the SQL
# install script is a copy of the canonical sql/meerkat.sql under the
# versioned name. The control template is kept header-free so git
# displays it as a rename of meerkat.control (the files must stay
# >50% similar for rename detection).
ext_control_data = configuration_data()
ext_control_data.set('VERSION', ext_version)

configure_file(
  input: meson.project_source_root() / 'meerkat.control.in',
  output: extension_name + '.control',
  configuration: ext_control_data,
  install: true,
  install_dir: pg_sharedir / 'extension',
)

# Per-version (secondary) control file: PostgreSQL substitutes this
# version's module_pathname for MODULE_PATHNAME in its install and
# upgrade scripts, so every installed version binds its own
# version-named library — including each step of an upgrade chain. A
# single generated line, no template. The primary control deliberately
# omits module_pathname: a missing secondary control (a packaging bug)
# then fails loudly at CREATE/ALTER EXTENSION instead of silently
# binding the newest version's library.
configure_file(
  output: extension_name + '--' + ext_version + '.control',
  command: ['echo', 'module_pathname = \'' + ext_module_pathname + '\''],
  capture: true,
  install: true,
  install_dir: pg_sharedir / 'extension',
)

# The canonical SQL is installed verbatim under the versioned name
# (pgvector-style): MODULE_PATHNAME and @extschema@ are resolved by
# PostgreSQL when the script runs, so nothing needs substituting.
custom_target('meerkat-sql',
  input: meson.project_source_root() / 'sql' / 'meerkat.sql',
  output: extension_name + '--' + ext_version + '.sql',
  command: ['cp', '@INPUT@', '@OUTPUT@'],
  build_by_default: true,
  install: true,
  install_dir: pg_sharedir / 'extension',
)

# Upgrade scripts (sql/meerkat--A--B.sql) are listed here
# explicitly as they are added; see sql/README.md.
ext_update_scripts = []
if ext_update_scripts.length() > 0
  install_data(ext_update_scripts, install_dir: pg_sharedir / 'extension')
endif
