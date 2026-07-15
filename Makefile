# Convenience wrapper around the meson build.
#
#   make              build the extension (release, against the
#                     installed PostgreSQL)
#   make install      configure + build + install; use sudo for a
#                     system PostgreSQL
#   make test         run the full test suite
#   make reconfigure  re-run meson setup after changing a variable
#                     below on an existing build directory
#   make clean        remove the build directory
#
# Variables (override on the command line):
#
#   PG_CONFIG=/path/to/pg_config   PostgreSQL to build against
#                                  (default: pg_config on PATH; a
#                                  missing one fails the setup — the
#                                  wrapper exists to build the
#                                  extension, so it never silently
#                                  skips it)
#   TOOLS=enabled                  also build the developer tools
#                                  (CLI, standalone library, unit
#                                  tests; default: extension only)
#   BUILDTYPE=debug                meson buildtype (default: release)
#   BUILDDIR=builddir-foo          build directory (default: builddir)
#   MESON_ARGS="-Dnative=true"     extra meson setup options
#
# Example:
#   make install PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config
#
# The wrapper covers the common build-and-install flow; anything more
# (coverage, formatting, per-target builds) is used directly through
# meson — see docs/development.md.

BUILDDIR   ?= builddir
BUILDTYPE  ?= release
PG_CONFIG  ?= pg_config
TOOLS      ?= disabled
MESON      ?= meson
MESON_ARGS ?=

# The wrapper exists to build the extension, so a missing PostgreSQL
# must be a hard error: -Dpostgresql=enabled fails the setup when
# pg_config or the server headers are absent, where meson's default
# auto-detection would quietly build everything except the extension.
MESON_SETUP_ARGS := --buildtype=$(BUILDTYPE)
MESON_SETUP_ARGS += -Dpostgresql=enabled
MESON_SETUP_ARGS += -Dpg_config=$(PG_CONFIG)
MESON_SETUP_ARGS += -Dtools=$(TOOLS)
MESON_SETUP_ARGS += $(MESON_ARGS)

.PHONY: all build setup reconfigure install test clean

all: build

$(BUILDDIR)/build.ninja:
	$(MESON) setup $(BUILDDIR) $(MESON_SETUP_ARGS)

setup: $(BUILDDIR)/build.ninja

build: $(BUILDDIR)/build.ninja
	$(MESON) compile -C $(BUILDDIR)

# meson setup runs once; variable changes after that need an explicit
# reconfigure (make cannot see them in the existing build directory).
reconfigure:
	$(MESON) setup --reconfigure $(BUILDDIR) $(MESON_SETUP_ARGS)

install: build
	$(MESON) install -C $(BUILDDIR)

test: build
	$(MESON) test -C $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)
