/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * library_guard.h - find the extension library among loaded libraries
 */

#ifndef VS_LIBRARY_GUARD_H
#define VS_LIBRARY_GUARD_H

#include "vs_config.h"

#include <string.h>

/*
 * Return the file name in `path` if it names the extension library, as
 * "<name>.<suffix>" or "<name>-<version>.<suffix>", and NULL otherwise. A
 * version starts with a digit, which keeps a sibling such as
 * "<name>-helper.so" from matching.
 *
 * Inline so the load guard calls no symbol that could resolve to another
 * loaded copy of the library.
 */
static inline const char *
vs_extension_library_file(const char *path)
{
	const size_t namelen = strlen(VS_MODULE_NAME);
	const char	*sep	 = strrchr(path, '/');
	const char	*file	 = sep ? sep + 1 : path;
	const char	*rest;

	if (strncmp(file, VS_MODULE_NAME, namelen) != 0)
		return NULL;

	rest = file + namelen;
	if (rest[0] == '-' && rest[1] >= '0' && rest[1] <= '9')
		rest = strchr(rest, '.');

	return rest != NULL && rest[0] == '.' && rest[1] != '\0' ? file : NULL;
}

/*
 * Access to the list of libraries loaded in the process. A module handle
 * is opaque and NULL ends the list.
 */
typedef struct VsModuleList
{
	void *(*first)(void);
	void *(*next)(void *module);
	const char *(*path)(void *module);
} VsModuleList;

/*
 * Return the file name of the first extension library in `list`, or NULL
 * if it holds none.
 */
static inline const char *
vs_find_extension_library(const VsModuleList *list)
{
	for (void *module = list->first(); module; module = list->next(module))
	{
		const char *file = vs_extension_library_file(list->path(module));

		if (file != NULL)
			return file;
	}

	return NULL;
}

#endif /* VS_LIBRARY_GUARD_H */
