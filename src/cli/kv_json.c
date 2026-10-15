/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * kv_json.c - JSON encoder
 */

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "core/log.h"
#include "core/memory.h"
#include "kv_impl.h"

int kv_saved_stdout = -1;

static bool
buf_put_quoted(Buf *b, const char *s)
{
	if (s == NULL)
		s = "";
	if (!kv_buf_putc(b, '"'))
		return false;
	if (s[0] == '\0')
		return kv_buf_putc(b, '"');
	for (const unsigned char *p = (const unsigned char *)s; *p; p++)
	{
		char hex[8];
		switch (*p)
		{
		case '"':
			if (!kv_buf_puts(b, "\\\""))
				return false;
			break;
		case '\\':
			if (!kv_buf_puts(b, "\\\\"))
				return false;
			break;
		case '\b':
			if (!kv_buf_puts(b, "\\b"))
				return false;
			break;
		case '\f':
			if (!kv_buf_puts(b, "\\f"))
				return false;
			break;
		case '\n':
			if (!kv_buf_puts(b, "\\n"))
				return false;
			break;
		case '\r':
			if (!kv_buf_puts(b, "\\r"))
				return false;
			break;
		case '\t':
			if (!kv_buf_puts(b, "\\t"))
				return false;
			break;
		default:
			if (*p < 0x20)
			{
				snprintf(hex, sizeof(hex), "\\u%04x", *p);
				if (!kv_buf_puts(b, hex))
					return false;
			}
			else if (!kv_buf_putc(b, (char)*p))
				return false;
			break;
		}
	}
	return kv_buf_putc(b, '"');
}

static bool
emit_json(Buf *b, const Kv *v)
{
	char tmp[64];

	if (v == NULL || v->cls == NULL)
		return kv_buf_puts(b, "null");
	if (v->cls == &kv_class_arr)
	{
		if (!kv_buf_putc(b, '['))
			return false;
		for (size_t i = 0; i < v->arr.n; i++)
		{
			if (i > 0 && !kv_buf_putc(b, ','))
				return false;
			if (!emit_json(b, v->arr.v[i]))
				return false;
		}
		return kv_buf_putc(b, ']');
	}
	if (v->cls == &kv_class_obj)
	{
		if (!kv_buf_putc(b, '{'))
			return false;
		for (size_t i = 0; i < v->obj.n; i++)
		{
			if (i > 0 && !kv_buf_putc(b, ','))
				return false;
			if (!buf_put_quoted(b, v->obj.k[i]) || !kv_buf_putc(b, ':') ||
				!emit_json(b, v->obj.v[i]))
				return false;
		}
		return kv_buf_putc(b, '}');
	}
	if (v->cls == &kv_class_str)
		return buf_put_quoted(b, v->s);
	if (v->cls == &kv_class_null)
		return kv_buf_puts(b, "null");
	if (v->cls == &kv_class_bool)
		return kv_buf_puts(b, v->b ? "true" : "false");
	if (v->cls == &kv_class_int)
	{
		snprintf(tmp, sizeof(tmp), "%lld", (long long)v->i);
		return kv_buf_puts(b, tmp);
	}
	if (v->cls == &kv_class_dbl)
	{
		if (!isfinite(v->d))
			return kv_buf_puts(b, "null");
		snprintf(tmp, sizeof(tmp), "%.17g", v->d);
		return kv_buf_puts(b, tmp);
	}
	return kv_buf_puts(b, "null");
}

char *
kv_to_json(const Kv *v)
{
	Buf b = {0};
	if (!emit_json(&b, v) || !kv_buf_reserve(&b, 0))
		return NULL;
	b.p[b.n] = '\0';
	return b.p;
}

/* Report a JSON write-open failure and free s. */
static int
cannot_write(const char *path, char *s)
{
	fprintf(stderr,
			"Error: cannot write JSON to '%s'\n",
			strcmp(path, "-") == 0 ? "stdout" : path);
	vs_free(s);
	return 1;
}

int
kv_write_json(const Kv *v, const char *path)
{
	if (path == NULL)
		return 1;

	char *s = kv_to_json(v);
	if (s == NULL)
	{
		fprintf(stderr, "Error: failed to stringify JSON\n");
		return 1;
	}

	FILE *f;
	bool  close_f = false;
	if (strcmp(path, "-") == 0)
	{
		int fd = kv_saved_stdout >= 0 ? kv_saved_stdout : STDOUT_FILENO;
		int wr = dup(fd);
		if (wr < 0)
			return cannot_write(path, s);
		f = fdopen(wr, "w");
		if (f == NULL)
		{
			close(wr);
			return cannot_write(path, s);
		}
		close_f = true;
	}
	else
	{
		f = fopen(path, "w");
		if (f == NULL)
			return cannot_write(path, s);
		close_f = true;
	}

	int rc = 0;
	if (fputs(s, f) == EOF || fputc('\n', f) == EOF || fflush(f) == EOF)
	{
		fprintf(stderr, "Error: writing JSON to '%s' failed\n", path);
		rc = 1;
	}
	if (close_f && fclose(f) == EOF)
		rc = 1;
	vs_free(s);
	return rc;
}

int
json_take_path(int *argc, char **argv, const char **out)
{
	*out  = NULL;
	int i = 1;
	while (i < *argc)
	{
		const char *a		= argv[i];
		const char *val		= NULL;
		int			consume = 0;

		if (a == NULL)
			break;
		if (strcmp(a, "--json") == 0)
		{
			const char *next = (i + 1 < *argc) ? argv[i + 1] : NULL;
			if (next != NULL && (next[0] != '-' || strcmp(next, "-") == 0))
			{
				val		= next;
				consume = 2;
			}
			else
			{
				val		= "-";
				consume = 1;
			}
		}
		else if (strncmp(a, "--json=", 7) == 0)
		{
			val		= a[7] != '\0' ? a + 7 : "-";
			consume = 1;
		}

		if (val != NULL)
		{
			*out	 = val;
			int rest = *argc - i - consume;
			memmove(&argv[i],
					&argv[i + consume],
					(size_t)rest * sizeof(char *));
			*argc -= consume;
			argv[*argc] = NULL;
			continue;
		}
		i++;
	}
	return 0;
}

int
json_begin_human(const char *json_path, bool quiet)
{
	bool json_stdout = json_path != NULL && strcmp(json_path, "-") == 0;

	if (!quiet && !json_stdout)
		return 0;

	fflush(stdout);
	/* Keep the saved fd off 0-2 so a closed stderr cannot alias it. */
	kv_saved_stdout = fcntl(STDOUT_FILENO, F_DUPFD, 3);
	if (kv_saved_stdout < 0)
	{
		fprintf(stderr, "Error: cannot save stdout\n");
		return 1;
	}

	if (quiet)
	{
		int nullfd = open("/dev/null", O_WRONLY);

		vs_standalone_quiet = 1;
		if (nullfd < 0 || dup2(nullfd, STDOUT_FILENO) < 0)
		{
			if (nullfd >= 0)
				close(nullfd);
			fprintf(stderr, "Error: cannot quiet stdout\n");
			return 1;
		}
		close(nullfd);
	}
	else if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
	{
		fprintf(stderr, "Error: cannot redirect stdout\n");
		return 1;
	}
	return 0;
}

void
json_end_human(void)
{
	vs_standalone_quiet = 0;
	if (kv_saved_stdout < 0)
		return;
	fflush(stdout);
	dup2(kv_saved_stdout, STDOUT_FILENO);
	close(kv_saved_stdout);
	kv_saved_stdout = -1;
}
