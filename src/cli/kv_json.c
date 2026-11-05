/*
 * Copyright (c) 2026 Tiger Data, Inc.
 * Licensed under the PostgreSQL License. See LICENSE for details.
 *
 * kv_json.c - JSON encoder and decoder
 */

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

/*
 * Decoder for kv_to_json output, not a JSON library. Strings are copied
 * as written (no unescape). Whitespace is allowed so a trailing newline
 * from kv_write_json is fine.
 */
/* Far deeper than any result tree; a bound so hostile nesting cannot
 * recurse the parser off the stack. */
#define JSON_MAX_DEPTH 64

typedef struct JsonParse
{
	const char *p;
	int			depth;
} JsonParse;

static void
jp_skip(JsonParse *j)
{
	while (*j->p == ' ' || *j->p == '\n' || *j->p == '\t' || *j->p == '\r')
		j->p++;
}

static Kv *jp_value(JsonParse *j);

static int
jp_hex(char c)
{
	if (c >= '0' && c <= '9')
		return c - '0';
	if (c >= 'a' && c <= 'f')
		return c - 'a' + 10;
	if (c >= 'A' && c <= 'F')
		return c - 'A' + 10;
	return -1;
}

/*
 * Interior of a JSON string with exactly the escapes the encoder
 * writes: \" \\ \b \f \n \r \t, and \u00XX for the other control
 * characters. Any other escape is an error, as are raw control
 * characters; JSON requires them escaped, and the encoder does.
 */
static char *
jp_string(JsonParse *j)
{
	const char *p;
	const char *q;
	char	   *out;
	size_t		n = 0;

	if (*j->p != '"')
		return NULL;
	p = j->p + 1;

	/* Find the closing quote; the decoded form never outgrows the
	 * escaped span. */
	for (q = p; *q != '\0' && *q != '"';)
	{
		if ((unsigned char)*q < 0x20)
			return NULL;
		if (*q == '\\')
		{
			if (q[1] == '\0')
				return NULL;
			q += 2;
		}
		else
			q++;
	}
	if (*q != '"')
		return NULL;
	out = vs_alloc((size_t)(q - p) + 1);
	if (out == NULL)
		return NULL;

	while (*p != '"')
	{
		if (*p != '\\')
		{
			out[n++] = *p++;
			continue;
		}
		p++;
		switch (*p)
		{
		case '"':
			out[n++] = '"';
			break;
		case '\\':
			out[n++] = '\\';
			break;
		case 'b':
			out[n++] = '\b';
			break;
		case 'f':
			out[n++] = '\f';
			break;
		case 'n':
			out[n++] = '\n';
			break;
		case 'r':
			out[n++] = '\r';
			break;
		case 't':
			out[n++] = '\t';
			break;
		case 'u':
		{
			/* Only the control range the encoder hex-escapes; a NUL
			 * cannot survive in a C string either way. Check one
			 * byte at a time so a truncated escape stops the reads
			 * before the end of the buffer. */
			int hi;
			int lo;
			int c;

			if (p[1] != '0' || p[2] != '0')
				return NULL;
			hi = jp_hex(p[3]);
			if (hi < 0)
				return NULL;
			lo = jp_hex(p[4]);
			if (lo < 0)
				return NULL;
			c = hi * 16 + lo;
			if (c == 0 || c >= 0x20)
				return NULL;
			out[n++] = (char)c;
			p += 4;
			break;
		}
		default:
			return NULL;
		}
		p++;
	}
	out[n] = '\0';
	j->p   = p + 1;
	return out;
}

/* 1 = another item, 0 = closed, -1 = error. */
static int
jp_more(JsonParse *j, char end)
{
	jp_skip(j);
	if (*j->p == ',')
	{
		j->p++;
		return 1;
	}
	if (*j->p == end)
	{
		j->p++;
		return 0;
	}
	return -1;
}

static Kv *
jp_array(JsonParse *j)
{
	Kv *a = kv_arr();

	if (++j->depth > JSON_MAX_DEPTH)
		return NULL;
	j->p++;
	jp_skip(j);
	if (*j->p == ']')
	{
		j->p++;
		j->depth--;
		return a;
	}
	for (;;)
	{
		Kv *v = jp_value(j);
		int m;

		if (v == NULL)
			return NULL;
		kv_arr_add(a, v);
		m = jp_more(j, ']');
		if (m < 0)
			return NULL;
		if (m == 0)
		{
			j->depth--;
			return a;
		}
	}
}

static Kv *
jp_object(JsonParse *j)
{
	Kv *o = kv_obj();

	if (++j->depth > JSON_MAX_DEPTH)
		return NULL;
	j->p++;
	jp_skip(j);
	if (*j->p == '}')
	{
		j->p++;
		j->depth--;
		return o;
	}
	for (;;)
	{
		char *key;
		Kv	 *v;
		int	  m;

		jp_skip(j);
		key = jp_string(j);
		jp_skip(j);
		if (key == NULL || *j->p != ':')
			return NULL;
		/* Readers disagree on which of two same-named keys wins, so
		 * an ambiguous document must not parse at all. */
		if (kv_get(o, key) != NULL)
			return NULL;
		j->p++;
		v = jp_value(j);
		if (v == NULL)
			return NULL;
		kv_obj_put(o, key, v);
		m = jp_more(j, '}');
		if (m < 0)
			return NULL;
		if (m == 0)
		{
			j->depth--;
			return o;
		}
	}
}

/*
 * A number by JSON's own grammar, which is stricter than strtod:
 * no leading +, no leading zeros, no bare or trailing dot, no hex,
 * and nothing non-finite.
 */
static Kv *
jp_number(JsonParse *j)
{
	const char *p		 = j->p;
	bool		integral = true;
	char	   *end;
	double		d;

	if (*p == '-')
		p++;
	if (*p == '0')
		p++;
	else if (*p >= '1' && *p <= '9')
		while (*p >= '0' && *p <= '9')
			p++;
	else
		return NULL;
	if (*p == '.')
	{
		integral = false;
		p++;
		if (*p < '0' || *p > '9')
			return NULL;
		while (*p >= '0' && *p <= '9')
			p++;
	}
	if (*p == 'e' || *p == 'E')
	{
		integral = false;
		p++;
		if (*p == '+' || *p == '-')
			p++;
		if (*p < '0' || *p > '9')
			return NULL;
		while (*p >= '0' && *p <= '9')
			p++;
	}

	/* A double would round an integer past 2^53, so keep what fits
	 * in the encoder's own integer type; wider spills to double. */
	if (integral)
	{
		errno = 0;

		long long i = strtoll(j->p, &end, 10);

		if (end == p && errno == 0)
		{
			j->p = p;
			return kv_int(i);
		}
	}

	d = strtod(j->p, &end);
	/* The grammar and strtod must agree on the token, and the encoder
	 * writes non-finite values as null, so an overflowing literal is
	 * malformed input here. */
	if (end != p || !isfinite(d))
		return NULL;
	j->p = p;
	return kv_dbl(d);
}

static Kv *
jp_value(JsonParse *j)
{
	jp_skip(j);
	if (*j->p == '{')
		return jp_object(j);
	if (*j->p == '[')
		return jp_array(j);
	if (*j->p == '"')
	{
		char *s = jp_string(j);

		return s != NULL ? kv_str(s) : NULL;
	}
	if (strncmp(j->p, "null", 4) == 0)
	{
		j->p += 4;
		return kv_null();
	}
	if (strncmp(j->p, "true", 4) == 0)
	{
		j->p += 4;
		return kv_bool(true);
	}
	if (strncmp(j->p, "false", 5) == 0)
	{
		j->p += 5;
		return kv_bool(false);
	}
	return jp_number(j);
}

Kv *
kv_from_json(const char *s)
{
	JsonParse j;
	Kv		 *v;

	if (s == NULL)
		return NULL;
	j.p		= s;
	j.depth = 0;
	v		= jp_value(&j);
	jp_skip(&j);
	if (v == NULL || *j.p != '\0')
		return NULL;
	return v;
}

/* Big enough for any bench result; a larger file fails the parse. */
#define JSON_FILE_CAP (1 << 20)

Kv *
kv_from_json_file(const char *path)
{
	FILE *f;
	char *buf;

	if (path == NULL)
		return NULL;
	f = fopen(path, "r");
	if (f == NULL)
	{
		fprintf(stderr, "Error: cannot read '%s'\n", path);
		return NULL;
	}
	buf = vs_alloc(JSON_FILE_CAP);
	if (buf == NULL)
	{
		fclose(f);
		return NULL;
	}

	size_t nread = fread(buf, 1, JSON_FILE_CAP - 1, f);

	if (ferror(f) != 0)
	{
		fprintf(stderr, "Error: cannot read '%s'\n", path);
		fclose(f);
		return NULL;
	}
	/* A file the cap truncates must not parse as its prefix. */
	if (nread == JSON_FILE_CAP - 1 && fgetc(f) != EOF)
	{
		fprintf(stderr, "Error: '%s' is too large\n", path);
		fclose(f);
		return NULL;
	}
	fclose(f);
	/* The parser stops at a NUL, so a stray one would hide whatever
	 * follows it rather than fail the parse. */
	if (memchr(buf, '\0', nread) != NULL)
	{
		fprintf(stderr, "Error: '%s' contains a NUL byte\n", path);
		return NULL;
	}
	/* In bounds: fread returns at most the JSON_FILE_CAP - 1 asked
	 * for. The analyzer cannot see that through the tainted count. */
	/* NOLINTNEXTLINE(clang-analyzer-security.ArrayBound) */
	buf[nread] = '\0';
	Kv *v	   = kv_from_json(buf);

	if (v == NULL)
		fprintf(stderr, "Error: bad JSON in '%s'\n", path);
	return v;
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
