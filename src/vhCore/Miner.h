#ifndef Miner_INCLUDE_ONCE
#define Miner_INCLUDE_ONCE

/*
 * Copyright 2018-2020 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#include <stdbool.h>
#include <inttypes.h>
#include <jansson.h>
#include <curl/curl.h>
#include <external/tinycthread/tinycthread.h>

#include <stdlib.h>
#include <stddef.h>

/* Define to the full name of this package. */
#define PACKAGE_NAME "VerthashMiner"
/* Define to the version of this package. */
#define PACKAGE_VERSION "0.6.0"

// alloca detection
#if !defined(alloca)
#if defined(__GLIBC__) || defined(__sun) || defined(__CYGWIN__)
#include <alloca.h>     // alloca
#elif defined(_WIN32)
#include <malloc.h>     // alloca
#if !defined(alloca)
#define alloca _alloca  // clang with MS Codegen
#endif
#else
#include <stdlib.h>     // alloca
#endif
#endif


#ifdef WIN32
#include <windows.h>
#elif _POSIX_C_SOURCE >= 199309L
#include <time.h>   // for nanosleep
#else
#include <unistd.h> // for usleep
#endif

inline static void sleep_ms(int milliseconds)
{
#ifdef WIN32
    Sleep(milliseconds);
#elif _POSIX_C_SOURCE >= 199309L
    struct timespec ts;
    ts.tv_sec = milliseconds / 1000;
    ts.tv_nsec = (milliseconds % 1000) * 1000000;
    nanosleep(&ts, NULL);
#else
    usleep(milliseconds * 1000);
#endif
}


#ifdef _MSC_VER 
//not #if defined(_WIN32) || defined(_WIN64) because we have strncasecmp in mingw
#define snprintf(...) _snprintf(__VA_ARGS__)
#define strdup(...) _strdup(__VA_ARGS__)
#define strncasecmp(x,y,z) _strnicmp(x,y,z)
#define strcasecmp(x,y) _stricmp(x,y)
typedef int ssize_t;
#endif


#undef unlikely
#undef likely
#if defined(__GNUC__) && (__GNUC__ > 2) && defined(__OPTIMIZE__)
#define unlikely(expr) (__builtin_expect(!!(expr), 0))
#define likely(expr) (__builtin_expect(!!(expr), 1))
#else
#define unlikely(expr) (expr)
#define likely(expr) (expr)
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

#if JANSSON_MAJOR_VERSION >= 2
#define JSON_LOADS(str, err_ptr) json_loads(str, 0, err_ptr)
#define JSON_LOAD_FILE(path, err_ptr) json_load_file(path, 0, err_ptr)
#else
#define JSON_LOADS(str, err_ptr) json_loads(str, err_ptr)
#define JSON_LOAD_FILE(path, err_ptr) json_load_file(path, err_ptr)
#endif

#define USER_AGENT PACKAGE_NAME "/" PACKAGE_VERSION

typedef enum
{
	LOG_ERR     = 0,
	LOG_WARNING = 1,
	LOG_INFO    = 2,
	LOG_DEBUG   = 3
} ELogPriority;

//-----------------------------------------------------------------------------
struct thread_q;

struct thr_info
{
	int		id;
	thrd_t	pth;
	struct thread_q	*q;
};

//-----------------------------------------------------------------------------
struct work_restart
{
	volatile unsigned long	restart;
	char padding[128 - sizeof(unsigned long)];
};

//-----------------------------------------------------------------------------
extern bool opt_debug;
extern bool opt_protocol;
extern bool opt_redirect;
extern int opt_longpoll_timeout;
extern bool want_longpoll;
extern bool have_longpoll;
extern bool have_stratum;
extern char *opt_cert;
extern char *opt_proxy;
extern long opt_proxy_type;
extern mtx_t applog_lock;
extern FILE* applog_file;
extern bool opt_log_file;
extern struct thr_info *thr_info;
extern int longpoll_thr_id;
extern int stratum_thr_id;
extern struct work_restart *work_restart;

extern volatile bool abort_flag;

//-----------------------------------------------------------------------------
#define JSON_RPC_LONGPOLL	(1 << 0)
#define JSON_RPC_QUIET_404	(1 << 1)

//-----------------------------------------------------------------------------
// Logger
extern void applog(ELogPriority prio, const char *fmt, ...);

//-----------------------------------------------------------------------------
// Utils
extern json_t *json_rpc_call(CURL *curl, const char *url, const char *user, const char *pass,
                             const char *rpc_req, int *curl_err, int flags);
void memrev(unsigned char *p, size_t len);
extern void bin2hex(char *s, const unsigned char *p, size_t len);
extern char *abin2hex(const unsigned char *p, size_t len);
extern bool hex2bin(unsigned char *p, const char *hexstr, size_t len);
extern int varint_encode(unsigned char *p, uint64_t n);
extern size_t address_to_script(unsigned char *out, size_t outsz, const char *addr);
extern bool fulltest(const uint32_t *hash, const uint32_t *target);
extern void diff_to_target(uint32_t *target, double diff);

//-----------------------------------------------------------------------------
// Stratum API
struct stratum_job
{
	char *job_id;
	unsigned char prevhash[32];
	size_t coinbase_size;
	unsigned char *coinbase;
	unsigned char *xnonce2;
	int merkle_count;
	unsigned char **merkle;
	unsigned char version[4];
	unsigned char nbits[4];
	unsigned char ntime[4];
	bool clean;
	double diff;
};

struct stratum_ctx
{
	char *url;

	CURL *curl;
	char *curl_url;
	char curl_err_str[CURL_ERROR_SIZE];
	curl_socket_t sock;
	size_t sockbuf_size;
	char *sockbuf;
	mtx_t sock_lock;
    curl_socket_t dummy_socket;

	double next_diff;

	char *session_id;
	size_t xnonce1_size;
	unsigned char *xnonce1;
	size_t xnonce2_size;
	struct stratum_job job;
	mtx_t work_lock;
};

bool stratum_socket_full(struct stratum_ctx *sctx, int timeout);
bool stratum_send_line(struct stratum_ctx *sctx, char *s);
char *stratum_recv_line(struct stratum_ctx *sctx);
bool stratum_connect(struct stratum_ctx *sctx, const char *url);
void stratum_disconnect(struct stratum_ctx *sctx);
bool stratum_subscribe(struct stratum_ctx *sctx);
bool stratum_authorize(struct stratum_ctx *sctx, const char *user, const char *pass);
bool stratum_handle_method(struct stratum_ctx *sctx, const char *s);
uint32_t stratum_get_block_height(struct stratum_ctx *sctx);

#endif // !Miner_INCLUDE_ONCE
