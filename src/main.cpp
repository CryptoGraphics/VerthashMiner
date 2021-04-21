/*
 * Copyright 2010 Jeff Garzik
 * Copyright 2012-2017 pooler
 * Copyright 2018-2021 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include <stdint.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <assert.h>

#include <vector>
#include <chrono>
#include <algorithm>
#include <atomic>

#include <jansson.h>
#include <curl/curl.h>

#ifdef WIN32
#include <windows.h>
#else
#include <errno.h>
#include <signal.h>
#include <sys/resource.h>
#if HAVE_SYS_SYSCTL_H
#include <sys/types.h>
#if HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
#include <sys/sysctl.h>
#endif
#include <netinet/in.h>
#endif

#define FOPEN_UTF8_IMPLEMENTATION
#include "external/fopen_utf8.h"

#include "vhCore/Miner.h"
#include "vhCore/SHA256.h"
#include "vhCore/Verthash.h"
#include "vhCore/ConfigFile.h"
#include "vhCore/ThreadQueue.h"

#include "vhDevice/DeviceUtils.h"
#include "vhDevice/ConfigGenerator.h"
// Monitoring
#include "vhDevice/ADLUtils.h"
#include "vhDevice/NVMLUtils.h"
#include "vhDevice/SYSFSUtils.h"

#ifdef _WIN32
#include <external/getopt/getopt.h>
#else
#include <getopt.h>
#endif

#define LP_SCANTIME 60

//-----------------------------------------------------------------------------
enum workio_commands
{
    WC_GET_WORK,
    WC_SUBMIT_WORK
};

struct workio_cmd
{
    enum workio_commands cmd;
    struct thr_info* thr;
    struct work *workInfo;
};

// Global objects
static verthash_info_t verthashInfo;

// Verthash data file hash in bytes for verification
// 0x48aa21d7afededb63976d48a8ff8ec29d5b02563af4a1110b056cd43e83155a5
static const uint8_t verthashDatFileHash_bytes[32] = { 0xa5, 0x55, 0x31, 0xe8, 0x43, 0xcd, 0x56, 0xb0,
                                                       0x10, 0x11, 0x4a, 0xaf, 0x63, 0x25, 0xb0, 0xd5,
                                                       0x29, 0xec, 0xf8, 0x8f, 0x8a, 0xd4, 0x76, 0x39,
                                                       0xb6, 0xed, 0xed, 0xaf, 0xd7, 0x21, 0xaa, 0x48 };


bool opt_debug = false;
bool opt_protocol = false;
static bool opt_benchmark = false;
bool opt_redirect = true;
bool want_longpoll = true;
bool have_longpoll = false;
bool have_stratum = false;
static int opt_retries = -1;
static int opt_fail_pause = 30;
int opt_longpoll_timeout = 0;
static int opt_scantime = 5;
static size_t opt_n_threads;
static char *rpc_url;
static char *rpc_user, *rpc_pass;
static int pk_script_size;
static unsigned char pk_script[42];
static char coinbase_sig[101] = "";
char *opt_cert;
char *opt_proxy;
long opt_proxy_type;
struct thr_info *thr_info;
static int work_thr_id;
int longpoll_thr_id = -1;
int stratum_thr_id = -1;
struct work_restart *work_restart = NULL;
static struct stratum_ctx stratum;

// Used to trigger application exit if all workers fail.
static std::atomic<int> numWorkersExited{0};

// logger
mtx_t applog_lock;
FILE* applog_file = NULL;
bool opt_log_file = false;

mtx_t stats_lock;



volatile bool abort_flag = false;

static unsigned long accepted_count = 0L;
static unsigned long rejected_count = 0L;
static double *thr_hashrates;

static char const usage[] = "\n"
PACKAGE_NAME " " PACKAGE_VERSION " by CryptoGraphics <CrGr@protonmail.com>\n"
"\n"
"Usage: VerthashMiner [options]\n"
"\n"
"Options:\n"
"=======\n"
"\n"
"--algo <algorithm>                                 (-a)\n\t"
    "Specify the algorithm to use:\n\t"
"- verthash (default)\n"
"\n"
"--url  <address>                                   (-o)\n\t"
    "Set URL of mining server in format (address:port).\n"
"\n"
"--user <user>                                      (-u)\n\t"
    "Set username for mining server.\n"
"\n"
"--pass <password>                                  (-p)\n\t"
    "Set password for mining server.\n"
"\n"
"--cert <File>\n\t"
    "Select certificate for mining server using SSL.\n"
"\n"
"--proxy <[PROTOCOL://]HOST[:PORT]>                 (-x)\n\t"
    "Connect through a proxy.\n"
"\n"
"--cl-devices <index,index,...>  (-d)\n\t"
    "Select specific OpenCL devices from the list, obtained by '-l' command.\n"
"\n"
"--all-cl-devices\n\t"
    "Use all available OpenCL devices from the list, obtained by '-l' command.\n\t"
    "This options as a priority over per device selection using '--cl-devices'\n"
"\n"
"--cu-devices <index,index,...>  (-D)\n\t"
    "Select specific CUDA devices from the list, obtained by '-l' command.\n"
"\n"
"--all-cu-devices\n\t"
    "Use all available CUDA devices from the list, obtained by '-l' command.\n\t"
    "This options as a priority over per device selection using '--cu-devices'\n"
"\n"
"--retries <N>                                      (-r)\n\t"
    "Set number of times to retry if a network call fails.\n\t"
    "(default: retry indefinitely)\n"
"\n"
"--retry-pause <N>                                  (-R)\n\t"
    "Time to pause between retries, in seconds.\n\t"
    "(default: 30)\n"
"\n"
"--timeout <N>                                      (-T)\n\t"
    "Timeout for long polling, in seconds.\n\t"
    "(default: none)\n"
"\n"
"--scantime <N>                                     (-s)\n\t"
    "Upper bound on time spent scanning current work when\n\t"
    "long polling is unavailable, in seconds.\n\t"
    "(default: 5)\n"
"\n"
"--coinbase-addr <ADDR>\n\t"
    "Specify a payout address for solo mining.\n"
"\n"
"--coinbase-sig <text>\n\t"
    "Data to insert in the coinbase when possible.\n"
"\n"
"--no-longpoll\n\t"
    "Disable long polling support.\n"
"\n"
"--no-redirect\n\t"
    "Ignore requests to change the URL of the mining server.\n"
"\n"
"--no-restrict-cuda\n\t"
    "Allow to use NVIDIA GPUs on OpenCL platform even if CUDA is available.\n"
"\n"
"--verbose\n\t"
    "Enable extra debug output.\n"
"\n"
"--protocol-dump                        (-P)\n\t"
    "Cerbose dump of protocol-level activities.\n"
"\n"
"--benchmark\n\t"
    "Run in offline benchmark mode.\n"
"\n"
"--config <File>                        (-c)\n\t"
    "Load a configuration file.\n"
"\n"
"--gen-conf <File>                      (-g)\n\t"
    "Generate a configuration file with pcie bus IDs(if possible) and exit.\n"
"\n"
"--gen-verthash-data <File>\n\t"
    "Generate a verthash data file and exit.\n"
"\n"
"--verthash-data <File>                 (-f)\n\t"
    "Specify verthash mining data file.\n"
"\n"
"--no-verthash-data_verification\n\t"
    "Disable verthash data file verification.\n"
"\n"
"--log-file\n\t"
    "Enables logging to file.\n"
"\n"
"--device-list                          (-l)\n\t"
    "Print all available device configurations and exit.\n"
"\n"
"--version                              (-v)\n\t"
    "Display version information and exit.\n"
"\n"
"--help                                 (-h)\n\t"
    "Display this help text and exit.\n"
"\n";

static char const short_options[] =
    "a:c:hp:Px:r:R:s:d:D:T:o:u:g:G:f:lv";

static struct option const options[] = {
    { "algo", 1, NULL, 'a' },
    { "benchmark", 0, NULL, 1005 },
    { "cert", 1, NULL, 1001 },
    { "coinbase-addr", 1, NULL, 1013 },
    { "coinbase-sig", 1, NULL, 1015 },
    { "no-restrict-cuda", 0, NULL, 1016 },
    { "config", 1, NULL, 'c' },
    { "verbose", 0, NULL, 1018 },
    { "help", 0, NULL, 'h' },
    { "no-longpoll", 0, NULL, 1003 },
    { "no-redirect", 0, NULL, 1009 },
    { "pass", 1, NULL, 'p' },
    { "protocol-dump", 0, NULL, 'P' },
    { "proxy", 1, NULL, 'x' },
    { "retries", 1, NULL, 'r' },
    { "retry-pause", 1, NULL, 'R' },
    { "scantime", 1, NULL, 's' },
    { "cl-devices", 1, NULL, 'd' },
    { "all-cl-devices", 0, NULL, 1019 },
    { "cu-devices", 1, NULL, 'D' },
    { "all-cu-devices", 0, NULL, 1020 },
    { "timeout", 1, NULL, 'T' },
    { "url", 1, NULL, 'o' },
    { "user", 1, NULL, 'u' },
    { "gen-conf", 1, NULL, 'g' },
    { "gen-verthash-data", 1, NULL, 1023 },
    { "verthash-data", 1, NULL, 'f' },
    { "no-verthash-data_verification", 0, NULL, 1021 },
    { "log-file", 0, NULL, 1022},
    { "device-list", 0, NULL, 'l' },
    { "version", 0, NULL, 'v' },
    { 0, 0, 0, 0 }
};

struct work {
    uint32_t data[32];
    uint32_t target[8];

    int height;
    char *txs;
    char *workid;

    char *job_id;
    size_t xnonce2_len;
    unsigned char *xnonce2;
};

static struct work g_work;
static time_t g_work_time;
static mtx_t g_work_lock;
static bool submit_old = false;
static char *lp_id;

static inline void work_free(struct work *w)
{
    free(w->txs);
    free(w->workid);
    free(w->job_id);
    free(w->xnonce2);
}

static inline void work_copy(struct work *dest, const struct work *src)
{
    memcpy(dest, src, sizeof(struct work));
    if (src->txs)
        dest->txs = strdup(src->txs);
    if (src->workid)
        dest->workid = strdup(src->workid);
    if (src->job_id)
        dest->job_id = strdup(src->job_id);
    if (src->xnonce2) {
        dest->xnonce2 = (unsigned char*)malloc(src->xnonce2_len);
        memcpy(dest->xnonce2, src->xnonce2, src->xnonce2_len);
    }
}

static bool jobj_binary(const json_t *obj, const char *key,
                        void *buf, size_t buflen)
{
    const char *hexstr;
    json_t *tmp;

    tmp = json_object_get(obj, key);
    if (unlikely(!tmp)) {
        applog(LOG_ERR, "JSON key '%s' not found", key);
        return false;
    }
    hexstr = json_string_value(tmp);
    if (unlikely(!hexstr)) {
        applog(LOG_ERR, "JSON key '%s' is not a string", key);
        return false;
    }
    if (!hex2bin((unsigned char*)buf, hexstr, buflen))
        return false;

    return true;
}

static bool gbt_work_decode(const json_t *val, struct work *work_info)
{
    if (opt_debug)
    {
        applog(LOG_DEBUG, "GBT_work_decode call");
    }

    int i;
    size_t n;
    uint32_t version, curtime, bits;
    uint32_t prevhash[8];
    uint32_t target[8];
    int blockHeight;
    int errorCode;
    int cbtx_size;
    unsigned char *cbtx = NULL;
    int tx_count, tx_size;
    unsigned char txc_vi[9];
    unsigned char (*merkle_tree)[32] = NULL;
    bool coinbase_append = false;
    bool submit_coinbase = false;
    bool segwit = false;
    json_t *tmp, *txa;
    bool rc = false;

    tmp = json_object_get(val, "rules");
    if (tmp && json_is_array(tmp)) {
        n = json_array_size(tmp);
        for (i = 0; i < n; i++) {
            const char *s = json_string_value(json_array_get(tmp, i));
            if (!s)
                continue;
            if (!strcmp(s, "segwit") || !strcmp(s, "!segwit"))
                segwit = true;
        }
    }

    tmp = json_object_get(val, "mutable");
    if (tmp && json_is_array(tmp)) {
        n = json_array_size(tmp);
        for (i = 0; i < n; i++) {
            const char *s = json_string_value(json_array_get(tmp, i));
            if (!s)
                continue;
            if (!strcmp(s, "coinbase/append"))
                coinbase_append = true;
            else if (!strcmp(s, "submit/coinbase"))
                submit_coinbase = true;
        }
    }

    tmp = json_object_get(val, "height");
    if (!tmp || !json_is_integer(tmp)) {
        applog(LOG_ERR, "JSON invalid height");
        goto out;
    }
    
    //-------------------------------------
    // Update verthash data if needed
    blockHeight = (int)json_integer_value(tmp);
    /*errorCode = verthash_info_update_data(&verthashInfo, blockHeight);
    if (errorCode != 0)
    {
        switch (errorCode)
        {
        case 1:
            {
                applog(LOG_ERR, "Failed to open verthash data file.");
            } break;
        case 2:
            {
                applog(LOG_ERR, "Invalid Verthash data file size. Must be >= lookup area.");
            } break;
        case 3:
            {
                applog(LOG_ERR, "Verthash data out of memory error.");
            } break;
        }

        // Trigger application exit.
        if (!abort_flag) { abort_flag = true; }

        goto out;
    }
    */
    work_info->height = blockHeight;

    if (opt_debug)
        applog(LOG_DEBUG, "block_height: %d", work_info->height);
    //-------------------------------------


    tmp = json_object_get(val, "version");
    if (!tmp || !json_is_integer(tmp)) {
        applog(LOG_ERR, "JSON invalid version");
        goto out;
    }
    version = (uint32_t)json_integer_value(tmp);

    if (unlikely(!jobj_binary(val, "previousblockhash", prevhash, sizeof(prevhash)))) {
        applog(LOG_ERR, "JSON invalid previousblockhash");
        goto out;
    }

    tmp = json_object_get(val, "curtime");
    if (!tmp || !json_is_integer(tmp)) {
        applog(LOG_ERR, "JSON invalid curtime");
        goto out;
    }
    curtime = (uint32_t)json_integer_value(tmp);

    if (unlikely(!jobj_binary(val, "bits", &bits, sizeof(bits)))) {
        applog(LOG_ERR, "JSON invalid bits");
        goto out;
    }

    /* find count and size of transactions */
    txa = json_object_get(val, "transactions");
    if (!txa || !json_is_array(txa)) {
        applog(LOG_ERR, "JSON invalid transactions");
        goto out;
    }
    tx_count = (int)json_array_size(txa);
    tx_size = 0;
    for (i = 0; i < tx_count; i++) {
        const json_t *tx = json_array_get(txa, i);
        const char *tx_hex = json_string_value(json_object_get(tx, "data"));
        if (!tx_hex) {
            applog(LOG_ERR, "JSON invalid transactions");
            goto out;
        }
        tx_size += strlen(tx_hex) / 2;
    }

    /* build coinbase transaction */
    tmp = json_object_get(val, "coinbasetxn");
    if (tmp)
    {
        const char *cbtx_hex = json_string_value(json_object_get(tmp, "data"));
        cbtx_size = cbtx_hex ? strlen(cbtx_hex) / 2 : 0;
        cbtx = (unsigned char*)malloc(cbtx_size + 100);
        if (cbtx_size < 60 || !hex2bin(cbtx, cbtx_hex, cbtx_size))
        {
            applog(LOG_ERR, "JSON invalid coinbasetxn");
            goto out;
        }
    }
    else
    {
        int64_t cbvalue;
        if (!pk_script_size)
        {
            applog(LOG_ERR, "No payout address provided");
            goto out;
        }
        tmp = json_object_get(val, "coinbasevalue");
        if (!tmp || !json_is_number(tmp))
        {
            applog(LOG_ERR, "JSON invalid coinbasevalue");
            goto out;
        }
        cbvalue = json_is_integer(tmp) ? json_integer_value(tmp) : json_number_value(tmp);
        cbtx = (unsigned char*)malloc(256);
        le32enc((uint32_t *)cbtx, 1); /* version */
        cbtx[4] = 1; /* in-counter */
        memset(cbtx+5, 0x00, 32); /* prev txout hash */
        le32enc((uint32_t *)(cbtx+37), 0xffffffff); /* prev txout index */
        cbtx_size = 43;
        /* BIP 34: height in coinbase */
        for (n = work_info->height; n; n >>= 8)
        {
            cbtx[cbtx_size++] = n & 0xff;
            if (n < 0x100 && n >= 0x80)
            {
                cbtx[cbtx_size++] = 0;
            }
        }
        cbtx[42] = cbtx_size - 43;
        cbtx[41] = cbtx_size - 42; /* scriptsig length */
        le32enc((uint32_t *)(cbtx+cbtx_size), 0xffffffff); /* sequence */
        cbtx_size += 4;
        cbtx[cbtx_size++] = segwit ? 2 : 1; /* out-counter */
        le32enc((uint32_t *)(cbtx+cbtx_size), (uint32_t)cbvalue); /* value */
        le32enc((uint32_t *)(cbtx+cbtx_size+4), cbvalue >> 32);
        cbtx_size += 8;
        cbtx[cbtx_size++] = pk_script_size; /* txout-script length */
        memcpy(cbtx+cbtx_size, pk_script, pk_script_size);
        cbtx_size += pk_script_size;
        if (segwit)
        {
            unsigned char (*wtree)[32] = (unsigned char(*)[32])calloc(tx_count + 2, 32);
            memset(cbtx+cbtx_size, 0, 8); /* value */
            cbtx_size += 8;
            cbtx[cbtx_size++] = 38; /* txout-script length */
            cbtx[cbtx_size++] = 0x6a; /* txout-script */
            cbtx[cbtx_size++] = 0x24;
            cbtx[cbtx_size++] = 0xaa;
            cbtx[cbtx_size++] = 0x21;
            cbtx[cbtx_size++] = 0xa9;
            cbtx[cbtx_size++] = 0xed;
            for (i = 0; i < tx_count; i++)
            {
                const json_t *tx = json_array_get(txa, i);
                const json_t *hash = json_object_get(tx, "hash");
                if (!hash || !hex2bin(wtree[1+i], json_string_value(hash), 32))
                {
                    applog(LOG_ERR, "JSON invalid transaction hash");
                    free(wtree);
                    goto out;
                }
                memrev(wtree[1+i], 32);
            }
            n = tx_count + 1;
            while (n > 1) {
                if (n % 2)
                    memcpy(wtree[n], wtree[n-1], 32);
                n = (n + 1) / 2;
                for (i = 0; i < n; i++)
                    sha256d(wtree[i], wtree[2*i], 64);
            }
            memset(wtree[1], 0, 32);  /* witness reserved value = 0 */
            sha256d(cbtx+cbtx_size, wtree[0], 64);
            cbtx_size += 32;
            free(wtree);
        }
        le32enc((uint32_t *)(cbtx+cbtx_size), 0); /* lock time */
        cbtx_size += 4;
        coinbase_append = true;
    }
    if (coinbase_append) {
        unsigned char xsig[100];
        int xsig_len = 0;
        if (*coinbase_sig) {
            n = strlen(coinbase_sig);
            if (cbtx[41] + xsig_len + n <= 100) {
                memcpy(xsig+xsig_len, coinbase_sig, n);
                xsig_len += n;
            } else {
                applog(LOG_WARNING, "Signature does not fit in coinbase, skipping");
            }
        }
        tmp = json_object_get(val, "coinbaseaux");
        if (tmp && json_is_object(tmp)) {
            void *iter = json_object_iter(tmp);
            while (iter) {
                unsigned char buf[100];
                const char *s = json_string_value(json_object_iter_value(iter));
                n = s ? strlen(s) / 2 : 0;
                if (!s || n > 100 || !hex2bin(buf, s, n)) {
                    applog(LOG_ERR, "JSON invalid coinbaseaux");
                    break;
                }
                if (cbtx[41] + xsig_len + n <= 100) {
                    memcpy(xsig+xsig_len, buf, n);
                    xsig_len += n;
                }
                iter = json_object_iter_next(tmp, iter);
            }
        }
        if (xsig_len) {
            unsigned char *ssig_end = cbtx + 42 + cbtx[41];
            int push_len = cbtx[41] + xsig_len < 76 ? 1 :
                           cbtx[41] + 2 + xsig_len > 100 ? 0 : 2;
            n = xsig_len + push_len;
            memmove(ssig_end + n, ssig_end, cbtx_size - 42 - cbtx[41]);
            cbtx[41] += n;
            if (push_len == 2)
                *(ssig_end++) = 0x4c; /* OP_PUSHDATA1 */
            if (push_len)
                *(ssig_end++) = xsig_len;
            memcpy(ssig_end, xsig, xsig_len);
            cbtx_size += n;
        }
    }

    n = varint_encode(txc_vi, 1 + tx_count);
    work_info->txs = (char*)malloc(2 * (n + cbtx_size + tx_size) + 1);
    bin2hex(work_info->txs, txc_vi, n);
    bin2hex(work_info->txs + 2*n, cbtx, cbtx_size);

    /* generate merkle root */
    merkle_tree = (unsigned char(*)[32])malloc(32 * ((1 + tx_count + 1) & ~1));
    sha256d(merkle_tree[0], cbtx, cbtx_size);
    for (i = 0; i < tx_count; i++) {
        tmp = json_array_get(txa, i);
        const char *tx_hex = json_string_value(json_object_get(tmp, "data"));
        const int tx_size = tx_hex ? strlen(tx_hex) / 2 : 0;
        if (segwit) {
            const char *txid = json_string_value(json_object_get(tmp, "txid"));
            if (!txid || !hex2bin(merkle_tree[1 + i], txid, 32)) {
                applog(LOG_ERR, "JSON invalid transaction txid");
                goto out;
            }
            memrev(merkle_tree[1 + i], 32);
        } else {
            unsigned char *tx = (unsigned char*)malloc(tx_size);
            if (!tx_hex || !hex2bin(tx, tx_hex, tx_size)) {
                applog(LOG_ERR, "JSON invalid transactions");
                free(tx);
                goto out;
            }
            sha256d(merkle_tree[1 + i], tx, tx_size);
            free(tx);
        }
        if (!submit_coinbase)
            strcat(work_info->txs, tx_hex);
    }
    n = 1 + tx_count;
    while (n > 1) {
        if (n % 2) {
            memcpy(merkle_tree[n], merkle_tree[n-1], 32);
            ++n;
        }
        n /= 2;
        for (i = 0; i < n; i++)
            sha256d(merkle_tree[i], merkle_tree[2*i], 64);
    }

    /* assemble block header */
    work_info->data[0] = swab32(version);
    for (i = 0; i < 8; i++)
        work_info->data[8 - i] = le32dec(prevhash + i);
    for (i = 0; i < 8; i++)
        work_info->data[9 + i] = be32dec((uint32_t *)merkle_tree[0] + i);
    work_info->data[17] = swab32(curtime);
    work_info->data[18] = le32dec(&bits);
    memset(work_info->data + 19, 0x00, 52);
    work_info->data[20] = 0x80000000;
    work_info->data[31] = 0x00000280;

    if (unlikely(!jobj_binary(val, "target", target, sizeof(target)))) {
        applog(LOG_ERR, "JSON invalid target");
        goto out;
    }
    for (i = 0; i < ARRAY_SIZE(work_info->target); i++)
        work_info->target[7 - i] = be32dec(target + i);

    tmp = json_object_get(val, "workid");
    if (tmp) {
        if (!json_is_string(tmp)) {
            applog(LOG_ERR, "JSON invalid workid");
            goto out;
        }
        work_info->workid = strdup(json_string_value(tmp));
    }

    /* Long polling */
    tmp = json_object_get(val, "longpollid");
    if (want_longpoll && json_is_string(tmp)) {
        free(lp_id);
        lp_id = strdup(json_string_value(tmp));
        if (!have_longpoll) {
            char *lp_uri;
            tmp = json_object_get(val, "longpolluri");
            lp_uri = strdup(json_is_string(tmp) ? json_string_value(tmp) : rpc_url);
            have_longpoll = true;
            tq_push(thr_info[longpoll_thr_id].q, lp_uri);
        }
    }

    rc = true;

out:
    free(cbtx);
    free(merkle_tree);
    return rc;
}

static void share_result(int result, const char* reject_reason)
{
    char s[345];
    double hashrate;
    int i;

    // record total hash-rate
    hashrate = 0.;
    mtx_lock(&stats_lock);
    for (i = 0; i < opt_n_threads; i++)
    {
        hashrate += thr_hashrates[i];
    }
    result ? accepted_count++ : rejected_count++;
    mtx_unlock(&stats_lock);
    
    // print result
    //sprintf(s, "%.2f Mhash/s", hashrate);
    sprintf(s, "%.2f kH/s", hashrate);

    applog(LOG_INFO, "accepted: %lu/%lu (%.2f%%), total hashrate: %s",
           accepted_count,
           accepted_count + rejected_count,
           100. * accepted_count / (accepted_count + rejected_count),
           hashrate != 0? s : "(pending...)");

    if (opt_debug && reject_reason)
    {
        applog(LOG_DEBUG, "reject reason: %s", reject_reason);
    }
}

static bool submit_upstream_work(CURL *curl, struct work *work)
{
    json_t *val, *res;
    char data_str[2 * sizeof(work->data) + 1];
    int i;
    bool rc = false;

    /* pass if the previous hash is not the current previous hash */
    if (!submit_old && memcmp(work->data + 1, g_work.data + 1, 32))
    {
        if (opt_debug)
        {
            applog(LOG_DEBUG, "stale work detected, discarding");
        }

        return true;
    }

    if (have_stratum)
    {
        if (opt_debug)
            applog(LOG_DEBUG, "stratum.submit");

        uint32_t ntime, nonce;
        char ntimestr[9], noncestr[9], *xnonce2str, *req;

        le32enc(&ntime, work->data[17]);
        be32enc(&nonce, work->data[19]); // Verthash
        bin2hex(ntimestr, (const unsigned char *)(&ntime), 4);
        bin2hex(noncestr, (const unsigned char *)(&nonce), 4);
        xnonce2str = abin2hex(work->xnonce2, work->xnonce2_len);
        req = (char*)malloc(256 + strlen(rpc_user) + strlen(work->job_id) + 2 * work->xnonce2_len);
        sprintf(req,
            "{\"method\": \"mining.submit\", \"params\": [\"%s\", \"%s\", \"%s\", \"%s\", \"%s\"], \"id\":4}",
            rpc_user, work->job_id, xnonce2str, ntimestr, noncestr);
        free(xnonce2str);

        rc = stratum_send_line(&stratum, req);
        free(req);
        if (unlikely(!rc))
        {
            applog(LOG_ERR, "submit_upstream_work stratum_send_line failed");
            goto out;
        }
    }
    else if (work->txs)
    {
        if (opt_debug)
            applog(LOG_DEBUG, "GBT submit");

        char *req;

        for (i = 0; i < ARRAY_SIZE(work->data); i++)
        {
            // TODO: move to another place
            if (i!=19)
            {
                be32enc(work->data + i, work->data[i]);
            }
        }
        bin2hex(data_str, (unsigned char *)work->data, 80);
        if (work->workid) {
            char *params;
            val = json_object();
            json_object_set_new(val, "workid", json_string(work->workid));
            params = json_dumps(val, 0);
            json_decref(val);
            req = (char*)malloc(128 + 2*80 + strlen(work->txs) + strlen(params));
            sprintf(req,
                "{\"method\": \"submitblock\", \"params\": [\"%s%s\", %s], \"id\":1}\r\n",
                data_str, work->txs, params);
            free(params);
        } else {
            req = (char*)malloc(128 + 2*80 + strlen(work->txs));
            sprintf(req,
                "{\"method\": \"submitblock\", \"params\": [\"%s%s\"], \"id\":1}\r\n",
                data_str, work->txs);
        }
        val = json_rpc_call(curl, rpc_url, rpc_user, rpc_pass, req, NULL, 0);
        free(req);
        if (unlikely(!val))
        {
            //applog(LOG_ERR, "submit_upstream_work json_rpc_call failed"); // abort_flag
            goto out;
        }

        res = json_object_get(val, "result");
        if (json_is_object(res)) {
            char *res_str;
            bool sumres = false;
            void *iter = json_object_iter(res);
            while (iter) {
                if (json_is_null(json_object_iter_value(iter))) {
                    sumres = true;
                    break;
                }
                iter = json_object_iter_next(res, iter);
            }
            res_str = json_dumps(res, 0);
            share_result(sumres, res_str);
            free(res_str);
        } else
            share_result(json_is_null(res), json_string_value(res));

        json_decref(val);
    }

    rc = true;

out:
    return rc;
}

#define GBT_CAPABILITIES "[\"coinbasetxn\", \"coinbasevalue\", \"longpoll\", \"workid\"]"
#define GBT_RULES "[\"segwit\"]"

static const char *gbt_req =
    "{\"method\": \"getblocktemplate\", \"params\": [{\"capabilities\": "
    GBT_CAPABILITIES ", \"rules\": " GBT_RULES "}], \"id\":0}\r\n";
static const char *gbt_lp_req =
    "{\"method\": \"getblocktemplate\", \"params\": [{\"capabilities\": "
    GBT_CAPABILITIES ", \"rules\": " GBT_RULES ", \"longpollid\": \"%s\"}], \"id\":0}\r\n";

static bool get_upstream_work(CURL *curl, struct work* work_info)
{
    json_t *val;
    int err;
    bool rc;

start:
    val = json_rpc_call(curl, rpc_url, rpc_user, rpc_pass,
                        gbt_req,
                        &err, JSON_RPC_QUIET_404);

    if (have_stratum)
    {
        if (val)
            json_decref(val);
        return true;
    }

    if (!val)
    {
        if (opt_debug) { applog(LOG_DEBUG, "get_upstream_work(), json_rpc_call returned NULL"); }
        return false;
    }

    rc = gbt_work_decode(json_object_get(val, "result"), work_info);
    if (!rc)
    {
        applog(LOG_DEBUG, "gbt_work_decode() failed");

        json_decref(val);
        goto start;
    }

    json_decref(val);

    return rc;
}

static void workio_cmd_free(struct workio_cmd *wc)
{
    if (!wc)
        return;

    switch (wc->cmd) {
    case WC_SUBMIT_WORK:
        work_free(wc->workInfo);
        free(wc->workInfo);
        break;
    default: /* do nothing */
        break;
    }

    memset(wc, 0, sizeof(*wc)); /* poison */
    free(wc);
}

static bool workio_get_work(struct workio_cmd *wc, CURL *curl)
{
    struct work *ret_work;
    int failures = 0;

    ret_work = (struct work*)calloc(1, sizeof(*ret_work));
    if (!ret_work)
        return false;

    /* obtain new work */
    while (!get_upstream_work(curl, ret_work))
    {
        if (abort_flag)
        {
            applog(LOG_ERR, "json_rpc_call: abort_flag, terminating workio thread");
            return false;
        }

        if (unlikely((opt_retries >= 0) && (++failures > opt_retries)))
        {
            applog(LOG_ERR, "json_rpc_call failed, terminating workio thread");
            free(ret_work);
            return false;
        }

        /* pause, then restart work-request loop */
        applog(LOG_ERR, "json_rpc_call failed, retry after %d seconds", opt_fail_pause);

        // Async wait allows to enter deinitialization phase faster especially if time between retries is very long.
        uint64_t waitTimeSec = (uint64_t)opt_fail_pause;
        uint64_t sec = 0;
        std::chrono::steady_clock::time_point hrTimerStart = std::chrono::steady_clock::now();
        while (sec < waitTimeSec)
        {
            sleep_ms(1);
            sec = std::chrono::duration_cast<std::chrono::seconds>( std::chrono::steady_clock::now() - hrTimerStart ).count();
            if (abort_flag) { return false; }
        }
    }

    if (!ret_work)
    {
        return false;
    }

    /* send work to requesting thread */
    if (!tq_push(wc->thr->q, ret_work))
        free(ret_work);

    return true;
}

static bool workio_submit_work(struct workio_cmd *wc, CURL *curl)
{
    int failures = 0;

    while (!submit_upstream_work(curl, wc->workInfo))
    {
        if (abort_flag)
        {
            //applog(LOG_ERR, "(submit)json_rpc_call: abort_flag, terminating workio thread");
            return false;
        }

        if (unlikely((opt_retries >= 0) && (++failures > opt_retries))) {
            applog(LOG_ERR, "...terminating workio thread");
            return false;
        }

        // pause, then restart work-request loop
        applog(LOG_ERR, "json_rpc_call failed, retry after %d seconds", opt_fail_pause);

        // Async wait allows to enter deinitialization phase faster especially if time between retries is very long.
        uint64_t waitTimeSec = (uint64_t)opt_fail_pause;
        uint64_t sec = 0;
        std::chrono::steady_clock::time_point hrTimerStart = std::chrono::steady_clock::now();
        while (sec < waitTimeSec)
        {
            sleep_ms(1);
            sec = std::chrono::duration_cast<std::chrono::seconds>( std::chrono::steady_clock::now() - hrTimerStart ).count();
            if (abort_flag) { return false; }
        }
    }

    return true;
}


static int workio_thread(void *userdata)
{
    struct thr_info* mythr = (struct thr_info*)userdata;
    CURL* curl = NULL;
    bool ok = true;

    curl = curl_easy_init();
    if (unlikely(!curl))
    {
        applog(LOG_ERR, "CURL initialization failed");
        return 1;
    }

    while (ok && !abort_flag)
    {
        struct workio_cmd* wc = NULL;

        // wait for workio_cmd sent to us, on our queue
        wc = (workio_cmd*)tq_pop(mythr->q, NULL);
        if (wc)
        {
            // process workio_cmd
            switch (wc->cmd)
            {
            case WC_GET_WORK:
                ok = workio_get_work(wc, curl);
                break;
            case WC_SUBMIT_WORK:
                ok = workio_submit_work(wc, curl);
                break;
            default:        // should never happen
                ok = false;
                break;
            }

            workio_cmd_free(wc);
        }
    }

    if (opt_debug) { applog(LOG_DEBUG, "Exit workIO thread"); }

    tq_freeze(mythr->q);
    curl_easy_cleanup(curl);

    return 0;
}

static bool get_work(struct thr_info *thr, struct work *work)
{
    struct workio_cmd *wc;
    struct work *work_heap;

    if (opt_benchmark)
    {
        memset(work->data, 0x55, 76);
        work->data[17] = swab32((uint32_t)time(NULL));
        memset(work->data + 19, 0x00, 52);
        work->data[20] = 0x80000000;
        work->data[31] = 0x00000280;
        memset(work->target, 0x00, sizeof(work->target));
        return true;
    }

    // fill out work request message
    wc = (struct workio_cmd*)calloc(1, sizeof(*wc));
    if (!wc)
    {
        return false;
    }

    wc->cmd = WC_GET_WORK;
    wc->thr = thr;

    // send work request to workio thread
    if (!tq_push(thr_info[work_thr_id].q, wc))
    {
        workio_cmd_free(wc);
        return false;
    }

    // wait for response, a unit of work
    work_heap = (struct work*)tq_pop(thr->q, NULL);
    if (!work_heap)
    {
        return false;
    }

    // copy returned work into storage provided by caller
    memcpy(work, work_heap, sizeof(*work));
    free(work_heap);

    return true;
}

static bool submit_work(struct thr_info *thr, const struct work *work_in)
{
    struct workio_cmd *wc;
    
    /* fill out work request message */
    wc = (struct workio_cmd*)calloc(1, sizeof(*wc));
    if (!wc)
        return false;

    wc->workInfo = (struct work*)malloc(sizeof(*work_in));
    if (!wc->workInfo)
        goto err_out;

    wc->cmd = WC_SUBMIT_WORK;
    wc->thr = thr;
    work_copy(wc->workInfo, work_in);

    /* send solution to workio thread */
    if (!tq_push(thr_info[work_thr_id].q, wc))
        goto err_out;

    return true;

err_out:
    workio_cmd_free(wc);
    return false;
}

static void stratum_gen_work(struct stratum_ctx *sctx, struct work *work)
{
    unsigned char merkle_root[64];
    int i;

    mtx_lock(&sctx->work_lock);

    free(work->job_id);
    work->job_id = strdup(sctx->job.job_id);
    work->xnonce2_len = sctx->xnonce2_size;
    work->xnonce2 = (unsigned char*)realloc(work->xnonce2, sctx->xnonce2_size);
    memcpy(work->xnonce2, sctx->job.xnonce2, sctx->xnonce2_size);

    /* Generate merkle root */
    sha256d(merkle_root, sctx->job.coinbase, sctx->job.coinbase_size);
    for (i = 0; i < sctx->job.merkle_count; i++) {
        memcpy(merkle_root + 32, sctx->job.merkle[i], 32);
        sha256d(merkle_root, merkle_root, 64);
    }
    
    /* Increment extranonce2 */
    for (i = 0; i < sctx->xnonce2_size && !++sctx->job.xnonce2[i]; i++);

    /* Assemble block header */
    memset(work->data, 0, 128);
    work->data[0] = le32dec(sctx->job.version);
    for (i = 0; i < 8; i++)
        work->data[1 + i] = le32dec((uint32_t *)sctx->job.prevhash + i);
    for (i = 0; i < 8; i++)
        work->data[9 + i] = be32dec((uint32_t *)merkle_root + i);
    work->data[17] = le32dec(sctx->job.ntime);
    work->data[18] = le32dec(sctx->job.nbits);
    work->data[20] = 0x80000000;
    work->data[31] = 0x00000280;

    mtx_unlock(&sctx->work_lock);

    if (opt_debug) {
        char *xnonce2str = abin2hex(work->xnonce2, work->xnonce2_len);
        applog(LOG_DEBUG, "job_id='%s' extranonce2=%s ntime=%08x",
               work->job_id, xnonce2str, swab32(work->data[17]));
        free(xnonce2str);
    }

    diff_to_target((unsigned char*)work->target, sctx->job.diff, 256.0);
}

static void restart_threads(void)
{
    for (int i = 0; i < opt_n_threads; ++i)
    {
        work_restart[i].restart = 1;
    }
}

// full host side nonce validation(slower & incompatible with VERTHASH_EXTENDED_VALIDATION)
//#define VERTHASH_FULL_VALIDATION

// extended 64 bit device side nonce validation
//#define VERTHASH_EXTENDED_VALIDATION

// using binary only kernels
//#define BINARY_KERNELS

//-----------------------------------------------------------------------------
// OpenCL worker thread
struct clworker_t
{
    struct thr_info* threadInfo;
    vh::cldevice_t cldevice;
    size_t workSize;
    uint32_t batchTimeMs;
    uint32_t occupancyPct;
    
    // monitoring
    nvmlDevice_t nvmlDevice;
    int adlAdapterIndex;
    int gpuTemperatureLimit;
    int deviceMonitor;
};

//! sha3/keccak state
struct kState { uint32_t v[50]; };
//! u32x8
struct u32x8 { uint32_t v[8]; };

//----------------------------------------------------------------------------
//! test u32x8 with target
// TODO: refactor/optimize, remove and use fullTest directly
inline bool fulltestUvec8(const u32x8& hash, const uint32_t *target)
{
    bool rc = true;

    for (int i = 7; i >= 0; i--)
    {
        if (hash.v[i] > target[i])
        {
            rc = false;
            break;
        }
        if (hash.v[i] < target[i])
        {
            rc = true;
            break;
        }
    }

    return rc;
}

//----------------------------------------------------------------------------
static int verthashOpenCL_thread(void *userdata)
{
    if (opt_debug)
    {
        applog(LOG_DEBUG, "Verthash OCL thread started");
    }

    //-------------------------------------
    // Memory error tracker
    int memTrackerEnabled = 1;
    uint32_t memErrorsDetected = 0;

    //-------------------------------------
    // Get thread data
    clworker_t* clworker = (clworker_t*)userdata;
    vh::cldevice_t& cldevice = clworker->cldevice;
    struct thr_info *mythr = clworker->threadInfo; 
    int thr_id = mythr->id;
    //-------------------------------------
    // Monitoring data
    int deviceMonitor = clworker->deviceMonitor;
    int gpuTemperatureLimit = clworker->gpuTemperatureLimit;
    bool throttling = false;
    // stats
    int temperature = 0;
    int power = 0;
    int fanSpeed = 0;

    // NVML
    nvmlDevice_t nvmlDevice = clworker->nvmlDevice;

#ifdef _WIN32
    // ADL
    int adlAdapterIndex = clworker->adlAdapterIndex;
    int overdriveVersion = 0;
    ADL_CONTEXT_HANDLE adlContext = NULL;

    if ((clworker->cldevice.vendor == vh::V_AMD) && (deviceMonitor != 0))
    {
        int adlRC = ADL2_Main_Control_Create(ADL_Main_Memory_Alloc, 1, &adlContext);
        if (adlRC == ADL_OK)
        {
            // Get overdrive capabilities (supported version etc)
            int oSupported = 0;
            int oEnabled = 0;
            int oVersion = 0;
            adlRC = ADL2_Overdrive_Caps(adlContext, adlAdapterIndex, &oSupported, &oEnabled, &oVersion);
            if (adlRC != ADL_OK)
            {
                applog(LOG_WARNING, "cl_device(%d):Failed to get ADL Overdrive capabilities!", thr_id);
            }
            else
            {
                if (oSupported == 0)
                {
                    applog(LOG_WARNING, "cl_device(%d):ADL Overdrive is not supported!", thr_id);
                    adlRC = ADL_ERR;
                }
                else if (oEnabled == 0)
                {
                    applog(LOG_WARNING, "cl_device(%d):ADL Overdrive is supported, but not enabled!", thr_id);
                    adlRC = ADL_ERR;
                }
                else if ((oVersion < 5) || (oVersion > 8))
                {
                    applog(LOG_WARNING, "cl_device(%d):ADL Overdrive(%d) integration is not available!", thr_id, oVersion);
                    adlRC = ADL_ERR;
                }
                else
                {
                    overdriveVersion = oVersion;
                    // check if required ADL functions are available
                    if(oVersion == 5)
                    {
                       if (pADL2_Overdrive5_Temperature_Get == NULL)
                           adlRC = ADL_ERR;
                    }
                    else if (oVersion == 6)
                    {
                       if (pADL2_Overdrive6_Temperature_Get == NULL)
                           adlRC = ADL_ERR;
                    }
                    else if (oVersion == 7)
                    {
                       if (pADL2_OverdriveN_Temperature_Get == NULL)
                           adlRC = ADL_ERR;
                    }
                    else if (oVersion == 8)
                    {
                        if (pADL2_New_QueryPMLogData_Get == NULL)
                           adlRC = ADL_ERR;
                    }
                }
            }
        }
        else
        {
            applog(LOG_WARNING, "cl_device(%d):Failed to create ADL context.", thr_id);
        }

        // disable ADL on errors
        if (adlRC != ADL_OK)
        {
            adlAdapterIndex = -1;
        }
    }

#elif defined __linux__
    // SYSFS

    char* sysFsPwm1Path = NULL;
    char* sysFsPwm1MaxPath = NULL;
    char* sysFsPower1AveragePath = NULL;
    char* sysFsTemp1InputPath = NULL;

    // SYSFS monitoring is limited to AMD devices
    if ((clworker->cldevice.vendor == vh::V_AMD) && (deviceMonitor != 0))
    {
        char *sysFsDevicePath = SYSFS_get_syspath_device(cldevice.pcieBusId,
                                                         cldevice.pcieDeviceId,
                                                         cldevice.pcieFunctionId);
        if (sysFsDevicePath != NULL)
        {
            char *sysFsHWMONPath = SYSFS_get_syspath_hwmon(sysFsDevicePath);
            if (sysFsHWMONPath != NULL)
            {
                sysFsPwm1Path = SYSFS_get_syspath_pwm1(sysFsHWMONPath);
                sysFsPwm1MaxPath = SYSFS_get_syspath_pwm1_max(sysFsHWMONPath);
                sysFsPower1AveragePath = SYSFS_get_syspath_power1_average(sysFsHWMONPath);
                sysFsTemp1InputPath = SYSFS_get_syspath_temp1_input(sysFsHWMONPath);
            }
        }
        
        free(sysFsDevicePath);
    }

#endif // __linux__


    // Disable monitoring(and GPU temperature limit) if no backend is available
    if ((deviceMonitor != 0))
    {
#ifdef _WIN32
        if ((nvmlDevice == NULL) && (adlAdapterIndex == -1))
        {
            applog(LOG_WARNING, "cl_device(%d):Monitoring has been disabled. No backends available!", thr_id);
            deviceMonitor = 0;
        }
#elif defined __linux__
        if ((sysFsPwm1Path == NULL) &&
            (sysFsPwm1MaxPath == NULL) &&
            (sysFsPower1AveragePath == NULL) &&
            (sysFsPower1AveragePath == NULL) &&
            (nvmlDevice == NULL))
        {
            applog(LOG_WARNING, "cl_device(%d):Monitoring has been disabled. No backends available!", thr_id);
            deviceMonitor = 0;
        }
#else
        //applog(LOG_WARNING, "cl_device(%d):Monitoring has been disabled. No backends available!", thr_id);
        //deviceMonitor = 0;
#endif
    }

    //-------------------------------------
    // Work related stuff
    struct work workInfo = {{0}};

    //-------------------------------------
    // Init CL data
    cl_int errorCode = CL_SUCCESS;

    // adaptive batch size flag
    const bool adaptiveBatchSize = (clworker->workSize == 0)? true : false;

    // batch size
    // 4096 is a starting batch size in adaptive mode
    size_t workSize = (adaptiveBatchSize == true)? 4096 : clworker->workSize;

    // occupancy percent(workSize must be adaptive)
    uint32_t occupancyPct = (adaptiveBatchSize == true) ? clworker->occupancyPct : 100;
    // extra check to prevent 0
    if ((occupancyPct == 0) || (occupancyPct > 100))
    {
        occupancyPct = 100;
    }
    const uint32_t occupancy = 100 - occupancyPct;

    // these can be changed at runtime if adaptive batch size is enabled
    size_t globalWorkSize1x = workSize;
    size_t globalWorkSize4x = workSize*4;

    // Init nonce range 
    const uint64_t numNoncesGlobal = 4294967296ULL; // global nonce range is [0..4294967295]
    const uint64_t maxBatches = (numNoncesGlobal / (uint64_t)workSize);
    uint64_t maxBatchesPerDevice = maxBatches;
    uint32_t firstNonce = 0;
    uint64_t maxNonce = numNoncesGlobal; // actually last nonce is maxNonce-1
    if (have_stratum == false) // GBT
    {
        // There is no extranonce2 on GBT. Split a single nonce range between workers.
        maxBatchesPerDevice = maxBatches / opt_n_threads;
        // begin nonce range
        firstNonce = (maxBatchesPerDevice * workSize) * thr_id;
        // Handle case when the number of workers is not power of 2
        if (thr_id == (opt_n_threads - 1))
        {
            maxBatchesPerDevice += (maxBatches % opt_n_threads);
        }
        // end nonce range
        maxNonce = firstNonce + (maxBatchesPerDevice * workSize);
    }

    //-------------------------------------
    // Adaptive batch size settings
    double maxMs = (double)vh::defaultBatchTimeMs; // maxBathTime
    if(clworker->batchTimeMs != 0)
    {
        maxMs = (double)clworker->batchTimeMs;
    }
    const uint64_t alignment = 256; // 256 minimum possible value (max local work size in the pipeline)
    const size_t minBatchSize = alignment;
    double elapsedTimeMs = maxMs; // batchTimeTimer in milliseconds
    // Max Work(Batch)Size, that can be used during the adaptive batch size.
    // It will be lowered automatically, in case of memory allocation errors, but will never go up.
    size_t maxBatchSize = 134217728;
    
    //-------------------------------------
    // Kernel configurations
    // local work batch sizes(mostly kernel specific)
    size_t localWorkSize = 64;
    size_t localWorkSize256 = 256;
    std::string buildOptions0;
    std::string buildOptions;
    // Local work size may change in future releases
    if (cldevice.vendor == vh::V_AMD)
    {
        buildOptions0 += " -DBIT_ALIGN";
        buildOptions += " -DBIT_ALIGN ";
    }
    buildOptions += " -DWORK_SIZE=64 ";
    // MDIV
    buildOptions += " -DMDIV=" + std::to_string(verthashInfo.bitmask);
    // result validation
#ifdef VERTHASH_FULL_VALIDATION
    buildOptions += " -DFULL_VALIDATION";
#elif defined VERTHASH_EXTENDED_VALIDATION
    buildOptions += " -DEXTENDED_VALIDATION";
#endif

#ifdef BINARY_KERNELS
    std::string deviceName;
    size_t infoSize = 0;
    deviceName.clear();
    clGetDeviceInfo(cldevice.clId, CL_DEVICE_NAME, 0, NULL, &infoSize);
    deviceName.resize(infoSize);
    clGetDeviceInfo(cldevice.clId, CL_DEVICE_NAME, infoSize, (void *)deviceName.data(), NULL);
    deviceName.pop_back();

    std::string fileName_sha3precompute = "kernels/sha3_512_precompute_"+deviceName+".bin";
    std::string fileName_sha3_512_256 = "kernels/sha3_512_256_"+deviceName+".bin";
    std::string fileName_verthash = "kernels/verthash_"+deviceName+".bin";
#endif


#ifdef VERTHASH_FULL_VALIDATION
    // Host side hash storage
    std::vector<u32x8> verthashIORES;
    verthashIORES.resize(workSize);
#else
    // HTarg result host side storage
    std::vector<uint32_t> results;
    std::vector<uint32_t> potentialResults; // used if (potentialNonceCount > 1)
#endif

    // init per device profiling data
    const size_t maxProfSamples = 16;
    size_t numSamples = 0;
    size_t sampleIndex = 0;
    std::vector<uint64_t> profSamples(maxProfSamples, 0);
    std::vector<uint64_t> batchSamples(maxProfSamples, 0);
    assert(profSamples.size() == batchSamples.size());

    // per hash-rate update timer 
    std::chrono::steady_clock::time_point hrTimerStart;
    // Hash-rate (console)report interval in seconds
    uint64_t hrTimerIntervalSec = 4; // TODO: make configurable

    //-------------------------------------
    // Init device
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)cldevice.clPlatformId,
        0
    };
    // pattern to fill buffers
    cl_uint zero = 0;
    // context and cmd queue
    cl_context clContext = NULL;
    cl_command_queue clCommandQueue = NULL;
    // buffers
    cl_mem clmemKStates = NULL;
    cl_mem clmemHeader = NULL;
    cl_mem clmemFullDat = NULL;
    cl_mem clmemResults = NULL;
    cl_mem clmemHTargetResults = NULL;
    // programs kernels
    cl_program clprogramSHA3_512_precompute = NULL;
    cl_kernel clkernelSHA3_512_precompute = NULL;
    cl_program clprogramSHA3_512_256 = NULL;
    cl_kernel clkernelSHA3_512_256 = NULL;
    cl_program clprogramVerthash = NULL;
    cl_kernel clkernelVerthash = NULL;

    // create 1 context for each device
    clContext = clCreateContext(contextProperties, 1, &cldevice.clId, nullptr, nullptr, &errorCode);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create an OpenCL context. error code: %d", thr_id, (int)errorCode); goto out; }

    // create an OpenCL command queue
    clCommandQueue = clCreateCommandQueue(clContext, cldevice.clId, 0, &errorCode);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create an OpenCL command queue. error code: %d", thr_id, (int)errorCode); goto out; }

    //-------------------------------------
    // device buffers
    //! 8 precomputed keccak states
    clmemKStates = clCreateBuffer(clContext, CL_MEM_READ_WRITE, 8 * sizeof(kState), nullptr, &errorCode);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create a SHA3 states buffer. error code: %d", thr_id, (int)errorCode); goto out; }

    //! block header for SHA3 reference or SHA3_precompute
    clmemHeader = clCreateBuffer(clContext, CL_MEM_READ_ONLY, 18 * sizeof(uint32_t), nullptr, &errorCode); 
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create a SHA3 headers buffer. error code: %d", thr_id, (int)errorCode); goto out; }

    //! hash results from IO pass
    clmemResults = clCreateBuffer(clContext, CL_MEM_READ_WRITE, workSize * sizeof(u32x8), nullptr, &errorCode);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create an IO results buffer. error code: %d", thr_id, (int)errorCode); goto out; }

    //! results against hash target.
    // Too much, but 100% robust: workBatchSize + num_actual_results(1)
    clmemHTargetResults = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(uint32_t) * (workSize + 1), nullptr, &errorCode);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create a hash target results buffer. error code: %d", thr_id, (int)errorCode); goto out; }

    // Some drivers do not initialize buffers to 0(e.g. AMD GPU Pro). At least "result counter"(first value) must be initialized to 0
    errorCode = clEnqueueFillBuffer(clCommandQueue, clmemHTargetResults, &zero, sizeof(uint32_t), 0, (sizeof(uint32_t) * 1), 0, NULL, NULL);
    // OpenCL 1.0 - 1.1
    //errorCode = clEnqueueWriteBuffer(clCommandQueue, clmemHTargetResults, CL_TRUE, 0, sizeof(cl_uint), &zero, 0, nullptr, nullptr);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to clear HTarget result buffer. error code: %d", thr_id, (int)errorCode); goto out; }

    //! Verthash data
    if (verthashInfo.dataSize != 0) 
    {
        if(opt_debug)
        {
            applog(LOG_DEBUG, "Load verthash data size: %llu", verthashInfo.dataSize);
        }

        //! verthash data
        clmemFullDat = clCreateBuffer(clContext, CL_MEM_READ_ONLY, verthashInfo.dataSize, nullptr, &errorCode);
        if (errorCode != CL_SUCCESS)
        {
            applog(LOG_ERR, "cl_device(%d):Failed to create verthash data buffer. error code: %d", thr_id, (int)errorCode);
            goto out;
        }

        // upload verthash data
        errorCode = clEnqueueWriteBuffer(clCommandQueue, clmemFullDat, CL_TRUE, 0,
                                         verthashInfo.dataSize, verthashInfo.data, 0, nullptr, nullptr);
        if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to copy Verthash data to the GPU memory. error code: %d", thr_id, (int)errorCode); goto out; }
    }
    else
    {
        applog(LOG_ERR, "Verthash data is empty!");
        goto out;
    }

    //-------------------------------------
    // load kernel modules

    // *SHA3_precompute
    // initial stage. precomputes Keccak/SHA3 states for SHA3 pass
#ifdef BINARY_KERNELS
    clprogramSHA3_512_precompute = vh::cluCreateProgramWithBinaryFromFile(clContext, cldevice.clId, fileName_sha3precompute.c_str()); 
#else
    clprogramSHA3_512_precompute = vh::cluCreateProgramFromFile(clContext, cldevice.clId, buildOptions0.c_str(), "kernels/sha3_512_precompute.cl");
#endif
    if(clprogramSHA3_512_precompute == NULL)
    {
        applog(LOG_ERR, "cl_device(%d):Failed to create a SHA3 precompute program.", thr_id);
        goto out; 
    }

    clkernelSHA3_512_precompute = clCreateKernel(clprogramSHA3_512_precompute, "sha3_512_precompute", &errorCode);
    if (errorCode != CL_SUCCESS)
    {
        applog(LOG_ERR, "cl_device(%d):Failed to create a SHA3 precompute kernel.", thr_id);
        goto out; 
    }

    errorCode = clSetKernelArg(clkernelSHA3_512_precompute, 0, sizeof(cl_mem), &clmemKStates);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(0) for SHA3 precompute kernel. error code: %d", thr_id, errorCode); goto out; }

    errorCode = clSetKernelArg(clkernelSHA3_512_precompute, 1, sizeof(cl_mem), &clmemHeader);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(1) for SHA3 precompute kernel. error code: %d", thr_id, errorCode); goto out; }

    // *SHA3_512_256
    // first stage.
#ifdef BINARY_KERNELS
    clprogramSHA3_512_256 = vh::cluCreateProgramWithBinaryFromFile(clContext, cldevice.clId, fileName_sha3_512_256.c_str()); 
#else
    clprogramSHA3_512_256 = vh::cluCreateProgramFromFile(clContext, cldevice.clId, buildOptions0.c_str(), "kernels/sha3_512_256.cl");
#endif
    if(clprogramSHA3_512_256 == NULL)
    {
        applog(LOG_ERR, "cl_device(%d):Failed to create a SHA3 precompute program.", thr_id);
        goto out; 
    }

    clkernelSHA3_512_256 = clCreateKernel(clprogramSHA3_512_256, "sha3_512_256", &errorCode);
    if (errorCode != CL_SUCCESS)
    {
        applog(LOG_ERR, "cl_device(%d):Failed to create a SHA3 precompute kernel.", thr_id);
        goto out; 
    }

    errorCode = clSetKernelArg(clkernelSHA3_512_256, 0, sizeof(cl_mem), &clmemResults);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(0) for SHA3_512_256 kernel. error code: %d", thr_id, errorCode); goto out; }

    errorCode = clSetKernelArg(clkernelSHA3_512_256, 1, sizeof(cl_mem), &clmemHeader);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(1) for SHA3_512_256 kernel. error code: %d", thr_id, errorCode); goto out; }


    // *Verthash pass
    // second stage

#ifdef BINARY_KERNELS
    clprogramVerthash = vh::cluCreateProgramWithBinaryFromFile(clContext, cldevice.clId, fileName_verthash.c_str()); 
#else
    clprogramVerthash = vh::cluCreateProgramFromFile(clContext, cldevice.clId, buildOptions.c_str(), "kernels/verthash.cl");
#endif

    if(clprogramVerthash == NULL) { applog(LOG_ERR, "cl_device(%d):Failed to create a Verthash program.", thr_id); goto out; }
    clkernelVerthash = clCreateKernel(clprogramVerthash, "verthash_4w", &errorCode);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create a Verthash kernel. error code: %d", thr_id, errorCode); goto out; }
    errorCode = clSetKernelArg(clkernelVerthash, 0, sizeof(cl_mem), &clmemResults);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(0) for Verthash kernel. error code: %d", thr_id, errorCode); goto out; }
    errorCode = clSetKernelArg(clkernelVerthash, 1, sizeof(cl_mem), &clmemKStates);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(1) for Verthash kernel. error code: %d", thr_id, errorCode); goto out; }
    errorCode = clSetKernelArg(clkernelVerthash, 2, sizeof(cl_mem), &clmemFullDat);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(2) for Verthash kernel. error code: %d", thr_id, errorCode); goto out; }
#ifndef VERTHASH_FULL_VALIDATION
    errorCode = clSetKernelArg(clkernelVerthash, 5, sizeof(cl_mem), &clmemHTargetResults);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(5) for Verthash kernel. error code: %d", thr_id, errorCode); goto out; }
#endif

    if (adaptiveBatchSize == false)
    {
        // print work size:
        applog(LOG_INFO, "cl_device(%d): WorkSize has been set to: %u", thr_id, (uint32_t)workSize);
    }


    //-------------------------------------
    // reset the hash-rate reporting timer
    hrTimerStart = std::chrono::steady_clock::now();

    //-------------------------------------
    // Main worker loop
    while (!abort_flag)
    {
        // Stratum
        if (have_stratum)
        {
            while (time(NULL) >= g_work_time + 120)
            {
                if (abort_flag) { goto out; }
                sleep_ms(1);
            }

            mtx_lock(&g_work_lock);
            stratum_gen_work(&stratum, &g_work);
        }
        else // GBT
        {
            // obtain new work from internal workio thread
            mtx_lock(&g_work_lock);

            work_free(&g_work);
            if (unlikely(!get_work(mythr, &g_work)))
            {
                if (!abort_flag) { applog(LOG_ERR, "cl_device(%d):Work retrieval failed, exiting mining thread %d", thr_id, thr_id); }
                mtx_unlock(&g_work_lock);
                goto out;
            }

            g_work_time = time(NULL);
        }

        // create a work copy
        work_free(&workInfo);
        work_copy(&workInfo, &g_work);
        workInfo.data[19] = 0;

        work_restart[thr_id].restart = 0;
        mtx_unlock(&g_work_lock);
        

        // Actual nonce is 32 bit, but we use 64 bit to prevent possible overflows
        uint64_t nonce64 = firstNonce;


        //-------------------------------------
        // Generate midstate
        uint32_t uheader[20] = {0};
        for (size_t i = 0; i < 20; ++i)
        {
            be32enc(&uheader[i], workInfo.data[i]);
        }

        //-------------------------------------
        // SHA3_precompute
        // update header for precompute pass
        errorCode = clEnqueueWriteBuffer(clCommandQueue, clmemHeader, CL_TRUE, 0, 72, uheader, 0, nullptr, nullptr);
        if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to update block header data.", thr_id); goto out; }
        //-------------------------------------
        // SHA3_256 kernel
        errorCode = clSetKernelArg(clkernelSHA3_512_256, 2, sizeof(uint32_t), &uheader[18]); // in18
        if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(3) for SHA3_256 kernel.", thr_id); goto out; }
        assert(errorCode == CL_SUCCESS);
        //-------------------------------------
        // Verthash kernel
        errorCode = clSetKernelArg(clkernelVerthash, 3, sizeof(uint32_t), &uheader[18]); // in18
        if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(3) for Verthash kernel.", thr_id); goto out; }
        assert(errorCode == CL_SUCCESS);
#ifndef VERTHASH_FULL_VALIDATION
        // Htarget result
    #ifdef VERTHASH_EXTENDED_VALIDATION
        uint64_t targetExt = ((uint64_t(workInfo.target[7])) << 32) | (uint64_t(workInfo.target[6]) & 0xFFFFFFFFUL);
        errorCode = clSetKernelArg(clkernelVerthash, 6, sizeof(uint64_t), &targetExt);
    #else
        errorCode = clSetKernelArg(clkernelVerthash, 6, sizeof(uint32_t), &workInfo.target[7]);
    #endif
        if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(6) for Verthash kernel.", thr_id); goto out; }
        assert(errorCode == CL_SUCCESS);
#endif

        //-------------------------------------
        // Kernel launch part 0
        // Hash block header
        const size_t preGlobalWorkSize1x = 8;
        const size_t localWorkSize8 = 1;
        errorCode = clEnqueueNDRangeKernel(clCommandQueue, clkernelSHA3_512_precompute, 1, nullptr,
                                           &preGlobalWorkSize1x, &localWorkSize8, 0, nullptr, nullptr);
        if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to start SHA3 precompute kernel.", thr_id); goto out; }


        // Every device works withing its own nonce range.
        //-------------------------------------
        // Compute hashes

        // abort flag is needed here to prevent an inconsistent behaviour
        // in case program termination was triggered between work generation and resetting work restart values
        while (!work_restart[thr_id].restart && (!abort_flag))
        {
            //---------------------------------------------------------------------
            // Temperature limit
            // 0 means handled by the device driver. This check is still required in case backend reports wrong temp.
            if ((gpuTemperatureLimit != 0) && (deviceMonitor != 0))
            {
                if (temperature > gpuTemperatureLimit)
                {
                    if (throttling == false)
                    {
                        throttling = true;
                        applog(LOG_WARNING, "cl_device(%d): gpu temperature limit has reached %dC(max:%uC), stopping.", thr_id, temperature, gpuTemperatureLimit);
                    }
                    // throttling
                    const int waitTimeMs = 100;
                    sleep_ms(waitTimeMs);

                    // emit empty sample(
                    profSamples[sampleIndex] = (uint64_t)waitTimeMs;
                    batchSamples[sampleIndex] = 0;
                    sampleIndex = (sampleIndex + 1) & (profSamples.size()-1);
                    ++numSamples;
                    if (numSamples > profSamples.size())
                    {
                        numSamples = profSamples.size();
                    }

                    // perform another temperature checks here
                    if (nvmlDevice != NULL)
                    {
                        unsigned int nvmlTemperature = 0;
                        nvmlReturn_t nvmlRC = nvmlDeviceGetTemperature(nvmlDevice, NVML_TEMPERATURE_GPU, &nvmlTemperature);
                        if (nvmlRC == NVML_SUCCESS)
                        {
                            temperature = (int)nvmlTemperature;
                        }
                        else
                        {
                            applog(LOG_ERR, "cl_device(%d):Failed to get temperature, when temperature limit is set. exiting.", thr_id);
                            goto out;
                        }
                    }
                    else
                    {
#ifdef _WIN32
                        // ADL
                        int adlRC = ADL_OK;
                        if (adlAdapterIndex != -1)
                        {
                            adlRC = ALD2_Overdrive_Temperature_Get(adlContext, adlAdapterIndex, overdriveVersion, &temperature);
                            if (adlRC != ADL_OK)
                            {
                                applog(LOG_ERR, "cl_device(%d):Failed to get temperature, when temperature limit is set. exiting.", thr_id);
                                goto out;
                            }
                        }
#elif defined __linux__
                        // perform another temperature checks here
                        int sysfsRC = 0;
                        sysfsRC = SYSFS_get_temperature(sysFsTemp1InputPath, &temperature);
                        if (sysfsRC != 0)
                        {
                            free(sysFsTemp1InputPath);
                            sysFsTemp1InputPath = NULL;

                            applog(LOG_ERR, "cl_device(%d):Failed to get temperature, when temperature limit is set. exiting.", thr_id);
                            goto out;
                        }
#endif // __linux__
                    }

                    // extra check for the throttling flag
                    if(temperature <= gpuTemperatureLimit)
                    {
                        applog(LOG_WARNING, "cl_device(%d): cooled down, resuming.", thr_id);
                        throttling = false;
                    }

                    // restart loop
                    continue;
                }
            }

            //---------------------------------------------------------------------
            // Adaptive batch size
            if (adaptiveBatchSize == true)
            {
                double ms = elapsedTimeMs;

                size_t newBatchSize = globalWorkSize1x;
                if(ms > maxMs)
                {
                    double p = ms / maxMs; // possible division by 0 is handled outside
                    newBatchSize /= p; // possible division by 0 is handled outside
                    newBatchSize = align_u64(newBatchSize, alignment);

                    globalWorkSize1x = newBatchSize;
                    globalWorkSize4x = newBatchSize * 4;

                    if (opt_debug) { applog(LOG_DEBUG, "cl_device(%d):Worksize decreased to %zu", thr_id, newBatchSize); }
                }
                else if (ms < maxMs)
                {
                    double p = maxMs / ms; // possible division by 0 is handled outside
                    newBatchSize *= p;
                    newBatchSize = align_u64(newBatchSize, alignment);

                    // resize buffers(if possible)
                    if ((newBatchSize > workSize)) // workSize in this case is a buffer work size
                    {
                        while (maxBatchSize > minBatchSize)
                        {
                            if (clmemResults != NULL) {clReleaseMemObject(clmemResults); }
                            //! hash results. can be used for full validation.
                            clmemResults = clCreateBuffer(clContext, CL_MEM_READ_WRITE, newBatchSize * sizeof(u32x8), nullptr, &errorCode);
                            if (errorCode == CL_SUCCESS)
                            {
                                // Some drivers use lazy allocation. Prove, that we have enough memory.
                                errorCode = clEnqueueFillBuffer(clCommandQueue, clmemResults, &zero, sizeof(uint32_t), 0, (sizeof(uint32_t) * 1), 0, nullptr, nullptr);
                            }

                            if (errorCode == CL_SUCCESS)
                            {
                                if (clmemHTargetResults != NULL) { clReleaseMemObject(clmemHTargetResults); }
                                clmemHTargetResults = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(uint32_t) * (newBatchSize + 1), nullptr, &errorCode);
                                if (errorCode == CL_SUCCESS)
                                {
                                    // Some drivers use lazy allocation. Prove, that we have enough memory.
                                    errorCode = clEnqueueFillBuffer(clCommandQueue, clmemHTargetResults, &zero, sizeof(uint32_t), 0, (sizeof(uint32_t) * 1), 0, nullptr, nullptr);
                                }

                                if (errorCode == CL_SUCCESS)
                                {
#ifdef VERTHASH_FULL_VALIDATION
                                    // resize CPU hash buffer for VERTHASH_FULL_VALIDATION
                                    verthashIORES.resize(workSize);
#endif

                                    workSize = newBatchSize;

                                    globalWorkSize1x = newBatchSize;
                                    globalWorkSize4x = newBatchSize * 4;

                                    break;
                                }
                                else
                                {
                                    if (opt_debug) { applog(LOG_DEBUG, "cl_device(%d):Failed to create an HTarget buffer, recreate with the old batch size", thr_id); }

                                    // out of memory error
                                    maxBatchSize >>= 1;
                                    if (maxBatchSize == 0) { maxBatchSize = minBatchSize; }

                                    //  failed to create a buffer, recreate with the old batch size
                                    clmemResults = clCreateBuffer(clContext, CL_MEM_READ_WRITE, workSize * sizeof(u32x8), nullptr, &errorCode);
                                    //  failed to create a buffer, recreate with the old batch size
                                    clmemHTargetResults = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(uint32_t) * (workSize + 1), nullptr, &errorCode);
                                }
                            }
                            else
                            {
                                if (opt_debug) { applog(LOG_DEBUG, "cl_device(%d):Failed to create a 'result' buffer, recreating with the old batch size...", thr_id); }

                                // out of memory error
                                maxBatchSize >>= 1;
                                if (maxBatchSize == 0) { maxBatchSize = minBatchSize; }

                                clmemResults = clCreateBuffer(clContext, CL_MEM_READ_WRITE, workSize * sizeof(u32x8), nullptr, &errorCode);
                            }
                        }

                        // Update kernel agruments for new buffers
                        clSetKernelArg(clkernelSHA3_512_256, 0, sizeof(cl_mem), &clmemResults);
                        clSetKernelArg(clkernelVerthash, 0, sizeof(cl_mem), &clmemResults);
                        clSetKernelArg(clkernelVerthash, 5, sizeof(cl_mem), &clmemHTargetResults);

                        if (opt_debug) { applog(LOG_DEBUG, "cl_device(%d):Kernel args and buffers have been updated for a new batchSize: %zu", thr_id, workSize); }
                    }
                    else
                    {
                        if (newBatchSize > maxBatchSize) // overflow check
                        {
                            maxBatchSize >>= 1;
                            if (maxBatchSize < minBatchSize) { maxBatchSize = minBatchSize; }

                            if (opt_debug) { applog(LOG_DEBUG, "cl_device(%d):Update max batch size to: %zu", thr_id, maxBatchSize); }
                        }
                        else
                        {
                            globalWorkSize1x = newBatchSize;
                            globalWorkSize4x = newBatchSize * 4;

                            if (opt_debug) { applog(LOG_DEBUG, "cl_device(%d):Worksize increased to %zu", thr_id, newBatchSize); }
                        }
                    }
                }

                // Handle overflow
                uint64_t nextNonce = nonce64 + globalWorkSize1x;
                if (nextNonce > maxNonce)
                {
                    globalWorkSize1x = globalWorkSize1x - (nextNonce - maxNonce);
                    globalWorkSize4x = globalWorkSize1x*4;
                }
            }
            //------------------- Adaptive batch size end ---------------------------------
            uint32_t nonce = (uint32_t)nonce64;

            auto start = std::chrono::steady_clock::now();

            // Update first nonce parameter
            errorCode = clSetKernelArg(clkernelSHA3_512_256, 3, sizeof(uint32_t), &nonce); // first_nonce
            assert(errorCode == CL_SUCCESS);
            errorCode = clSetKernelArg(clkernelVerthash, 4, sizeof(uint32_t), &nonce); // first_nonce
            assert(errorCode == CL_SUCCESS);

            //-------------------------------------
            // Start pipeline

            // p1. sha3_256 stage.
            errorCode = clEnqueueNDRangeKernel(clCommandQueue, clkernelSHA3_512_256, 1, nullptr, 
                                               &globalWorkSize1x, &localWorkSize256, 0, nullptr, nullptr);
            // p2. IO stage
            errorCode = clEnqueueNDRangeKernel(clCommandQueue, clkernelVerthash, 1, nullptr, 
                                               &globalWorkSize4x, &localWorkSize, 0, nullptr, nullptr);
            if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed start pipeline. error code: %d.", thr_id, errorCode); goto out; }

            //-----------------------------------
            // Asynchronous processing
            uint64_t hrSec = std::chrono::duration_cast<std::chrono::seconds>( std::chrono::steady_clock::now() - hrTimerStart ).count();
            if ((hrSec >= hrTimerIntervalSec) && (numSamples > 0))
            {
                char sGPUTemperature[32] = {};
                char sPower[32] = {};
                char sFanSpeed[32] = {};

                // monitoring
                if (deviceMonitor != 0)
                {
                    if (nvmlDevice != NULL)
                    {
                        unsigned int nvmlTemperature = 0;
                        unsigned int nvmlPower = 0;
                        unsigned int nvmlFanSpeed = 0;

                        // get data from NVML backend
                        nvmlReturn_t nvmlRC = nvmlDeviceGetTemperature(nvmlDevice, NVML_TEMPERATURE_GPU, &nvmlTemperature);
                        if (nvmlRC == NVML_SUCCESS)
                        {
                            temperature = (int)nvmlTemperature;
                            snprintf(sGPUTemperature, sizeof(sGPUTemperature), " temp:%dC,", temperature);
                        }
                        else
                        {
                            if(gpuTemperatureLimit != 0)
                            {
                                applog(LOG_ERR, "cl_device(%d):Failed to get temperature, when temperature limit is set. exiting.", thr_id);
                                goto out;
                            }
                        }
                        
                        nvmlRC = nvmlDeviceGetPowerUsage(nvmlDevice, &nvmlPower);
                        if (nvmlRC == NVML_SUCCESS)
                        {
                            power = (int)(nvmlPower / 1000);
                            snprintf(sPower, sizeof(sPower), " power:%dW,", power);
                        }

                        nvmlRC = nvmlDeviceGetFanSpeed(nvmlDevice, &nvmlFanSpeed);
                        if (nvmlRC == NVML_SUCCESS)
                        {
                            fanSpeed = (int)nvmlFanSpeed;
                            snprintf(sFanSpeed, sizeof(sFanSpeed), " fan:%d%%,", fanSpeed);
                        }
                    }
                    else
                    {
#ifdef WIN32
                        // ADL
                        int adlRC = ADL_OK;
                        if (adlAdapterIndex != -1)
                        {
                            adlRC = ALD2_Overdrive_Temperature_Get(adlContext, adlAdapterIndex, overdriveVersion, &temperature);
                            if (adlRC == ADL_OK)
                            {
                                snprintf(sGPUTemperature, sizeof(sGPUTemperature), " temp:%dC,", temperature);
                            }
                            else
                            {
                                if(gpuTemperatureLimit != 0)
                                {
                                    applog(LOG_ERR, "cl_device(%d):Failed to get temperature, when temperature limit is set. exiting.", thr_id);
                                    goto out;
                                }
                            }
                        }
#elif defined __linux__
                        // get data from SYSFS backend
                        int sysfsRC = 0;
                        sysfsRC = SYSFS_get_temperature(sysFsTemp1InputPath, &temperature);
                        if (sysfsRC == 0)
                        {
                            snprintf(sGPUTemperature, sizeof(sGPUTemperature), " temp:%dC,", temperature);
                        }
                        else
                        {
                            free(sysFsTemp1InputPath);
                            sysFsTemp1InputPath = NULL;
                            if(gpuTemperatureLimit != 0)
                            {
                                applog(LOG_ERR, "cl_device(%d):Failed to get temperature, when temperature limit is set. exiting.", thr_id);
                                goto out;
                            }
                        }
                        
                        sysfsRC = SYSFS_get_power_usage(sysFsPower1AveragePath, &power);
                        if (sysfsRC == 0)
                        {
                            snprintf(sPower, sizeof(sPower), " power:%dW,", power);
                        }
                        else
                        {
                            free(sysFsPower1AveragePath);
                            sysFsPower1AveragePath = NULL;
                        }

                        sysfsRC = SYSFS_get_fan_speed(sysFsPwm1Path, sysFsPwm1MaxPath, &fanSpeed);
                        if (sysfsRC == 0)
                        {
                            snprintf(sFanSpeed, sizeof(sFanSpeed), " fan:%d%%,", fanSpeed);
                        }
                        else
                        {
                            free(sysFsPwm1MaxPath);
                            sysFsPwm1MaxPath = NULL;
                            free(sysFsPwm1Path);
                            sysFsPwm1Path = NULL;
                        }
#endif // __linux__
                    }
                }

                // memory error tracker
                char sMemErrors[32] = {};
                if (memTrackerEnabled != 0)
                {
                    snprintf(sMemErrors, sizeof(sMemErrors), " err:%u,", memErrorsDetected);
                }

                // compute average time from samples asynchronously
                double avgHr = 0;
                int t = 1;
                for (size_t i = 0; i < numSamples; ++i)
                {
                    double timeSec = ((double)profSamples[i]) * 0.000000001;
                    double hashesPerSec = ((double)batchSamples[i]) / timeSec;
                    double hs = hashesPerSec *0.001;

                    // compute avg
                    avgHr += (hs - avgHr) / t;
                    ++t;
                }

                applog(LOG_INFO, "cl_device(%d):%s%s%s%s hashrate: %.02f kH/s",
                           thr_id, sMemErrors, sGPUTemperature, sPower, sFanSpeed, avgHr);

                // update total hash-rate
                mtx_lock(&stats_lock);
                thr_hashrates[thr_id] = avgHr;
                mtx_unlock(&stats_lock);

                // reset timer
                hrTimerStart = std::chrono::steady_clock::now();
            }


            if (deviceMonitor != 0)
            {
                // perform another temperature checks here
                if (nvmlDevice != NULL)
                {
                    unsigned int nvmlTemperature = 0;
                    nvmlReturn_t nvmlRC = nvmlDeviceGetTemperature(nvmlDevice, NVML_TEMPERATURE_GPU, &nvmlTemperature);
                    if (nvmlRC == NVML_SUCCESS)
                    {
                        temperature = (int)nvmlTemperature;
                    }
                    else
                    {
                        applog(LOG_ERR, "cl_device(%d):Failed to get temperature, when temperature limit is set. exiting.", thr_id);
                        goto out;
                    }
                }
                else
                {
#ifdef _WIN32
                    // ADL
                    int adlRC = ADL_OK;
                    if (adlAdapterIndex != -1)
                    {
                        adlRC = ALD2_Overdrive_Temperature_Get(adlContext, adlAdapterIndex, overdriveVersion, &temperature);
                        if (adlRC != ADL_OK)
                        {
                            applog(LOG_ERR, "cl_device(%d):Failed to get temperature, when temperature limit is set. exiting.", thr_id);
                            goto out;
                        }
                    }
#elif defined __linux__
                    // perform another temperature checks here
                    int sysfsRC = 0;
                    sysfsRC = SYSFS_get_temperature(sysFsTemp1InputPath, &temperature);
                    if (sysfsRC != 0)
                    {
                        free(sysFsTemp1InputPath);
                        sysFsTemp1InputPath = NULL;

                        applog(LOG_ERR, "cl_device(%d):Failed to get temperature, when temperature limit is set. exiting.", thr_id);
                        goto out;
                    }
#endif // __linux__
                }
            }


            //-----------------------------------
            // Wait pipeline to finish
            errorCode = clFinish(clCommandQueue);
            if (errorCode != CL_SUCCESS)
            {
                applog(LOG_ERR, "cl_device(%d):Device not responding. error code: %d. terminating...", thr_id, errorCode);
                goto out;
            }

            //-----------------------------------
            // Handle occupancy
            const uint64_t waitTime = (occupancy * (uint32_t)elapsedTimeMs) / 100; // prevent overflow with u64
            if (waitTime != 0) { sleep_ms((int)waitTime); }

#ifdef VERTHASH_FULL_VALIDATION
            //-------------------------------------
            // Retrieve device data
            verthashIORES.clear();
            errorCode = clEnqueueReadBuffer(clCommandQueue, clmemResults, CL_TRUE, 0, globalWorkSize1x * sizeof(u32x8), verthashIORES.data(), 0, nullptr, nullptr);
            if (errorCode != CL_SUCCESS)
            {
                applog(LOG_ERR, "cl_device(%d):Failed to read a 'hash_result' buffer.", thr_id);
                goto out;
            }

            //-------------------------------------
            // record a profiler sample
            auto end = std::chrono::steady_clock::now();
            profSamples[sampleIndex] = std::chrono::duration<uint64_t, std::nano>(end - start).count();
            batchSamples[sampleIndex] = (uint64_t)globalWorkSize1x;
            elapsedTimeMs = ((double)profSamples[sampleIndex]) * 0.000001;
            sampleIndex = (sampleIndex + 1) & (profSamples.size()-1);
            ++numSamples;
            if (numSamples > profSamples.size()) { numSamples = profSamples.size(); }

            //-------------------------------------
            // Submit results
            for (size_t i = 0; i < globalWorkSize1x; ++i)
            {
                if (fulltestUvec8(verthashIORES[i], workInfo.target)) 
                {
                    // update nonce
                    workInfo.data[19] = nonce+i;

                    if (!submit_work(mythr, &workInfo))
                    {
                        // Submit work can fail due to abort flag.
                        if (!abort_flag)
                        {
                            applog(LOG_ERR, "cl_device(%d):Failed to submit work!", thr_id);
                            continue;
                        }
                        else
                        {
                            // Stop submitting to prevent spam.
                            break;
                        }
                    }

                    if (!have_stratum)
                    {
                        // We need to restart threads at this point if longpoll is not enabled to prevent inconclusive etc errors.
                        // If longpolling is enabled, then restart threads request will come from the longpoll thread
                        if (!have_longpoll) { restart_threads(); }
                        // on GBT result is one so exit...
                        break;
                    }
                }
            }
#else // HTarg test
            //-------------------------------------
            // Get potential result(num results + first one) fast path
            uint32_t testResult[2] = { 0 };
            errorCode = clEnqueueReadBuffer(clCommandQueue, clmemHTargetResults, CL_TRUE, 0, 2 * sizeof(uint32_t), &testResult[0], 0, nullptr, nullptr);
            if (errorCode != CL_SUCCESS)
            {
                applog(LOG_ERR, "cl_device(%d):Failed to read a 'hash_target_results' buffer. error code: %d", thr_id, errorCode);
                goto out;
            }
            uint32_t potentialResultCount = testResult[0];
            uint32_t potentialResult = testResult[1];

            //-------------------------------------
            // Check if at least 1 potential nonce was found
            if (potentialResultCount != 0)
            {
                if (opt_debug)
                {
                    applog(LOG_DEBUG, "cl_device(%d):Potential result count = %u", thr_id, potentialResultCount);
                }

                u32x8 hashResult;
                // get latest hash result from device
                clEnqueueReadBuffer(clCommandQueue, clmemResults, CL_TRUE, sizeof(u32x8)*potentialResult, sizeof(u32x8), &hashResult, 0, nullptr, nullptr);
                //-------------------------------------
                // test it against target
                bool gpuTest = false;
                if (fulltestUvec8(hashResult, workInfo.target))
                {
                    gpuTest = true;
                }

                // GPU memory error tracker
                bool cpuTest = false;
                if (memTrackerEnabled != 0)
                {
                    u32x8 hashResultCPU;
                    uint32_t uhTemp = uheader[19];
                    uheader[19] = potentialResult + nonce;
                    verthash_hash(verthashInfo.data,
                        verthashInfo.dataSize,
                        (const unsigned char(*)[VH_HEADER_SIZE])uheader,
                        (unsigned char(*)[VH_HASH_OUT_SIZE])hashResultCPU.v);
                    uheader[19] = uhTemp;

                    if (fulltestUvec8(hashResultCPU, workInfo.target)) { cpuTest = true; }
                    else
                    {
                        if (gpuTest == true)
                        {
                            applog(LOG_ERR, "cl_device(%d): Memory errors have been detected! Check your overclocking settings!", thr_id);
                            memErrorsDetected++;
                        }
                    }
                }
                else
                {
                    cpuTest = true;
                }

                if ((gpuTest == true) && (cpuTest == true))
                {
                    // add nonce local offset
                    results.push_back(potentialResult + nonce);
                }
                //-------------------------------------

                // continue with remaining potential results if they were found
                if (potentialResultCount > 1)
                {
                    size_t numRemainingNonces = potentialResultCount-1; // skip first result(it was handled previously)
                    // get remaining potential results
                    size_t offsetElem = 2; // potentialResultCount and potentialResult
                    if(potentialResults.size() < numRemainingNonces)
                    {
                        potentialResults.resize(numRemainingNonces);    
                    }
                    clEnqueueReadBuffer(clCommandQueue, clmemHTargetResults, CL_TRUE,
                                        offsetElem * sizeof(uint32_t), numRemainingNonces*sizeof(uint32_t), potentialResults.data(), 0, nullptr, nullptr);

                    for (size_t g = 0; g < numRemainingNonces; ++g)
                    {
                        // get latest hash result from device
                        clEnqueueReadBuffer(clCommandQueue, clmemResults, CL_TRUE, sizeof(u32x8)*potentialResults[g], sizeof(u32x8), &hashResult, 0, nullptr, nullptr);
                        //-------------------------------------
                        gpuTest = false;
                        if (fulltestUvec8(hashResult, workInfo.target))
                        {
                            gpuTest = true;
                        }

                        // GPU memory error tracker
                        cpuTest = false;
                        if (memTrackerEnabled != 0)
                        {
                            u32x8 hashResultCPU;
                            uint32_t uhTemp = uheader[19];
                            uheader[19] = potentialResults[g] + nonce;
                            verthash_hash(verthashInfo.data,
                                verthashInfo.dataSize,
                                (const unsigned char(*)[VH_HEADER_SIZE])uheader,
                                (unsigned char(*)[VH_HASH_OUT_SIZE])hashResultCPU.v);
                            uheader[19] = uhTemp;

                            if (fulltestUvec8(hashResultCPU, workInfo.target)) { cpuTest = true; }
                            else
                            {
                                if (gpuTest == true)
                                {
                                    applog(LOG_ERR, "cu_device(%d): Memory errors have been detected! Check your overclocking settings!", thr_id);
                                    memErrorsDetected++;
                                }
                            }
                        }
                        else
                        {
                            cpuTest = true;
                        }

                        if ((gpuTest == true) && (cpuTest == true))
                        {
                            // add nonce local offset
                            results.push_back(potentialResults[g] + nonce);
                        }
                        //-------------------------------------
                    }
                    // clear potential nonces. Not needed anymore.
                    potentialResults.clear();
                }

                // Clear result to prevent duplicate shares. Clear only potentialResultCount value.
                clEnqueueFillBuffer(clCommandQueue, clmemHTargetResults, &zero, sizeof(uint32_t), 0, (sizeof(uint32_t) * 1), 0, nullptr, nullptr);
                // OpenCL 1.0 - 1.1
                // clEnqueueWriteBuffer(clCommandQueue, clmemHTargetResults, CL_TRUE, 0, sizeof(cl_uint), &zero, 0, nullptr, nullptr);
            }

            //-------------------------------------
            // record a profiler sample
            auto end = std::chrono::steady_clock::now();
            profSamples[sampleIndex] = std::chrono::duration<uint64_t, std::nano>(end - start).count();
            batchSamples[sampleIndex] = (uint64_t)globalWorkSize1x;
            elapsedTimeMs = ((double)profSamples[sampleIndex]) * 0.000001;
            sampleIndex = (sampleIndex + 1) & (profSamples.size()-1);
            ++numSamples;
            if (numSamples > profSamples.size()) { numSamples = profSamples.size(); }

            //-------------------------------------
            // Send results
            for (size_t i = 0; i < results.size(); ++i)
            {
                workInfo.data[19] = results[i]; // HTarget results

                if (opt_debug)
                {
                    applog(LOG_ERR, "cl_device(%d):Submit work(nonce: %u)", thr_id, results[i]);
                }

                if (!submit_work(mythr, &workInfo))
                {
                    // Submit work can fail due to abort flag.
                    if (!abort_flag)
                    {
                        applog(LOG_ERR, "cl_device(%d):Failed to submit work!", thr_id);
                        continue;
                    }
                    else
                    {
                        // Stop submitting to prevent spam.
                        break;
                    }
                }

                if (!have_stratum)
                {
                    // We need to restart threads at this point if longpoll is not enabled to prevent inconclusive etc errors.
                    // If longpolling is enabled, then restart threads request will come from the longpoll thread
                    if (!have_longpoll) { restart_threads(); }
                    // on GBT result is one so exit...
                    break;
                }
            }

            // clear results. Not needed anymore.
            results.clear();
#endif // HTarget result tests


            //-------------------------------------
            // max nonce limit
            nonce64 += globalWorkSize1x;
            if (nonce64 >= maxNonce)
            {
                applog(LOG_INFO, "cl_device(%d):Device has completed its nonce range.", thr_id);
                break;
            }

        } // end is running

    } // end is thread running

out:

    applog(LOG_INFO, "cl_device(%d):Exiting worker thread id(%d)...", thr_id, thr_id);

    // reset per thread hash-rate. Inactive device shouldn't contribute to the total hash-rate
    mtx_lock(&stats_lock);
    thr_hashrates[thr_id] = 0;
    mtx_unlock(&stats_lock);

    //-------------------------------------
    // Destroy worker
    // buffers
    if (clmemKStates != NULL) { clReleaseMemObject(clmemKStates); }
    if (clmemHeader != NULL) { clReleaseMemObject(clmemHeader); }
    if (clmemFullDat != NULL) { clReleaseMemObject(clmemFullDat); }
    if (clmemResults != NULL) { clReleaseMemObject(clmemResults); }
    if (clmemHTargetResults != NULL) { clReleaseMemObject(clmemHTargetResults); }
    // programs and kernels
    // SHA3_512_precompute(initial pass)
    if (clkernelSHA3_512_precompute != NULL) { clReleaseKernel(clkernelSHA3_512_precompute); }
    if (clprogramSHA3_512_precompute != NULL) { clReleaseProgram(clprogramSHA3_512_precompute); }
    // SHA3_512_256(first pass)
    if (clkernelSHA3_512_256 != NULL) { clReleaseKernel(clkernelSHA3_512_256); }
    if (clprogramSHA3_512_256 != NULL) { clReleaseProgram(clprogramSHA3_512_256); }
    // Verthash IO(second pass)
    if (clkernelVerthash != NULL) { clReleaseKernel(clkernelVerthash); }
    if (clprogramVerthash != NULL) { clReleaseProgram(clprogramVerthash); }
    // misc
    if (clCommandQueue != NULL) { clReleaseCommandQueue(clCommandQueue); }
    if (clContext != NULL) { clReleaseContext(clContext); }

    //-------------------------------------
    // Free monitoing data
#ifdef _WIN32
    if (adlContext != NULL)
    {
        int adlRC = ADL2_Main_Control_Destroy(adlContext);
        if (adlRC != ADL_OK)
        {
            applog(LOG_ERR, "cl_device(%d):Failed to destroy ADL context", thr_id);
        }
    }
#elif defined __linux__

    free(sysFsPwm1Path);
    free(sysFsPwm1MaxPath);
    free(sysFsPower1AveragePath);
    free(sysFsTemp1InputPath);
#endif // __linux__

    //-------------------------------------
    // Exit thread
    tq_freeze(mythr->q);

    ++numWorkersExited;
    if (numWorkersExited == opt_n_threads) 
    {
        if (!abort_flag) { abort_flag = true; }
        // Trigger workio thread to exit
        tq_push(thr_info[work_thr_id].q, NULL);

        applog(LOG_INFO, "All worker threads have been exited.");
    }

    return 0;
}


// This option gets automatically set during CMake configuration.
//#define HAVE_CUDA

#ifdef HAVE_CUDA

#include <cuda_runtime.h>

// Usually equals clworkers.size()
static uint32_t cuDeviceIndexOffset = 0;

//-----------------------------------------------------------------------------
// CUDA kernel prototypes

// sha3 precompute kernel
extern "C" void sha3_512_precompute_cuda(size_t blocksPerGrid, size_t threadsPerBlock,
                                         uint32_t* output, uint32_t* hheader);

extern "C" void sha3_512_256_cuda(int blocksPerGrid, int threadsPerBlock,
                                  uint32_t* output,
                                  uint32_t in18,
                                  uint32_t firstNonce);

#ifdef VERTHASH_FULL_VALIDATION
extern "C" void verthash_cuda(size_t blocksPerGrid, size_t threadsPerBlock,
                              uint32_t* output,
                              uint32_t* kStates,
                              uint32_t* memory,
                              uint32_t in18,
                              uint32_t firstNonce);
#else
// verthashIO monolithic kernel
extern "C" void verthash_cuda(size_t blocksPerGrid, size_t threadsPerBlock,
                              uint32_t* output,
                              uint32_t* kStates,
                              uint32_t* memory,
                              uint32_t in18,
                              uint32_t firstNonce,
                              uint32_t* htargetResults,
        #ifdef VERTHASH_EXTENDED_VALIDATION
                              uint64_t target);
        #else
                              uint32_t target);
        #endif

#endif // !VERTHASH_FULL_VALIDATION

//-----------------------------------------------------------------------------
// CUDA worker thread
struct cuworker_t
{
    struct thr_info* threadInfo;
    vh::cudevice_t cudevice;
    size_t workSize;
    uint32_t batchTimeMs;
    uint32_t occupancyPct;

    // monitoring
    nvmlDevice_t nvmlDevice;
    int gpuTemperatureLimit;
    int deviceMonitor;
};

static int verthashCuda_thread(void *userdata)
{
    if (opt_debug)
    {
        applog(LOG_DEBUG, "Verthash CUDA thread started");
    }

    //-------------------------------------
    // Get thread data
    cuworker_t* cuworker = (cuworker_t*)userdata;
    vh::cudevice_t& cudevice = cuworker->cudevice;
    int cuWorkerIndex = cuworker->threadInfo->id - cuDeviceIndexOffset; 
    struct thr_info *mythr = cuworker->threadInfo;
    int thr_id = mythr->id;

    //-------------------------------------
    // Memory error tracker
    int memTrackerEnabled = 1;
    uint32_t memErrorsDetected = 0;

    //-------------------------------------
    // Monitoring data
    int deviceMonitor = cuworker->deviceMonitor;
    int gpuTemperatureLimit = cuworker->gpuTemperatureLimit;
    bool throttling = false;
    // stats
    int temperature = 0;
    int power = 0;
    int fanSpeed = 0;

    // NVML
    nvmlDevice_t nvmlDevice = cuworker->nvmlDevice;
    if ((nvmlDevice == NULL) && (cuworker->deviceMonitor != 0))
    {
        applog(LOG_WARNING, "cu_device(%d):Monitoring has been disabled. No backends available!", cuWorkerIndex);
        deviceMonitor = 0;
    }
    //-------------------------------------
    // Work related stuff
    struct work workInfo = { { 0 } };

    // adaptive batch size flag
    const bool adaptiveBatchSize = (cuworker->workSize == 0)? true : false;

    // batch size
    // 4096 is a starting batch size in adaptive mode
    size_t workSize = (adaptiveBatchSize == true)? 4096 : cuworker->workSize;

    // occupancy percent(workSize must be adaptive)
    uint32_t occupancyPct = (adaptiveBatchSize == true) ? cuworker->occupancyPct : 100;
    // extra check to prevent 0
    if ((occupancyPct == 0) || (occupancyPct > 100))
    {
        occupancyPct = 100;
    }
    const uint32_t occupancy = 100 - occupancyPct;

    size_t globalWorkSize1x = workSize;

    //-------------------------------------
    // Init nonce range 
    const uint64_t numNoncesGlobal = 4294967296ULL; // global nonce range is [0..4294967295]
    const uint64_t maxBatches = (numNoncesGlobal / (uint64_t)workSize);
    uint64_t maxBatchesPerDevice = maxBatches;
    uint32_t firstNonce = 0;
    uint64_t maxNonce = numNoncesGlobal; // actually last nonce is maxNonce-1
    if (have_stratum == false) // GBT
    {
        // There is no extranonce2 on GBT. Split a single nonce range between workers.
        maxBatchesPerDevice = maxBatches / opt_n_threads;
        // begin nonce range
        firstNonce = (maxBatchesPerDevice * workSize) * thr_id;
        // Handle case when the number of workers is not power of 2
        if (thr_id == (opt_n_threads - 1))
        {
            maxBatchesPerDevice += (maxBatches % opt_n_threads);
        }
        // end nonce range
        maxNonce = firstNonce + (maxBatchesPerDevice * workSize);
    }

    //-------------------------------------
    // Adaptive batch size settings
    double maxMs = (double)vh::defaultBatchTimeMs; // maxBathTime
    if(cuworker->batchTimeMs != 0)
    {
        maxMs = (double)cuworker->batchTimeMs;
    }
    const uint64_t alignment = 256; // 256 minimum possible value (max local work size in the pipeline)
    const size_t minBatchSize = alignment;
    double elapsedTimeMs = maxMs; // batchTimeTimer in milliseconds
    // Max Work(Batch)Size, that can be used during the adaptive batch size.
    // It will be lowered automatically, in case of memory allocation errors, but will never go up.
    size_t maxBatchSize = 134217728;


#ifdef VERTHASH_FULL_VALIDATION
    // Host side hash storage
    std::vector<u32x8> verthashIORES;
    verthashIORES.resize(workSize);
#else
    // HTarg result host side storage
    std::vector<uint32_t> results;
    std::vector<uint32_t> potentialResults; // used if (potentialNonceCount > 1)
#endif

    // init per device profiling data
    const size_t maxProfSamples = 16;
    size_t numSamples = 0;
    size_t sampleIndex = 0;
    std::vector<uint64_t> profSamples(maxProfSamples, 0);
    std::vector<uint64_t> batchSamples(maxProfSamples, 0);

    // per hash-rate update timer 
    std::chrono::steady_clock::time_point hrTimerStart;
    // Hash-rate (console)report interval in seconds
    uint64_t hrTimerIntervalSec = 4; // TODO: make configurable
    
    //------------------------------
    // Init CUDA data
    cudaError_t cuerr;
    // buffers
    uint32_t* dmemKStates = NULL;
    uint32_t* dmemFullDat = NULL;
    uint32_t* dmemResults = NULL;
    uint32_t* dmemHTargetResults = NULL;

    cuerr = cudaSetDevice(cuworker->cudevice.cudeviceHandle);
    if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to assign a CUDA device to thread. error code: %d", cuWorkerIndex, cuerr); goto out; }

    // remove busy waiting, but reduce performance by 0.1%
    cuerr = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to set CUDA device flags. error code: %d", cuWorkerIndex, cuerr); goto out; }

    //-------------------------------------
    // Create buffers

    //! 8 precomputed sha3/keccak states
    cuerr = cudaMalloc((void**)&dmemKStates, 8 * sizeof(kState));
    if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to create a SHA3/Keccak states buffer. error code: %d", cuWorkerIndex, cuerr); goto out; }

    //! hash results
    cuerr = cudaMalloc((void**)&dmemResults, workSize * sizeof(u32x8));
    if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to create a hash results buffer. error code: %d", cuWorkerIndex, cuerr); goto out; }

    //! results against hash target.
    // Too much, but 100% robust: workSize + num_actual_results + 1
    cuerr = cudaMalloc((void**)&dmemHTargetResults, sizeof(uint32_t) * (workSize + 1));
    if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to create a hash target results buffer. error code: %d", cuWorkerIndex, cuerr); goto out; }

    cuerr = cudaMemset(dmemHTargetResults, 0, (sizeof(uint32_t) * 1));
    if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to clean a hash target results buffer. error code: %d", cuWorkerIndex, cuerr); goto out; }

    // Upload verthash data
    if (verthashInfo.dataSize != 0) 
    {
        if(opt_debug)
        {
            applog(LOG_DEBUG, "Load verthash data size: %llu", verthashInfo.dataSize);
        }

        // create buffer and upload verthash data to GPU memory.
        cuerr = cudaMalloc((void**)&dmemFullDat, verthashInfo.dataSize);
        if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to create a verthash data buffer. error code: %d", cuWorkerIndex, cuerr); goto out; }

        // upload
        cuerr = cudaMemcpy(dmemFullDat, verthashInfo.data, verthashInfo.dataSize, cudaMemcpyHostToDevice);
        if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to copy a verthash data to device memory. error code: %d", cuWorkerIndex, cuerr); goto out; }
    }
    else
    {
        applog(LOG_ERR, "Verthash data is empty!"); 
    }


    if (adaptiveBatchSize == false)
    {
        // print work size:
        applog(LOG_INFO, "cu_device(%d): WorkSize has been set to: %u", cuWorkerIndex, (uint32_t)workSize);
    }

    //-------------------------------------
    // reset hash-rate reporting timer
    hrTimerStart = std::chrono::steady_clock::now();
    
    while (!abort_flag)
    {
        uint32_t maxRuns, runs, nonce;

        // Stratum
        if (have_stratum)
        {
            while (time(NULL) >= g_work_time + 120)
            {
                if (abort_flag) { goto out; }
                sleep_ms(1);
            }
            mtx_lock(&g_work_lock);
            stratum_gen_work(&stratum, &g_work);
        }
        else // GBT
        {
            // obtain new work from internal workio thread
            mtx_lock(&g_work_lock);

            work_free(&g_work);
            if (unlikely(!get_work(mythr, &g_work)))
            {
                if (!abort_flag) { applog(LOG_ERR, "cu_device(%d):Work retrieval failed, exiting mining thread %d", cuWorkerIndex, mythr->id); goto out; }
                mtx_unlock(&g_work_lock);
                goto out;
            }

            g_work_time = time(NULL);
        }

        // create a work copy
        work_free(&workInfo);
        work_copy(&workInfo, &g_work);
        workInfo.data[19] = 0;
        work_restart[thr_id].restart = 0;

        mtx_unlock(&g_work_lock);

        // Actual nonce is 32 bit, but we use 64 bit to prevent possible overflows
        uint64_t nonce64 = firstNonce;


#ifdef VERTHASH_EXTENDED_VALIDATION
        uint64_t wtarget = ((uint64_t(workInfo.target[7])) << 32) | (uint64_t(workInfo.target[6]) & 0xFFFFFFFFUL);
#else
        uint32_t wtarget = workInfo.target[7];
#endif
        //-------------------------------------
        // Generate midstate
        uint32_t uheader[20] = {0};
        for (size_t i = 0; i < 20; ++i)
        {
            be32enc(&uheader[i], workInfo.data[i]);
        }


        /*printf("GBTwork.target: %u, %u, %u, %u, %u, %u, %u, %u\n",
               workInfo.target[0], workInfo.target[1], workInfo.target[2], workInfo.target[3],
               workInfo.target[4], workInfo.target[5], workInfo.target[6], workInfo.target[7]);

        // print dec
        //printf("GBTwork.data: %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u\n",
        //       gbtWork.data[0], gbtWork.data[1], gbtWork.data[2], gbtWork.data[3], gbtWork.data[4],
        //       gbtWork.data[5], gbtWork.data[6], gbtWork.data[7], gbtWork.data[8], gbtWork.data[9],
        //       gbtWork.data[10], gbtWork.data[11], gbtWork.data[12], gbtWork.data[13], gbtWork.data[14],
        //       gbtWork.data[15], gbtWork.data[16], gbtWork.data[17], gbtWork.data[18], gbtWork.data[19]);

        // print in hex
        printf("GBTwork.data: 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x,"
                              "0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x, 0x%08x\n",
               workInfo.data[0], workInfo.data[1], workInfo.data[2], workInfo.data[3], workInfo.data[4],
               workInfo.data[5], workInfo.data[6], workInfo.data[7], workInfo.data[8], workInfo.data[9],
               workInfo.data[10], workInfo.data[11], workInfo.data[12], workInfo.data[13], workInfo.data[14],
               workInfo.data[15], workInfo.data[16], workInfo.data[17], workInfo.data[18], workInfo.data[19]);

        fflush(stdout);*/

        //-------------------------------------
        // SHA3_precompute

        //check_cuda(cudaMemcpy(dmemHeaders, uheader, 68, cudaMemcpyHostToDevice));
        //sha3_512_precompute(8,1, dmemKStates, dmemHeaders, uheader[17], uheader[18]);
        // header as constant
        sha3_512_precompute_cuda(8,1, dmemKStates, uheader);

        cuerr = cudaDeviceSynchronize();
        if (cuerr != cudaSuccess) { applog(LOG_ERR, "Device not responding. cu_device(%d) error code: %d", cuWorkerIndex, cuerr); goto out; }

        //-------------------------------------
        // Compute hashes

        // abort flag is needed here to prevent an inconsistent behaviour
        // in case program termination was triggered between work generation and resetting work restart values
        while (!work_restart[thr_id].restart && (!abort_flag))
        {
            //---------------------------------------------------------------------
            // Temperature limit
            // 0 means handled by the device driver. This check is still required in case backend reports wrong temp.
            if ((gpuTemperatureLimit != 0) && (deviceMonitor != 0))
            {
                if (temperature > gpuTemperatureLimit)
                {
                    if (throttling == false)
                    {
                        throttling = true;
                        applog(LOG_WARNING, "cu_device(%d): gpu temperature limit has reached %dC(max:%uC), stopping.", cuWorkerIndex, temperature, gpuTemperatureLimit);
                    }
                    // throttling
                    const int waitTimeMs = 100;
                    sleep_ms(waitTimeMs);

                    // emit empty sample(
                    profSamples[sampleIndex] = (uint64_t)waitTimeMs;
                    batchSamples[sampleIndex] = 0;
                    sampleIndex = (sampleIndex + 1) & (profSamples.size()-1);
                    ++numSamples;
                    if (numSamples > profSamples.size())
                    {
                        numSamples = profSamples.size();
                    }

                    // perform another temperature checks here
                    if (nvmlDevice != NULL)
                    {
                        unsigned int nvmlTemperature = 0;
                        nvmlReturn_t nvmlRC = nvmlDeviceGetTemperature(nvmlDevice, NVML_TEMPERATURE_GPU, &nvmlTemperature);
                        if (nvmlRC == NVML_SUCCESS)
                        {
                            temperature = (int)nvmlTemperature;
                        }
                        else
                        {
                            applog(LOG_ERR, "cu_device(%d):Failed to get temperature, when temperature limit is set. exiting.", cuWorkerIndex);
                            goto out;
                        }
                    }

                    // extra check for the throttling flag
                    if(temperature <= gpuTemperatureLimit)
                    {
                        applog(LOG_WARNING, "cu_device(%d): cooled down, resuming.", cuWorkerIndex);
                        throttling = false;
                    }

                    // restart loop
                    continue;
                }
            }

            //---------------------------------------------------------------------
            // Adaptive batch size
            //---------------------------------------------------------------------
            if (adaptiveBatchSize == true)
            {
                double ms = elapsedTimeMs;

                size_t newBatchSize = globalWorkSize1x;
                if(ms > maxMs)
                {
                    double p = ms / maxMs; // possible division by 0 is handled outside
                    newBatchSize /= p; // possible division by 0 is handled outside
                    newBatchSize = align_u64(newBatchSize, alignment);

                    globalWorkSize1x = newBatchSize;

                    if (opt_debug) { applog(LOG_DEBUG, "cu_device(%d):Worksize decreased to %zu", cuWorkerIndex, newBatchSize); }
                }
                else if (ms < maxMs)
                {
                    double p = maxMs / ms; // possible division by 0 is handled outside
                    newBatchSize *= p;
                    newBatchSize = align_u64(newBatchSize, alignment);

                    // resize buffers(if possible)
                    if ((newBatchSize > workSize)) // workSize in this case is a buffer work size
                    {
                        while (maxBatchSize > minBatchSize)
                        {
                            if (dmemResults != NULL) { cudaFree(dmemResults); }
                            cuerr = cudaMalloc((void**)&dmemResults, newBatchSize * sizeof(u32x8));
                            if (cuerr == cudaSuccess)
                            {
                                // Some drivers use lazy allocation. Prove, that we have enough memory.
                                cuerr = cudaMemset(dmemResults, 0, (sizeof(uint32_t) * 1));
                            }

                            if (cuerr == cudaSuccess)
                            {
                                if (dmemHTargetResults != NULL) { cudaFree(dmemHTargetResults); }
                                cuerr = cudaMalloc((void**)&dmemHTargetResults, sizeof(uint32_t) * (newBatchSize + 1));
                                if (cuerr == cudaSuccess)
                                {
                                    // Some drivers use lazy allocation. Prove, that we have enough memory.
                                    cuerr = cudaMemset(dmemHTargetResults, 0, (sizeof(uint32_t) * 1));
                                }

                                if (cuerr == cudaSuccess)
                                {
#ifdef VERTHASH_FULL_VALIDATION
                                    // resize CPU hash buffer for VERTHASH_FULL_VALIDATION
                                    verthashIORES.resize(workSize);
#endif

                                    workSize = newBatchSize;
                                    globalWorkSize1x = newBatchSize;
                                    break;
                                }
                                else
                                {
                                    if (opt_debug) { applog(LOG_DEBUG, "cu_device(%d):Failed to create an HTarget buffer, recreate with the old batch size", cuWorkerIndex); }

                                    // out of memory error
                                    maxBatchSize >>= 1;
                                    if (maxBatchSize == 0) { maxBatchSize = minBatchSize; }

                                    //  failed to create a buffer, recreate with the old batch size
                                    cudaMalloc((void**)&dmemResults, workSize * sizeof(u32x8));
                                    //  failed to create a buffer, recreate with the old batch size
                                    cudaMalloc((void**)&dmemHTargetResults, sizeof(uint32_t) * (workSize + 1));
                                }
                            }
                            else
                            {
                                if (opt_debug) { applog(LOG_DEBUG, "cu_device(%d):Failed to create a 'result' buffer, recreating with the old batch size...", cuWorkerIndex); }

                                // out of memory error
                                maxBatchSize >>= 1;
                                if (maxBatchSize == 0) { maxBatchSize = minBatchSize; }

                                cudaMalloc((void**)&dmemResults, workSize * sizeof(u32x8));
                            }
                        }

                        if (opt_debug) { applog(LOG_DEBUG, "cu_device(%d): Buffers have been updated for a new batchSize: %zu", cuWorkerIndex, workSize); }
                    }
                    else
                    {
                        if (newBatchSize > maxBatchSize) // overflow check
                        {
                            maxBatchSize >>= 1;
                            if (maxBatchSize < minBatchSize) { maxBatchSize = minBatchSize; }

                            if (opt_debug) { applog(LOG_DEBUG, "cu_device(%d): Update max batch size to: %zu", cuWorkerIndex, maxBatchSize); }
                        }
                        else
                        {
                            globalWorkSize1x = newBatchSize;

                            if (opt_debug) { applog(LOG_DEBUG, "cu_device(%d): Worksize increased to %zu", cuWorkerIndex, newBatchSize); }
                        }
                    }
                }

                // Handle overflow
                uint64_t nextNonce = nonce64 + globalWorkSize1x;
                if (nextNonce > maxNonce)
                {
                    globalWorkSize1x = globalWorkSize1x - (nextNonce - maxNonce);
                }
            }
            //------------------- Adaptive batch size end ---------------------------------
            uint32_t nonce = (uint32_t)nonce64;

            // verthash kernel run configuration
            int cudaThreadsPerBlock = 64;
            int cudaBlocksPerGrid = (globalWorkSize1x + cudaThreadsPerBlock - 1) / cudaThreadsPerBlock;

            auto start = std::chrono::steady_clock::now();

            //-------------------------------------
            // Run
            sha3_512_256_cuda(cudaBlocksPerGrid, cudaThreadsPerBlock, 
                              dmemResults, uheader[18], nonce);
#ifdef VERTHASH_FULL_VALIDATION
            verthash_cuda(cudaBlocksPerGrid*4, cudaThreadsPerBlock,
                          dmemResults, dmemKStates, dmemFullDat,
                          uheader[18], nonce);
#else
            verthash_cuda(cudaBlocksPerGrid*4, cudaThreadsPerBlock,
                          dmemResults, dmemKStates, dmemFullDat,
                          uheader[18], nonce, dmemHTargetResults, wtarget);
#endif

            //-----------------------------------
            // Asynchronous processing
            uint64_t hrSec = std::chrono::duration_cast<std::chrono::seconds>( std::chrono::steady_clock::now() - hrTimerStart ).count();
            if ((hrSec >= hrTimerIntervalSec) && (numSamples > 0))
            {
                char sGPUTemperature[32] = {};
                char sPower[32] = {};
                char sFanSpeed[32] = {};

                // monitoring
                if (deviceMonitor != 0)
                {
                    if (nvmlDevice != NULL)
                    {
                        unsigned int nvmlTemperature = 0;
                        unsigned int nvmlPower = 0;
                        unsigned int nvmlFanSpeed = 0;

                        // get data from NVML backend
                        nvmlReturn_t nvmlRC = nvmlDeviceGetTemperature(nvmlDevice, NVML_TEMPERATURE_GPU, &nvmlTemperature);
                        if (nvmlRC == NVML_SUCCESS)
                        {
                            temperature = (int)nvmlTemperature;
                            snprintf(sGPUTemperature, sizeof(sGPUTemperature), " temp:%dC,", temperature);
                        }
                        else
                        {
                            if(gpuTemperatureLimit != 0)
                            {
                                applog(LOG_ERR, "cu_device(%d):Failed to get temperature, when temperature limit is set. exiting.", cuWorkerIndex);
                                goto out;
                            }
                        }

                        nvmlRC = nvmlDeviceGetPowerUsage(nvmlDevice, &nvmlPower);
                        if (nvmlRC == NVML_SUCCESS)
                        {
                            power = (int)(nvmlPower / 1000);
                            snprintf(sPower, sizeof(sPower), " power:%dW,", power);
                        }

                        nvmlRC = nvmlDeviceGetFanSpeed(nvmlDevice, &nvmlFanSpeed);
                        if (nvmlRC == NVML_SUCCESS)
                        {
                            fanSpeed = (int)nvmlFanSpeed;
                            snprintf(sFanSpeed, sizeof(sFanSpeed), " fan:%d%%,", fanSpeed);
                        }
                    }
                }

                // memory error tracker
                char sMemErrors[32] = {};
                if (memTrackerEnabled != 0)
                {
                    snprintf(sMemErrors, sizeof(sMemErrors), " err:%u,", memErrorsDetected);
                }

                // compute average time from samples asynchronously
                double avgHr = 0;
                int t = 1;
                for (size_t i = 0; i < numSamples; ++i)
                {
                    double timeSec = ((double)profSamples[i]) * 0.000000001;
                    double hashesPerSec = ((double)batchSamples[i]) / timeSec;
                    double hs = hashesPerSec *0.001;
                    // compute avg
                    avgHr += (hs - avgHr) / t;
                    ++t;
                }

                applog(LOG_INFO, "cu_device(%d):%s%s%s%s hashrate: %.02f kH/s",
                           cuWorkerIndex, sMemErrors, sGPUTemperature, sPower, sFanSpeed, avgHr);

                // update total hash-rate
                mtx_lock(&stats_lock);
                thr_hashrates[thr_id] = avgHr;
                mtx_unlock(&stats_lock);

                // reset timer
                hrTimerStart = std::chrono::steady_clock::now();
            }


            if (deviceMonitor != 0)
            {
                // get data from NVML backend
                unsigned int nvmlTemperature = 0;
                nvmlReturn_t nvmlRC = nvmlDeviceGetTemperature(nvmlDevice, NVML_TEMPERATURE_GPU, &nvmlTemperature);
                if (nvmlRC == NVML_SUCCESS)
                {
                    temperature = (int)nvmlTemperature;
                }
                else
                {
                    if(gpuTemperatureLimit != 0)
                    {
                        applog(LOG_ERR, "cu_device(%d):Failed to get temperature, when temperature limit is set. exiting.", cuWorkerIndex);
                        goto out;
                    }
                }
            }


            //-----------------------------------
            // Wait pipeline to finish
            cuerr = cudaDeviceSynchronize();
            if (cuerr != cudaSuccess)
            {
                applog(LOG_ERR, "cu_device(%d):Device not responding. error code: %d. terminating...", cuWorkerIndex, cuerr);
                goto out;
            }

            //-----------------------------------
            // Handle occupancy
            const uint64_t waitTime = (occupancy * (uint32_t)elapsedTimeMs) / 100; // prevent overflow with u64
            if (waitTime != 0) { sleep_ms((int)waitTime); }

#ifdef VERTHASH_FULL_VALIDATION
            //-------------------------------------
            // Retrieve device data
            verthashIORES.clear();
            cuerr = cudaMemcpy(verthashIORES.data(), dmemResults, globalWorkSize1x * sizeof(u32x8), cudaMemcpyDeviceToHost);
            if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to download hash data. error code: %d", cuWorkerIndex, cuerr); goto out; }

            //-------------------------------------
            // Record a profiler sample
            auto end = std::chrono::steady_clock::now();
            profSamples[sampleIndex] = std::chrono::duration<uint64_t, std::nano>(end - start).count();
            batchSamples[sampleIndex] = (uint64_t)globalWorkSize1x;
            elapsedTimeMs = ((double)profSamples[sampleIndex]) * 0.000001;
            sampleIndex = (sampleIndex + 1) & (profSamples.size()-1);
            ++numSamples;
            if (numSamples > profSamples.size()) { numSamples = profSamples.size(); }

            //-------------------------------------
            // Submit results
            for (size_t i = 0; i < globalWorkSize1x; ++i)
            {
                if (fulltestUvec8(verthashIORES[i], workInfo.target)) 
                {
                    // update nonce
                    workInfo.data[19] = nonce+i;

                    if (!submit_work(mythr, &workInfo))
                    {
                        // Submit work can fail due to abort flag.
                        if (!abort_flag)
                        {
                            applog(LOG_ERR, "cu_device(%d):Failed to submit work!", cuWorkerIndex);
                            continue;
                        }
                        else
                        {
                            // Stop submitting to prevent spam.
                            break;
                        }
                    }

                    if (!have_stratum)
                    {
                        // We need to restart threads at this point if longpoll is not enabled to prevent inconclusive etc errors.
                        // If longpolling is enabled, then restart threads request will come from the longpoll thread
                        if (!have_longpoll) { restart_threads(); }
                        // on GBT result is one so exit...
                        break;
                    }
                }
            }
#else // HTarg test
            //-------------------------------------
            // Get potential result(num results + first one) fast path
            uint32_t testResult[2] = { 0 };
            cuerr = cudaMemcpy(&testResult[0], dmemHTargetResults, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to get a potential result. error code: %d", cuerr); goto out; }

            uint32_t potentialResultCount = testResult[0];
            uint32_t potentialResult = testResult[1];


            //-------------------------------------
            // Check if at least 1 potential nonce was found
            if (potentialResultCount != 0)
            {
                if(opt_debug)
                    applog(LOG_DEBUG, "cu_device(%d):Potential result count = %u", cuWorkerIndex, potentialResultCount);

                u32x8 hashResult;
                // get latest hash result from device
                cuerr = cudaMemcpy(&hashResult, dmemResults+(potentialResult*8), sizeof(u32x8), cudaMemcpyDeviceToHost);
                if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to get a potential hash result. error code: %d", cuWorkerIndex, cuerr); goto out; }
                //-------------------------------------
                // test it against target
                bool gpuTest = false;
                if (fulltestUvec8(hashResult, workInfo.target))
                {
                    gpuTest = true;
                }

                // GPU memory error tracker
                bool cpuTest = false;
                if (memTrackerEnabled != 0)
                {
                    u32x8 hashResultCPU;
                    uint32_t uhTemp = uheader[19];
                    uheader[19] = potentialResult + nonce;
                    verthash_hash(verthashInfo.data,
                                  verthashInfo.dataSize,
                                  (const unsigned char(*)[VH_HEADER_SIZE])uheader,
                                  (unsigned char(*)[VH_HASH_OUT_SIZE])hashResultCPU.v);
                    uheader[19] = uhTemp;

                    if (fulltestUvec8(hashResultCPU, workInfo.target)) { cpuTest = true; }
                    else
                    {
                        if (gpuTest == true)
                        {
                            applog(LOG_ERR, "cu_device(%d): Memory errors have been detected! Check your overclocking settings!", cuWorkerIndex);
                            memErrorsDetected++;
                        }
                    }
                }
                else
                {
                    cpuTest = true;
                }

                if ((gpuTest == true) && (cpuTest == true))
                {
                    // add nonce local offset
                    results.push_back(potentialResult + nonce);
                }
                //-------------------------------------

                // continue with remaining potential results if they were found
                if (potentialResultCount > 1)
                {
                    size_t numRemainingNonces = potentialResultCount-1; // skip first result(it was handled previously)
                    // get remaining potential results
                    size_t offsetElem = 2; // potentialResultCount and potentialResult
                    if(potentialResults.size() < numRemainingNonces)
                    {
                        potentialResults.resize(numRemainingNonces);    
                    }
                    cuerr = cudaMemcpy(potentialResults.data(), dmemHTargetResults+offsetElem, numRemainingNonces*sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to get remaining potential results. error code: %d", cuWorkerIndex, cuerr); goto out; }

                    for (size_t g = 0; g < numRemainingNonces; ++g)
                    {
                        // get latest hash result from device
                        cuerr = cudaMemcpy(&hashResult, dmemResults+(potentialResults[g]*8), sizeof(u32x8), cudaMemcpyDeviceToHost);
                        if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to get a potential hash result(%llu). error code: %d", cuWorkerIndex, cuerr, (uint64_t)g); goto out; }

                        //-------------------------------------
                        gpuTest = false;
                        if (fulltestUvec8(hashResult, workInfo.target))
                        {
                            gpuTest = true;
                        }

                        // GPU memory error tracker
                        cpuTest = false;
                        if (memTrackerEnabled != 0)
                        {
                            u32x8 hashResultCPU;
                            uint32_t uhTemp = uheader[19];
                            uheader[19] = potentialResults[g] + nonce;
                            verthash_hash(verthashInfo.data,
                                          verthashInfo.dataSize,
                                          (const unsigned char(*)[VH_HEADER_SIZE])uheader,
                                          (unsigned char(*)[VH_HASH_OUT_SIZE])hashResultCPU.v);
                            uheader[19] = uhTemp;

                            if (fulltestUvec8(hashResultCPU, workInfo.target)) { cpuTest = true; }
                            else
                            {
                                if (gpuTest == true)
                                {
                                    applog(LOG_ERR, "cu_device(%d): Memory errors have been detected! Check your overclocking settings!", cuWorkerIndex);
                                    memErrorsDetected++;
                                }
                            }
                        }
                        else
                        {
                            cpuTest = true;
                        }

                        if ((gpuTest == true) && (cpuTest == true))
                        {
                            // add nonce local offset
                            results.push_back(potentialResults[g] + nonce);
                        }
                        //-------------------------------------

                    }
                    // clear potential nonces. Not needed anymore.
                    potentialResults.clear();
                }

                // Clear result to prevent duplicate shares. Clear only potentialResultCount value.
                cuerr = cudaMemset(dmemHTargetResults, 0, (sizeof(uint32_t) * 1));
                if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to clean a hash target results buffer. error code: %d", cuWorkerIndex, cuerr); goto out; }
            }

            //-------------------------------------
            // Record a profiler sample
            auto end = std::chrono::steady_clock::now();
            profSamples[sampleIndex] = std::chrono::duration<uint64_t, std::nano>(end - start).count();
            batchSamples[sampleIndex] = (uint64_t)globalWorkSize1x;
            elapsedTimeMs = ((double)profSamples[sampleIndex]) * 0.000001;
            sampleIndex = (sampleIndex + 1) & (profSamples.size()-1);
            ++numSamples;
            if (numSamples > profSamples.size()) { numSamples = profSamples.size(); }

            //-------------------------------------
            // Submit results
            for (size_t i = 0; i < results.size(); ++i)
            {
                workInfo.data[19] = results[i]; // HTarget results

                if (!submit_work(mythr, &workInfo))
                {
                    // Submit work can fail due to abort flag.
                    if (!abort_flag)
                    {
                        applog(LOG_ERR, "cu_device(%d):Failed to submit work!", cuWorkerIndex);
                        continue;
                    }
                    else
                    {
                        // Stop submitting to prevent spam.
                        break;
                    }
                }

                if (!have_stratum)
                {
                    // We need to restart threads at this point if longpoll is not enabled to prevent inconclusive etc errors.
                    // If longpolling is enabled, then restart threads request will come from the longpoll thread
                    if (!have_longpoll) { restart_threads(); }
                    // on GBT result is one so exit...
                    break;
                }
            }

            // clear results. Not needed anymore.
            results.clear();
#endif // HTarget result tests

            //-------------------------------------
            // max nonce limit
            nonce64 += globalWorkSize1x;
            if (nonce64 >= maxNonce)
            {
                applog(LOG_INFO, "cu_device(%d):Device has completed its nonce range!!!", cuWorkerIndex);
                break;
            }

        } // end is running

    } // end is thread running

out:

    applog(LOG_INFO, "Exiting worker thread id(%d)...", thr_id);

    // reset per thread hash-rate. Inactive device shouldn't contribute to the total hash-rate
    mtx_lock(&stats_lock);
    thr_hashrates[thr_id] = 0;
    mtx_unlock(&stats_lock);

    //-------------------------------------
    // Destroy
    if (dmemHTargetResults != NULL) { cudaFree(dmemHTargetResults); }
    if (dmemResults != NULL) { cudaFree(dmemResults); }
    if (dmemFullDat != NULL) { cudaFree(dmemFullDat); }
    if (dmemKStates != NULL) { cudaFree(dmemKStates); }

    //-------------------------------------
    // Exit thread
    tq_freeze(mythr->q);

    ++numWorkersExited;
    if (numWorkersExited >= opt_n_threads)
    {
        if (!abort_flag) { abort_flag = true; }
        // trigger WorkIO thread to exit
        tq_push(thr_info[work_thr_id].q, NULL);

        applog(LOG_INFO, "All worker threads have been exited.");
    }

    return 0;
}

#endif // HAVE_CUDA





static int longpoll_thread(void *userdata)
{
    if (opt_debug)
    {
        applog(LOG_DEBUG, "Longpoll thread started");
    }

    struct thr_info *mythr = (struct thr_info*)userdata;
    CURL *curl = NULL;
    char *copy_start, *hdr_path = NULL, *lp_url = NULL;
    bool need_slash = false;

    curl = curl_easy_init();
    if (unlikely(!curl))
    {
        applog(LOG_ERR, "CURL initialization failed");
        goto out;
    }

start:
    hdr_path = (char*)tq_pop(mythr->q, NULL);
    if (!hdr_path)
    {
        goto out;
    }

    // full URL
    if (strstr(hdr_path, "://"))
    {
        lp_url = hdr_path;
        hdr_path = NULL;
    }
    else
    {
        // absolute path, on current server
        copy_start = (*hdr_path == '/') ? (hdr_path + 1) : hdr_path;
        if (rpc_url[strlen(rpc_url) - 1] != '/')
        {
            need_slash = true;
        }

        lp_url = (char*)malloc(strlen(rpc_url) + strlen(copy_start) + 2);
        if (!lp_url)
        {
            goto out;
        }

        sprintf(lp_url, "%s%s%s", rpc_url, need_slash ? "/" : "", copy_start);
    }

    applog(LOG_INFO, "Long-polling activated for %s", lp_url);

    while (!abort_flag)
    {
        json_t *val, *res, *soval;
        char *req = NULL;
        int err;

        req = (char*)malloc(strlen(gbt_lp_req) + strlen(lp_id) + 1);
        sprintf(req, gbt_lp_req, lp_id);

        val = json_rpc_call(curl, lp_url, rpc_user, rpc_pass,
                            req,
                            &err,
                            JSON_RPC_LONGPOLL);
        free(req);
        if (have_stratum)
        {
            if (val)
            {
                json_decref(val);
            }
            goto out;
        }
        if (likely(val))
        {
            bool rc;
            applog(LOG_INFO, "LONGPOLL pushed new work");
            res = json_object_get(val, "result");
            soval = json_object_get(res, "submitold");
            submit_old = soval ? json_is_true(soval) : false;
            mtx_lock(&g_work_lock);
            work_free(&g_work);

            rc = gbt_work_decode(res, &g_work);
            if (rc)
            {
                time(&g_work_time);
                restart_threads();
            }
            mtx_unlock(&g_work_lock);
            json_decref(val);
        }
        else
        {
            mtx_lock(&g_work_lock);
            g_work_time -= LP_SCANTIME;
            mtx_unlock(&g_work_lock);
            if (err == CURLE_OPERATION_TIMEDOUT)
            {
                restart_threads();
            }
            else if ((err == CURLE_ABORTED_BY_CALLBACK) && (abort_flag == true))
            {
                if (opt_debug) { applog(LOG_DEBUG, "Trigger workIO exit from longpoll thread"); }
                tq_push(thr_info[work_thr_id].q, NULL);
                goto out;
            }
            else
            {
                if (opt_debug) { applog(LOG_DEBUG, "Have_longpoll = false"); }

                have_longpoll = false;
                restart_threads();
                free(hdr_path);
                free(lp_url);
                lp_url = NULL;

                // Async wait allows to enter deinitialization phase faster especially if time between retries is very long.
                uint64_t waitTimeSec = (uint64_t)opt_fail_pause;
                uint64_t sec = 0;
                std::chrono::steady_clock::time_point hrTimerStart = std::chrono::steady_clock::now();
                while (sec < waitTimeSec)
                {
                    sleep_ms(1);
                    sec = std::chrono::duration_cast<std::chrono::seconds>( std::chrono::steady_clock::now() - hrTimerStart ).count();
                    if (abort_flag) { break; }
                }

                goto start;
            }
        }
    }

out:
    free(hdr_path);
    free(lp_url);
    tq_freeze(mythr->q);
    if (curl)
    {
        curl_easy_cleanup(curl);
    }

    if (opt_debug)
    {
        applog(LOG_DEBUG, "Exit longpoll thread");
    }

    return 0;
}

static bool stratum_handle_response(char *buf)
{
    json_t *val, *err_val, *res_val, *id_val;
    json_error_t err;
    bool ret = false;

    val = JSON_LOADS(buf, &err);
    if (!val)
    {
        applog(LOG_INFO, "JSON decode failed(%d): %s", err.line, err.text);
        goto out;
    }

    res_val = json_object_get(val, "result");
    err_val = json_object_get(val, "error");
    id_val = json_object_get(val, "id");

    if (!id_val || json_is_null(id_val) || !res_val)
        goto out;

    share_result(json_is_true(res_val),
                 err_val ? json_string_value(json_array_get(err_val, 1)) : NULL);

    ret = true;
out:
    if (val)
        json_decref(val);

    return ret;
}

static int stratum_thread(void *userdata)
{
    struct thr_info *mythr = (struct thr_info*)userdata;
    char *s;

    stratum.url = (char*)tq_pop(mythr->q, NULL);
    if (!stratum.url)
    {
        goto out;
    }
    applog(LOG_INFO, "Starting Stratum on %s", stratum.url);

    while (!abort_flag)
    {
        int failures = 0;

        while (!stratum.curl && !abort_flag)
        {
            mtx_lock(&g_work_lock);
            g_work_time = 0;
            restart_threads();
            mtx_unlock(&g_work_lock);
            
            if (!stratum_connect(&stratum, stratum.url) ||
                !stratum_subscribe(&stratum) ||
                !stratum_authorize(&stratum, rpc_user, rpc_pass)) {
                stratum_disconnect(&stratum);
                if (opt_retries >= 0 && ++failures > opt_retries) {
                    applog(LOG_ERR, "...terminating workio thread");
                    goto out;
                }
                applog(LOG_ERR, "...retry after %d seconds", opt_fail_pause);

                // Async wait allows to enter deinitialization phase faster especially if time between retries is very long.
                uint64_t waitTimeSec = (uint64_t)opt_fail_pause;
                uint64_t sec = 0;
                std::chrono::steady_clock::time_point hrTimerStart = std::chrono::steady_clock::now();
                while (sec < waitTimeSec)
                {
                    sleep_ms(1);
                    sec = std::chrono::duration_cast<std::chrono::seconds>( std::chrono::steady_clock::now() - hrTimerStart ).count();
                    if (abort_flag) { goto out; }
                }
            }
        }

        int jobIDsNotEq = 1;
        if ((g_work.job_id != NULL) && (stratum.job.job_id != NULL))
        {
            jobIDsNotEq = strcmp(stratum.job.job_id, g_work.job_id);
        }

        if ((g_work_time == 0) || (jobIDsNotEq != 0))
        {
            mtx_lock(&g_work_lock);

            // Due to session restore feature, first mining.notify message may not request miner to clean previous jobs.
            // Thus 0 block height check is needed here too.
            if (stratum.job.clean)
            {
                uint32_t blockHeight = stratum_get_block_height(&stratum);
                if (stratum.blockHeight != blockHeight)
                {
                    stratum.blockHeight = blockHeight;
                    applog(LOG_INFO, "Verthash block: %u", blockHeight);
                }
            }

            restart_threads();

            time(&g_work_time);
            mtx_unlock(&g_work_lock);
        }
        
        s = NULL;
        if (!stratum_socket_full(&stratum, 120))
        {
            applog(LOG_ERR, "Stratum connection timed out");
        }
        else if (!abort_flag) // stratum_socket_full may set abort_flag
        {
            s = stratum_recv_line(&stratum);
        }

        if (!s)
        {
            stratum_disconnect(&stratum);
            applog(LOG_ERR, "Stratum connection interrupted");
            continue;
        }

        if (!stratum_handle_method(&stratum, s))
        {
            stratum_handle_response(s);
        }
        free(s);
    }

out:
    if (!abort_flag) { abort_flag = true; }
    tq_push(thr_info[work_thr_id].q, NULL);

    applog(LOG_DEBUG, "Stratum thread exit");

    return 0;
}

static void show_version_and_exit(void)
{
    printf(PACKAGE_NAME " " PACKAGE_VERSION "\nbuilt on " __DATE__ "\n");
    printf("%s\n", curl_version());
#ifdef JANSSON_VERSION
    printf("libjansson %s\n", JANSSON_VERSION);
#endif
#ifdef HAVE_CUDA
    printf("CUDA: %d\n", CUDART_VERSION);
#endif
    exit(0);
}

static void show_usage_and_exit(int status)
{
    if (status)
        fprintf(stderr, "Try 'VerthashMiner --help' for more information.\n");
    else
        printf(usage);
    exit(status);
}

static void strhide(char *s)
{
    if (*s) *s++ = 'x';
    while (*s) *s++ = '\0';
}


struct cmd_device_select_t
{
    uint32_t deviceIndex;
    uint32_t workSize;
    uint32_t batchTimeMs;
    uint32_t occupancyPct;
    int gpuTemperatureLimit;
    int deviceMonitor;
};

struct cmd_result_t
{
    // print device list in raw mode
    bool printDeviceList;

    // automatic config generation
    bool generateConfigFile;
    char* generateConfigFileName;

    // select config file
    bool useConfigFile;
    char* useConfigFileName;

    // restrict Nvidia GPUs to CUDA backend
    bool restrictNVGPUToCUDA;

    // verthash
    char* verthashDataFileName;
    char* verthashDataFileNameToGenerate;
    bool disableVerthashDataFileVerification;
    
    // overwrite configuration options
    // cmd line has a priority over configuration file
    bool overwrite_rpcUser;
    bool overwrite_rpcPass;
    bool overwrite_rpcUrlPort;
    bool overwrite_protocolDebug;
    bool overwrite_reconnectRetries;
    bool overwrite_retryPause;
    bool overwrite_scantime;
    bool overwrite_longpollTimeout;
    bool overwrite_proxy;
    bool overwrite_cert;
    bool overwrite_benchmark;
    bool overwrite_wantLongpoll;
    bool overwrite_redirect;
    bool overwrite_coinbaseAddr;
    bool overwrite_coinbaseSig;
    bool overwrite_debug;

    // device cmd-line configuration overwrite
    bool selectAllCLDevices;
    std::vector<cmd_device_select_t> selectedCLDevices;
    bool selectAllCUDevices;
    std::vector<cmd_device_select_t> selectedCUDevices;
};

//-----------------------------------------------------------------------------
inline void cmd_result_init(cmd_result_t* cmdr)
{
    assert(cmdr != NULL);

    cmdr->printDeviceList = false;

    cmdr->generateConfigFile = false;
    cmdr->generateConfigFileName = NULL;

    cmdr->useConfigFile = false;
    cmdr->useConfigFileName = NULL;

#ifdef HAVE_CUDA
    cmdr->restrictNVGPUToCUDA = true;
#else
    cmdr->restrictNVGPUToCUDA = false;
#endif

    cmdr->verthashDataFileName = NULL;
    cmdr->verthashDataFileNameToGenerate = NULL;
    cmdr->disableVerthashDataFileVerification = false;

    cmdr->overwrite_rpcUser = false;
    cmdr->overwrite_rpcPass = false;
    cmdr->overwrite_rpcUrlPort = false;
    cmdr->overwrite_protocolDebug = false;
    cmdr->overwrite_reconnectRetries = false;
    cmdr->overwrite_retryPause = false;
    cmdr->overwrite_scantime = false;
    cmdr->overwrite_longpollTimeout = false;
    cmdr->overwrite_proxy = false;
    cmdr->overwrite_cert = false;
    cmdr->overwrite_benchmark = false;
    cmdr->overwrite_wantLongpoll = false;
    cmdr->overwrite_redirect = false;
    cmdr->overwrite_coinbaseAddr = false;
    cmdr->overwrite_coinbaseSig = false;
    cmdr->overwrite_debug = false;

    cmdr->selectAllCLDevices = false;
    cmdr->selectedCLDevices.clear();
    cmdr->selectAllCUDevices = false;
    cmdr->selectedCUDevices.clear();
}

//-----------------------------------------------------------------------------
inline void cmd_result_free(cmd_result_t* cmdr)
{
    assert(cmdr != NULL);

    free(cmdr->generateConfigFileName);
    free(cmdr->useConfigFileName);
    free(cmdr->verthashDataFileName);
    free(cmdr->verthashDataFileNameToGenerate);
    cmdr->selectedCLDevices.clear();
    cmdr->selectedCUDevices.clear();
}

//-----------------------------------------------------------------------------
inline void cmd_result_update(cmd_result_t* cmdr, int argc, char *argv[])
{
    assert(cmdr != NULL);

    int key;
    // Can be used to detect option priorities.
    int numArgs = 0;

    while (1)
    {
        key = getopt_long(argc, argv, short_options, options, NULL);

        if (key < 0)
        {
            break;
        }

        //-------------------------------------
        // parse arguments
        char* const arg = optarg;
        char* const pname = argv[0];

        char *p;
        int v;

        // device selection
        const char* delims1 = ",";
        const char* delims2 = ":";
        char* tokenBase;
        char* token;

        switch(key)
        {
        case 'l':
            cmdr->printDeviceList = true;
            break;
        case 'g':
            cmdr->generateConfigFile = true;
            free(cmdr->generateConfigFileName); 
            cmdr->generateConfigFileName = strdup(arg);
            break;
        case 'a':
            // TODO:
            //fprintf(stderr, "Algorithm selection through cmd params are not supported\n");
            break;
        case 'c':
            cmdr->useConfigFile = true;
            free(cmdr->useConfigFileName);
            cmdr->useConfigFileName = strdup(arg);
            break;
        case 'p':
            cmdr->overwrite_rpcPass = true;
            free(rpc_pass);
            rpc_pass = strdup(arg);
            strhide(arg);
            break;
        case 'P':
            cmdr->overwrite_protocolDebug = true; 
            opt_protocol = true;
            break;
        case 'r':
            cmdr->overwrite_reconnectRetries = true; 
            v = atoi(arg);
            if (v < -1 || v > 9999) // sanity check
                show_usage_and_exit(1);
            opt_retries = v;
            break;
        case 'R':
            cmdr->overwrite_retryPause = true; 
            v = atoi(arg);
            if (v < 1 || v > 9999)  // sanity check
                show_usage_and_exit(1);
            opt_fail_pause = v;
            break;
        case 's':
            cmdr->overwrite_scantime = true; 
            v = atoi(arg);
            if (v < 1 || v > 9999)  // sanity check
                show_usage_and_exit(1);
            opt_scantime = v;
            break;
        case 'T':
            cmdr->overwrite_longpollTimeout = true;
            v = atoi(arg);
            if (v < 1 || v > 99999) // sanity check
                show_usage_and_exit(1);
            opt_longpoll_timeout = v;
            break;
        case 'd': // --cl-devices
            p = arg;
            while (0 != (tokenBase = strsep(&p, delims1)))
            {
               // printf("\t Select device: %s\n", tokenBase);
                size_t paramIndex = 0;
                cmd_device_select_t sel; 
                sel.workSize = vh::defaultWorkSize;
                sel.batchTimeMs = vh::defaultBatchTimeMs;
                sel.deviceMonitor = vh::defaultDeviceMonitor;
                sel.gpuTemperatureLimit = vh::defaultGPUTemperatureLimit;
                sel.occupancyPct = vh::defaultOccupancyPct;
                while (0 != (token = strsep(&tokenBase, delims2)) && (paramIndex < 5))
                {
                    if (paramIndex == 0)
                    {
                        sel.deviceIndex = std::stoul(token, nullptr, 0); 
                    }
                    else
                    {
                        char t = token[0];
                        switch (t)
                        {
                        case 'w':
                            {
                                size_t len = strlen(token);
                                if (len > 1) { sel.workSize = std::stoul(token + 1, nullptr, 0); }
                            } break;
                        case 'b':
                            {
                                size_t len = strlen(token);
                                if (len > 1) { sel.batchTimeMs = std::stoul(token + 1, nullptr, 0); }
                            } break;
                        case 'm':
                            {
                                size_t len = strlen(token);
                                if (len > 1) { sel.deviceMonitor = std::stoul(token + 1, nullptr, 0); }
                            } break;
                        case 't':
                            {
                                size_t len = strlen(token);
                                if (len > 1) { sel.gpuTemperatureLimit = std::stoul(token + 1, nullptr, 0); }
                            } break;
                        case 'o':
                            {
                                size_t len = strlen(token);
                                if (len > 1) { sel.occupancyPct = std::stoul(token + 1, nullptr, 0); }
                            } break;
                        default:
                            {
                                fprintf(stderr, "Unknown prefix '%c'. Device settings must be specified prefix:\n", t);
                                fprintf(stderr, "Example: 0:w131072:m1 Select device 0, set workSize to 131072 and enable monitoring\n");
                            } break;
                        }
                    }

                    ++paramIndex;
                }
                cmdr->selectedCLDevices.push_back(sel); 
            }
            break;
        case 'D': // --cu-devices
            p = arg;
            while (0 != (tokenBase = strsep(&p, delims1)))
            {
                size_t paramIndex = 0;
                cmd_device_select_t sel; 
                sel.workSize = vh::defaultWorkSize;
                sel.batchTimeMs = vh::defaultBatchTimeMs;
                sel.deviceMonitor = vh::defaultDeviceMonitor;
                sel.gpuTemperatureLimit = vh::defaultGPUTemperatureLimit;
                sel.occupancyPct = vh::defaultOccupancyPct;
                while (0 != (token = strsep(&tokenBase, delims2)) && (paramIndex < 5))
                {
                    if (paramIndex == 0)
                    {
                        sel.deviceIndex = std::stoul(token, nullptr, 0); 
                    }
                    else
                    {
                        char t = token[0];
                        switch (t)
                        {
                        case 'w':
                            {
                                size_t len = strlen(token);
                                if (len > 1) { sel.workSize = std::stoul(token + 1, nullptr, 0); }
                            } break;
                        case 'b':
                            {
                                size_t len = strlen(token);
                                if (len > 1) { sel.batchTimeMs = std::stoul(token + 1, nullptr, 0); }
                            } break;
                        case 'm':
                            {
                                size_t len = strlen(token);
                                if (len > 1) { sel.deviceMonitor = std::stoul(token + 1, nullptr, 0); }
                            } break;
                        case 't':
                            {
                                size_t len = strlen(token);
                                if (len > 1) { sel.gpuTemperatureLimit = std::stoul(token + 1, nullptr, 0); }
                            } break;
                        case 'o':
                            {
                                size_t len = strlen(token);
                                if (len > 1) { sel.occupancyPct = std::stoul(token + 1, nullptr, 0); }
                            } break;
                        default:
                            {
                                fprintf(stderr, "Unknown prefix '%c'. Device settings must be specified prefix:\n", t);
                                fprintf(stderr, "Example: 0:w131072:m1 Select device 0, set workSize to 131072 and enable monitoring\n");
                            } break;
                        }
                    }

                    ++paramIndex;
                }
                cmdr->selectedCUDevices.push_back(sel); 
            }
            break;
        case 'u':
            cmdr->overwrite_rpcUser = true;
            free(rpc_user);
            rpc_user = strdup(arg);
            break;
        case 'o': {         // --url // TODO: refactor ap/hp
            cmdr->overwrite_rpcUrlPort = true;
            char *ap, *hp;
            ap = strstr(arg, "://");
            ap = ap ? ap + 3 : arg;
            hp = ap;
            if (ap != arg) {
                if (strncasecmp(arg, "http://", 7) &&
                    strncasecmp(arg, "https://", 8) &&
                    strncasecmp(arg, "stratum+tcp://", 14) &&
                    strncasecmp(arg, "stratum+tcps://", 15)) {
                    fprintf(stderr, "Error: %s: unknown protocol -- '%s'\n",
                           pname, arg);
                    cmdr->overwrite_rpcUrlPort = false;
                }
                free(rpc_url);
                rpc_url = strdup(arg);
                strcpy(rpc_url + (ap - arg), hp);
            } else {
                if (*hp == '\0' || *hp == '/') {
                    applog(LOG_ERR, "%s: invalid URL -- '%s'\n",
                            pname, arg);
                    cmdr->overwrite_rpcUrlPort = false;
                }
                free(rpc_url);
                rpc_url = (char*)malloc(strlen(hp) + 8);
                sprintf(rpc_url, "http://%s", hp);
            }
            have_stratum = !opt_benchmark && !strncasecmp(rpc_url, "stratum", 7);
            break;
        }
        case 'x': // --proxy
            cmdr->overwrite_proxy = true;
            if (!strncasecmp(arg, "socks4://", 9))
                opt_proxy_type = CURLPROXY_SOCKS4;
            else if (!strncasecmp(arg, "socks5://", 9))
                opt_proxy_type = CURLPROXY_SOCKS5;
    #if LIBCURL_VERSION_NUM >= 0x071200
            else if (!strncasecmp(arg, "socks4a://", 10))
                opt_proxy_type = CURLPROXY_SOCKS4A;
            else if (!strncasecmp(arg, "socks5h://", 10))
                opt_proxy_type = CURLPROXY_SOCKS5_HOSTNAME;
    #endif
            else
                opt_proxy_type = CURLPROXY_HTTP;
            free(opt_proxy);
            opt_proxy = strdup(arg);
            break;
        case 1001:
            cmdr->overwrite_cert = true;
            free(opt_cert);
            opt_cert = strdup(arg);
            break;
        case 1005:
            // TODO:
            opt_benchmark = true;
            want_longpoll = false;
            have_stratum = false;
            break;
        case 1003:
            cmdr->overwrite_wantLongpoll = true;
            want_longpoll = false;
            break;
        case 1009:
            cmdr->overwrite_redirect = true;
            opt_redirect = false;
            break;
        case 1013:          // --coinbase-addr
            cmdr->overwrite_coinbaseAddr = true; 
            pk_script_size = address_to_script(pk_script, sizeof(pk_script), arg);
            if (!pk_script_size)
            {
                fprintf(stderr, "Error: %s: invalid address -- '%s'\n", pname, arg);
                cmdr->overwrite_coinbaseAddr = false;
            }
            break;
        case 1015:          // --coinbase-sig
            cmdr->overwrite_coinbaseSig = true; 
            if (strlen(arg) + 1 > sizeof(coinbase_sig))
            {
                fprintf(stderr, "Error: %s: coinbase signature is too long\n", pname);
                cmdr->overwrite_coinbaseSig = false;
            }
            strcpy(coinbase_sig, arg);
            break;
        case 1016:          // --no-restrict-cuda
            cmdr->restrictNVGPUToCUDA = false;
            break;
        case 'f':          // --verthash-data
            free(cmdr->verthashDataFileName);
            cmdr->verthashDataFileName = strdup(arg);
            break;
        case 1018:          // --verbose
            cmdr->overwrite_debug = true;
            opt_debug = true;
            break;
        case 1019:          // --all-cl-devices
            cmdr->selectAllCLDevices = true; 
            break;
        case 1020:          // --all-cu-devices
            cmdr->selectAllCUDevices = true;
            break;
        case 1021:          // --no-verthash-data-verification
            cmdr->disableVerthashDataFileVerification = true;
            break;
        case 1022:          // --log-file
            opt_log_file = true;
            break;
        case 1023:          // --gen-verthash-data
            free(cmdr->verthashDataFileNameToGenerate);
            cmdr->verthashDataFileNameToGenerate = strdup(arg);
            break;
        case 'v':
            show_version_and_exit();
        case 'h':
            show_usage_and_exit(0);
        default:
            show_usage_and_exit(1);
        }

        ++numArgs;
    }

    if (optind < argc)
    {
        fprintf(stderr, "%s: unsupported non-option argument -- '%s'\n",
            argv[0], argv[optind]);
        show_usage_and_exit(1);
    }

    if (numArgs == 0)
    {
        show_usage_and_exit(0);
    }
}


//-----------------------------------------------------------------------------
// signal handler callbacks
#ifndef _WIN32
static void on_signal(int signum)
{
    switch (signum)
    {
    case SIGHUP:
    case SIGINT:
    case SIGTERM:
        {
            signal(signum, SIG_IGN);
            applog(LOG_INFO, "signal(id:%d) received, exiting...", signum);
            if (!abort_flag) { abort_flag = true; }
            if (thr_info)
            {
                if (thr_info[work_thr_id].q)
                {
                    tq_push(thr_info[work_thr_id].q, NULL);
                    //tq_freeze(thr_info[work_thr_id].q);
                }
            }
            if (have_stratum)
            {
                send(stratum.dummy_socket, NULL, 0, 0);
            }
        } break;
    }
}
#else // WIN32
BOOL WINAPI on_consoleEvent(DWORD dwType)
{
    switch (dwType)
    {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        applog(LOG_INFO, "event(%d) received, exiting...", (int)dwType);
        if (!abort_flag) { abort_flag = true; }
        // trigger workio thread to exit if it was stuck at waiting
        if (thr_info)
        {
            if (thr_info[work_thr_id].q)
            {
                tq_push(thr_info[work_thr_id].q, NULL);
            }
        }

        if (have_stratum)
        {
            send(stratum.dummy_socket, NULL, 0, 0);
        }
        break;
    default:
        return false;
    }
    return true;
}
#endif

int utf8_main(int argc, char *argv[])
{
    // Init global lock data
    mtx_init(&applog_lock, mtx_plain);
    mtx_init(&stats_lock, mtx_plain);
    mtx_init(&g_work_lock, mtx_plain);
    mtx_init(&stratum.sock_lock, mtx_plain);
    mtx_init(&stratum.work_lock, mtx_plain);

    // Enables signal handlers to exit application properly
#ifndef _WIN32
    signal(SIGHUP, on_signal);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
#else // WIN32
    SetConsoleCtrlHandler((PHANDLER_ROUTINE)on_consoleEvent, TRUE);
#endif

    //-------------------------------------
    // command line parameters have higher priority over configuration file
    cmd_result_t cmdr;
    cmd_result_init(&cmdr);
    cmd_result_update(&cmdr, argc, argv);

    //-------------------------------------
    // NOTE: File logger is getting configured before everything else,
    // so as much logging data as possible can be saved in file
    if (opt_log_file)
    {
        // 4+1+2+1+2+1+2+1+2+1+2+4+1
        // YYYY-MM-DD HH-MM-SS.txt\0
        char logFileName[24] = { 0 };
        struct tm* sTm;

        time_t now = time(0);
        sTm = gmtime(&now);

        strftime(logFileName, sizeof(logFileName), "%Y-%m-%d %H-%M-%S.txt", sTm);
        applog_file = fopen_utf8(logFileName, "w");
        if (applog_file == NULL)
        {
            applog(LOG_WARNING, "Failed to open file for logging(%s).", logFileName);
            applog(LOG_WARNING, "Logging to file is not available.");
        }
    }

    //-------------------------------------
    // Verthash data file generation (if it was requested)
    if (cmdr.verthashDataFileNameToGenerate != NULL)
    {
        applog(LOG_INFO, "Generating a verthash data file: %s", cmdr.verthashDataFileNameToGenerate);
        applog(LOG_INFO, "This may take a while...");
        int res = verthash_generate_data_file(cmdr.verthashDataFileNameToGenerate);
        if (res != 0)
        {
            applog(LOG_ERR, "Failed to generate a verthash data file!");
        }

        applog(LOG_INFO, "Verthash data file has been generated!", cmdr.verthashDataFileNameToGenerate);

        cmd_result_free(&cmdr);
        return 0;
    }

    //-------------------------------------
    // OpenCL init
    // get raw device list options, which can be modified later depending on supported extensions
    cl_int errorCode = CL_SUCCESS;
    //-------------------------------------
    // get platform IDs
    cl_uint numCLPlatformIDs = 0;
    std::vector<cl_platform_id> clplatformIds;
    errorCode = clGetPlatformIDs(0, nullptr, &numCLPlatformIDs);
    if (errorCode != CL_SUCCESS || numCLPlatformIDs <= 0)
    {
#ifdef HAVE_CUDA
        applog(LOG_WARNING, "Failed to find any OpenCL platforms.");
        // Look for CUDA platforms...
#else
        applog(LOG_WARNING, "Failed to find any OpenCL platforms. Exiting...");
        // exit application only if compiled without CUDA support
        cmd_result_free(&cmdr);
        return 1;
#endif
    }
    else
    {
        clplatformIds.resize(numCLPlatformIDs);
        clGetPlatformIDs(numCLPlatformIDs, clplatformIds.data(), nullptr);
    }
    //-------------------------------------
    // get logical device list
    // AMDCL2(Windows) and ROCm.(Mesa and others are listed as V_Other)
    const std::string platformVendorAMD("Advanced Micro Devices");
    const std::string platformVendorNV("NVIDIA Corporation");
    std::vector<vh::cldevice_t> cldevices;
    for (size_t i = 0; i < (size_t)numCLPlatformIDs; ++i)
    {
        // check if platform is supported
        size_t infoSize = 0;
        clGetPlatformInfo(clplatformIds[i], CL_PLATFORM_VENDOR, 0, nullptr, &infoSize);
        std::string infoString(infoSize, ' ');
        clGetPlatformInfo(clplatformIds[i], CL_PLATFORM_VENDOR, infoSize, (void*)infoString.data(), nullptr);

        vh::EVendor vendor = vh::V_OTHER;
        if (infoString.find(platformVendorAMD) != std::string::npos)
        {
            vendor = vh::V_AMD;
        }
        else if (infoString.find(platformVendorNV) != std::string::npos) 
        {
            vendor = vh::V_NVIDIA;
        }

        // check if platform version
        infoSize = 0;
        clGetPlatformInfo(clplatformIds[i], CL_PLATFORM_VERSION, 0, nullptr, &infoSize);
        std::string infoVersion(infoSize, ' ');
        clGetPlatformInfo(clplatformIds[i], CL_PLATFORM_VERSION, infoSize, (void*)infoVersion.data(), nullptr);

        if ( (infoVersion.find("OpenCL 1.0") != std::string::npos) ||
             (infoVersion.find("OpenCL 1.1") != std::string::npos) )
        {
            applog(LOG_WARNING, "Platform doesn't meet version requirements (index: %u, %s)", i, infoString.c_str());
            applog(LOG_WARNING, "OpenCL 1.2 or higher is required.");
            continue;
        }

        // get devices available on this platform
        cl_uint numDeviceIDs = 0;
        errorCode = clGetDeviceIDs(clplatformIds[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDeviceIDs);
        if (errorCode != CL_SUCCESS || numCLPlatformIDs <= 0)
        {
            applog(LOG_WARNING, "No GPU devices available on platform:");
            applog(LOG_WARNING, "index: %u, %s", i, infoString.c_str()); 
            continue;
        }

        std::vector<cl_device_id> deviceIds(numDeviceIDs);
        clGetDeviceIDs(clplatformIds[i], CL_DEVICE_TYPE_GPU, numDeviceIDs, deviceIds.data(), nullptr);

        clu_device_topology_amd topology;
        for (size_t j = 0; j < deviceIds.size(); ++j)
        {
            int32_t pcieBusId = -1;
            int32_t pcieDeviceId = -1;
            int32_t pcieFunctionId = -1;
        
            if (vendor == vh::V_AMD)
            {
                cl_int status = clGetDeviceInfo(deviceIds[j], CL_DEVICE_TOPOLOGY_AMD,
                                                sizeof(clu_device_topology_amd), &topology, nullptr);
                if(status == CL_SUCCESS)
                {
                    if (topology.raw.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD)
                    {
                        pcieBusId = (int32_t)topology.pcie.bus;
                        pcieDeviceId = (int32_t)topology.pcie.device;
                        pcieFunctionId = (int32_t)topology.pcie.function;
                    }
                }
                else // if extension is not supported
                {
                    applog(LOG_WARNING, "Failed to get CL_DEVICE_TOPOLOGY_AMD info"
                                        "(possibly unsupported extension). Platform index: %u", i);
                }
            }
            else if (vendor == vh::V_NVIDIA)
            {
#ifdef HAVE_CUDA
                // Handle Nvidia GPU to CUDA restrictions
                if (cmdr.restrictNVGPUToCUDA)
                {
                    // If device compute capability is not supported by CUDA backend,
                    // transfer to OpenCL
                    cl_int nvCCMajor = -1;
                    cl_int status0 = clGetDeviceInfo(deviceIds[j], CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_int), &nvCCMajor, NULL);
                    cl_int nvCCMinor = -1;
                    cl_int status1 = clGetDeviceInfo(deviceIds[j], CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_int), &nvCCMinor, NULL);
                    if((status0 == CL_SUCCESS) && (status1 == CL_SUCCESS))
                    {
                        // CUDA 11 or higher
                        if (CUDART_VERSION >= 11000)
                        {
                            // CUDA 11 removed SM 3.0 support.
                            // Transfer SM 3.0 to the OpenCL backend
                            if (!((nvCCMajor == 3) && (nvCCMinor == 0)))
                            {
                                continue;
                            }
                        }
                        else // other versions
                        {
                            continue;
                        }
                    }
                }
#endif
                cl_int nvpciBus = -1;
                cl_int status0 = clGetDeviceInfo(deviceIds[j], CL_DEVICE_PCI_BUS_ID_NV, sizeof(cl_int), &nvpciBus, NULL);
                cl_int nvpciSlot = -1;
                cl_int status1 = clGetDeviceInfo(deviceIds[j], CL_DEVICE_PCI_SLOT_ID_NV, sizeof(cl_int), &nvpciSlot, NULL);
                if((status0 == CL_SUCCESS) && (status1 == CL_SUCCESS))
                {
                    pcieBusId = (int32_t)nvpciBus;
                    pcieDeviceId = (int32_t)nvpciSlot;
                    pcieFunctionId = 0;
                }
                else
                {
                    applog(LOG_WARNING, "Failed to get NV_PCIE info"
                                        "(possibly unsupported extension). Platform index: %u", i);
                }
            }
            else // V_OTHER
            {
                // NO PCIe info extensions available
            }

            vh::cldevice_t cldevice;
            cldevice.clPlatformId = clplatformIds[i];
            cldevice.clId = deviceIds[j];
            cldevice.platformIndex = (int32_t)i;
            cldevice.binaryFormat = vh::BF_None;
            cldevice.asmProgram = vh::AP_None;
            cldevice.vendor = vendor;
            cldevice.pcieBusId = pcieBusId;
            cldevice.pcieDeviceId = pcieDeviceId;
            cldevice.pcieFunctionId = pcieFunctionId;

            cldevices.push_back(cldevice);
        }
    }

    if(numCLPlatformIDs > 0)
    {
        applog(LOG_INFO, "Found %" PRIu64 " OpenCL devices.", (uint64_t)cldevices.size());
    }

#ifdef HAVE_CUDA
    //-------------------------------------
    // CUDA init
    int cudeviceListSize = 0;
    cudaError_t cuerr = cudaGetDeviceCount(&cudeviceListSize);

    std::vector<vh::cudevice_t> cudevices;
    if (cuerr == cudaSuccess)
    {
        for (int i = 0; i < cudeviceListSize; ++i)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);

            if (CUDART_VERSION > 8000)
            {
                if ((prop.major == 2) && ((prop.minor == 1) || (prop.minor == 0)))
                {
                    // SM 2.0 and 2.1 have been removed in CUDA 9
                    applog(LOG_WARNING, "Found an unsupported SM %d.%d device: %s. Skipping...",
                        prop.major, prop.minor, prop.name);
                    applog(LOG_WARNING, "To use this device on CUDA backend, software must be compiled with CUDA 8.0 or lower!");

                    continue;
                }
                else if (CUDART_VERSION >= 11000)
                {
                    // SM 3.0 has been removed in CUDA 11
                    if ((prop.major == 3) && (prop.minor == 0))
                    {
                        applog(LOG_WARNING, "Unsupported SM %d.%d device %s has been transfered to the OpenCL backend",
                            prop.major, prop.minor, prop.name);
                        applog(LOG_WARNING, "To use this device on CUDA backend, software must be compiled with CUDA 10.2 or lower!");

                        continue;
                    }                    
                }
            }


            // SM 1.x is not supported
            if ((prop.major == 1) && ((prop.minor == 3) || (prop.minor == 2) || (prop.minor == 1) || (prop.minor == 0)))
            {
                applog(LOG_WARNING, "Found an unsupported SM %d.%d device: %s. Skipping...",
                       prop.major, prop.minor, prop.name);

                continue;
            }


            vh::cudevice_t cudevice;
            cudevice.cudeviceHandle = i;
            cudevice.pcieBusId = prop.pciBusID;
            cudevice.pcieDeviceId = prop.pciDeviceID;

            cudevices.push_back(cudevice);
        }

        applog(LOG_INFO, "Found %" PRIu64 " CUDA devices", (uint64_t)cudevices.size());
    }
    else
    {
        applog(LOG_WARNING, "Failed to initialize CUDA. error code: %d", cuerr);
        if(numCLPlatformIDs > 0)
        {
            applog(LOG_WARNING, "CUDA devices will be unavailable.");
        }
        else
        {
            applog(LOG_WARNING, "Both OpenCL and CUDA modules are not available! Exiting...");
            cmd_result_free(&cmdr);
            return 1;            
        }
    }
#endif

    if (cmdr.printDeviceList)
    {
        puts("\nDevice list:");
        puts("==================");

        // print OpenCL devices
        if (cldevices.size() != 0)
        {
            puts("OpenCL devices:");
            size_t infoSize;
            std::string infoString0;
            std::string infoString1;
            for (size_t i = 0; i < cldevices.size(); ++i)
            {
                // get device name
                infoString0.clear();
                clGetDeviceInfo(cldevices[i].clId, CL_DEVICE_NAME, 0, NULL, &infoSize);
                infoString0.resize(infoSize);
                clGetDeviceInfo(cldevices[i].clId, CL_DEVICE_NAME, infoSize, (void *)infoString0.data(), NULL);
                infoString0.pop_back();

                // get platform name
                infoString1.clear();
                clGetPlatformInfo(cldevices[i].clPlatformId, CL_PLATFORM_VENDOR, 0, NULL, &infoSize);
                infoString1.resize(infoSize);
                clGetPlatformInfo(cldevices[i].clPlatformId, CL_PLATFORM_VENDOR, infoSize, (void *)infoString1.data(), NULL);
                infoString1.pop_back();


                char pcieStr[16] = { };
                if (cldevices[i].pcieBusId == -1)
                {
                    snprintf (pcieStr, 15, "not avilable");
                }
                else
                {
                    snprintf (pcieStr, 15, "%02x:%02x:%01x",
                              cldevices[i].pcieBusId, cldevices[i].pcieDeviceId, cldevices[i].pcieFunctionId);
                }


                printf("\tIndex: %u. Name: %s\n\t"
                    "          Platform index: %u\n\t"
                    "          Platform name: %s\n\t"
                    "          pcieId: %s\n\n",
                    (uint32_t)i, infoString0.c_str(), cldevices[i].platformIndex, infoString1.c_str(), pcieStr);
            }
        }
        else
        {
            puts("OpenCL devices: None\n");
        }
#ifdef HAVE_CUDA
        // print CUDA devices
        
        if (cudevices.size() != 0)
        {
            puts("CUDA devices:");
            for (size_t i = 0; i < cudevices.size(); ++i)
            {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, cudevices[i].cudeviceHandle);

                char pcieStr[16] = { };
                if (cudevices[i].pcieBusId == -1)
                {
                    snprintf (pcieStr, 15, "not avilable");
                }
                else
                {
                    snprintf (pcieStr, 15, "%02x:%02x:0",
                              cudevices[i].pcieBusId, cudevices[i].pcieDeviceId);
                }

                printf("\tIndex: %u. Name: %s. pcieId: %s\n", (uint32_t)i, prop.name, pcieStr);
            }
        }
        else
        {
            puts("CUDA devices: None\n");
        }
#endif

        fflush(stdout);

        cmd_result_free(&cmdr);

        return 0;
    }

    //-----------------------------------------------------------------------------
    // Config generation
    if (cmdr.generateConfigFile)
    {
        // Miner configration
        std::string configText("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
                               "# Global settings:\n"
                               "#\n"
                               "#    VerthashDataFile\n"
                               "#        Set mining data file for Verthash algorithm.\n"
                               "#        It can be (usually) found inside \"Data\" folder when using a core wallet.\n"
                               "#        If verthash.dat is specified directly from the data folder and\n"
                               "#        core wallet is running along with the miner, then mining data will be\n"
                               "#        automatically updated if it becomes outdated\n"
                               "#        Default: verthash.dat\n"
                               "#\n"
                               "#    VerthashDataFileVerification\n"
                               "#        Enable verthash data file verification.\n"
                               "#        Default: true\n"
                               "#\n"
                               "#    Debug\n"
                               "#        Enable extra debugging output.\n"
                               "#        Default: false\n"
                               "#\n"
                               "#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
                               "\n"
                               "<Global VerthashDataFile = \"verthash.dat\"\n"
                               "        VerthashDataFileVerification = \"true\"\n"
                               "        Debug = \"false\">\n"
                               "\n"
                               "#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
                               "# Connection setup:\n"
                               "#\n"
                               "#    Algorithm\n"
                               "#        Select mining algorithm.\n"
                               "#        Default: Verthash\n"
                               "#\n"
                               "#    Url\n"
                               "#        URL of mining server.\n"
                               "#        Example stratum: \"stratum+tcp://example.com:port\"\n"
                               "#        Example GBT: \"http://127.0.0.1:port\"\n"
                               "#\n"
                               "#    Username\n"
                               "#        Username for mining server.\n"
                               "#\n"
                               "#    Password\n"
                               "#        Password for mining server. If server doesn't require it. Set: \"x\"\n"
                               "#\n"
                               "#    CoinbaseAddress\n"
                               "#        Payout address for GBT solo mining\n"
                               "#\n"
                               "#    CoinbaseSignature\n"
                               "#        Data to insert in the coinbase when possible.\n"
                               "#\n"
                               "#    SSLCertificateFileName\n"
                               "#        Certificate to connect to mining server with SSL.\n"
                               "#\n"
                               "#    Proxy\n"
                               "#        Connect through a proxy.\n"
                               "#        Example: [PROTOCOL://]HOST[:PORT]\n"
                               "#\n"
                               "#    Redirect\n"
                               "#        Allow(true) or Ignore(false) requests to change the URL of the mining server.\n"
                               "#        Default: true\n"
                               "#\n"
                               "#    LongPoll\n"
                               "#        Enable/Disable long polling.\n"
                               "#        Default: true\n"
                               "#\n"
                               "#    LongPollTimeout\n"
                               "#        Timeout for long polling, in seconds.\n"
                               "#        Default: 0\n"
                               "#\n"
                               "#    Scantime\n"
                               "#        Upper bound on time spent scanning current work when\n"
                               "#        long polling is unavailable, in seconds.\n"
                               "#        Default: 5\n"
                               "#\n"
                               "#    Retries\n"
                               "#        Number of times to retry if a network call fails\n"
                               "#        Default: -1 (retry indefinitely)\n"
                               "#\n"
                               "#    RetryPause\n"
                               "#        Time to pause between retries, in seconds.\n"
                               "#        Default: 30\n"
                               "#\n"
                               "#    ProtocolDump\n"
                               "#        Verbose dump of protocol-level activities.\n"
                               "#        Default: false\n"
                               "#\n"
                               "#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
                               "\n"
                               "<Connection Algorithm = \"Verthash\"\n"
                               "            Url = \"stratum+tcp://example.com:port\"\n"
                               "            Username = \"user\"\n"
                               "            Password = \"x\"\n"
                               "            CoinbaseAddress = \"\"\n"
                               "            CoinbaseSignature = \"\"\n"
                               "            SSLCertificateFileName = \"\"\n"
                               "            Proxy = \"\"\n"
                               "            Redirect = \"true\"\n"
                               "            LongPoll = \"true\"\n"
                               "            LongPollTimeout = \"0\"\n"
                               "            Scantime = \"5\"\n"
                               "            Retries = \"-1\"\n"
                               "            RetryPause = \"30\"\n"
                               "            ProtocolDump = \"false\">\n"
                               "\n");

        // OpenCL devices
        vh::generateCLDeviceConfig(clplatformIds, cldevices, configText); 
#ifdef HAVE_CUDA
        configText += "\n";
        // CUDA devices
        vh::generateCUDADeviceConfig(cudevices, configText);
#endif
        // save a config file.
        if (fileExists(cmdr.generateConfigFileName) == true)
        {
            applog(LOG_ERR, "Failed to create a configuration file. %s already exists.", cmdr.generateConfigFileName);
            cmd_result_free(&cmdr);
            return 1;
        }

        FILE* cfgOutput = fopen_utf8(cmdr.generateConfigFileName, "w");
        if (cfgOutput == NULL)
        {
            applog(LOG_ERR, "Failed to create a configuration file. %s already exists.", cmdr.generateConfigFileName);
            cmd_result_free(&cmdr);
            return 1;
        }
        fwrite(configText.c_str(), 1, configText.length(), cfgOutput);
        fclose(cfgOutput);


        applog(LOG_INFO, "Configuration file: (%s) has been generated.", cmdr.generateConfigFileName);

        cmd_result_free(&cmdr);
        
        return 0;
    }

    // device workers
    std::vector<clworker_t> clworkers;

    // configure OpenCL devices from cmd-line
    if (!cmdr.selectAllCLDevices) 
    {
        for (size_t i = 0; i < cmdr.selectedCLDevices.size(); ++i)
        {
            // validate parameters
            if (cmdr.selectedCLDevices[i].deviceIndex >= cldevices.size()) 
            {
                applog(LOG_ERR, "Invalid CL device index: %u", cmdr.selectedCLDevices[i].deviceIndex);
                cmd_result_free(&cmdr);
                return 1;
            }

            uint32_t workSize = cmdr.selectedCLDevices[i].workSize;
            if ((workSize % 256) != 0)
            {
                applog(LOG_WARNING, "Invalid CL Device \"WorkSize\"(w) parameter(index: %u, workSize: %u)",
                       cmdr.selectedCLDevices[i].deviceIndex, workSize);
                applog(LOG_WARNING, "It must be multiple of 256. Using default: %u", vh::defaultWorkSize);

                workSize = vh::defaultWorkSize;
            }

            uint32_t batchTimeMs = cmdr.selectedCLDevices[i].batchTimeMs;
            if (batchTimeMs == 0)
            {
                applog(LOG_WARNING, "Invalid CL Device \"BatchTime\"(b) parameter(index: %u, batchTime: %u)",
                       cmdr.selectedCLDevices[i].deviceIndex, batchTimeMs);
                applog(LOG_WARNING, "It must be above 0. Using default: %u", vh::defaultBatchTimeMs);
                batchTimeMs = vh::defaultBatchTimeMs;
            }

            uint32_t occupancyPct = cmdr.selectedCLDevices[i].occupancyPct;
            if ((occupancyPct == 0) || (occupancyPct > 100))
            {
                applog(LOG_WARNING, "Invalid CL Device \"OccupancyPct\"(o) parameter(index: %u, occupancyPct: %u)",
                       cmdr.selectedCLDevices[i].deviceIndex, occupancyPct);
                applog(LOG_WARNING, "It must be above 0 and less than or equal to 100. Using default: %u", vh::defaultOccupancyPct);

                occupancyPct = vh::defaultOccupancyPct;
            }


            clworker_t clworker;
            clworker.nvmlDevice = NULL;
            clworker.adlAdapterIndex = -1;
            clworker.cldevice = cldevices[cmdr.selectedCLDevices[i].deviceIndex];
            clworker.workSize = workSize;
            clworker.batchTimeMs = batchTimeMs;
            clworker.occupancyPct = occupancyPct;
            clworker.deviceMonitor = cmdr.selectedCLDevices[i].deviceMonitor;
            clworker.gpuTemperatureLimit = cmdr.selectedCLDevices[i].gpuTemperatureLimit;
            // AsmProgram will be detected on context init
            clworker.cldevice.binaryFormat = vh::BF_AUTO; 
            clworker.cldevice.asmProgram = vh::AP_None; 
            clworkers.push_back(clworker);
        }
    }
    else // add all CL devices
    {
        for (size_t i = 0; i < cldevices.size(); ++i)
        {
            clworker_t clworker;
            clworker.nvmlDevice = NULL;
            clworker.adlAdapterIndex = -1;
            clworker.cldevice = cldevices[i];
            clworker.workSize = vh::defaultWorkSize;
            clworker.batchTimeMs = vh::defaultBatchTimeMs;
            clworker.occupancyPct = vh::defaultOccupancyPct;
            clworker.deviceMonitor = vh::defaultDeviceMonitor;
            clworker.gpuTemperatureLimit = vh::defaultGPUTemperatureLimit;
            // AsmProgram will be detected on context init
            clworker.cldevice.binaryFormat = vh::BF_AUTO; 
            clworker.cldevice.asmProgram = vh::AP_None; 
            clworkers.push_back(clworker);
        }
    }

#ifdef HAVE_CUDA
    std::vector<cuworker_t> cuworkers;

    // configure CUDA devices from cmd-line
    if (!cmdr.selectAllCUDevices) 
    {
        for (size_t i = 0; i < cmdr.selectedCUDevices.size(); ++i)
        {
            // validate parameters
            if (cmdr.selectedCUDevices[i].deviceIndex >= cudeviceListSize) 
            {
                applog(LOG_ERR, "Invalid CUDA device index: %u", cmdr.selectedCUDevices[i].deviceIndex);
                cmd_result_free(&cmdr);
                return 1;
            }

            uint32_t workSize = cmdr.selectedCUDevices[i].workSize;
            if ((workSize % 256) != 0)
            {
                applog(LOG_WARNING, "Invalid CUDA Device \"WorkSize\"(w) parameter(index: %u, workSize: %u)",
                       cmdr.selectedCUDevices[i].deviceIndex, workSize);
                applog(LOG_WARNING, "It must be multiple of 256. Using default: %u", vh::defaultWorkSize);

                workSize = vh::defaultWorkSize;
            }

            uint32_t batchTimeMs = cmdr.selectedCUDevices[i].batchTimeMs;
            if (batchTimeMs == 0)
            {
                applog(LOG_WARNING, "Invalid CUDA Device \"BatchTime\"(b) parameter(index: %u, batchTime: %u)",
                       cmdr.selectedCUDevices[i].deviceIndex, batchTimeMs);
                applog(LOG_WARNING, "It must be above 0. Using default: %u", vh::defaultBatchTimeMs);
                batchTimeMs = vh::defaultBatchTimeMs;
            }

            uint32_t occupancyPct = cmdr.selectedCUDevices[i].occupancyPct;
            if ((occupancyPct == 0) || (occupancyPct > 100))
            {
                applog(LOG_WARNING, "Invalid CUDA Device \"OccupancyPct\"(o) parameter(index: %u, occupancyPct: %u)",
                       cmdr.selectedCUDevices[i].deviceIndex, occupancyPct);
                applog(LOG_WARNING, "It must be above 0 and less than or equal to 100. Using default: %u", vh::defaultOccupancyPct);

                occupancyPct = vh::defaultOccupancyPct;
            }


            cuworker_t cuworker;
            cuworker.nvmlDevice = NULL;
            cuworker.cudevice = cudevices[cmdr.selectedCUDevices[i].deviceIndex];
            cuworker.workSize = workSize;
            cuworker.batchTimeMs = batchTimeMs;
            cuworker.occupancyPct = occupancyPct;
            cuworker.deviceMonitor = cmdr.selectedCUDevices[i].deviceMonitor;
            cuworker.gpuTemperatureLimit = cmdr.selectedCUDevices[i].gpuTemperatureLimit;
            cuworkers.push_back(cuworker);
        }
    }
    else // add all CUDA devices
    {
        for (size_t i = 0; i < cudevices.size(); ++i)
        {
            cuworker_t cuworker;
            cuworker.nvmlDevice = NULL;
            cuworker.cudevice = cudevices[i];
            cuworker.workSize = vh::defaultWorkSize;
            cuworker.batchTimeMs = vh::defaultBatchTimeMs;
            cuworker.occupancyPct = vh::defaultOccupancyPct;
            cuworker.deviceMonitor = vh::defaultDeviceMonitor;
            cuworker.gpuTemperatureLimit = vh::defaultGPUTemperatureLimit;
            cuworkers.push_back(cuworker);
        }
    }
#endif

    if (cmdr.useConfigFile) 
    {
        //-----------------------------------------------------------------------------
        // Setup miner from the configuration file and CMD line(by overwriting config file options).
        vh::ConfigFile cf;
        if (cf.setSource(cmdr.useConfigFileName, true))
        {
            applog(LOG_INFO, "Using a configuration file (%s)", cmdr.useConfigFileName);
        }
        else
        {
            applog(LOG_ERR, "Failed to load a configuration file. (%s)", cmdr.useConfigFileName);
            cmd_result_free(&cmdr);
            return 1;
        }

        vh::ConfigSetting* csetting = nullptr;

        int cfWarnings = 0;
        int cfErrors = 0;

        //-----------------------------------------------------------------------------
        // Global config section

        //-------------------------------------
        // get Verthash data file parameter and initialize (if possible)
        int vhLoadResult = -1;
        if (!cmdr.verthashDataFileName) // if data file has been specified using cmd line
        {
            // TODO: check if algorithm == Verthash
            csetting = cf.getSetting("Global", "VerthashDataFile");
            if (csetting)
            {
                applog(LOG_INFO, "Loading verthash data file...");
                vhLoadResult = verthash_info_init(&verthashInfo, csetting->AsString.c_str());
            }
            else
            {
                applog(LOG_ERR, "Failed to get a \"VerthashDataFile\" option inside \"Global\" section.");
                ++cfErrors;
            }
        }
        else // use data from command line
        {
            applog(LOG_INFO, "Loading verthash data file...");
            vhLoadResult = verthash_info_init(&verthashInfo, cmdr.verthashDataFileName);
        }

        //-------------------------------------
        // Check Verthash initialization status and verify data file(if enabled)
        if (vhLoadResult == 0) // No Error
        {
            applog(LOG_INFO, "Verthash data file has been loaded succesfully!");

            // set verification flag
            bool verifyDataFile = true;
            if (!cmdr.disableVerthashDataFileVerification)
            {
                csetting = cf.getSetting("Global", "VerthashDataFileVerification");
                if (csetting)
                {
                    verifyDataFile = csetting->AsBool;
                }
                else
                {
                    applog(LOG_WARNING, "Failed to get a \"VerthashDataFileVerification\" option inside \"Global\" section. Setting to \"true\"");
                    ++cfWarnings;
                }
            }
            else
            {
                verifyDataFile = false;
            }

            // verify data file if enabled
            if (verifyDataFile)
            {
                applog(LOG_INFO, "Verifying verthash data file...");
                uint8_t vhDataFileHash[32] = { 0 };
                sha256s(vhDataFileHash, verthashInfo.data, verthashInfo.dataSize);
                if (memcmp(vhDataFileHash, verthashDatFileHash_bytes, sizeof(verthashDatFileHash_bytes)) == 0)
                {
                    applog(LOG_INFO, "Verthash data file has been verified succesfully!");
                }
                else
                {
                    applog(LOG_ERR, "Verthash data file verification has failed!");
                    ++cfErrors;
                }
            }
            else
            {
                applog(LOG_WARNING, "Verthash data file verification stage is disabled!");
                ++cfWarnings;
            }
        }
        else
        {
            if (vhLoadResult == 1)
                applog(LOG_ERR, "Verthash data file name is invalid");
            else if (vhLoadResult == 2)
                applog(LOG_ERR, "Failed to allocate memory for Verthash data");
            else // for debugging purposes
                applog(LOG_ERR, "Verthash data initialization unknown error code: %d", vhLoadResult);
            ++cfErrors;
        }


        //-------------------------------------
        // get Debug parameter
        if (!cmdr.overwrite_debug) 
        {
            csetting = cf.getSetting("Global", "Debug"); 
            if (csetting)
            {
                opt_debug = csetting->AsBool;
            }
            else
            {
                opt_debug = false;

                applog(LOG_WARNING, "Failed to get a \"Debug\" option inside \"Connection\" section. Setting to \"false\"");
                ++cfWarnings;
            }
        }


        //-----------------------------------------------------------------------------
        // Connection section.
        // get algorithm
        // TODO:...

        //-------------------------------------
        // get url
        if (!cmdr.overwrite_rpcUrlPort) 
        {
            csetting = cf.getSetting("Connection", "Url");
            if (csetting)
            {
                if (strncasecmp(csetting->AsString.c_str(), "http://", 7) &&
                    strncasecmp(csetting->AsString.c_str(), "https://", 8) &&
                    strncasecmp(csetting->AsString.c_str(), "stratum+tcp://", 14) &&
                    strncasecmp(csetting->AsString.c_str(), "stratum+tcps://", 15))
                {
                    applog(LOG_ERR, "Unknown protocol -- '%s'", csetting->AsString.c_str());
                    ++cfErrors;
                }
                free(rpc_url);
                rpc_url = strdup(csetting->AsString.c_str());

                have_stratum = !opt_benchmark && !strncasecmp(rpc_url, "stratum", 7);

                applog(LOG_INFO, "Using protocol: %s.", have_stratum?"Stratum":"getblocktemplate");
            }
            else
            {
                applog(LOG_ERR, "Failed to get an \"Url\" option inside \"Connection\" section.");
                ++cfErrors;
            }
        }

        //-------------------------------------
        // get username
        if (!cmdr.overwrite_rpcUser)
        {
            csetting = cf.getSetting("Connection", "Username");
            if (csetting)
            {
                if (csetting->AsString.length() > 0) 
                {
                    free(rpc_user);
                    rpc_user = strdup(csetting->AsString.c_str()); 
                }
                else
                {
                    applog(LOG_ERR, "\"Username\" option inside \"Connection\" section is empty.");
                    ++cfErrors;
                }
            }
            else
            {
                applog(LOG_ERR, "Failed to get a \"Username\" option inside \"Connection\" section.");
                ++cfErrors;
            }
        }

        //-------------------------------------
        // get password
        if (!cmdr.overwrite_rpcPass) 
        {
            csetting = cf.getSetting("Connection", "Password");
            if (csetting)
            {
                if (csetting->AsString.length() > 0) 
                {
                    free(rpc_pass);
                    rpc_pass = strdup(csetting->AsString.c_str());
                }
                else
                {
                    applog(LOG_ERR, "\"Password\" option inside \"Connection\" section is empty.");
                    ++cfErrors;
                }
            }
            else
            {
                applog(LOG_ERR, "Failed to get a \"Password\" option inside \"Connection\" section.");
                ++cfErrors;
            }
        }

        //-------------------------------------
        // get coinbase address. Required for GBT solo mining.
        if (!cmdr.overwrite_coinbaseAddr)
        {
            csetting = cf.getSetting("Connection", "CoinbaseAddress");
            if (csetting)
            {
                if (csetting->AsString.length() > 0) 
                {
                    pk_script_size = address_to_script(pk_script, sizeof(pk_script), csetting->AsString.c_str());
                    if (!pk_script_size)
                    {
                        applog(LOG_ERR, "Invalid payout address: '%s'", csetting->AsString.c_str()); 
                        ++cfErrors;
                    }
                }
                else if (!have_stratum)
                {
                    applog(LOG_ERR, "\"CoinbaseAddress\" option inside \"Connection\" section is empty.");
                    ++cfErrors;
                }
            }
            else
            {
                // have_statum have been set during "Url" field validation
                if (have_stratum)
                {
                    // CoinbaseAddress is not needed for stratum
                    applog(LOG_WARNING, "Failed to get a \"CoinbaseAddress\" option inside \"Connection\" section. Setting to \"\"");
                    ++cfWarnings;
                }
                else // CoinbaseAddress is mandatory for GBT solo mining
                {
                    applog(LOG_ERR, "Failed to get a \"CoinbaseAddress\" option inside \"Connection\" section.");
                    ++cfErrors;
                }
            }
        }

        //-------------------------------------
        // get Coinbase signature
        if (!cmdr.overwrite_coinbaseSig)
        {
            csetting = cf.getSetting("Connection", "CoinbaseSignature");
            if (csetting)
            {
                if (csetting->AsString.length() + 1 > sizeof(coinbase_sig)) 
                {
                    applog(LOG_ERR, "Coinbase signature is too long");
                    ++cfErrors;
                }
                else if (csetting->AsString.length() > 0)
                {
                    strncpy(coinbase_sig, csetting->AsString.c_str(), csetting->AsString.length()); 
                }
            }
            else
            {
                applog(LOG_WARNING, "Failed to get a \"CoinbaseSignature\" option inside \"Connection\" section. Setting to \"\"");
                ++cfWarnings;
            }
        }

        //-------------------------------------
        // get SSL certificate file name
        if (!cmdr.overwrite_cert) 
        {
            csetting = cf.getSetting("Connection", "SSLCertificateFileName"); 
            if (csetting)
            {
                // TODO: what if url uses TLS/SSL protocol? Warn empty field?
                if (csetting->AsString.length() > 0)
                {
                    free(opt_cert);
                    opt_cert = strdup(csetting->AsString.c_str()); 
                }
            }
            else
            {
                // TODO: what if url uses TLS/SSL protocol? Warn empty field?
                applog(LOG_WARNING, "Failed to get a \"SSLCertificateFileName\" option inside \"Connection\" section. Setting to \"\"");
                ++cfWarnings;
            }
        }

        //-------------------------------------
        // get Proxy
        if (!cmdr.overwrite_proxy) 
        {
            csetting = cf.getSetting("Connection", "Proxy"); 
            if (csetting)
            {
                if (csetting->AsString.length() > 0)
                {
                    if (!strncasecmp(csetting->AsString.c_str(), "socks4://", 9))
                    {
                        opt_proxy_type = CURLPROXY_SOCKS4;
                    }
                    else if (!strncasecmp(csetting->AsString.c_str(), "socks5://", 9))
                    {
                        opt_proxy_type = CURLPROXY_SOCKS5;
                    }
            #if LIBCURL_VERSION_NUM >= 0x071200
                    else if (!strncasecmp(csetting->AsString.c_str(), "socks4a://", 10))
                    {
                        opt_proxy_type = CURLPROXY_SOCKS4A;
                    }
                    else if (!strncasecmp(csetting->AsString.c_str(), "socks5h://", 10))
                    {
                        opt_proxy_type = CURLPROXY_SOCKS5_HOSTNAME;
                    }
            #endif
                    else
                    {
                        opt_proxy_type = CURLPROXY_HTTP;
                    }
                    free(opt_proxy);
                    opt_proxy = strdup(csetting->AsString.c_str());
                }
            }
            else
            {
                applog(LOG_WARNING, "Failed to get a \"Proxy\" option inside \"Connection\" section. Setting to \"\"");
                ++cfWarnings;
            }
        }

        //-------------------------------------
        // get Redirect parameter
        if (!cmdr.overwrite_redirect) 
        {
            csetting = cf.getSetting("Connection", "Redirect"); 
            if (csetting)
            {
                opt_redirect = csetting->AsBool;
            }
            else
            {
                opt_redirect = true;

                applog(LOG_WARNING, "Failed to get a \"Redirect\" option inside \"Connection\" section. Setting to \"true\"");
                ++cfWarnings;
            }
        }

        //-------------------------------------
        // get LongPoll parameter
        if (!cmdr.overwrite_wantLongpoll) 
        {
            csetting = cf.getSetting("Connection", "LongPoll"); 
            if (csetting)
            {
                want_longpoll = csetting->AsBool;
            }
            else
            {
                want_longpoll = true;

                applog(LOG_WARNING, "Failed to get a \"LongPoll\" option inside \"Connection\" section. Setting to \"true\"");
                ++cfWarnings;
            }
        }

        //-------------------------------------
        // get LongPollTimeout parameter
        if (!cmdr.overwrite_longpollTimeout) 
        {
            csetting = cf.getSetting("Connection", "LongPollTimeout"); 
            if (csetting)
            {
                opt_longpoll_timeout = csetting->AsInt;
            }
            else
            {
                opt_longpoll_timeout = 0;

                applog(LOG_WARNING, "Failed to get a \"LongPollTimeout\" option inside \"Connection\" section. Setting to \"0\"");
                ++cfWarnings;
            }
        }

        //-------------------------------------
        // get Scantime parameter
        if (!cmdr.overwrite_scantime) 
        {
            csetting = cf.getSetting("Connection", "Scantime"); 
            if (csetting)
            {
                opt_scantime = csetting->AsInt;
            }
            else
            {
                opt_scantime = 5;

                applog(LOG_WARNING, "Failed to get a \"Scantime\" option inside \"Connection\" section. Setting to \"5\"");
                ++cfWarnings;
            }
        }

        //-------------------------------------
        // get Retries parameter
        if (!cmdr.overwrite_reconnectRetries) 
        {
            csetting = cf.getSetting("Connection", "Retries"); 
            if (csetting)
            {
                opt_retries = csetting->AsInt;
            }
            else
            {
                opt_retries = -1;

                applog(LOG_WARNING, "Failed to get a \"Retries\" option inside \"Connection\" section. Setting to \"-1\"");
                ++cfWarnings;
            }
        }

        //-------------------------------------
        // get RetryPause parameter
        if (!cmdr.overwrite_retryPause) 
        {
            csetting = cf.getSetting("Connection", "RetryPause");
            if (csetting)
            {
                opt_fail_pause = csetting->AsInt;
            }
            else
            {
                opt_fail_pause = 5;

                applog(LOG_WARNING, "Failed to get a \"RetryPause\" option inside \"Connection\" section. Setting to \"5\"");
                ++cfWarnings;
            }
        }

        //-------------------------------------
        // get ProtocolDump parameter
        if (!cmdr.overwrite_protocolDebug) 
        {
            csetting = cf.getSetting("Connection", "ProtocolDump");
            if (csetting)
            {
                opt_protocol = csetting->AsBool;
            }
            else
            {
                opt_protocol = false;

                applog(LOG_WARNING, "Failed to get a \"ProtocolDump\" option inside \"Connection\" section. Setting to \"false\"");
                ++cfWarnings;
            }
        }


        // Check if no devices are selected are using command line paramaters
        if ((!cmdr.selectAllCLDevices) && (!cmdr.selectedCLDevices.size()))
        {
            // Configure OpenCL devices from the configuration file.
            size_t deviceBlockIndex = 0;
            std::string dCLBlockName("CL_Device");
            std::string binaryFormatName;
            std::string asmProgramName;
            std::string batchSizeStr;
            std::string deviceBlock = dCLBlockName + std::to_string(deviceBlockIndex);
            
            while( (csetting = cf.getSetting(deviceBlock.c_str(), "DeviceIndex")) )
            {
                size_t deviceIndex = (size_t)csetting->AsInt;
                if (deviceIndex >= cldevices.size())
                {
                    applog(LOG_WARNING, "\"DeviceIndex\" is invalid inside \"%s\" section. Skipping device...", deviceBlock.c_str());
                    ++cfWarnings;

                    // move to the next device
                    ++deviceBlockIndex;
                    deviceBlock = dCLBlockName + std::to_string(deviceBlockIndex);

                    continue;
                }

#ifdef HAVE_CUDA
                // Handle Nvidia GPU to CUDA restrictions
                if (cmdr.restrictNVGPUToCUDA)
                {
                    if (cldevices[deviceIndex].vendor == vh::V_NVIDIA)
                    {
                        cl_int nvCCMajor = -1;
                        cl_int status0 = clGetDeviceInfo(cldevices[deviceIndex].clId, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_int), &nvCCMajor, NULL);
                        cl_int nvCCMinor = -1;
                        cl_int status1 = clGetDeviceInfo(cldevices[deviceIndex].clId, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_int), &nvCCMinor, NULL);
                        if((status0 == CL_SUCCESS) && (status1 == CL_SUCCESS))
                        {
                            // CUDA 11 or higher
                            if (CUDART_VERSION >= 11000)
                            {
                                // CUDA 11 removed SM 3.0 support.
                                // Transfer SM 3.0 to the OpenCL backend
                                if (!((nvCCMajor == 3) && (nvCCMinor == 0)))
                                {
                                    applog(LOG_ERR, "It looks like this configuration file has been generated with --no-restrict-cuda parameter!");
                                    applog(LOG_ERR, "Please restart application with '--no-restrict-cuda'. Exiting.");
                                    return 1;
                                }
                            }
                        }
                    }
                }
#endif

                // setting to default values
                uint32_t workSize = vh::defaultWorkSize;
                uint32_t batchTimeMs = vh::defaultBatchTimeMs;
                uint32_t occupancyPct = vh::defaultOccupancyPct;
                int deviceMonitor = vh::defaultDeviceMonitor;
                int gpuTemperatureLimit = vh::defaultGPUTemperatureLimit;
                vh::EBinaryFormat binaryFormat = vh::BF_None;
                vh::EAsmProgram asmProgram = vh::AP_None; 

                // TODO: We don't validate these 2 parameters because they are not used for now
                // get program binary format
                csetting = cf.getSetting(deviceBlock.c_str(), "BinaryFormat"); 
                if (csetting)
                {
                    binaryFormatName = csetting->AsString;
                    binaryFormat = vh::getBinaryFormatFromName(binaryFormatName);
                }

                // get asm program name
                csetting = cf.getSetting(deviceBlock.c_str(), "AsmProgram"); 
                if (csetting)
                {
                    asmProgramName = csetting->AsString;
                    asmProgram = vh::getAsmProgramName(asmProgramName);
                }

                // get work size parameter and validate it
                csetting = cf.getSetting(deviceBlock.c_str(), "WorkSize"); 
                if (csetting)
                {
                    workSize = (uint32_t)csetting->AsInt;
                    if ((workSize % 256) != 0)
                    {
                        applog(LOG_WARNING, "\"WorkSize\" parameter is incorrect inside \"%s\" section."
                                            " It must be multiple of 256. Using default: %u.",
                               deviceBlock.c_str(), vh::defaultWorkSize);
                        ++cfWarnings;
                    }
                }
                else
                {
                    applog(LOG_WARNING, "Failed to get a \"WorkSize\" option inside \"%s\" section. Setting to: \"%u\".",
                           deviceBlock.c_str(), vh::defaultWorkSize);
                    ++cfWarnings;
                }

                // get batch time and validate it
                csetting = cf.getSetting(deviceBlock.c_str(), "BatchTimeMs"); 
                if (csetting)
                {
                    batchTimeMs = (uint32_t)csetting->AsInt;
                    if (batchTimeMs == 0)
                    {
                        applog(LOG_WARNING, "\"BatchTime\" parameter is incorrect inside \"%s\" section."
                                            " It must be above 0. Using default: %u",
                               deviceBlock.c_str(), vh::defaultBatchTimeMs);
                        ++cfWarnings;
                    }
                }
                else
                {
                    applog(LOG_WARNING, "Failed to get a \"BatchTime\" option inside \"%s\" section. Setting to: \"%u\".",
                           deviceBlock.c_str(), vh::defaultBatchTimeMs);
                    ++cfWarnings;
                }

                // get occupancy percent and validate it
                csetting = cf.getSetting(deviceBlock.c_str(), "OccupancyPct"); 
                if (csetting)
                {
                    occupancyPct = (uint32_t)csetting->AsInt;
                    if ((occupancyPct == 0) || (occupancyPct > 100))
                    {
                        applog(LOG_WARNING, "\"OccupancyPct\" parameter is incorrect inside \"%s\" section."
                                            " It must be above 0 and less than or equal to 100. Using default: %u",
                               deviceBlock.c_str(), vh::defaultOccupancyPct);
                        ++cfWarnings;
                    }
                }
                else
                {
                    applog(LOG_WARNING, "Failed to get a \"OccupancyPct\" option inside \"%s\" section. Setting to: \"%u\".",
                           deviceBlock.c_str(), vh::defaultOccupancyPct);
                    ++cfWarnings;
                }

                // get device monitoring level
                csetting = cf.getSetting(deviceBlock.c_str(), "DeviceMonitor"); 
                if (csetting)
                {
                    deviceMonitor = csetting->AsInt;
                }
                else
                {
                    applog(LOG_WARNING, "Failed to get a \"DeviceMonitor\" option inside \"%s\" section. Setting to: \"%u\".",
                           deviceBlock.c_str(), vh::defaultDeviceMonitor);
                    ++cfWarnings;
                }

                // get GPU temperature limit level
                csetting = cf.getSetting(deviceBlock.c_str(), "GPUTemperatureLimit"); 
                if (csetting)
                {
                    gpuTemperatureLimit = csetting->AsInt;
                }
                else
                {
                    applog(LOG_WARNING, "Failed to get a \"GPUTemperatureLimit\" option inside \"%s\" section. Setting to: \"%u\".",
                           deviceBlock.c_str(), vh::defaultGPUTemperatureLimit);
                    ++cfWarnings;
                }

                // check if device index is correct
                clworker_t clworker;
                clworker.nvmlDevice = NULL;
                clworker.adlAdapterIndex = -1;
                clworker.cldevice = cldevices[deviceIndex];
                clworker.workSize = workSize;
                clworker.batchTimeMs = batchTimeMs;
                clworker.occupancyPct = occupancyPct;
                clworker.gpuTemperatureLimit = gpuTemperatureLimit;
                clworker.deviceMonitor = deviceMonitor;
                // AsmProgram will be detected on context init
                clworker.cldevice.binaryFormat = binaryFormat;
                clworker.cldevice.asmProgram = asmProgram;
                clworkers.push_back(clworker);

                ++deviceBlockIndex;
                deviceBlock = dCLBlockName + std::to_string(deviceBlockIndex);
            }
        } // end ((!cmdr.selectAllDevices) && (!cmdr.selectedCLDevices.size()))


#ifdef HAVE_CUDA
        // Check if no devices are selected are using command line paramaters
        if ((!cmdr.selectAllCUDevices) && (!cmdr.selectedCUDevices.size()))
        {
            // Configure CUDA devices from the configuration file.
            size_t deviceBlockIndex = 0;
            std::string dCUBlockName("CU_Device");
            std::string deviceBlock = dCUBlockName + std::to_string(deviceBlockIndex);
            std::string batchSizeStr;

            while( (csetting = cf.getSetting(deviceBlock.c_str(), "DeviceIndex")) )
            {
                int deviceIndex = csetting->AsInt;
                if ((deviceIndex >= cudeviceListSize) || (deviceIndex < 0))
                {
                    applog(LOG_WARNING, "\"DeviceIndex\" is invalid inside \"%s\" section. Skipping device...", deviceBlock.c_str());
                    ++cfWarnings;

                    // move to the next device
                    ++deviceBlockIndex;
                    deviceBlock = dCUBlockName + std::to_string(deviceBlockIndex);

                    continue;
                }

                // set to default values
                uint32_t workSize = vh::defaultWorkSize;
                uint32_t batchTimeMs = vh::defaultBatchTimeMs;
                uint32_t occupancyPct = vh::defaultOccupancyPct;
                int deviceMonitor = vh::defaultDeviceMonitor;
                int gpuTemperatureLimit = vh::defaultGPUTemperatureLimit;

                // get work size parameter and validate it
                csetting = cf.getSetting(deviceBlock.c_str(), "WorkSize"); 
                if (csetting)
                {
                    workSize = (uint32_t)csetting->AsInt;
                    if ((workSize % 256) != 0)
                    {
                        applog(LOG_WARNING, "\"WorkSize\" parameter is incorrect inside \"%s\" section."
                                            " It must be multiple of 256. Using default: %u.",
                               deviceBlock.c_str(), vh::defaultWorkSize);
                        ++cfWarnings;
                    }
                }
                else
                {
                    applog(LOG_WARNING, "Failed to get a \"WorkSize\" option inside \"%s\" section. Setting to: \"%u\".",
                           deviceBlock.c_str(), vh::defaultWorkSize);
                    ++cfWarnings;
                }

                // get batch time and validate it
                csetting = cf.getSetting(deviceBlock.c_str(), "BatchTimeMs"); 
                if (csetting)
                {
                    batchTimeMs = (uint32_t)csetting->AsInt;
                    if (batchTimeMs == 0)
                    {
                        applog(LOG_WARNING, "\"BatchTime\" parameter is incorrect inside \"%s\" section."
                                            " It must be above 0. Using default: %u",
                               deviceBlock.c_str(), vh::defaultBatchTimeMs);
                        ++cfWarnings;
                    }
                }
                else
                {
                    applog(LOG_WARNING, "Failed to get a \"BatchTime\" option inside \"%s\" section. Setting to: \"%u\".",
                           deviceBlock.c_str(), vh::defaultBatchTimeMs);
                    ++cfWarnings;
                }

                // get occupancy percent and validate it
                csetting = cf.getSetting(deviceBlock.c_str(), "OccupancyPct"); 
                if (csetting)
                {
                    occupancyPct = (uint32_t)csetting->AsInt;
                    if ((occupancyPct == 0) || (occupancyPct > 100))
                    {
                        applog(LOG_WARNING, "\"OccupancyPct\" parameter is incorrect inside \"%s\" section."
                                            " It must be above 0 and less than or equal to 100. Using default: %u",
                               deviceBlock.c_str(), vh::defaultOccupancyPct);
                        ++cfWarnings;
                    }
                }
                else
                {
                    applog(LOG_WARNING, "Failed to get a \"OccupancyPct\" option inside \"%s\" section. Setting to: \"%u\".",
                           deviceBlock.c_str(), vh::defaultOccupancyPct);
                    ++cfWarnings;
                }

                // get device monitoring level
                csetting = cf.getSetting(deviceBlock.c_str(), "DeviceMonitor"); 
                if (csetting)
                {
                    deviceMonitor = csetting->AsInt;
                }
                else
                {
                    applog(LOG_WARNING, "Failed to get a \"DeviceMonitor\" option inside \"%s\" section. Setting to: \"%u\".",
                           deviceBlock.c_str(), vh::defaultDeviceMonitor);
                    ++cfWarnings;
                }

                // get GPU temperature limit level
                csetting = cf.getSetting(deviceBlock.c_str(), "GPUTemperatureLimit"); 
                if (csetting)
                {
                    gpuTemperatureLimit = csetting->AsInt;
                }
                else
                {
                    applog(LOG_WARNING, "Failed to get a \"GPUTemperatureLimit\" option inside \"%s\" section. Setting to: \"%u\".",
                           deviceBlock.c_str(), vh::defaultGPUTemperatureLimit);
                    ++cfWarnings;
                }

                // create CUDA worker
                cuworker_t cuworker;
                cuworker.nvmlDevice = NULL;
                cuworker.cudevice = cudevices[deviceIndex];
                cuworker.workSize = workSize;
                cuworker.batchTimeMs = batchTimeMs;
                cuworker.occupancyPct = occupancyPct;
                cuworker.gpuTemperatureLimit = gpuTemperatureLimit;
                cuworker.deviceMonitor = deviceMonitor;
                cuworkers.push_back(cuworker);

                // move to the next device
                ++deviceBlockIndex;
                deviceBlock = dCUBlockName + std::to_string(deviceBlockIndex);
            }
        }
#endif

        // Final configuration validation.
        if (cfErrors > 0)
        {
            applog(LOG_ERR, "Miner configuration failed! (Errors: %d, Warnings: %d)", cfErrors, cfWarnings);
            cmd_result_free(&cmdr);
            return 1;
        }
        else
        {
            applog(LOG_INFO, "Miner has been successfully configured! (Errors: %d, Warnings: %d)", cfErrors, cfWarnings);
        }

    } // end usingConfigFile
    else
    {
        // Pure cmd line configuration.
        // Validate parameters
        int cmdErrors = 0;
        int cmdWarnings = 0;

        // address
        if (!cmdr.overwrite_rpcUrlPort) 
        {
            applog(LOG_ERR, "Connection parameter address(url:port) is not set");
            ++cmdErrors;
        }

        // user
        if (!cmdr.overwrite_rpcUser) 
        {
            applog(LOG_ERR, "Connection parameter 'user' is not set");
            ++cmdErrors;
        }

        // password
        if (!cmdr.overwrite_rpcPass) 
        {
            applog(LOG_ERR, "Connection parameter 'pass' is not set");
            ++cmdErrors;
        }

        // verthash data file
        int vhLoadResult = verthash_info_init(&verthashInfo, cmdr.verthashDataFileName);
        // Check Verthash initialization status
        if (vhLoadResult == 0) // No Error
        {
            applog(LOG_INFO, "Verthash data file has been loaded succesfully!");

            //  and verify data file(if it was enabled)
            if (!cmdr.disableVerthashDataFileVerification)
            {
                uint8_t vhDataFileHash[32] = { 0 };
                sha256s(vhDataFileHash, verthashInfo.data, verthashInfo.dataSize);
                if (memcmp(vhDataFileHash, verthashDatFileHash_bytes, sizeof(verthashDatFileHash_bytes)) == 0)
                {
                    applog(LOG_INFO, "Verthash data file has been verified succesfully!");
                }
                else
                {
                    applog(LOG_ERR, "Verthash data file verification has failed!");
                    ++cmdErrors;
                }
            }
            else
            {
                applog(LOG_WARNING, "Verthash data file verification stage is disabled!");
                ++cmdWarnings;
            }
        }
        else
        {
            // Handle Verthash error codes
            if (vhLoadResult == 1)
                applog(LOG_ERR, "Verthash data file name is invalid");
            else if (vhLoadResult == 2)
                applog(LOG_ERR, "Failed to allocate memory for Verthash data");
            else // for debugging purposes
                applog(LOG_ERR, "Verthash data initialization unknown error code: %d", vhLoadResult);
            ++cmdErrors;
        }

        // coinbase address is needed only for stratum values
        if (!have_stratum)
        {
            if (!cmdr.overwrite_coinbaseAddr) 
            {
                applog(LOG_ERR, "Using GBT protocol, while 'coinbase-addr' is not set");
                ++cmdErrors;
            }
        }

        // Final configuration validation.
        if (cmdErrors > 0)
        {
            applog(LOG_ERR, "Miner configuration failed! (Errors: %d, Warnings: %d)", cmdErrors, cmdWarnings);
            cmd_result_free(&cmdr);
            return 1;
        }
        else
        {
            applog(LOG_INFO, "Miner has been successfully configured! (Errors: %d, Warnings: %d)", cmdErrors, cmdWarnings);
        }
    }

    // free cmd-line result data
    cmd_result_free(&cmdr);

    //-------------------------------------
    // Calculate the number of workers
#ifdef HAVE_CUDA
    opt_n_threads = clworkers.size() + cuworkers.size();
    cuDeviceIndexOffset = clworkers.size();
    if (opt_n_threads > 0)
        applog(LOG_INFO, "Configured %llu(CL) and %llu(CUDA) workers", clworkers.size(), cuworkers.size());
    else
    {
        applog(LOG_WARNING, "Found 0 configured workers. Exiting...");
        return 1;
    }
    
#else
    opt_n_threads = clworkers.size();
    if (opt_n_threads > 0)
        applog(LOG_INFO, "Configured %llu(CL) workers", clworkers.size());
    else
    {
        applog(LOG_WARNING, "Found 0 configured workers. Exiting...");
        return 1;
    }
#endif

    //-----------------------------------------------------------------------------
    // GPU Monitoring

    //-------------------------------------
    // NVML backend
    bool NVMLNeeded = false;
    for (size_t i = 0; i < clworkers.size(); ++i)
    {
        // Skip non AMD devices
        if (clworkers[i].cldevice.vendor == vh::V_NVIDIA)
        {
            if (clworkers[i].deviceMonitor != 0)
            {
                NVMLNeeded = true;
                break;
            }
            
        }
    }

#ifdef HAVE_CUDA
    for (size_t i = 0; i < cuworkers.size(); ++i)
    {
        if (cuworkers[i].deviceMonitor != 0)
        {
            NVMLNeeded = true;
            break;
        }
    }
#endif

    int libNVMLRC = -1;
    if (NVMLNeeded == true)
    {
        libNVMLRC = nvmlInitApi();
        if (libNVMLRC == 0)
        {
            // First initialize NVML library
            nvmlReturn_t nvmlRC = nvmlInitWithFlags(0);
            if (nvmlRC != NVML_SUCCESS)
            { 
                applog(LOG_WARNING, "Failed to initialize NVML: %s", nvmlErrorString(nvmlRC));
                // goto exit TODO:
            }

            unsigned int nvml_device_count = 0;
            nvmlRC = nvmlDeviceGetCount(&nvml_device_count);
            if (nvmlRC != NVML_SUCCESS)
            { 
                applog(LOG_WARNING, "Failed to query device count: %s", nvmlErrorString(nvmlRC));
                // goto exit TODO:
            }
            applog(LOG_DEBUG, "Found %u NVML device%s", nvml_device_count, nvml_device_count != 1 ? "s" : "");


            nvmlDevice_t nvmlDevices[64] = {};
            for (unsigned int i = 0; i < nvml_device_count; i++)
            {
                nvmlRC = nvmlDeviceGetHandleByIndex(i, &nvmlDevices[(size_t)i]);
                if (nvmlRC != NVML_SUCCESS)
                {
                    applog(LOG_WARNING, "Failed to get handle for device %u: %s", i, nvmlErrorString(nvmlRC));
                }
            }

            // configure all NVML cappable OpenCL workers
            for (size_t i = 0; i < clworkers.size(); ++i)
            {
                // Skip non NVIDIA devices
                if (clworkers[i].cldevice.vendor != vh::V_NVIDIA)
                {
                    continue;
                }

                for (size_t n = 0; n < (size_t)nvml_device_count; ++n)
                {
                    nvmlPciInfo_t pci;
                    nvmlDeviceGetPciInfo(nvmlDevices[n], &pci);

                    if (clworkers[i].cldevice.pcieBusId == pci.bus &&
                        clworkers[i].cldevice.pcieDeviceId == pci.device)
                    {
                        clworkers[i].nvmlDevice = nvmlDevices[n];
                        break;
                    }
                }
            }

    #ifdef HAVE_CUDA
            // configure all NVML cappable CUDA workers
            for (size_t i = 0; i < cuworkers.size(); ++i)
            {
                for (size_t n = 0; n < (size_t)nvml_device_count; ++n)
                {
                    nvmlPciInfo_t pci;
                    nvmlDeviceGetPciInfo(nvmlDevices[n], &pci);

                    if (cuworkers[i].cudevice.pcieBusId == pci.bus &&
                        cuworkers[i].cudevice.pcieDeviceId == pci.device)
                    {
                        cuworkers[i].nvmlDevice = nvmlDevices[n];
                        break;
                    }
                }
            }
    #endif
        }
        else
        {
            applog(LOG_WARNING, "Failed to initalize NVML API");
        }
    }




    //-------------------------------------
    // ADL backend
    bool ADLNeeded = false;
    for (size_t i = 0; i < clworkers.size(); ++i)
    {
        // Skip non AMD devices
        if (clworkers[i].cldevice.vendor == vh::V_AMD)
        {
            if (clworkers[i].deviceMonitor == 1)
            {
                ADLNeeded = true;
                break;
            }
        }
    }

    if (ADLNeeded)
    {
        int libADLRC = adlInitApi();
        if (libADLRC == 0)
        {
            ADL_CONTEXT_HANDLE adlContext = NULL;
            int adlRC = ADL2_Main_Control_Create(ADL_Main_Memory_Alloc, 1, &adlContext);
            if (adlRC == ADL_OK)
            {
                int numAdapters = 0;
                adlRC = ADL2_Adapter_NumberOfAdapters_Get(adlContext, &numAdapters);
                if (adlRC != ADL_OK)
                {
                    applog(LOG_WARNING, "Failed to get the number of ADL adapters");
                    // goto exit TODO:
                }

                if (numAdapters > 0)
                {
                    AdapterInfo *adapterInfos = (AdapterInfo *)malloc(sizeof(AdapterInfo) * numAdapters);
                    if (!adapterInfos)
                    {
                        applog(LOG_ERR, "Failed to allocate memory for ADL adapterInfos");
                        adlRC = ADL2_Main_Control_Destroy(adlContext);
                        if (adlRC != ADL_OK)
                        {
                            applog(LOG_ERR, "Failed to destroy ADL context");
                        }
                        return 1;
                    }

                    // get ADL adapter infos
                    ADL2_Adapter_AdapterInfo_Get(adlContext, adapterInfos, sizeof(AdapterInfo)* numAdapters);

                    // get unique and suitable adapter indices
                    std::vector<size_t> adapterInfoIndices;
                    int lastAdapterId = -1;
                    for (size_t i = 0; i < (size_t)numAdapters; ++i)
                    {
                        int adapterId;
                        adlRC = ADL2_Adapter_ID_Get(adlContext, adapterInfos[i].iAdapterIndex, &adapterId);
                        if (adlRC != ADL_OK)
                        {
                            if ((adapterInfos[i].iVendorID == 1002) && (errorCode == ADL_ERR_DISABLED_ADAPTER))
                            {
                                applog(LOG_WARNING, "Found an AMD adapter, but it is disabled(infoIdx:%u)", (uint32_t)i);
                            }
                            continue;
                        }

                        // Filter out non AMD adapters(in case they didn't trigger ADL2_Adapter_ID_Get errors)
                        if (adapterInfos[i].iVendorID != 1002)
                        {
                            continue;
                        }

                        // Each adapter may have multiple entries
                        if (adapterId == lastAdapterId)
                        {
                            continue;
                        }

                        adapterInfoIndices.push_back(i);

                        lastAdapterId = adapterId;
                    }


                    // configure all ADL cappable OpenCL workers
                    for (size_t i = 0; i < clworkers.size(); ++i)
                    {
                        // Skip non AMD devices
                        if (clworkers[i].cldevice.vendor != vh::V_AMD)
                        {
                            continue;
                        }

                        for (size_t n = 0; n < adapterInfoIndices.size(); ++n)
                        {
                            const size_t ainfIndex = adapterInfoIndices[n];

                            if (clworkers[i].cldevice.pcieBusId == adapterInfos[ainfIndex].iBusNumber &&
                                clworkers[i].cldevice.pcieDeviceId == adapterInfos[ainfIndex].iDeviceNumber &&
                                clworkers[i].cldevice.pcieFunctionId == adapterInfos[ainfIndex].iFunctionNumber)
                            {
                                clworkers[i].adlAdapterIndex = adapterInfos[ainfIndex].iAdapterIndex;
                                break;
                            }
                        }
                    }

                    //-------------------------------------
                    // Free memory
                    ADL_Main_Memory_Free((void**)&adapterInfos);

                } // numAdapters > 0

                adlRC = ADL2_Main_Control_Destroy(adlContext);
                if (adlRC != ADL_OK)
                {
                    applog(LOG_ERR, "Failed to destroy ADL context");
                }
            }
            else
            {
                applog(LOG_WARNING, "Failed to create ADL context");
            }
        }
        else
        {
            applog(LOG_WARNING, "Failed to load ADL functions.");
        }
    }


    //-------------------------------------
    // Init cURL
    long flags = opt_benchmark || (strncasecmp(rpc_url, "https://", 8) &&
                                   strncasecmp(rpc_url, "stratum+tcps://", 15))
                ? (CURL_GLOBAL_ALL & ~CURL_GLOBAL_SSL) : CURL_GLOBAL_ALL;
    if (curl_global_init(flags))
    {
        applog(LOG_ERR, "CURL initialization failed");
        return 1;
    }

    //-------------------------------------
    // Allocate thread handles
    struct thr_info* thr;

    work_restart = (struct work_restart*)calloc(opt_n_threads, sizeof(*work_restart));
    if (!work_restart) { return 1; }

    thr_info = (struct thr_info*)calloc(opt_n_threads + 3, sizeof(*thr));
    if (!thr_info) { return 1; }
    
    thr_hashrates = (double *) calloc(opt_n_threads, sizeof(double));
    if (!thr_hashrates) { return 1; }

    // Init workio thread info.
    work_thr_id = opt_n_threads;
    thr = &thr_info[work_thr_id];
    thr->id = work_thr_id;
    thr->q = tq_new();
    if (!thr->q)
    {
        return 1;
    }

    // Start work I/O thread
    if (thrd_create(&thr->pth, workio_thread, thr) != thrd_success)
    {
        applog(LOG_ERR, "workio thread create failed");
        return 1;
    }

    if (want_longpoll && !have_stratum)
    {
        // Init longpoll thread info
        longpoll_thr_id = opt_n_threads + 1;
        thr = &thr_info[longpoll_thr_id];
        thr->id = longpoll_thr_id;
        thr->q = tq_new();
        if (!thr->q)
        {
            return 1;
        }

        // Start longpoll thread
        if (unlikely(thrd_create(&thr->pth, longpoll_thread, thr) != thrd_success))
        {
            applog(LOG_ERR, "longpoll thread create failed");
            return 1;
        }
    }
    if (have_stratum)
    {
        // Init stratum thread info
        stratum_thr_id = opt_n_threads + 2;
        thr = &thr_info[stratum_thr_id];
        thr->id = stratum_thr_id;
        thr->q = tq_new();
        if (!thr->q)
        {
            return 1;
        }

        // Create a dummy socket(used to trigger de-initialization stage from stratum thread)
        struct sockaddr_in addr;
        stratum.dummy_socket = socket(AF_INET, SOCK_DGRAM, 0); 
        if (stratum.dummy_socket < 0)
        {
            applog(LOG_ERR, "Dummy socket create failed");
            return 1;
        }
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(0);
        if (bind(stratum.dummy_socket, (struct sockaddr *)&addr, sizeof (addr)) < 0)
        {
            applog(LOG_ERR, "Dummy socket bind failed");
            return 1;
        }

        struct sockaddr_in my_addr;
        memset(&my_addr, 0, sizeof(my_addr));

#ifdef _MSC_VER // TODO: or just WIN32?
        int len = sizeof(my_addr);
#else
        socklen_t len = sizeof(my_addr);
#endif
        int s_rc = getsockname(stratum.dummy_socket, (struct sockaddr *) &my_addr, &len);
        if (s_rc == -1)
        {
            applog(LOG_ERR, "Dummy socket creation failed (getsockname stage)");
            return 1;
        }

        memset(&addr, 0, sizeof (addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        //addr.sin_port = srv->s_port;
        addr.sin_port = my_addr.sin_port;
        if (connect(stratum.dummy_socket, (struct sockaddr*)&addr, sizeof(addr)) < 0)
        {
            applog(LOG_ERR, "Dummy socket creation failed (connect stage)");
            return 1;
        }

        // start stratum thread
        if (unlikely(thrd_create(&thr->pth, stratum_thread, thr) != thrd_success))
        {
            applog(LOG_ERR, "stratum thread create failed");
            return 1;
        }

        if (have_stratum)
        {
            tq_push(thr_info[stratum_thr_id].q, strdup(rpc_url));
        }
    }

    //-------------------------------------
    // OpenCL workers
    for (size_t i = 0; i < clworkers.size(); i++)
    {
        clworkers[i].threadInfo = &thr_info[i];
        clworkers[i].threadInfo->id = (int)i;
        clworkers[i].threadInfo->q = tq_new(); 
        if (!clworkers[i].threadInfo->q)
        {
            return 1;
        }

        if (unlikely(thrd_create(&clworkers[i].threadInfo->pth, verthashOpenCL_thread, &clworkers[i]) != thrd_success))
        {
            applog(LOG_ERR, "thread %d create failed", i);
            return 1;
        }
    }

#ifdef HAVE_CUDA
    //-------------------------------------
    // CUDA workers
    for (size_t i = 0; i < cuworkers.size(); i++)
    {
        cuworkers[i].threadInfo = &thr_info[clworkers.size()+i];
        cuworkers[i].threadInfo->id = (int)(clworkers.size()+i);
        cuworkers[i].threadInfo->q = tq_new(); 
        if (!cuworkers[i].threadInfo->q)
        {
            return 1;
        }

        if (unlikely(thrd_create(&cuworkers[i].threadInfo->pth, verthashCuda_thread, &cuworkers[i]) != thrd_success))
        {
            applog(LOG_ERR, "thread %d create failed", i);
            return 1;
        }
    }
#endif


    applog(LOG_INFO, "%d miner threads started, using Verthash algorithm.", opt_n_threads);

    //-------------------------------------
    // Wait all threads to exit

    // DEBUG: stale work detected, discarding
    // Can happen with GBT when result has been submited but haven't notified about new block

    // workio thread(main loop)
    thrd_join(thr_info[work_thr_id].pth, NULL);
    tq_freeze(thr_info[work_thr_id].q);
    applog(LOG_INFO, "WorkIO thread has been finished.");

    // set abort flag if it wasn't set.(in case if workIO thread exit wasn't triggered by user)
    if (!abort_flag) { abort_flag = true; }
    // if stratum is enabled trigger stratum thread to exit
    if (have_stratum) { send(stratum.dummy_socket, NULL, 0, 0); }

    applog(LOG_INFO, "Waiting for worker threads to exit...");
    // worker threads
    // disable all queues first
    for (size_t i = 0; i < opt_n_threads; ++i) { tq_freeze(thr_info[i].q); }
    // wait
    for (size_t i = 0; i < opt_n_threads; ++i) { thrd_join(thr_info[i].pth, NULL); }

    // stratum & longpoll threads
    if (have_stratum)
    {
        applog(LOG_INFO, "Waiting for stratum thread to exit...");
        tq_freeze(thr_info[stratum_thr_id].q);
        thrd_join(thr_info[stratum_thr_id].pth, NULL);
    }
    else if (want_longpoll)
    {
        applog(LOG_INFO, "Waiting for longpoll thread to exit...");
        tq_freeze(thr_info[longpoll_thr_id].q);
        thrd_join(thr_info[longpoll_thr_id].pth, NULL);
    }

    applog(LOG_INFO, "Freeing allocated memory...");

    //-----------------------------------------------------------------------------
    // Free allocated data
    verthash_info_free(&verthashInfo);

    // queues
    tq_free(thr_info[work_thr_id].q);
    for (size_t i = 0; i < opt_n_threads; ++i) { tq_free(thr_info[i].q); }
    if (have_stratum) { tq_free(thr_info[stratum_thr_id].q); }
    else if (want_longpoll) { tq_free(thr_info[longpoll_thr_id].q); }

    // NVML
    if (libNVMLRC == 0)
    {
        nvmlShutdown();
    }

    applog(LOG_INFO, "Application has been exited gracefully.");

    // close file logger if it was enabled
    if (applog_file)
    {
        fclose(applog_file);
    }

    return 0;
}

int main(int argc, char * argv[])
{
#ifdef _WIN32
    int win_argc;
    char ** win_argv;

    wchar_t ** wargv = CommandLineToArgvW(GetCommandLineW(), &win_argc);
    win_argv         = (char**)malloc((win_argc + 1) * sizeof(char*));

    for(size_t i = 0; i < (size_t)win_argc; i++)
    {
        int32_t n   = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
        win_argv[i] = (char*)malloc(n);
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, win_argv[i], n, NULL, NULL);
    }
    LocalFree(wargv);

    int ret = utf8_main(win_argc, win_argv);

    for(size_t i = 0; i < (size_t)win_argc; i++)
    {
        free(win_argv[i]);
    }
    free(win_argv);

    return ret;
#else
    return utf8_main(argc, argv);
#endif
}

