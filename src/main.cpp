/*
 * Copyright 2010 Jeff Garzik
 * Copyright 2012-2017 pooler
 * Copyright 2018-2020 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <inttypes.h>
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

#include "vhCore/Miner.h"
#include "vhCore/SHA256.h"
#include "vhCore/Verthash.h"
#include "vhCore/ConfigFile.h"
#include "vhCore/ThreadQueue.h"

#include "vhDevice/CLUtils.h"
#include "vhDevice/ConfigGenerator.h"
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

static mtx_t stats_lock;



volatile bool abort_flag = false;

static unsigned long accepted_count = 0L;
static unsigned long rejected_count = 0L;
static double *thr_hashrates;

static char const usage[] = PACKAGE_NAME " " PACKAGE_VERSION " by CryptoGraphics <CrGr@protonmail.com>\n"
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
"--cl-devices <index:wWorkSize index:wWorkSize...>  (-d)\n\t"
    "Select specific OpenCL devices from the list, obtained by '-l' command.\n"
"\n"
"--all-cl-devices\n\t"
    "Use all available OpenCL devices from the list, obtained by '-l' command.\n\t"
    "This options as a priority over per device selection using '--cl-devices'\n"
"\n"
"--cu-devices <index:wWorkSize index:wWorkSize...>  (-D)\n\t"
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
"--gen-conf-raw <File>                  (-G)\n\t"
    "Generate a configuration file with raw device list format and exit.\n"
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
    "Print all available device configurations in raw device list format and exit.\n"
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
    { "gen-conf-raw", 1, NULL, 'G' },
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
        le32enc(&nonce, work->data[19]);
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
        applog(LOG_DEBUG, "DEBUG: job_id='%s' extranonce2=%s ntime=%08x",
               work->job_id, xnonce2str, swab32(work->data[17]));
        free(xnonce2str);
    }

    diff_to_target(work->target, sctx->job.diff / 256.0);
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
};

//! sha3/keccak state
struct kState { uint32_t v[50]; };
//! uvec8
struct uvec8 { uint32_t v[8]; };

//----------------------------------------------------------------------------
//! test uvec8 with target
// TODO: refactor/optimize, remove and use fullTest directly
inline bool fulltestUvec8(const uvec8& hash, const uint32_t *target)
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
        applog(LOG_DEBUG, "Verthash OCL thread started");

    clworker_t* clworker = (clworker_t*)userdata;
    vh::cldevice_t& cldevice = clworker->cldevice;
    //-------------------------------------
    // Init CL data
    cl_int errorCode = CL_SUCCESS;

    // num runs to reach max nonce
    const uint64_t numNoncesGlobal = 4294967296ULL;
    uint32_t maxRunsGlobal = (uint32_t)(numNoncesGlobal / (uint32_t)clworker->workSize);
    size_t workSize = clworker->workSize;
    const size_t globalWorkSize1x = workSize;
    const size_t globalWorkSize4x = workSize*4;
    size_t localWorkSize = 64;
    size_t localWorkSize256 = 256;
    std::string buildOptions0;
    std::string buildOptions;
    // Local work size may change in future releases
    if (cldevice.vendor == vh::V_NVIDIA)
    {
        localWorkSize = 64;
        buildOptions += " -DWORK_SIZE=64 ";
    }
    else if (cldevice.vendor == vh::V_AMD)
    {
        localWorkSize = 64;
        buildOptions0 += " -DBIT_ALIGN";
        buildOptions += " -DWORK_SIZE=64 -DBIT_ALIGN ";
    }
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

    struct thr_info *mythr = clworker->threadInfo; 
    int thr_id = mythr->id;

    const uint64_t numNoncesPerDevice = numNoncesGlobal / (uint64_t)opt_n_threads;
    uint32_t maxRunsPerDevice = (uint32_t)(numNoncesPerDevice / (uint32_t)workSize);
    // last device
    if (thr_id == (opt_n_threads-1))
    {
        maxRunsPerDevice += (uint32_t)(numNoncesPerDevice % opt_n_threads);
    }
    //-------------------------------------
    struct work workInfo = {{0}};
    
    uint32_t numRuns = 0;

#ifdef VERTHASH_FULL_VALIDATION
    // Host side hash storage
    std::vector<uvec8> verthashIORES;
    verthashIORES.resize(workSize);
#else
    // HTarg result host side storage
    std::vector<uint32_t> results;
    std::vector<uint32_t> potentialResults; // used if (potentialNonceCount > 1)
#endif

    // init per device profiling data
    const size_t maxProfSamples = 32;
    size_t numSamples = 0;
    size_t sampleIndex = 0;
    std::vector<uint64_t> profSamples(maxProfSamples, 0);

    // per hash-rate update timer 
    std::chrono::steady_clock::time_point hrTimerStart;
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
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create an OpenCL context.", thr_id); goto out; }

    // create an OpenCL command queue
    clCommandQueue = clCreateCommandQueue(clContext, cldevice.clId, 0, &errorCode);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create an OpenCL command queue.", thr_id); goto out; }

    //-------------------------------------
    // device buffers
    //! 8 precomputed keccak states
    clmemKStates = clCreateBuffer(clContext, CL_MEM_READ_WRITE, 8 * sizeof(kState), nullptr, &errorCode);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create a SHA3 states buffer.", thr_id); goto out; }

    //! block header for SHA3 reference or SHA3_precompute
    clmemHeader = clCreateBuffer(clContext, CL_MEM_READ_ONLY, 18 * sizeof(uint32_t), nullptr, &errorCode); 
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create a SHA3 headers buffer.", thr_id); goto out; }

    //! hash results from IO pass
    clmemResults = clCreateBuffer(clContext, CL_MEM_READ_WRITE, workSize * sizeof(uvec8), nullptr, &errorCode);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create an IO results buffer.", thr_id); goto out; }

    //! results against hash target.
    // Too much, but 100% robust: workBatchSize + num_actual_results(1)
    clmemHTargetResults = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(uint32_t) * (workSize + 1), nullptr, &errorCode);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create a hash target results buffer.", thr_id); goto out; }

    // Some drivers do not initialize buffers to 0(e.g. AMD GPU Pro). At least "result counter"(first value) must be initialized to 0
    errorCode = clEnqueueFillBuffer(clCommandQueue, clmemHTargetResults, &zero, sizeof(uint32_t), 0, (sizeof(uint32_t) * 1), 0, nullptr, nullptr);
    // OpenCL 1.0 - 1.1
    //errorCode = clEnqueueWriteBuffer(clCommandQueue, clmemHTargetResults, CL_TRUE, 0, sizeof(cl_uint), &zero, 0, nullptr, nullptr);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to clear HTarget result buffer.", thr_id); goto out; }

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
            applog(LOG_ERR, "cl_device(%d):Failed to create verthash data buffer.", thr_id);
            goto out;
        }

        // upload verthash data
        errorCode = clEnqueueWriteBuffer(clCommandQueue, clmemFullDat, CL_TRUE, 0,
                                         verthashInfo.dataSize, verthashInfo.data, 0, nullptr, nullptr);
        if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to copy Verthash data to the GPU memory.", thr_id); goto out; }
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
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(0) for SHA3 precompute kernel.", thr_id); goto out; }

    errorCode = clSetKernelArg(clkernelSHA3_512_precompute, 1, sizeof(cl_mem), &clmemHeader);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(1) for SHA3 precompute kernel.", thr_id); goto out; }

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
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(0) for SHA3_512_256 kernel.", thr_id); goto out; }

    errorCode = clSetKernelArg(clkernelSHA3_512_256, 1, sizeof(cl_mem), &clmemHeader);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(1) for SHA3_512_256 kernel.", thr_id); goto out; }


    // *Verthash pass
    // second stage

#ifdef BINARY_KERNELS
    clprogramVerthash = vh::cluCreateProgramWithBinaryFromFile(clContext, cldevice.clId, fileName_verthash.c_str()); 
#else
    clprogramVerthash = vh::cluCreateProgramFromFile(clContext, cldevice.clId, buildOptions.c_str(), "kernels/verthash.cl");
#endif

    if(clprogramVerthash == NULL) { applog(LOG_ERR, "cl_device(%d):Failed to create a Verthash program.", thr_id); goto out; }
    clkernelVerthash = clCreateKernel(clprogramVerthash, "verthash_4w", &errorCode);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to create a Verthash kernel.", thr_id); goto out; }
    errorCode = clSetKernelArg(clkernelVerthash, 0, sizeof(cl_mem), &clmemResults);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(0) for Verthash kernel.", thr_id); goto out; }
    errorCode = clSetKernelArg(clkernelVerthash, 1, sizeof(cl_mem), &clmemKStates);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(1) for Verthash kernel.", thr_id); goto out; }
    errorCode = clSetKernelArg(clkernelVerthash, 2, sizeof(cl_mem), &clmemFullDat);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(2) for Verthash kernel.", thr_id); goto out; }
#ifndef VERTHASH_FULL_VALIDATION
    errorCode = clSetKernelArg(clkernelVerthash, 5, sizeof(cl_mem), &clmemHTargetResults);
    if (errorCode != CL_SUCCESS) { applog(LOG_ERR, "cl_device(%d):Failed to set arg(5) for Verthash kernel.", thr_id); goto out; }
#endif


    //-------------------------------------
    // compute max runs
    // 4294967295 max nonce

    // reset the hash-rate reporting timer
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
            //if (memcmp(workInfo.data, g_work.data, 76))
            {
                stratum_gen_work(&stratum, &g_work);
            }

            maxRuns = maxRunsGlobal;
            runs = 0;
            nonce = 0;
        }
        else // GBT
        {
            // obtain new work from internal workio thread
            mtx_lock(&g_work_lock);

            work_free(&g_work);
            if (unlikely(!get_work(mythr, &g_work)))
            {
                if (!abort_flag) { applog(LOG_ERR, "cl_device(%d):Work retrieval failed, exiting mining thread %d", mythr->id, mythr->id); }
                mtx_unlock(&g_work_lock);
                goto out;
            }

            g_work_time = time(NULL);

            const uint32_t offsetN = (maxRunsPerDevice * workSize)*thr_id;
            const uint32_t first_nonce = offsetN + (numRuns * workSize);

            maxRuns = maxRunsPerDevice;
            runs = 0;
            nonce = first_nonce;
        }

        // create a work copy
        work_free(&workInfo);
        work_copy(&workInfo, &g_work);
        workInfo.data[19] = 0;

        mtx_unlock(&g_work_lock);
        work_restart[thr_id].restart = 0;
        
        //-------------------------------------
        // Generate midstate
        // TODO: optimize. Check on every new block
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
        // Not needed for stratum.
        //-------------------------------------
        // Compute hashes

        // abort flag is needed here to prevent an inconsistent behaviour
        // in case program termination was triggered between work generation and resetting work restart values
        while (!work_restart[thr_id].restart && (!abort_flag))
        {
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
            // compute average time from samples asynchronously
            uint64_t hrSec = std::chrono::duration_cast<std::chrono::seconds>( std::chrono::steady_clock::now() - hrTimerStart ).count();
            if ((hrSec >= hrTimerIntervalSec) && (numSamples > 0))
            {
                uint64_t avg = computeAverage(profSamples.data(), profSamples.size(), numSamples);
                double timeSec = ((double)avg) * 0.000000001;
                double hashesPerSec = ((double)workSize) / timeSec;
                
                // Mhash/s version. Not precise enough with the new algorithm version.
                //double hs = hashesPerSec * 0.000001;
                //applog(LOG_INFO, "cl_device(%d): hashrate: %.02f Mhash/s", thr_id, hs);
                
                double hs = hashesPerSec *0.001;                
                applog(LOG_INFO, "cl_device(%d): hashrate: %.02f kH/s", thr_id, hs);

                // update total hash-rate
                mtx_lock(&stats_lock);
                thr_hashrates[thr_id] = hs;
                mtx_unlock(&stats_lock);

                // reset timer
                hrTimerStart = std::chrono::steady_clock::now();
            }

            //-----------------------------------
            // Wait pipeline to finish
            errorCode = clFinish(clCommandQueue);
            if (errorCode != CL_SUCCESS)
            {
                applog(LOG_ERR, "cl_device(%d):Device not responding. error code: %d. Terminaning worker...", thr_id, errorCode);
                goto out;
            }

#ifdef VERTHASH_FULL_VALIDATION
            //-------------------------------------
            // Retrieve device data
            verthashIORES.clear();
            errorCode = clEnqueueReadBuffer(clCommandQueue, clmemResults, CL_TRUE, 0, workSize * sizeof(uvec8), verthashIORES.data(), 0, nullptr, nullptr);
            if (errorCode != CL_SUCCESS)
            {
                applog(LOG_ERR, "cl_device(%d):Failed to read a 'hash_result' buffer.", thr_id);
                goto out;
            }

            //-------------------------------------
            // record a profiler sample
            auto end = std::chrono::steady_clock::now();
            profSamples[sampleIndex] = std::chrono::duration<uint64_t, std::nano>(end - start).count();
            sampleIndex = (sampleIndex + 1) & (profSamples.size()-1);
            ++numSamples;
            if (numSamples > profSamples.size()) { numSamples = profSamples.size(); }

            //-------------------------------------
            // Submit results
            for (size_t i = 0; i < workSize; ++i)
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
                applog(LOG_ERR, "cl_device(%d):Failed to read a 'hash_target_results' buffer.", thr_id);
                goto out;
            }
            uint32_t potentialResultCount = testResult[0];
            uint32_t potentialResult = testResult[1];


            //-------------------------------------
            // Check if at least 1 potential nonce was found
            if (potentialResultCount != 0)
            {
                if (opt_debug)
                    applog(LOG_DEBUG, "cl_device(%d):Potential result count = %u", thr_id, potentialResultCount);

                uvec8 hashResult;
                // get latest hash result from device
                clEnqueueReadBuffer(clCommandQueue, clmemResults, CL_TRUE, sizeof(uvec8)*potentialResult, sizeof(uvec8), &hashResult, 0, nullptr, nullptr);
                // test it against target
                if (fulltestUvec8(hashResult, workInfo.target))
                {
                    // add nonce local offset
                    results.push_back(potentialResult + nonce);
                }

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
                        clEnqueueReadBuffer(clCommandQueue, clmemResults, CL_TRUE, sizeof(uvec8)*potentialResults[g], sizeof(uvec8), &hashResult, 0, nullptr, nullptr);

                        if (fulltestUvec8(hashResult, workInfo.target))
                        {
                            // add nonce local offset
                            results.push_back(potentialResults[g] + nonce); 
                        }
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
            sampleIndex = (sampleIndex + 1) & (profSamples.size()-1);
            ++numSamples;
            if (numSamples > profSamples.size()) { numSamples = profSamples.size(); }

            //-------------------------------------
            // Send results
            for (size_t i = 0; i < results.size(); ++i)
            {
                workInfo.data[19] = results[i]; // HTarget results

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
            // max runs limit
            ++runs;
            if (runs >= maxRuns)
            {
                applog(LOG_INFO, "cl_device(%d):Device has completed its nonce range.", thr_id);
                fflush(stdout);

                break;
            }
            // update nonce
            nonce += workSize;

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
// verthashIO monolothic kernel
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
    int cudevice;
    size_t workSize;
};

static int verthashCuda_thread(void *userdata)
{
    cuworker_t* cuworker = (cuworker_t*)userdata;
    int cudevice = cuworker->cudevice;
    int cuWorkerIndex = cuworker->threadInfo->id - cuDeviceIndexOffset; 
    
    //-------------------------------------
    // num runs to reach max nonce
    const uint64_t numNoncesGlobal = 4294967296ULL;
    uint32_t maxRunsGlobal = (uint32_t)(numNoncesGlobal / (uint32_t)cuworker->workSize);
    size_t workSize = cuworker->workSize;
    // verthash kernel run configuration
    int cudaThreadsPerBlock = 64;
    int cudaBlocksPerGrid = (workSize + cudaThreadsPerBlock - 1) / cudaThreadsPerBlock;

    //------------------------------
    struct thr_info *mythr = cuworker->threadInfo;
    int thr_id = mythr->id;
    struct work workInfo = { { 0 } };
    
    const uint64_t numNoncesPerDevice = 4294967296ULL / (uint64_t)opt_n_threads;
    uint32_t maxRunsPerDevice = (uint32_t)(numNoncesPerDevice / (uint32_t)workSize);
    // last device
    if (thr_id == (opt_n_threads - 1))
    {
        maxRunsPerDevice += (uint32_t)(numNoncesPerDevice % opt_n_threads);
    }
    
    uint32_t numRuns = 0;

#ifdef VERTHASH_FULL_VALIDATION
    // Host side hash storage
    std::vector<uvec8> verthashIORES;
    verthashIORES.resize(workSize);
#else
    // HTarg result host side storage
    std::vector<uint32_t> results;
    std::vector<uint32_t> potentialResults; // used if (potentialNonceCount > 1)
#endif

    // init per device profiling data
    const size_t maxProfSamples = 32;
    size_t numSamples = 0;
    size_t sampleIndex = 0;
    std::vector<uint64_t> profSamples(maxProfSamples, 0);

    // per hash-rate update timer 
    std::chrono::steady_clock::time_point hrTimerStart;
    uint64_t hrTimerIntervalSec = 4; // TODO: make configurable
    
    //------------------------------
    // Init CUDA data
    cudaError_t cuerr;
    // buffers
    uint32_t* dmemKStates = NULL;
    uint32_t* dmemFullDat = NULL;
    uint32_t* dmemResults = NULL;
    uint32_t* dmemHTargetResults = NULL;

    cuerr = cudaSetDevice(cuworker->cudevice);
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
    cuerr = cudaMalloc((void**)&dmemResults, workSize * sizeof(uvec8));
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
            //if (memcmp(workInfo.data, g_work.data, 76))
            {
                stratum_gen_work(&stratum, &g_work);
            }

            maxRuns = maxRunsGlobal;
            runs = 0;
            nonce = 0;
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

            const uint32_t offsetN = (maxRunsPerDevice * workSize)*thr_id;
            const uint32_t first_nonce = offsetN + (numRuns * workSize);

            maxRuns = maxRunsPerDevice;
            runs = 0;
            nonce = first_nonce;
        }

        // create a work copy
        work_free(&workInfo);
        work_copy(&workInfo, &g_work);
        workInfo.data[19] = 0; // TODO: needed?

        mtx_unlock(&g_work_lock);
        work_restart[thr_id].restart = 0;

#ifdef VERTHASH_EXTENDED_VALIDATION
        uint64_t wtarget = ((uint64_t(workInfo.target[7])) << 32) | (uint64_t(workInfo.target[6]) & 0xFFFFFFFFUL);
#else
        uint32_t wtarget = workInfo.target[7];
#endif
        //-------------------------------------
        // Generate midstate
        // TODO: optimize. Check on every new block
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


        // TODO: sort runs and nonce is the same, but different metrics
        //uint32_t runs = 0;
        //uint32_t nonce = 0;

        //-------------------------------------
        // Compute hashes

        // abort flag is needed here to prevent an inconsistent behaviour
        // in case program termination was triggered between work generation and resetting work restart values
        while (!work_restart[thr_id].restart && (!abort_flag))
        {
            //printf("tid %d, First nonce: %u, maxRuns: %u\n", thr_id, nonce, maxRuns);

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
            // compute average time from samples asynchronously
            uint64_t hrSec = std::chrono::duration_cast<std::chrono::seconds>( std::chrono::steady_clock::now() - hrTimerStart ).count();
            if ((hrSec >= hrTimerIntervalSec) && (numSamples > 0))
            {
                uint64_t avg = computeAverage(profSamples.data(), profSamples.size(), numSamples);
                double timeSec = ((double)avg) * 0.000000001;
                double hashesPerSec = ((double)workSize) / timeSec;
                //double hs = hashesPerSec * 0.000001;
                //applog(LOG_INFO, "cu_device(%d): hashrate: %.02f Mhash/s", thr_id, hs);

                double hs = hashesPerSec * 0.001;
                applog(LOG_INFO, "cu_device(%d): hashrate: %.02f kH/s", thr_id, hs);

                // update total hash-rate
                mtx_lock(&stats_lock);
                thr_hashrates[thr_id] = hs;
                mtx_unlock(&stats_lock);

                // reset timer
                hrTimerStart = std::chrono::steady_clock::now();
            }

            //-----------------------------------
            // Wait pipeline to finish
            cuerr = cudaDeviceSynchronize();
            if (cuerr != cudaSuccess)
            {
                applog(LOG_ERR, "cu_device(%d):Device not responding. error code: %d. Terminaning worker...", cuWorkerIndex, cuerr);
                goto out;
            }


#ifdef VERTHASH_FULL_VALIDATION
            //-------------------------------------
            // Retrieve device data
            verthashIORES.clear();
            cuerr = cudaMemcpy(verthashIORES.data(), dmemResults, workSize * sizeof(uvec8), cudaMemcpyDeviceToHost);
            if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to download hash data. error code: %d", cuWorkerIndex, cuerr); goto out; }

            //-------------------------------------
            // Record a profiler sample
            auto end = std::chrono::steady_clock::now();
            profSamples[sampleIndex] = std::chrono::duration<uint64_t, std::nano>(end - start).count();
            sampleIndex = (sampleIndex + 1) & (profSamples.size()-1);
            ++numSamples;
            if (numSamples > profSamples.size()) { numSamples = profSamples.size(); }

            //-------------------------------------
            // Submit results
            for (size_t i = 0; i < workSize; ++i)
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

                uvec8 hashResult;
                // get latest hash result from device
                cuerr = cudaMemcpy(&hashResult, dmemResults+(potentialResult*8), sizeof(uvec8), cudaMemcpyDeviceToHost);
                if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to get a potential hash result. error code: %d", cuWorkerIndex, cuerr); goto out; }
                // test it against target
                if (fulltestUvec8(hashResult, workInfo.target))
                {
                    // add nonce local offset
                    results.push_back(potentialResult + nonce);
                }

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
                        cuerr = cudaMemcpy(&hashResult, dmemResults+(potentialResults[g]*8), sizeof(uvec8), cudaMemcpyDeviceToHost);
                        if (cuerr != cudaSuccess) { applog(LOG_ERR, "cu_device(%d):Failed to get a potential hash result(%llu). error code: %d", cuWorkerIndex, cuerr, (uint64_t)g); goto out; }

                        if (fulltestUvec8(hashResult, workInfo.target))
                        {
                            // add nonce local offset
                            results.push_back(potentialResults[g] + nonce); 
                        }
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
            // max runs limit
            ++runs;
            if (runs >= maxRuns)
            {
                applog(LOG_INFO, "cu_device(%d):Device has completed its nonce range!!!", cuWorkerIndex);
                break;
            }
            // update nonce
            nonce += workSize;

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
        goto out;
    applog(LOG_INFO, "Starting Stratum on %s", stratum.url);

    while (!abort_flag)
    {
        int failures = 0;

        while (!stratum.curl && !abort_flag)
        {
            mtx_lock(&g_work_lock);
            g_work_time = 0;
            mtx_unlock(&g_work_lock);
            restart_threads();

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

        if (stratum.job.job_id &&
            (!g_work_time || strcmp(stratum.job.job_id, g_work.job_id)))
        {
            mtx_lock(&g_work_lock);
            stratum_gen_work(&stratum, &g_work);

            // Due to session restore feature, first mining.notify message may not request miner to clean previous jobs.
            // Thus 0 block height check is needed here too.
            if (stratum.job.clean /*|| (verthashInfo.blockHeight == 0)*/)
            {
                //-------------------------------------
                // Update verthash data if needed
               /* uint32_t blockHeight = stratum_get_block_height(&stratum);
                int errorCode = verthash_info_update_data(&verthashInfo, blockHeight);
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

                    if (!abort_flag) { abort_flag = true; }
                    mtx_unlock(&g_work_lock);

                    goto out;
                }*/
                //-------------------------------------

                applog(LOG_INFO, "Stratum requested work restart");
                restart_threads();
            }

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
};

struct cmd_result_t
{
    // print device list in raw mode
    bool printDeviceList;

    // automatic config generation
    bool generateConfigFile;
    bool rawDeviceList;
    char* generateConfigFileName;

    // select config file
    bool useConfigFile;
    char* useConfigFileName;

    // restrict Nvidia GPUs to CUDA backend
    bool restrictNVGPUToCUDA;

    // verthash
    char* verthashDataFileName;
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
    cmdr->rawDeviceList = false;
    cmdr->generateConfigFileName = NULL;

    cmdr->useConfigFile = false;
    cmdr->useConfigFileName = NULL;

#ifdef HAVE_CUDA
    cmdr->restrictNVGPUToCUDA = true;
#else
    cmdr->restrictNVGPUToCUDA = false;
#endif

    cmdr->verthashDataFileName = NULL;
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
        case 'G':
            cmdr->rawDeviceList = true;
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
                sel.workSize = vh::defaultWorkSize; // default work size
                while (0 != (token = strsep(&tokenBase, delims2)) && (paramIndex < 2))
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
                sel.workSize = vh::defaultWorkSize; // default work size
                while (0 != (token = strsep(&tokenBase, delims2)) && (paramIndex < 2))
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
                fprintf(stderr, "Error: %s: invalid address -- '%s'\n",
                       pname, arg);
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


int main(int argc, char *argv[])
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
        applog_file = fopen(logFileName, "w");
        if (!applog_file)
        {
            applog(LOG_WARNING, "Failed to open file for logging(%s).", logFileName);
            applog(LOG_WARNING, "Logging to file is not available.");
        }
    }

    // get raw device list options, which can be modified later depending on supported extensions
    bool rawDeviceList = cmdr.rawDeviceList;
    //-------------------------------------
    // OpenCL init
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
    // logical device list sorted by PCIe bus ID
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
            if (cmdr.restrictNVGPUToCUDA)
            {
                applog(LOG_WARNING, "Skipping CL platform (index: %u, %s)", i, infoString.c_str());
                continue;
            }
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

        cl_device_topology_amd topology;
        for (size_t j = 0; j < deviceIds.size(); ++j)
        {
            vh::cldevice_t cldevice;
            cldevice.clPlatformId = clplatformIds[i];
            cldevice.clId = deviceIds[j];
            cldevice.platformIndex = (int32_t)i;
            cldevice.binaryFormat = vh::BF_None;
            cldevice.asmProgram = vh::AP_None;
            cldevice.vendor = vendor;
        
            if (cldevice.vendor == vh::V_AMD)
            {
                cl_int status = clGetDeviceInfo(deviceIds[j], CL_DEVICE_TOPOLOGY_AMD,
                                                sizeof(cl_device_topology_amd), &topology, nullptr);
                if(status == CL_SUCCESS)
                {
                    if (topology.raw.type == CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD)
                    {
                        cldevice.pcieBusId = (int32_t)topology.pcie.bus;
                    }
                }
                else // if extension is not supported
                {
                    applog(LOG_WARNING, "Failed to get CL_DEVICE_TOPOLOGY_AMD info"
                                        "(possibly unsupported extension). Platform index: %u", i);
                    cldevice.pcieBusId = -1;
                    // fallback to raw device list
                    rawDeviceList = true;
                }
            }
            else if (cldevice.vendor == vh::V_NVIDIA) 
            {
                cl_int nvpciBus = -1;
                cl_int status = clGetDeviceInfo(deviceIds[j], 0x4008, sizeof(cl_int), &nvpciBus, NULL);
                if(status == CL_SUCCESS)
                {
                    cldevice.pcieBusId = (int32_t)nvpciBus;
                }
                else
                {
                    applog(LOG_WARNING, "Failed to get NV_PCIE_BUS_ID info"
                                        "(possibly unsupported extension). Platform index: %u", i);
                    cldevice.pcieBusId = -1;
                    // fallback to raw device list
                    rawDeviceList = true;
                }
            }
            else // V_OTHER
            {
                cldevice.pcieBusId = -1;
                // fallback to raw device list
                rawDeviceList = true;
            }
                
            cldevices.push_back(cldevice);
        }
    }

    if(numCLPlatformIDs > 0)
    {
        //-----------------------------------------------------------------------------
        // sort device list based on pcieBusID -> platformID
        std::sort(cldevices.begin(), cldevices.end(), vh::compareLogicalDevices);

        applog(LOG_INFO, "Found %llu OpenCL devices.", (uint64_t)cldevices.size());
    }

#ifdef HAVE_CUDA
    //-------------------------------------
    // CUDA init
    int cudeviceListSize = 0;
    cudaError_t cuerr = cudaGetDeviceCount(&cudeviceListSize);
    if (cuerr == cudaSuccess)
    {
        applog(LOG_INFO, "Found %d CUDA devices", cudeviceListSize);
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
        puts("\nDevice list(raw):");
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

                printf("\tIndex: %u. Name: %s\n\t"
                    "          Platform index: %u\n\t"
                    "          Platform name: %s\n\t"
                    "          pcieBusId: %d\n\n",
                    (uint32_t)i, infoString0.c_str(), cldevices[i].platformIndex, infoString1.c_str(), cldevices[i].pcieBusId);
            }
        }
        else
        {
            puts("OpenCL devices: None\n");
        }
#ifdef HAVE_CUDA
        // print CUDA devices
        
        if (cudeviceListSize > 0)
        {
            puts("CUDA devices:");
            for (int i = 0; i < cudeviceListSize; ++i)
            {
                cudaDeviceProp cudeviceProp;
                cudaGetDeviceProperties(&cudeviceProp, i);
                printf("\tIndex: %d. Name: %s\n", i, cudeviceProp.name);
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

    // check for duplicate PCIeBusIds(rare case), only if raw device list is disabled
    if (!rawDeviceList)
    {
        int32_t prevPlatformIndex = -1;
        int32_t prevPcieId = -1;
        for (size_t i = 0; i < cldevices.size(); ++i)
        {
            if((cldevices[i].pcieBusId == prevPcieId) &&
               (cldevices[i].platformIndex == prevPlatformIndex))
            {
                if (cmdr.generateConfigFile)
                {
                    applog(LOG_WARNING, "Failed to group logical devices by PCIe slot ID. Duplicates Found!");
                }
                rawDeviceList = true;
                break;
            }
            else
            {
                prevPlatformIndex = cldevices[i].platformIndex;
                prevPcieId = cldevices[i].pcieBusId;
            }
        }
    }

    // Config generation
    if (cmdr.generateConfigFile)
    {
        if (rawDeviceList)
        {
            applog(LOG_INFO, "Configuration file device list format: \"raw device list\"");
            applog(LOG_WARNING, "All possible OpenCL device/platform combinations will be listed.");
        }
        else
        {
            applog(LOG_INFO, "Configuration file device list format: \"pcie bus id\"");
        }

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
        vh::generateCLDeviceConfig(clplatformIds, cldevices, rawDeviceList, configText); 
#ifdef HAVE_CUDA
        configText += "\n";
        // CUDA devices
        vh::generateCUDADeviceConfig(cudeviceListSize, rawDeviceList, configText);
#endif
        // save a config file.
        if (fileExists(cmdr.generateConfigFileName))
        {
            applog(LOG_ERR, "Failed to create a configuration file. %s already exists.", cmdr.generateConfigFileName);
            cmd_result_free(&cmdr);
            return 1;
        }
        std::ofstream cfgOutput(cmdr.generateConfigFileName); 
        cfgOutput << configText;
        cfgOutput.close();

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
            uint32_t workSize = cmdr.selectedCLDevices[i].workSize;
            // validate parameters
            if (cmdr.selectedCLDevices[i].deviceIndex >= cldevices.size()) 
            {
                applog(LOG_ERR, "Invalid CL device index: %u", cmdr.selectedCLDevices[i].deviceIndex);
                cmd_result_free(&cmdr);
                return 1;

            }
            else if ( (workSize == 0) || ((workSize % 256) != 0))
            {
                // TODO: rewrite
                applog(LOG_WARNING, "Invalid CL Device \"WorkSize\" parameter(index: %u, workSize: %u)",
                       cmdr.selectedCLDevices[i].deviceIndex, workSize);
                applog(LOG_WARNING, "Setting CL Device\"WorkSize\" to default: %u", vh::defaultWorkSize);
                workSize = vh::defaultWorkSize;
            }

            clworker_t clworker;
            clworker.cldevice = cldevices[cmdr.selectedCLDevices[i].deviceIndex];
            clworker.workSize = workSize;
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
            clworker.cldevice = cldevices[i];
            clworker.workSize = vh::defaultWorkSize;
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
            uint32_t workSize = cmdr.selectedCUDevices[i].workSize;
            // validate parameters
            if (cmdr.selectedCUDevices[i].deviceIndex >= cudeviceListSize) 
            {
                applog(LOG_ERR, "Invalid CUDA device index: %u", cmdr.selectedCUDevices[i].deviceIndex);
                cmd_result_free(&cmdr);
                return 1;

            }
            else if ( (workSize == 0) || ((workSize % 256) != 0))
            {
                // TODO: rewrite
                applog(LOG_WARNING, "Invalid CUDA Device \"WorkSize\" parameter(index: %u, workSize: %u)",
                       cmdr.selectedCUDevices[i].deviceIndex, workSize);
                applog(LOG_WARNING, "Setting CUDA Device\"WorkSize\" to default: %u", vh::defaultWorkSize);
                workSize = vh::defaultWorkSize;
            }

            cuworker_t cuworker;
            cuworker.cudevice = cmdr.selectedCUDevices[i].deviceIndex;
            cuworker.workSize = workSize;
            cuworkers.push_back(cuworker);
        }
    }
    else // add all CUDA devices
    {
        for (int i = 0; i < cudeviceListSize; ++i)
        {
            cuworker_t cuworker;
            cuworker.cudevice = i;
            cuworker.workSize = vh::defaultWorkSize;
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


        if ((!cmdr.selectAllCLDevices) && (!cmdr.selectedCLDevices.size()))
        {
            //-----------------------------------------------------------------------------
            // OpenCL device section.


            // TODO: AsmProgram and BinaryFormat automatic detection

            //-------------------------------------
            // setup devices
            size_t deviceBlockIndex = 0;
            std::string dCLBlockName("CL_Device");
            std::string binaryFormatName;
            std::string asmProgramName;
            std::string deviceBlock = dCLBlockName + std::to_string(deviceBlockIndex);
            
            // check if configuration file is in "raw device list" format
            csetting = cf.getSetting(deviceBlock.c_str(), "PCIeBusId");
            if (!csetting)
                rawDeviceList = true;

            // PCIeBusId is the only required setting, others are optional
            if (!rawDeviceList)
            {
                while ( (csetting = cf.getSetting(deviceBlock.c_str(), "PCIeBusId")) )
                {
                    int pcieBusId = csetting->AsInt;
                    int platformIndex = -1;
                    int workSize = 0;
                    vh::EBinaryFormat binaryFormat = vh::BF_None;
                    vh::EAsmProgram asmProgram = vh::AP_None; 

                    // get platform index
                    csetting = cf.getSetting(deviceBlock.c_str(), "PlatformIndex"); 
                    if (csetting) platformIndex = csetting->AsInt;

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

                    // get work size flag
                    csetting = cf.getSetting(deviceBlock.c_str(), "WorkSize"); 
                    if (csetting) workSize = csetting->AsInt;

                    // check if pcieID and platfromIndex are correct
                    ptrdiff_t foundByPCIeBusId = -1;
                    ptrdiff_t foundByPlatformIndex = -1;
                    for (size_t i = 0; i < cldevices.size(); ++i)
                    {
                        vh::cldevice_t& dv = cldevices[i];
                        if (dv.pcieBusId == pcieBusId) 
                        {
                            foundByPCIeBusId = i;
                            if (dv.platformIndex == platformIndex) 
                            {
                                foundByPlatformIndex = i;
                                break;
                            }
                        }
                        // check all logical devices in case they were sorted wrong...
                    }

                    // add a configured device if it was found
                    if (foundByPCIeBusId >= 0)
                    {
                        clworker_t clworker;
                        if (foundByPlatformIndex >= 0)
                            clworker.cldevice = cldevices[foundByPlatformIndex];
                        else
                        {
                            // unlike pcieBusId, platform index may not be static.
                            applog(LOG_WARNING, "\"PlatformIndex\" parameter is not set or incorrect"
                                                "ver inside \"%s\" section. Using default(%d).",
                                   deviceBlock.c_str(), cldevices[foundByPCIeBusId].platformIndex);

                            clworker.cldevice = cldevices[foundByPCIeBusId];
                        }

                        // check if workSize is set correct.
                        // TODO: review for other vendors/drivers
                        if ( (workSize > 0) && ((workSize % 256) == 0) )
                            clworker.workSize = workSize;
                        else
                        {
                            applog(LOG_WARNING, "\"WorkSize\" parameter is not set or incorrect inside \"%s\" section."
                                                "It must be multiple of 256. Using default(%u).",
                                   deviceBlock.c_str(), vh::defaultWorkSize);

                            clworker.workSize = vh::defaultWorkSize;
                        }

                        // AsmProgram will be detected on context init
                        clworker.cldevice.binaryFormat = binaryFormat;
                        clworker.cldevice.asmProgram = asmProgram;

                        clworkers.push_back(clworker);
                    }
                    else
                    {
                        if ((platformIndex >= 0) && (platformIndex < clplatformIds.size()))
                        {
                            // check (maybe config file was generated with --no-restrict-cuda)
                            size_t pVendorStringSize = 0;
                            clGetPlatformInfo(clplatformIds[platformIndex], CL_PLATFORM_VENDOR, 0, nullptr, &pVendorStringSize);
                            std::string pVendorString(pVendorStringSize, ' ');
                            clGetPlatformInfo(clplatformIds[platformIndex], CL_PLATFORM_VENDOR, pVendorStringSize, (void*)pVendorString.data(), nullptr);

                            if (pVendorString.find(platformVendorNV) != std::string::npos) 
                            {
                                if (!cmdr.restrictNVGPUToCUDA)
                                {
                                    applog(LOG_WARNING, "It looks like %s was configured with --no-restrict-cuda parameter. Skipping device...", deviceBlock.c_str());
                                    applog(LOG_WARNING, "Please restart application with '--no-restrict-cuda'");
                                }
                            }
                        }
                        else
                        {
                            // It seems config file is just wrong.
                            applog(LOG_WARNING, "\"PCIeBusId\" is invalid inside \"%s\" section. Skipping device...",
                                   deviceBlock.c_str());
                        }
                    }

                    ++deviceBlockIndex;
                    deviceBlock = dCLBlockName + std::to_string(deviceBlockIndex);
                }
            }
            else
            {
                while( (csetting = cf.getSetting(deviceBlock.c_str(), "DeviceIndex")) )
                {
                    size_t deviceIndex = (size_t)csetting->AsInt;
                    int workSize = 0;
                    vh::EBinaryFormat binaryFormat = vh::BF_None;
                    vh::EAsmProgram asmProgram = vh::AP_None; 

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

                    // get work size flag
                    csetting = cf.getSetting(deviceBlock.c_str(), "WorkSize"); 
                    if (csetting) workSize = csetting->AsInt;

                    // check if pcieBusId and platfromIndex are correct
                    if ((deviceIndex < cldevices.size()) && (deviceIndex >= 0))
                    {
                        clworker_t clworker;
                        clworker.cldevice = cldevices[deviceIndex];

                        // check if workSize is set correct.
                        // TODO: review for other vendors/drivers
                        if ( (workSize > 0) && ((workSize % 256) == 0) )
                            clworker.workSize = workSize;
                        else
                        {
                            applog(LOG_WARNING, "\"WorkSize\" parameter is not set or incorrect inside \"%s\" section."
                                                " It must be multiple of 256. Using default(%u).",
                                   deviceBlock.c_str(), vh::defaultWorkSize);

                            clworker.workSize = vh::defaultWorkSize;
                        }

                        // AsmProgram will be detected on context init
                        clworker.cldevice.binaryFormat = binaryFormat;
                        clworker.cldevice.asmProgram = asmProgram;
                        clworkers.push_back(clworker);
                    }
                    else
                    {
                        applog(LOG_WARNING, "\"DeviceIndex\" is invalid inside \"%s\" section. Skipping device...", deviceBlock.c_str());
                    }

                    ++deviceBlockIndex;
                    deviceBlock = dCLBlockName + std::to_string(deviceBlockIndex);
                }
            } // end !rawDeviceList
        } // end ((!cmdr.selectAllDevices) && (!cmdr.selectedCLDevices.size()))


#ifdef HAVE_CUDA
        if ((!cmdr.selectAllCUDevices) && (!cmdr.selectedCUDevices.size()))
        {
            //-----------------------------------------------------------------------------
            // CUDA device section.
            size_t deviceBlockIndex = 0;
            std::string dCUBlockName("CU_Device");
            std::string deviceBlock = dCUBlockName + std::to_string(deviceBlockIndex);

            while( (csetting = cf.getSetting(deviceBlock.c_str(), "DeviceIndex")) )
            {
                int deviceIndex = (size_t)csetting->AsInt;
                int workSize = 0;

                // get work size flag
                csetting = cf.getSetting(deviceBlock.c_str(), "WorkSize"); 
                if (csetting) workSize = csetting->AsInt;

                // check if pcieBusId and platfromIndex are correct
                if ((deviceIndex < cudeviceListSize) && (deviceIndex >= 0))
                {
                    cuworker_t cuworker;
                    cuworker.cudevice = deviceIndex;

                    // check if workSize is set correct.
                    // TODO: review for other vendors/drivers
                    if ( (workSize > 0) && ((workSize % 256) == 0) )
                        cuworker.workSize = workSize;
                    else
                    {
                        applog(LOG_WARNING, "\"WorkSize\" parameter is not set or incorrect inside \"%s\" section."
                                            " It must be multiple of 256. Using default(%u).",
                               deviceBlock.c_str(), vh::defaultWorkSize);

                        cuworker.workSize = vh::defaultWorkSize;
                    }

                    // AsmProgram will be detected on context init
                    cuworkers.push_back(cuworker);
                }
                else
                {
                    applog(LOG_WARNING, "\"DeviceIndex\" is invalid inside \"%s\" section. Skipping device...", deviceBlock.c_str());
                }

                ++deviceBlockIndex;
                deviceBlock = dCUBlockName + std::to_string(deviceBlockIndex);
            }
        }
#endif
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

        // CRASH, TODO: inster opencl worker
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

        // CRASH, TODO: inster opencl worker
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

    applog(LOG_INFO, "Application has been exited gracefully.");

    // close file logger if it was enabled
    if (applog_file)
    {
        fclose(applog_file);
    }

    return 0;
}
