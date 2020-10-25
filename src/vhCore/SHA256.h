/*
 * Copyright 2020 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#ifndef SHA256_INCLUDE_ONCE
#define SHA256_INCLUDE_ONCE

#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string>

#include "Utils.h"

#define SHA256_OUTPUT_SIZE 32

struct sha256_state_t
{
    uint32_t s[8];
    unsigned char buf[64];
    uint64_t bytes;
};

inline void sha256_init(sha256_state_t* state);
inline void sha256_process(sha256_state_t* state, const unsigned char* data, size_t data_size);
inline void sha256_done(sha256_state_t* state, unsigned char hash[SHA256_OUTPUT_SIZE]);


// Internal implementation code.

inline uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) { return z ^ (x & (y ^ z)); }
inline uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (z & (x | y)); }
inline uint32_t Sigma0(uint32_t x) { return (x >> 2 | x << 30) ^ (x >> 13 | x << 19) ^ (x >> 22 | x << 10); }
inline uint32_t Sigma1(uint32_t x) { return (x >> 6 | x << 26) ^ (x >> 11 | x << 21) ^ (x >> 25 | x << 7); }
inline uint32_t sigma0(uint32_t x) { return (x >> 7 | x << 25) ^ (x >> 18 | x << 14) ^ (x >> 3); }
inline uint32_t sigma1(uint32_t x) { return (x >> 17 | x << 15) ^ (x >> 19 | x << 13) ^ (x >> 10); }

/** One round of SHA-256. */
void inline Round(uint32_t a, uint32_t b, uint32_t c, uint32_t& d, uint32_t e, uint32_t f, uint32_t g, uint32_t& h, uint32_t k)
{
    uint32_t t1 = h + Sigma1(e) + Ch(e, f, g) + k;
    uint32_t t2 = Sigma0(a) + Maj(a, b, c);
    d += t1;
    h = t1 + t2;
}

/** Perform a number of SHA-256 transformations, processing 64-byte chunks. */
inline void Transform(uint32_t* s, const unsigned char* chunk, size_t blocks)
{
    while (blocks--)
    {
        uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
        uint32_t w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15;

        Round(a, b, c, d, e, f, g, h, 0x428a2f98 + (w0 = be32dec(chunk + 0)));
        Round(h, a, b, c, d, e, f, g, 0x71374491 + (w1 = be32dec(chunk + 4)));
        Round(g, h, a, b, c, d, e, f, 0xb5c0fbcf + (w2 = be32dec(chunk + 8)));
        Round(f, g, h, a, b, c, d, e, 0xe9b5dba5 + (w3 = be32dec(chunk + 12)));
        Round(e, f, g, h, a, b, c, d, 0x3956c25b + (w4 = be32dec(chunk + 16)));
        Round(d, e, f, g, h, a, b, c, 0x59f111f1 + (w5 = be32dec(chunk + 20)));
        Round(c, d, e, f, g, h, a, b, 0x923f82a4 + (w6 = be32dec(chunk + 24)));
        Round(b, c, d, e, f, g, h, a, 0xab1c5ed5 + (w7 = be32dec(chunk + 28)));
        Round(a, b, c, d, e, f, g, h, 0xd807aa98 + (w8 = be32dec(chunk + 32)));
        Round(h, a, b, c, d, e, f, g, 0x12835b01 + (w9 = be32dec(chunk + 36)));
        Round(g, h, a, b, c, d, e, f, 0x243185be + (w10 = be32dec(chunk + 40)));
        Round(f, g, h, a, b, c, d, e, 0x550c7dc3 + (w11 = be32dec(chunk + 44)));
        Round(e, f, g, h, a, b, c, d, 0x72be5d74 + (w12 = be32dec(chunk + 48)));
        Round(d, e, f, g, h, a, b, c, 0x80deb1fe + (w13 = be32dec(chunk + 52)));
        Round(c, d, e, f, g, h, a, b, 0x9bdc06a7 + (w14 = be32dec(chunk + 56)));
        Round(b, c, d, e, f, g, h, a, 0xc19bf174 + (w15 = be32dec(chunk + 60)));

        Round(a, b, c, d, e, f, g, h, 0xe49b69c1 + (w0 += sigma1(w14) + w9 + sigma0(w1)));
        Round(h, a, b, c, d, e, f, g, 0xefbe4786 + (w1 += sigma1(w15) + w10 + sigma0(w2)));
        Round(g, h, a, b, c, d, e, f, 0x0fc19dc6 + (w2 += sigma1(w0) + w11 + sigma0(w3)));
        Round(f, g, h, a, b, c, d, e, 0x240ca1cc + (w3 += sigma1(w1) + w12 + sigma0(w4)));
        Round(e, f, g, h, a, b, c, d, 0x2de92c6f + (w4 += sigma1(w2) + w13 + sigma0(w5)));
        Round(d, e, f, g, h, a, b, c, 0x4a7484aa + (w5 += sigma1(w3) + w14 + sigma0(w6)));
        Round(c, d, e, f, g, h, a, b, 0x5cb0a9dc + (w6 += sigma1(w4) + w15 + sigma0(w7)));
        Round(b, c, d, e, f, g, h, a, 0x76f988da + (w7 += sigma1(w5) + w0 + sigma0(w8)));
        Round(a, b, c, d, e, f, g, h, 0x983e5152 + (w8 += sigma1(w6) + w1 + sigma0(w9)));
        Round(h, a, b, c, d, e, f, g, 0xa831c66d + (w9 += sigma1(w7) + w2 + sigma0(w10)));
        Round(g, h, a, b, c, d, e, f, 0xb00327c8 + (w10 += sigma1(w8) + w3 + sigma0(w11)));
        Round(f, g, h, a, b, c, d, e, 0xbf597fc7 + (w11 += sigma1(w9) + w4 + sigma0(w12)));
        Round(e, f, g, h, a, b, c, d, 0xc6e00bf3 + (w12 += sigma1(w10) + w5 + sigma0(w13)));
        Round(d, e, f, g, h, a, b, c, 0xd5a79147 + (w13 += sigma1(w11) + w6 + sigma0(w14)));
        Round(c, d, e, f, g, h, a, b, 0x06ca6351 + (w14 += sigma1(w12) + w7 + sigma0(w15)));
        Round(b, c, d, e, f, g, h, a, 0x14292967 + (w15 += sigma1(w13) + w8 + sigma0(w0)));

        Round(a, b, c, d, e, f, g, h, 0x27b70a85 + (w0 += sigma1(w14) + w9 + sigma0(w1)));
        Round(h, a, b, c, d, e, f, g, 0x2e1b2138 + (w1 += sigma1(w15) + w10 + sigma0(w2)));
        Round(g, h, a, b, c, d, e, f, 0x4d2c6dfc + (w2 += sigma1(w0) + w11 + sigma0(w3)));
        Round(f, g, h, a, b, c, d, e, 0x53380d13 + (w3 += sigma1(w1) + w12 + sigma0(w4)));
        Round(e, f, g, h, a, b, c, d, 0x650a7354 + (w4 += sigma1(w2) + w13 + sigma0(w5)));
        Round(d, e, f, g, h, a, b, c, 0x766a0abb + (w5 += sigma1(w3) + w14 + sigma0(w6)));
        Round(c, d, e, f, g, h, a, b, 0x81c2c92e + (w6 += sigma1(w4) + w15 + sigma0(w7)));
        Round(b, c, d, e, f, g, h, a, 0x92722c85 + (w7 += sigma1(w5) + w0 + sigma0(w8)));
        Round(a, b, c, d, e, f, g, h, 0xa2bfe8a1 + (w8 += sigma1(w6) + w1 + sigma0(w9)));
        Round(h, a, b, c, d, e, f, g, 0xa81a664b + (w9 += sigma1(w7) + w2 + sigma0(w10)));
        Round(g, h, a, b, c, d, e, f, 0xc24b8b70 + (w10 += sigma1(w8) + w3 + sigma0(w11)));
        Round(f, g, h, a, b, c, d, e, 0xc76c51a3 + (w11 += sigma1(w9) + w4 + sigma0(w12)));
        Round(e, f, g, h, a, b, c, d, 0xd192e819 + (w12 += sigma1(w10) + w5 + sigma0(w13)));
        Round(d, e, f, g, h, a, b, c, 0xd6990624 + (w13 += sigma1(w11) + w6 + sigma0(w14)));
        Round(c, d, e, f, g, h, a, b, 0xf40e3585 + (w14 += sigma1(w12) + w7 + sigma0(w15)));
        Round(b, c, d, e, f, g, h, a, 0x106aa070 + (w15 += sigma1(w13) + w8 + sigma0(w0)));

        Round(a, b, c, d, e, f, g, h, 0x19a4c116 + (w0 += sigma1(w14) + w9 + sigma0(w1)));
        Round(h, a, b, c, d, e, f, g, 0x1e376c08 + (w1 += sigma1(w15) + w10 + sigma0(w2)));
        Round(g, h, a, b, c, d, e, f, 0x2748774c + (w2 += sigma1(w0) + w11 + sigma0(w3)));
        Round(f, g, h, a, b, c, d, e, 0x34b0bcb5 + (w3 += sigma1(w1) + w12 + sigma0(w4)));
        Round(e, f, g, h, a, b, c, d, 0x391c0cb3 + (w4 += sigma1(w2) + w13 + sigma0(w5)));
        Round(d, e, f, g, h, a, b, c, 0x4ed8aa4a + (w5 += sigma1(w3) + w14 + sigma0(w6)));
        Round(c, d, e, f, g, h, a, b, 0x5b9cca4f + (w6 += sigma1(w4) + w15 + sigma0(w7)));
        Round(b, c, d, e, f, g, h, a, 0x682e6ff3 + (w7 += sigma1(w5) + w0 + sigma0(w8)));
        Round(a, b, c, d, e, f, g, h, 0x748f82ee + (w8 += sigma1(w6) + w1 + sigma0(w9)));
        Round(h, a, b, c, d, e, f, g, 0x78a5636f + (w9 += sigma1(w7) + w2 + sigma0(w10)));
        Round(g, h, a, b, c, d, e, f, 0x84c87814 + (w10 += sigma1(w8) + w3 + sigma0(w11)));
        Round(f, g, h, a, b, c, d, e, 0x8cc70208 + (w11 += sigma1(w9) + w4 + sigma0(w12)));
        Round(e, f, g, h, a, b, c, d, 0x90befffa + (w12 += sigma1(w10) + w5 + sigma0(w13)));
        Round(d, e, f, g, h, a, b, c, 0xa4506ceb + (w13 += sigma1(w11) + w6 + sigma0(w14)));
        Round(c, d, e, f, g, h, a, b, 0xbef9a3f7 + (w14 + sigma1(w12) + w7 + sigma0(w15)));
        Round(b, c, d, e, f, g, h, a, 0xc67178f2 + (w15 + sigma1(w13) + w8 + sigma0(w0)));

        s[0] += a;
        s[1] += b;
        s[2] += c;
        s[3] += d;
        s[4] += e;
        s[5] += f;
        s[6] += g;
        s[7] += h;
        chunk += 64;
    }
}


//-----------------------------------------------------------------------------
// SHA256 functions
//-----------------------------------------------------------------------------
void sha256_init(sha256_state_t* state)
{
    assert(state != NULL);

    state->s[0] = 0x6a09e667ul;
    state->s[1] = 0xbb67ae85ul;
    state->s[2] = 0x3c6ef372ul;
    state->s[3] = 0xa54ff53aul;
    state->s[4] = 0x510e527ful;
    state->s[5] = 0x9b05688cul;
    state->s[6] = 0x1f83d9abul;
    state->s[7] = 0x5be0cd19ul;

    memset(state->buf, 0, sizeof(state->buf));
    state->bytes = 0;
}

//-----------------------------------------------------------------------------
void sha256_process(sha256_state_t* state, const unsigned char* data, size_t data_size)
{
    assert(state != NULL);

    const unsigned char* end = data + data_size;
    size_t bufsize = state->bytes & 63; // bytes % 64

    if (bufsize && bufsize + data_size >= 64)
    {
        // Fill the buffer, and process it.
        memcpy(state->buf + bufsize, data, 64 - bufsize);
        state->bytes += 64 - bufsize;
        data += 64 - bufsize;
        Transform(state->s, state->buf, 1);
        bufsize = 0;
    }

    if (end - data >= 64)
    {
        size_t blocks = (end - data) / 64;
        Transform(state->s, data, blocks);
        data += (blocks << 6); // 64 * blocks;
        state->bytes += (blocks << 6); // 64 * blocks;
    }

    if (end > data)
    {
        // Fill the buffer with what remains.
        memcpy(state->buf + bufsize, data, end - data);
        state->bytes += end - data;
    }
}

//-----------------------------------------------------------------------------
void sha256_done(sha256_state_t* state, unsigned char hash[SHA256_OUTPUT_SIZE])
{
    static const unsigned char pad[64] = { 0x80 };
    unsigned char sizedesc[8];
    be64enc(sizedesc, state->bytes << 3);

    //st_sha256_update(state, pad, 1 + ((119 - (state->bytes % 64)) % 64));
    sha256_process(state, pad, 1 + ((119 - (state->bytes & 63)) & 63));

    sha256_process(state, sizedesc, 8);
    be32enc(hash, state->s[0]);
    be32enc(hash + 4, state->s[1]);
    be32enc(hash + 8, state->s[2]);
    be32enc(hash + 12, state->s[3]);
    be32enc(hash + 16, state->s[4]);
    be32enc(hash + 20, state->s[5]);
    be32enc(hash + 24, state->s[6]);
    be32enc(hash + 28, state->s[7]);
}

//-----------------------------------------------------------------------------

inline void sha256s(unsigned char hash[SHA256_OUTPUT_SIZE], const unsigned char* data, size_t data_size)
{
    sha256_state_t state;
    sha256_init(&state);
    sha256_process(&state, data, data_size);
    sha256_done(&state, hash);
}

inline void sha256d(unsigned char hash[SHA256_OUTPUT_SIZE], const unsigned char* data, size_t data_size)
{
    sha256s(hash, data, data_size);
    sha256s(hash, hash, SHA256_OUTPUT_SIZE);
}

#endif // !SHA256_INCLUDE_ONCE