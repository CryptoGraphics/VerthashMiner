
/*
 * Copyright 2018-2020 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#ifndef Utils_INCLUDE_ONCE
#define Utils_INCLUDE_ONCE

#include <sys/stat.h>
#include <assert.h>

#if ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
#define WANT_BUILTIN_BSWAP
#else
#define bswap_32(x) ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) \
                   | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#endif

static inline uint32_t swab32(uint32_t v)
{
#ifdef WANT_BUILTIN_BSWAP
    return __builtin_bswap32(v);
#else
    return bswap_32(v);
#endif
}

#ifdef HAVE_SYS_ENDIAN_H
#include <sys/endian.h>
#endif

#if !HAVE_DECL_BE64DEC
inline uint64_t be64dec(const void* pp)
{
    const uint8_t* p = (uint8_t const*)pp;

    return ((uint64_t)(p[7]) + ((uint64_t)(p[6]) << 8) +
        ((uint64_t)(p[5]) << 16) + ((uint64_t)(p[4]) << 24) +
        ((uint64_t)(p[3]) << 32) + ((uint64_t)(p[2]) << 40) +
        ((uint64_t)(p[1]) << 48) + ((uint64_t)(p[0]) << 56));
}
#endif

#if !HAVE_DECL_BE64ENC
inline void be64enc(void* pp, uint64_t x)
{
    uint8_t* p = (uint8_t*)pp;

    p[7] = x & 0xff;
    p[6] = (x >> 8) & 0xff;
    p[5] = (x >> 16) & 0xff;
    p[4] = (x >> 24) & 0xff;
    p[3] = (x >> 32) & 0xff;
    p[2] = (x >> 40) & 0xff;
    p[1] = (x >> 48) & 0xff;
    p[0] = (x >> 56) & 0xff;
}
#endif

#if !HAVE_DECL_BE32DEC
static inline uint32_t be32dec(const void *pp)
{
    const uint8_t *p = (uint8_t const *)pp;
    return ((uint32_t)(p[3]) + ((uint32_t)(p[2]) << 8) +
        ((uint32_t)(p[1]) << 16) + ((uint32_t)(p[0]) << 24));
}
#endif

#if !HAVE_DECL_LE32DEC
static inline uint32_t le32dec(const void *pp)
{
    const uint8_t *p = (uint8_t const *)pp;
    return ((uint32_t)(p[0]) + ((uint32_t)(p[1]) << 8) +
        ((uint32_t)(p[2]) << 16) + ((uint32_t)(p[3]) << 24));
}
#endif

#if !HAVE_DECL_BE32ENC
static inline void be32enc(void *pp, uint32_t x)
{
    uint8_t *p = (uint8_t *)pp;
    p[3] = x & 0xff;
    p[2] = (x >> 8) & 0xff;
    p[1] = (x >> 16) & 0xff;
    p[0] = (x >> 24) & 0xff;
}
#endif

#if !HAVE_DECL_LE32ENC
static inline void le32enc(void *pp, uint32_t x)
{
    uint8_t *p = (uint8_t *)pp;
    p[0] = x & 0xff;
    p[1] = (x >> 8) & 0xff;
    p[2] = (x >> 16) & 0xff;
    p[3] = (x >> 24) & 0xff;
}
#endif

#if !HAVE_DECL_LE16DEC
inline uint16_t le16dec(const void * pp)
{
    const uint8_t * p = (uint8_t const *)pp;

    return ((uint16_t)(p[0]) + ((uint16_t)(p[1]) << 8));
}
#endif

//----------------------------------------------------------------------------
inline bool fileExists(const char* file_name)
{
    struct stat buffer;   
    return (stat (file_name, &buffer) == 0); 
}
//----------------------------------------------------------------------------
inline uint64_t computeAverage(const uint64_t* samples, size_t max_samples, size_t num_samples)
{
    assert((num_samples > 0) && (num_samples <= max_samples));

    uint64_t avg = 0;
    for (size_t i = 0; i < num_samples; ++i)
    {
        avg += samples[i] / num_samples;
    }

    uint64_t avg2 = 0;
    for (size_t i = 0; i < num_samples; ++i)
    {
        avg2 += samples[i] % num_samples;
    }

    avg = avg + (avg2 / num_samples);
    return avg;
}
//----------------------------------------------------------------------------
inline char *strsep(char **str, const char *sep)
{
	char *s = *str, *end;
	if (!s)
        return NULL;

    end = s + strcspn(s, sep);

    if (*end)
        *end++ = 0;
	else
        end = 0;

    *str = end;

	return s;
}
//----------------------------------------------------------------------------

#endif // !Utils_INCLUDE_ONCE
