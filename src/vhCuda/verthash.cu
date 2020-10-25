
/*
 * Copyright 2020 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

typedef unsigned int uint;
typedef unsigned long long uint64_t;

static __device__ __forceinline__ uint2 operator^ (uint2 a, uint2 b) { return make_uint2(a.x ^ b.x, a.y ^ b.y); }
static __device__ __forceinline__ uint2 operator& (uint2 a, uint2 b) { return make_uint2(a.x & b.x, a.y & b.y); }
static __device__ __forceinline__ uint2 operator~ (uint2 a) { return make_uint2(~a.x, ~a.y); }
static __device__ __forceinline__ void operator^= (uint2 &a, uint2 b) { a = a ^ b; }

static __device__ __forceinline__ uint64_t make_ulonglong(uint lo, uint hi)
{
#if __CUDA_ARCH__ >= 130
    return __double_as_longlong(__hiloint2double(hi, lo));
#else
    return (uint64_t)lo | (((uint64_t)hi) << 32);
#endif
}

__device__ uint2 rotl64(const uint2 w, const int offset)
{
    uint2 result;
    if (offset < 32)
    {
        result.y = ((w.y << offset) | (w.x >> (32 - offset)));
        result.x = ((w.x << offset) | (w.y >> (32 - offset)));
    }
    else
    {
        result.y = ((w.x << (offset - 32)) | (w.y >> (64 - offset)));
        result.x = ((w.y << (offset - 32)) | (w.x >> (64 - offset)));
    }

    return result;
}

__constant__ uint2 keccakf_rndc[24] = {
    { 0x00000001, 0x00000000 },{ 0x00008082, 0x00000000 },{ 0x0000808a, 0x80000000 },{ 0x80008000, 0x80000000 },
    { 0x0000808b, 0x00000000 },{ 0x80000001, 0x00000000 },{ 0x80008081, 0x80000000 },{ 0x00008009, 0x80000000 },
    { 0x0000008a, 0x00000000 },{ 0x00000088, 0x00000000 },{ 0x80008009, 0x00000000 },{ 0x8000000a, 0x00000000 },
    { 0x8000808b, 0x00000000 },{ 0x0000008b, 0x80000000 },{ 0x00008089, 0x80000000 },{ 0x00008003, 0x80000000 },
    { 0x00008002, 0x80000000 },{ 0x00000080, 0x80000000 },{ 0x0000800a, 0x00000000 },{ 0x8000000a, 0x80000000 },
    { 0x80008081, 0x80000000 },{ 0x00008080, 0x80000000 },{ 0x80000001, 0x00000000 },{ 0x80008008, 0x80000000 }
};

static __device__ __forceinline__  uint fnv1a(const uint a, const uint b)
{
    uint res = (a ^ b) * 0x1000193U;
    return res;
}

static __device__ __forceinline__  uint rotl32(const uint x, const uint n)
{
    return (((x) << (n)) | ((x) >> (32 - (n))));
}

// SHA3 precompute result
typedef union
{
    uint u[50];
    uint2 u2[25];
} kstate_t;

typedef union {
    uint2 u2[50];
} kstate2x_t;

// Hash result(used as output inside sha3_512_32 kernel and in/out inside IO kernel)
typedef union
{
    uint u[8];
    uint2 u2[4];
} hash8_t;

// Combined hash from 8 SHA3 passes
struct sha3_state_t
{
    union
    {
        uint u[128];
        uint2 u2[64];
        uint4 u4[32];
    };
};

// Header used both in sha3_512_precompute & sha3_512_32 kernels
__constant__ uint header[19];


__global__ void cu_sha3_512_precompute(uint* output)
{
    uint globalThId = blockDim.x * blockIdx.x + threadIdx.x;
    uint h0 = header[0]+(globalThId&7)+1;
    
    //-------------------------------------
    // sha3 init
    uint2 st[25] = { 0 };

    // header 72 bytes
    st[0].x = h0;         st[0].y = header[ 1];
    st[1].x = header[ 2]; st[1].y = header[ 3];
    st[2].x = header[ 4]; st[2].y = header[ 5];
    st[3].x = header[ 6]; st[3].y = header[ 7];
    st[4].x = header[ 8]; st[4].y = header[ 9];
    st[5].x = header[10]; st[5].y = header[11];
    st[6].x = header[12]; st[6].y = header[13];
    st[7].x = header[14]; st[7].y = header[15];
    st[8].x = header[16]; st[8].y = header[17];
    

    uint2 t[5];
    uint2 u[5];
    uint2 v, w;

    for (int r = 0; r < 24; r++)
    {
        // Theta
        t[0] = (st[0] ^ (st[5] ^ (st[10] ^ (st[15] ^ st[20]))));
        t[1] = (st[1] ^ (st[6] ^ (st[11] ^ (st[16] ^ st[21]))));
        t[2] = (st[2] ^ (st[7] ^ (st[12] ^ (st[17] ^ st[22]))));
        t[3] = (st[3] ^ (st[8] ^ (st[13] ^ (st[18] ^ st[23]))));
        t[4] = (st[4] ^ (st[9] ^ (st[14] ^ (st[19] ^ st[24]))));

        u[0] = rotl64(t[0], 1) ^ t[3];
        u[1] = rotl64(t[1], 1) ^ t[4];
        u[2] = rotl64(t[2], 1) ^ t[0];
        u[3] = rotl64(t[3], 1) ^ t[1];
        u[4] = rotl64(t[4], 1) ^ t[2];

        st[4] ^= u[0]; st[9] ^= u[0]; st[14] ^= u[0]; st[19] ^= u[0]; st[24] ^= u[0];
        st[0] ^= u[1]; st[5] ^= u[1]; st[10] ^= u[1]; st[15] ^= u[1]; st[20] ^= u[1];
        st[1] ^= u[2]; st[6] ^= u[2]; st[11] ^= u[2]; st[16] ^= u[2]; st[21] ^= u[2];
        st[2] ^= u[3]; st[7] ^= u[3]; st[12] ^= u[3]; st[17] ^= u[3]; st[22] ^= u[3];
        st[3] ^= u[4]; st[8] ^= u[4]; st[13] ^= u[4]; st[18] ^= u[4]; st[23] ^= u[4];

        // Rho Pi
        v = st[1];
        st[1] = rotl64(st[6], 44);
        st[6] = rotl64(st[9], 20);
        st[9] = rotl64(st[22], 61);
        st[22] = rotl64(st[14], 39);
        st[14] = rotl64(st[20], 18);
        st[20] = rotl64(st[2], 62);
        st[2] = rotl64(st[12], 43);
        st[12] = rotl64(st[13], 25);
        st[13] = rotl64(st[19], 8);
        st[19] = rotl64(st[23], 56);
        st[23] = rotl64(st[15], 41);
        st[15] = rotl64(st[4], 27);
        st[4] = rotl64(st[24], 14);
        st[24] = rotl64(st[21], 2);
        st[21] = rotl64(st[8], 55);
        st[8] = rotl64(st[16], 45);
        st[16] = rotl64(st[5], 36);
        st[5] = rotl64(st[3], 28);
        st[3] = rotl64(st[18], 21);
        st[18] = rotl64(st[17], 15);
        st[17] = rotl64(st[11], 10);
        st[11] = rotl64(st[7], 6);
        st[7] = rotl64(st[10], 3);
        st[10] = rotl64(v, 1);

        //  Chi
        v = st[0]; w = st[1]; st[0] = v ^ ((~w) & st[2]); st[1] = w ^ ((~st[2]) & st[3]); st[2] = st[2] ^ ((~st[3]) & st[4]); st[3] = st[3] ^ ((~st[4]) & v); st[4] = st[4] ^ ((~v) & w);
        v = st[5]; w = st[6]; st[5] = v ^ ((~w) & st[7]); st[6] = w ^ ((~st[7]) & st[8]); st[7] = st[7] ^ ((~st[8]) & st[9]); st[8] = st[8] ^ ((~st[9]) & v); st[9] = st[9] ^ ((~v) & w);
        v = st[10]; w = st[11]; st[10] = v ^ ((~w) & st[12]); st[11] = w ^ ((~st[12]) & st[13]); st[12] = st[12] ^ ((~st[13]) & st[14]); st[13] = st[13] ^ ((~st[14]) & v); st[14] = st[14] ^ ((~v) & w);
        v = st[15]; w = st[16]; st[15] = v ^ ((~w) & st[17]); st[16] = w ^ ((~st[17]) & st[18]); st[17] = st[17] ^ ((~st[18]) & st[19]); st[18] = st[18] ^ ((~st[19]) & v); st[19] = st[19] ^ ((~v) & w);
        v = st[20]; w = st[21]; st[20] = v ^ ((~w) & st[22]); st[21] = w ^ ((~st[22]) & st[23]); st[22] = st[22] ^ ((~st[23]) & st[24]); st[23] = st[23] ^ ((~st[24]) & v); st[24] = st[24] ^ ((~v) & w);

        //  Iota
        st[0] ^= keccakf_rndc[r];
    }

    kstate_t* kstate = (kstate_t *)(output + (50 * (globalThId)));

    // store result
    kstate->u2[0] = st[0];
    kstate->u2[1] = st[1];
    kstate->u2[2] = st[2];
    kstate->u2[3] = st[3];
    kstate->u2[4] = st[4];
    kstate->u2[5] = st[5];
    kstate->u2[6] = st[6];
    kstate->u2[7] = st[7];
    kstate->u2[8] = st[8];
    kstate->u2[9] = st[9];
    kstate->u2[10] = st[10];
    kstate->u2[11] = st[11];
    kstate->u2[12] = st[12];
    kstate->u2[13] = st[13];
    kstate->u2[14] = st[14];
    kstate->u2[15] = st[15];
    kstate->u2[16] = st[16];
    kstate->u2[17] = st[17];
    kstate->u2[18] = st[18];
    kstate->u2[19] = st[19];
    kstate->u2[20] = st[20];
    kstate->u2[21] = st[21];
    kstate->u2[22] = st[22];
    kstate->u2[23] = st[23];
    kstate->u2[24] = st[24];
}



__global__ void cu_sha3_512_256(uint* output, const uint in18, const uint firstNonce)
{
    uint globalThId = blockDim.x * blockIdx.x + threadIdx.x;
    uint nonce = firstNonce + globalThId;
    
    //-------------------------------------
    // sha3 init
    uint2 st[25] = { 0 };

    // header 72 bytes
    st[0].x = header[ 0]; st[0].y = header[ 1];
    st[1].x = header[ 2]; st[1].y = header[ 3];
    st[2].x = header[ 4]; st[2].y = header[ 5];
    st[3].x = header[ 6]; st[3].y = header[ 7];
    st[4].x = header[ 8]; st[4].y = header[ 9];
    st[5].x = header[10]; st[5].y = header[11];
    st[6].x = header[12]; st[6].y = header[13];
    st[7].x = header[14]; st[7].y = header[15];
    st[8].x = header[16]; st[8].y = header[17];
    
    // output size 32 bytes
    st[9].x ^= in18; st[9].y ^= nonce;
    st[10].x ^= 0x00000006U; // byte[80] ^= 0x06
    st[16].y ^= 0x80000000U; // byte[135] ^= 0x80


    uint2 t[5];
    uint2 u[5];
    uint2 v, w;

    for (int r = 0; r < 24; r++)
    {
        // Theta
        t[0] = (st[0] ^ (st[5] ^ (st[10] ^ (st[15] ^ st[20]))));
        t[1] = (st[1] ^ (st[6] ^ (st[11] ^ (st[16] ^ st[21]))));
        t[2] = (st[2] ^ (st[7] ^ (st[12] ^ (st[17] ^ st[22]))));
        t[3] = (st[3] ^ (st[8] ^ (st[13] ^ (st[18] ^ st[23]))));
        t[4] = (st[4] ^ (st[9] ^ (st[14] ^ (st[19] ^ st[24]))));

        u[0] = rotl64(t[0], 1) ^ t[3];
        u[1] = rotl64(t[1], 1) ^ t[4];
        u[2] = rotl64(t[2], 1) ^ t[0];
        u[3] = rotl64(t[3], 1) ^ t[1];
        u[4] = rotl64(t[4], 1) ^ t[2];

        st[4] ^= u[0]; st[9] ^= u[0]; st[14] ^= u[0]; st[19] ^= u[0]; st[24] ^= u[0];
        st[0] ^= u[1]; st[5] ^= u[1]; st[10] ^= u[1]; st[15] ^= u[1]; st[20] ^= u[1];
        st[1] ^= u[2]; st[6] ^= u[2]; st[11] ^= u[2]; st[16] ^= u[2]; st[21] ^= u[2];
        st[2] ^= u[3]; st[7] ^= u[3]; st[12] ^= u[3]; st[17] ^= u[3]; st[22] ^= u[3];
        st[3] ^= u[4]; st[8] ^= u[4]; st[13] ^= u[4]; st[18] ^= u[4]; st[23] ^= u[4];

        // Rho Pi
        v = st[1];
        st[1] = rotl64(st[6], 44);
        st[6] = rotl64(st[9], 20);
        st[9] = rotl64(st[22], 61);
        st[22] = rotl64(st[14], 39);
        st[14] = rotl64(st[20], 18);
        st[20] = rotl64(st[2], 62);
        st[2] = rotl64(st[12], 43);
        st[12] = rotl64(st[13], 25);
        st[13] = rotl64(st[19], 8);
        st[19] = rotl64(st[23], 56);
        st[23] = rotl64(st[15], 41);
        st[15] = rotl64(st[4], 27);
        st[4] = rotl64(st[24], 14);
        st[24] = rotl64(st[21], 2);
        st[21] = rotl64(st[8], 55);
        st[8] = rotl64(st[16], 45);
        st[16] = rotl64(st[5], 36);
        st[5] = rotl64(st[3], 28);
        st[3] = rotl64(st[18], 21);
        st[18] = rotl64(st[17], 15);
        st[17] = rotl64(st[11], 10);
        st[11] = rotl64(st[7], 6);
        st[7] = rotl64(st[10], 3);
        st[10] = rotl64(v, 1);

        //  Chi
        v = st[0]; w = st[1]; st[0] = v ^ ((~w) & st[2]); st[1] = w ^ ((~st[2]) & st[3]); st[2] = st[2] ^ ((~st[3]) & st[4]); st[3] = st[3] ^ ((~st[4]) & v); st[4] = st[4] ^ ((~v) & w);
        v = st[5]; w = st[6]; st[5] = v ^ ((~w) & st[7]); st[6] = w ^ ((~st[7]) & st[8]); st[7] = st[7] ^ ((~st[8]) & st[9]); st[8] = st[8] ^ ((~st[9]) & v); st[9] = st[9] ^ ((~v) & w);
        v = st[10]; w = st[11]; st[10] = v ^ ((~w) & st[12]); st[11] = w ^ ((~st[12]) & st[13]); st[12] = st[12] ^ ((~st[13]) & st[14]); st[13] = st[13] ^ ((~st[14]) & v); st[14] = st[14] ^ ((~v) & w);
        v = st[15]; w = st[16]; st[15] = v ^ ((~w) & st[17]); st[16] = w ^ ((~st[17]) & st[18]); st[17] = st[17] ^ ((~st[18]) & st[19]); st[18] = st[18] ^ ((~st[19]) & v); st[19] = st[19] ^ ((~v) & w);
        v = st[20]; w = st[21]; st[20] = v ^ ((~w) & st[22]); st[21] = w ^ ((~st[22]) & st[23]); st[22] = st[22] ^ ((~st[23]) & st[24]); st[23] = st[23] ^ ((~st[24]) & v); st[24] = st[24] ^ ((~v) & w);

        //  Iota
        st[0] ^= keccakf_rndc[r];
    }

    hash8_t* hash = (hash8_t *)(output + (8 * (globalThId)));

    // store result
    hash->u2[0] = st[0];
    hash->u2[1] = st[1];
    hash->u2[2] = st[2];
    hash->u2[3] = st[3];
}


// Local work size. It can vary between hardware architectures.
#define WORK_SIZE 64

// Computed from the Verthash data file. NOTE!!! MUST BE UPDATED IF VERTHASH.DAT file size has been changed!!!
#define MDIV 80216063

// Extended validation uses 64 bit GPU side validation instead of 32 bit.
// It can be slightly more efficient with higher diff
//#define VERTHASH_EXTENDED_VALIDATION

//! Full host side validation
//#define VERTHASH_FULL_VALIDATION

//! Sync value_accumulator using shared memory(otherwise warp shuffle instruction will be used)
#define SYNC_SHARED

__global__ void cu_verthash(uint2* io_hashes,
                            const kstate2x_t* __restrict__ const kStates,
                            const uint2* __restrict__ const memory,
                            const uint in18,
                            const uint firstNonce
#ifdef VERTHASH_FULL_VALIDATION
                          )
#else
                          , uint* targetResults,
    #ifdef VERTHASH_EXTENDED_VALIDATION                          
                          const uint64_t target)
    #else // !VERTHASH_EXTENDED_VALIDATION
                          const uint target)
    #endif
#endif // !VERTHASH_FULL_VALIDATION
{
    // global id (1x work id)
    uint globalThId = blockDim.x * blockIdx.x + threadIdx.x;
    // 4x lane group index(local)
    uint lgr4id = (globalThId & (WORK_SIZE-1)) >> 2;
    // 4x lane group index(global) used as nonce result
    uint gr4id = globalThId >> 2;
    // sub group id(of 4x lane group)
    uint gr4e = globalThId & 3;

    //-----------------------------------------------------------------------------
    // SHA3 stage
    const kstate2x_t* const kstate = &kStates[gr4e];
    
    __shared__ struct sha3_state_t sha3St[WORK_SIZE / 4];
    uint nonce = firstNonce + gr4id;

    // 4 way kernel running 8xSHA3 passes(2x each lane)    
    #pragma unroll
    for (int s3s = 0; s3s < 2; ++s3s)
    {
        uint2 st[25];// = { 0 };
        // load state
        #pragma unroll
        for (int i = 0; i < 25; ++i)
        {
            st[i] = kstate->u2[25 * s3s + i];
        }

        uint2 t[5];
        uint2 u[5];
        uint2 v, w;

        st[0].x ^= in18;
        st[0].y ^= nonce;

        st[1].x ^= 0x00000006U; // byte[8] ^= 0x06
        st[8].y ^= 0x80000000U; // byte[71] ^= 0x80

        for (int r = 0; r < 24; r++)
        {
            // Theta
            t[0] = (st[0] ^ (st[5] ^ (st[10] ^ (st[15] ^ st[20]))));
            t[1] = (st[1] ^ (st[6] ^ (st[11] ^ (st[16] ^ st[21]))));
            t[2] = (st[2] ^ (st[7] ^ (st[12] ^ (st[17] ^ st[22]))));
            t[3] = (st[3] ^ (st[8] ^ (st[13] ^ (st[18] ^ st[23]))));
            t[4] = (st[4] ^ (st[9] ^ (st[14] ^ (st[19] ^ st[24]))));

            u[0] = rotl64(t[0], 1) ^ t[3];
            u[1] = rotl64(t[1], 1) ^ t[4];
            u[2] = rotl64(t[2], 1) ^ t[0];
            u[3] = rotl64(t[3], 1) ^ t[1];
            u[4] = rotl64(t[4], 1) ^ t[2];

            st[4] ^= u[0]; st[9] ^= u[0]; st[14] ^= u[0]; st[19] ^= u[0]; st[24] ^= u[0];
            st[0] ^= u[1]; st[5] ^= u[1]; st[10] ^= u[1]; st[15] ^= u[1]; st[20] ^= u[1];
            st[1] ^= u[2]; st[6] ^= u[2]; st[11] ^= u[2]; st[16] ^= u[2]; st[21] ^= u[2];
            st[2] ^= u[3]; st[7] ^= u[3]; st[12] ^= u[3]; st[17] ^= u[3]; st[22] ^= u[3];
            st[3] ^= u[4]; st[8] ^= u[4]; st[13] ^= u[4]; st[18] ^= u[4]; st[23] ^= u[4];

            // Rho Pi
            v = st[1];
            st[1] = rotl64(st[6], 44);
            st[6] = rotl64(st[9], 20);
            st[9] = rotl64(st[22], 61);
            st[22] = rotl64(st[14], 39);
            st[14] = rotl64(st[20], 18);
            st[20] = rotl64(st[2], 62);
            st[2] = rotl64(st[12], 43);
            st[12] = rotl64(st[13], 25);
            st[13] = rotl64(st[19], 8);
            st[19] = rotl64(st[23], 56);
            st[23] = rotl64(st[15], 41);
            st[15] = rotl64(st[4], 27);
            st[4] = rotl64(st[24], 14);
            st[24] = rotl64(st[21], 2);
            st[21] = rotl64(st[8], 55);
            st[8] = rotl64(st[16], 45);
            st[16] = rotl64(st[5], 36);
            st[5] = rotl64(st[3], 28);
            st[3] = rotl64(st[18], 21);
            st[18] = rotl64(st[17], 15);
            st[17] = rotl64(st[11], 10);
            st[11] = rotl64(st[7], 6);
            st[7] = rotl64(st[10], 3);
            st[10] = rotl64(v, 1);

            //  Chi
            v = st[0]; w = st[1]; st[0] = v ^ ((~w) & st[2]); st[1] = w ^ ((~st[2]) & st[3]); st[2] = st[2] ^ ((~st[3]) & st[4]); st[3] = st[3] ^ ((~st[4]) & v); st[4] = st[4] ^ ((~v) & w);
            v = st[5]; w = st[6]; st[5] = v ^ ((~w) & st[7]); st[6] = w ^ ((~st[7]) & st[8]); st[7] = st[7] ^ ((~st[8]) & st[9]); st[8] = st[8] ^ ((~st[9]) & v); st[9] = st[9] ^ ((~v) & w);
            v = st[10]; w = st[11]; st[10] = v ^ ((~w) & st[12]); st[11] = w ^ ((~st[12]) & st[13]); st[12] = st[12] ^ ((~st[13]) & st[14]); st[13] = st[13] ^ ((~st[14]) & v); st[14] = st[14] ^ ((~v) & w);
            v = st[15]; w = st[16]; st[15] = v ^ ((~w) & st[17]); st[16] = w ^ ((~st[17]) & st[18]); st[17] = st[17] ^ ((~st[18]) & st[19]); st[18] = st[18] ^ ((~st[19]) & v); st[19] = st[19] ^ ((~v) & w);
            v = st[20]; w = st[21]; st[20] = v ^ ((~w) & st[22]); st[21] = w ^ ((~st[22]) & st[23]); st[22] = st[22] ^ ((~st[23]) & st[24]); st[23] = st[23] ^ ((~st[24]) & v); st[24] = st[24] ^ ((~v) & w);

            //  Iota
            st[0] ^= keccakf_rndc[r];
        }

        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) + 0] = st[0];
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) + 1] = st[1];
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) + 2] = st[2];
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) + 3] = st[3];
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) + 4] = st[4];
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) + 5] = st[5];
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) + 6] = st[6];
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) + 7] = st[7];
    }

    __syncthreads();


    //-----------------------------------------------------------------------------
    // Verthash IO memory seek stage

    // get SHA3 256 input
    uint2 up1;
    up1 = io_hashes[globalThId];

    // local array used to sync between lanes
#ifdef SYNC_SHARED
    __shared__ hash8_t sHash[WORK_SIZE / 4];
#endif


    uint value_accumulator = 0x811c9dc5;

    // reference computed on the host side.
    //const uint mdiv = ((datfile_sz - HASH_OUT_SIZE)/BYTE_ALIGNMENT) + 1;

    // 71303125 is by default, but can change in future releases
    const uint mdiv = MDIV;

    for (uint i = 0; i < 4096; ++i)
    {
        // generate seek indexes at runtime
        // After each iteration SHA3 combined state requires a bit rotate operation
        // Note that some hardware don't support bit rotate by dynamic amount in a single instruction
        // v1 uses Load -> rotate by 1 -> store
        // v2 uses Load -> rotate by dynamic factor(depends on iteration)

        // v1. Rotate by constant amount
        //uint s3idx0 = i & 127;
        //uint seek_index = sha3St[lgr4id].u[s3idx0];
        //uint state0mod = rotl32(seek_index, 1);
        //sha3St[lgr4id].u[s3idx0] = state0mod;
        //__syncthreads();
        //const uint offset = (fnv1a(seek_index, value_accumulator) % mdiv) << 1;

        // v2. Rotate by dynamic amount
        uint s3idx0 = i & 127;
        uint rfactor = i >> 7;
        uint state0 = sha3St[lgr4id].u[s3idx0];
        uint seek_index = rotl32(state0, rfactor);
        const uint offset = (fnv1a(seek_index, value_accumulator) % mdiv) << 1;

        // 4 way memory lookup
        const uint2 vvalue = memory[offset + gr4e];

        // update up1
        up1.x = fnv1a(up1.x, vvalue.x);
        up1.y = fnv1a(up1.y, vvalue.y);

        // update value accumulator and synchronize it between lanes



#ifdef SYNC_SHARED
        sHash[lgr4id].u2[gr4e] = vvalue;
        __syncthreads();
        uint2 uu0 = sHash[lgr4id].u2[0];
        uint2 uu1 = sHash[lgr4id].u2[1];
        uint2 uu2 = sHash[lgr4id].u2[2];
        uint2 uu3 = sHash[lgr4id].u2[3];
        value_accumulator = fnv1a(value_accumulator, uu0.x);
        value_accumulator = fnv1a(value_accumulator, uu0.y);
        value_accumulator = fnv1a(value_accumulator, uu1.x);
        value_accumulator = fnv1a(value_accumulator, uu1.y);
        value_accumulator = fnv1a(value_accumulator, uu2.x);
        value_accumulator = fnv1a(value_accumulator, uu2.y);
        value_accumulator = fnv1a(value_accumulator, uu3.x);
        value_accumulator = fnv1a(value_accumulator, uu3.y);
#else
        value_accumulator = fnv1a(value_accumulator, __shfl_sync(0xffffffff, vvalue.x, 0, 4));
        value_accumulator = fnv1a(value_accumulator, __shfl_sync(0xffffffff, vvalue.y, 0, 4));
        value_accumulator = fnv1a(value_accumulator, __shfl_sync(0xffffffff, vvalue.x, 1, 4));
        value_accumulator = fnv1a(value_accumulator, __shfl_sync(0xffffffff, vvalue.y, 1, 4));
        value_accumulator = fnv1a(value_accumulator, __shfl_sync(0xffffffff, vvalue.x, 2, 4));
        value_accumulator = fnv1a(value_accumulator, __shfl_sync(0xffffffff, vvalue.y, 2, 4));
        value_accumulator = fnv1a(value_accumulator, __shfl_sync(0xffffffff, vvalue.x, 3, 4));
        value_accumulator = fnv1a(value_accumulator, __shfl_sync(0xffffffff, vvalue.y, 3, 4));
#endif
    }

    // store result
    io_hashes[globalThId] = up1;

#ifndef VERTHASH_FULL_VALIDATION
    //---------------------------------------------------
    // Save result as HTarg
    if (gr4e == 3)
    {
#ifdef VERTHASH_EXTENDED_VALIDATION
        uint64_t up1_64 = make_ulonglong(up1.x, up1.y);
        if (up1_64 <= target)
#else
        if (up1.y <= target)
#endif
        {
            int ai = atomicAdd(targetResults, 1);
            // only nonce offset gets saved(instead of firstNonce + offset).
            // It will be used as index for validation pass.
            targetResults[ai + 1] = gr4id; // nonce
        }
    }
#endif // !VERTHASH_FULL_VALIDATION
}

//-----------------------------------------------------------------------------
extern "C" void sha3_512_precompute_cuda(int blocksPerGrid, int threadsPerBlock,
                                         uint* output,
                                         uint* hheader)
{
    cudaMemcpyToSymbol(header, hheader, sizeof(uint) * 19);
    cu_sha3_512_precompute << <blocksPerGrid, threadsPerBlock >> >(output);
}

//-----------------------------------------------------------------------------
extern "C" void sha3_512_256_cuda(int blocksPerGrid, int threadsPerBlock,
                                 uint* output,
                                 uint in18,
                                 uint firstNonce)
{
    cu_sha3_512_256 << <blocksPerGrid, threadsPerBlock >> >(output, in18, firstNonce);
}


//-----------------------------------------------------------------------------
// verthashIO Host side interface
#ifdef VERTHASH_FULL_VALIDATION
extern "C" void verthash_cuda(int blocksPerGrid, int threadsPerBlock,
                              uint* output,
                              uint* kStates,
                              uint* memory,
                              uint in18,
                              uint firstNonce)
{
    cu_verthash<<<blocksPerGrid, threadsPerBlock>>>((uint2*)output,
                                                    (kstate2x_t*)kStates,
                                                    (uint2*)memory,
                                                    in18,
                                                    firstNonce);
}

#else // Simplified validation

extern "C" void verthash_cuda(int blocksPerGrid, int threadsPerBlock,
                              uint* output,
                              uint* kStates,
                              uint* memory,
                              uint in18,
                              uint firstNonce,
                              uint* htargetResults,
#ifdef VERTHASH_EXTENDED_VALIDATION
                              uint64_t target)
#else
                              uint target)
#endif
{
    cu_verthash<<<blocksPerGrid, threadsPerBlock>>>((uint2*)output,
                                                    (kstate2x_t*)kStates,
                                                    (uint2*)memory,
                                                    in18,
                                                    firstNonce,
                                                    htargetResults,
                                                    target);
}

#endif // !FULL_VALIDATION

