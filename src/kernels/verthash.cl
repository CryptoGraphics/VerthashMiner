/*
 * Copyright 2020 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */


// AMD specific optimization for 64 bit rotate
#ifdef BIT_ALIGN
#define rotr64(x, n) ((n) < 32 ? (amd_bitalign((uint)((x) >> 32), (uint)(x), (uint)(n)) | ((ulong)amd_bitalign((uint)(x), (uint)((x) >> 32), (uint)(n)) << 32)) : (amd_bitalign((uint)(x), (uint)((x) >> 32), (uint)(n) - 32) | ((ulong)amd_bitalign((uint)((x) >> 32), (uint)(x), (uint)(n) - 32) << 32)))
#else
#define rotr64(x, n) rotate(x, (ulong)(64-n))
#endif

inline uint rotl32(uint x, uint n)
{
    return (((x) << (n)) | ((x) >> (32 - (n))));
}

inline uint fnv1a(const uint a, const uint b)
{
    uint res = (a ^ b) * 0x1000193U;
    return res;
}

// 2x precomputed SHA3 states
typedef union {
    ulong ul[50];
} kstate2x_t;

// shared hash to exchange between lanes during memory seeks stage
typedef union {
    uint2 u2[4];
} hash8_t;

// A combined SHA3 result used during memory seeks stage
typedef union {
    uint u[128];
    uint2 u2[64];
} sha3_state_t;

// Keccak constants
__constant ulong keccakf_rndc[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

// configured from host side(here just for unit testing)
//-----------------------------------------------------------------------------
// Kernel is configured from host side.

// Local work size. It can vary between hardware architectures.
//#define WORK_SIZE 64 // AMD
//#define WORK_SIZE 64 // NV

// Computed from the Verthash data file
//#define MDIV 71303125

// Extended validation uses 64 bit GPU side validation instead of 32 bit.
// It can be slightly more efficient with higher diff
//#define EXTENDED_VALIDATION

//! Full host side validation
//#define FULL_VALIDATION

__attribute__((reqd_work_group_size(WORK_SIZE, 1, 1)))
__kernel void verthash_4w(__global uint2* io_hashes,
                          __global kstate2x_t* restrict kStates,
                          __global uint2* restrict memory,
                          const uint in18,
                          const uint firstNonce
#ifdef FULL_VALIDATION
                          )
#else
                          , __global uint* targetResults,
    #ifdef EXTENDED_VALIDATION                          
                          const ulong target)
    #else // !EXTENDED_VALIDATION
                          const uint target)
    #endif
#endif // !FULL_VALIDATION
{
    // global id (1x work id)
    uint gid = get_global_id(0);
    // 4x lane group index(local)
    uint lgr4id = get_local_id(0) >> 2;
    // 4x lane group index(global) used as nonce result
    uint gr4id = gid >> 2;
    // sub group id(of 4x lane group)
    uint gr4e = gid & 3;

    //-----------------------------------------------------------------------------
    // SHA3 stage
    __global kstate2x_t* kstate = &kStates[gr4e];
    
    __local sha3_state_t sha3St[WORK_SIZE/4];
    uint nonce = firstNonce + gr4id;
    
    // 4 way kernel running 8xSHA3 passes(2x each lane)    
    #pragma unroll
    for(int s3s = 0; s3s < 2; ++s3s)
    {
        ulong st[25] = { 0 };
        // load state
        #pragma unroll
        for(int i = 0; i < 25; ++i)
        {
            st[ i] = kstate->ul[25 * s3s + i];
        }

        // variables
        st[0] ^= as_ulong((uint2)(in18, nonce));
   
        st[1] ^= 0x06UL;
        st[8] ^= 0x8000000000000000UL;
    
        ulong u[5];
        ulong v,w;

        for (int r = 0; r < 24; r++)
        {
            // Theta
            v    = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
            u[2] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
            u[3] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
            u[4] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
            w    = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

            u[0] = rotr64(u[2], 63) ^    w;
            u[1] = rotr64(u[3], 63) ^    v;
            u[2] = rotr64(u[4], 63) ^ u[2];
            u[3] = rotr64(   w, 63) ^ u[3];
            u[4] = rotr64(   v, 63) ^ u[4];

            st[0] ^= u[0]; st[5] ^= u[0]; st[10] ^= u[0]; st[15] ^= u[0]; st[20] ^= u[0];
            st[1] ^= u[1]; st[6] ^= u[1]; st[11] ^= u[1]; st[16] ^= u[1]; st[21] ^= u[1];
            st[2] ^= u[2]; st[7] ^= u[2]; st[12] ^= u[2]; st[17] ^= u[2]; st[22] ^= u[2];
            st[3] ^= u[3]; st[8] ^= u[3]; st[13] ^= u[3]; st[18] ^= u[3]; st[23] ^= u[3];
            st[4] ^= u[4]; st[9] ^= u[4]; st[14] ^= u[4]; st[19] ^= u[4]; st[24] ^= u[4];

            // Rho Pi
            v = st[1];
            st[ 1] = rotr64(st[ 6], 20);
            st[ 6] = rotr64(st[ 9], 44);
            st[ 9] = rotr64(st[22],  3);
            st[22] = rotr64(st[14], 25);
            st[14] = rotr64(st[20], 46);
            st[20] = rotr64(st[ 2],  2);
            st[ 2] = rotr64(st[12], 21);
            st[12] = rotr64(st[13], 39);
            st[13] = rotr64(st[19], 56);
            st[19] = rotr64(st[23],  8);
            st[23] = rotr64(st[15], 23);
            st[15] = rotr64(st[ 4], 37);
            st[ 4] = rotr64(st[24], 50);
            st[24] = rotr64(st[21], 62);
            st[21] = rotr64(st[ 8],  9);
            st[ 8] = rotr64(st[16], 19);
            st[16] = rotr64(st[ 5], 28);
            st[ 5] = rotr64(st[ 3], 36);
            st[ 3] = rotr64(st[18], 43);
            st[18] = rotr64(st[17], 49);
            st[17] = rotr64(st[11], 54);
            st[11] = rotr64(st[ 7], 58);
            st[ 7] = rotr64(st[10], 61);
            st[10] = rotr64(v, 63);

            //  Chi
            v = st[ 0]; w = st[ 1]; st[ 0] = bitselect(st[ 0] ^ st[ 2], st[ 0], st[ 1]); st[ 1] = bitselect(st[ 1] ^ st[ 3], st[ 1], st[ 2]); st[ 2] = bitselect(st[ 2] ^ st[ 4], st[ 2], st[ 3]); st[ 3] = bitselect(st[ 3] ^ v, st[ 3], st[ 4]); st[ 4] = bitselect(st[ 4] ^ w, st[ 4], v);
            v = st[ 5]; w = st[ 6]; st[ 5] = bitselect(st[ 5] ^ st[ 7], st[ 5], st[ 6]); st[ 6] = bitselect(st[ 6] ^ st[ 8], st[ 6], st[ 7]); st[ 7] = bitselect(st[ 7] ^ st[ 9], st[ 7], st[ 8]); st[ 8] = bitselect(st[ 8] ^ v, st[ 8], st[ 9]); st[ 9] = bitselect(st[ 9] ^ w, st[ 9], v);
            v = st[10]; w = st[11]; st[10] = bitselect(st[10] ^ st[12], st[10], st[11]); st[11] = bitselect(st[11] ^ st[13], st[11], st[12]); st[12] = bitselect(st[12] ^ st[14], st[12], st[13]); st[13] = bitselect(st[13] ^ v, st[13], st[14]); st[14] = bitselect(st[14] ^ w, st[14], v);
            v = st[15]; w = st[16]; st[15] = bitselect(st[15] ^ st[17], st[15], st[16]); st[16] = bitselect(st[16] ^ st[18], st[16], st[17]); st[17] = bitselect(st[17] ^ st[19], st[17], st[18]); st[18] = bitselect(st[18] ^ v, st[18], st[19]); st[19] = bitselect(st[19] ^ w, st[19], v);
            v = st[20]; w = st[21]; st[20] = bitselect(st[20] ^ st[22], st[20], st[21]); st[21] = bitselect(st[21] ^ st[23], st[21], st[22]); st[22] = bitselect(st[22] ^ st[24], st[22], st[23]); st[23] = bitselect(st[23] ^ v, st[23], st[24]); st[24] = bitselect(st[24] ^ w, st[24], v);
                
            //  Iota
            st[0] ^= keccakf_rndc[r];
        }
    
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) +  0] = as_uint2(st[0]);
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) +  1] = as_uint2(st[1]);
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) +  2] = as_uint2(st[2]);
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) +  3] = as_uint2(st[3]);
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) +  4] = as_uint2(st[4]);
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) +  5] = as_uint2(st[5]);
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) +  6] = as_uint2(st[6]);
        sha3St[lgr4id].u2[(gr4e * 16) + (s3s * 8) +  7] = as_uint2(st[7]);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
   

    //-----------------------------------------------------------------------------
    // Verthash IO memory seek stage
    
    // get SHA3 256 input
    uint2 up1;
    up1 = io_hashes[gid];

    // local array used to sync between lanes
    __local hash8_t sHash[WORK_SIZE/4];
    
    uint value_accumulator = 0x811c9dc5;

    // reference computed on the host side.
    //const uint mdiv = ((datfile_sz - HASH_OUT_SIZE)/BYTE_ALIGNMENT) + 1;
    
    // 71303125 is by default, but can change in future releases
    const uint mdiv = MDIV;
    
    for(uint i = 0; i < 4096; ++i)
    {
        // generate seek indexes at runtime
        // After each iteration SHA3 combined state requires a bit rotate operation
        // Note that some hardware don't support bit rotate by dynamic amount in a single instruction
        // v1 uses Load -> rotate by 1 -> store
        // v2 uses Load -> rotate by dynamic factor(depends on iteration)
        
        // v1. Rotate by constant amount
        uint s3idx0 = i & 127;
        uint seek_index = sha3St[lgr4id].u[s3idx0];
        uint state0mod = rotl32(seek_index, 1);
        sha3St[lgr4id].u[s3idx0] = state0mod;
        barrier(CLK_LOCAL_MEM_FENCE);
 
        const uint offset = (fnv1a(seek_index, value_accumulator) % mdiv) << 1;

        // v2. Rotate by dynamic amount
       // uint s3idx0 = i & 127;
       // uint rfactor = i >> 7;
       // uint state0 = sha3St[lgr4id].u[s3idx0];
       // uint seek_index = rotl32(state0, rfactor);
       // const uint offset = (fnv1a(seek_index, value_accumulator) % mdiv) << 1;

        // 4 way memory lookup
        const uint2 vvalue = memory[offset + gr4e];

        // update up1
        up1.x = fnv1a(up1.x, vvalue.x);
        up1.y = fnv1a(up1.y, vvalue.y);

        // update value accumulator and synchronize it between lanes
        sHash[lgr4id].u2[gr4e] = vvalue;
        barrier(CLK_LOCAL_MEM_FENCE);
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
    }
    
    // store result
    io_hashes[gid] = up1;

#ifndef FULL_VALIDATION
    //---------------------------------------------------
    // Save result as HTarg
    if(gr4e == 3)
    {
#ifdef EXTENDED_VALIDATION
        ulong up1_64 = as_ulong(up1);
        if(up1_64 <= target)
#else
        if(up1.y <= target)
#endif
        {
            uint ai = atomic_inc(targetResults);
            targetResults[ai+1] = gr4id; // final nonce
        }
    }
#endif // !FULL_VALIDATION

    barrier(CLK_GLOBAL_MEM_FENCE);
}
