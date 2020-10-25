/*
 * Copyright 2020 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#ifdef BIT_ALIGN
#define rotr64(x, n) ((n) < 32 ? (amd_bitalign((uint)((x) >> 32), (uint)(x), (uint)(n)) | ((ulong)amd_bitalign((uint)(x), (uint)((x) >> 32), (uint)(n)) << 32)) : (amd_bitalign((uint)(x), (uint)((x) >> 32), (uint)(n) - 32) | ((ulong)amd_bitalign((uint)((x) >> 32), (uint)(x), (uint)(n) - 32) << 32)))
#else
#define rotr64(x, n) rotate(x, (ulong)(64-n))
#endif

typedef union
{
    ulong ul[25];
} kstate_t;

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

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void sha3_512_precompute(__global ulong* output, __global uint* header)
{
    uint gid = get_global_id(0);
    uint h0 = header[0]+(gid&7)+1;
    
    //-------------------------------------
    // sha3 init
    ulong st[25] = { 0 };

    //-------------------------------------
    // sha3 update(header 72 bytes)
    st[0] = as_ulong((uint2)(h0,         header[ 1]));
    st[1] = as_ulong((uint2)(header[ 2], header[ 3]));
    st[2] = as_ulong((uint2)(header[ 4], header[ 5]));
    st[3] = as_ulong((uint2)(header[ 6], header[ 7]));
    st[4] = as_ulong((uint2)(header[ 8], header[ 9]));
    st[5] = as_ulong((uint2)(header[10], header[11]));
    st[6] = as_ulong((uint2)(header[12], header[13]));
    st[7] = as_ulong((uint2)(header[14], header[15]));
    st[8] = as_ulong((uint2)(header[16], header[17]));


    ulong u[5];
    ulong v,w;

    for (int r = 0; r < 24; ++r)
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
    
    //-------------------------------------
    // store result
    __global kstate_t *kstate = (__global kstate_t *)(output + (25* (gid & 7)));
    kstate->ul[ 0] = st[ 0];
    kstate->ul[ 1] = st[ 1];
    kstate->ul[ 2] = st[ 2];
    kstate->ul[ 3] = st[ 3];
    kstate->ul[ 4] = st[ 4];
    kstate->ul[ 5] = st[ 5];
    kstate->ul[ 6] = st[ 6];
    kstate->ul[ 7] = st[ 7];
    kstate->ul[ 8] = st[ 8];
    kstate->ul[ 9] = st[ 9];
    kstate->ul[10] = st[10];
    kstate->ul[11] = st[11];
    kstate->ul[12] = st[12];
    kstate->ul[13] = st[13];
    kstate->ul[14] = st[14];
    kstate->ul[15] = st[15];
    kstate->ul[16] = st[16];
    kstate->ul[17] = st[17];
    kstate->ul[18] = st[18];
    kstate->ul[19] = st[19];
    kstate->ul[20] = st[20];
    kstate->ul[21] = st[21];
    kstate->ul[22] = st[22];
    kstate->ul[23] = st[23];
    kstate->ul[24] = st[24];
}
