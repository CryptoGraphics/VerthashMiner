/*
 * Copyright 2018-2020 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#ifndef Verthash_INCLUDE_ONCE
#define Verthash_INCLUDE_ONCE

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Verthash constants used to compute bitmask, used inside kernel during IO pass
#define VH_HASH_OUT_SIZE 32
#define VH_BYTE_ALIGNMENT 16

//-----------------------------------------------------------------------------
// Verthash data
//! Verthash C api for data maniputation.
typedef struct VerthashInfo
{
    char* fileName;
    uint8_t* data;
    uint64_t dataSize;
    uint32_t bitmask;
} verthash_info_t;

//! Must be called before usage. Reset all fields and set a mining data file name.
//! Error codes
//! 0 - Success(No error).
//! 1 - File name is invalid.
//! 2 - Memory allocation error
int verthash_info_init(verthash_info_t* info, const char* file_name)
{
    // init fields to 0
    info->fileName = NULL;
    info->data = NULL;
    info->dataSize = 0;
    info->bitmask = 0;

    // get name
    if (file_name == NULL) { return 1; }
    size_t fileNameLen = strlen(file_name);
    if (fileNameLen == 0) { return 1; }

    info->fileName = (char*)malloc(fileNameLen+1);
    memset(info->fileName, 0, fileNameLen+1);
    memcpy(info->fileName, file_name, fileNameLen);

    // Load data
    FILE *fileMiningData = fopen(info->fileName, "rb");
    // Failed to open file for reading
    if (!fileMiningData) { return 1; }

    // Get file size
    fseek(fileMiningData, 0, SEEK_END);
    uint64_t fileSize = (uint64_t)ftell(fileMiningData);
    fseek(fileMiningData, 0, SEEK_SET);

    // Allocate data
    info->data = (uint8_t *)malloc(fileSize);
    if (!info->data)
    {
        fclose(fileMiningData);
        // Memory allocation fatal error.
        return 2;
    }

    // Load data
    fread(info->data, fileSize, 1, fileMiningData);
    fclose(fileMiningData);

    // Update fields
    info->bitmask = ((fileSize - VH_HASH_OUT_SIZE)/VH_BYTE_ALIGNMENT) + 1;
    info->dataSize = fileSize;

    return 0;
}

//! Reset all fields and free allocated data
void verthash_info_free(verthash_info_t* info)
{
    free(info->fileName);
    free(info->data);
    info->dataSize = 0;
    info->bitmask = 0;
}

#endif // !Verthash_INCLUDE_ONCE

