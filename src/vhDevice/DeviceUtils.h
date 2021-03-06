/*
 * Copyright 2018-2021 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#ifndef CLUtils_INCLUDE_ONCE
#define CLUtils_INCLUDE_ONCE

// remove OpenCL deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#define CL_TARGET_OPENCL_VERSION 120

#include "../external/CL/opencl.h"
#include <string>

//-----------------------------------------------------------------------------
// cl_amd_device_attribute_query can be undefined inside standard OpenCL headers
#ifndef CL_DEVICE_TOPOLOGY_AMD
#define CL_DEVICE_TOPOLOGY_AMD                      0x4037
#endif

#ifndef CL_DEVICE_BOARD_NAME_AMD
#define CL_DEVICE_BOARD_NAME_AMD                    0x4038
#endif

typedef union
{
    struct { cl_uint type; cl_uint data[5]; } raw;
    struct { cl_uint type; cl_char unused[17]; cl_char bus; cl_char device; cl_char function; } pcie;
} clu_device_topology_amd;

#ifndef CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD
#define CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD            1
#endif
//-----------------------------------------------------------------------------
// NVIDIA OpenCL stuff
#ifndef CL_DEVICE_PCI_BUS_ID_NV 
#define CL_DEVICE_PCI_BUS_ID_NV                     0x4008
#endif

#ifndef CL_DEVICE_PCI_SLOT_ID_NV
#define CL_DEVICE_PCI_SLOT_ID_NV                    0x4009
#endif

#ifndef CL_DEVICE_PCI_DOMAIN_ID_NV
#define CL_DEVICE_PCI_DOMAIN_ID_NV                  0x400A
#endif

namespace vh
{
    //-----------------------------------------------------------------------------
    typedef enum
    {
        AP_None  = 0,
        AP_GFX6  = 1,
        AP_GFX7  = 2,
        AP_GFX8  = 3,
        AP_GFX9  = 4
    } EAsmProgram;
    //-----------------------------------------------------------------------------
    typedef enum
    {
        BF_None    = 0,
        BF_AMDCL2  = 1,
        BF_ROCm    = 2,
        BF_AUTO    = 3
    } EBinaryFormat;
    //-----------------------------------------------------------------------------
    typedef enum
    {
        V_AMD    = 0,
        V_NVIDIA = 1,
        V_OTHER  = 2
    } EVendor;
    //-----------------------------------------------------------------------------
    //! OpenCL logical device
    struct cldevice_t
    {
        // global OCL stuff
        cl_platform_id clPlatformId;
        cl_device_id clId;
        int32_t platformIndex;

        // PCIe stuff
        int32_t pcieBusId;
        int32_t pcieDeviceId;
        int32_t pcieFunctionId; // 0 for NVIDIA devices

        // ASM program handling
        EAsmProgram asmProgram;
        EBinaryFormat binaryFormat;

        // General
        EVendor vendor;
    };
#ifdef HAVE_CUDA
    //-----------------------------------------------------------------------------
    //! CUDA logical device
    struct cudevice_t
    {
        int cudeviceHandle;
        // PCIe stuff
        int32_t pcieBusId;
        int32_t pcieDeviceId;
    };
#endif
    //-----------------------------------------------------------------------------
    //! Compare cl devices by PCIe Bus Id.
    inline bool compareLogicalDevices(const cldevice_t & d1, const cldevice_t& d2)
    {
        // TODO: wrong. Must be compared by whole PCIeId, not just bus
        return (d1.pcieBusId < d2.pcieBusId); 
    }
    //-----------------------------------------------------------------------------
    inline void getAsmProgramNameFromDeviceName(const std::string& device_name, std::string& out_asm_name)
    {
        //------------------------------------------
        // GCN 1.0-1.1
        if (!device_name.compare("Capeverde"))
            out_asm_name = "gfx6";
        else if (!device_name.compare("Hainan"))
            out_asm_name = "gfx6";
        else if (!device_name.compare("Oland"))
            out_asm_name = "gfx6";
        else if (!device_name.compare("Pitcairn"))
            out_asm_name = "gfx6";
        else if (!device_name.compare("Tahiti"))
            out_asm_name = "gfx6";
        //------------------------------------------
        // GCN 1.2
        else if (!device_name.compare("Bonaire"))
            out_asm_name = "gfx7";
        else if (!device_name.compare("Hawaii"))
            out_asm_name = "gfx7";
        else if (!device_name.compare("Kalindi"))
            out_asm_name = "gfx7";
        else if (!device_name.compare("Mullins"))
            out_asm_name = "gfx7";
        else if (!device_name.compare("Spectre"))
            out_asm_name = "gfx7";
        else if (!device_name.compare("Spooky"))
            out_asm_name = "gfx7";
        //------------------------------------------
        // GCN 1.3
        else if (!device_name.compare("Baffin"))
            out_asm_name = "gfx8";
        else if (!device_name.compare("Iceland"))
            out_asm_name = "gfx8";
        else if (!device_name.compare("Ellesmere"))
            out_asm_name = "gfx8";
        else if (!device_name.compare("Fiji"))
            out_asm_name = "gfx8";
        else if (!device_name.compare("Tonga"))
            out_asm_name = "gfx8";
        else if (!device_name.compare("gfx803"))
            out_asm_name = "gfx8";
        else if (!device_name.compare("gfx804"))
            out_asm_name = "gfx8";
        else if (!device_name.compare("Carrizo"))
            out_asm_name = "gfx8";
        else if (!device_name.compare("Stoney"))
            out_asm_name = "gfx8";
        //------------------------------------------
        // GCN 1.4
        else if (!device_name.compare("gfx900"))
            out_asm_name = "gfx9";
        else if (!device_name.compare("gfx902"))
            out_asm_name = "gfx9";
        else if (!device_name.compare("gfx903"))
            out_asm_name = "gfx9";
        else if (!device_name.compare("gfx904"))
            out_asm_name = "gfx9";
        else if (!device_name.compare("gfx905"))
            out_asm_name = "gfx9";
        else if (!device_name.compare("gfx906"))
            out_asm_name = "gfx9";
        else if (!device_name.compare("gfx907"))
            out_asm_name = "gfx9";
        //------------------------------------------
        // Unsupported/Untested/Not detected.
        else
            out_asm_name = "none";
    }
    //-----------------------------------------------------------------------------
    inline EAsmProgram getAsmProgramName(const std::string &arch_name)
    {
        EAsmProgram result;
        if (arch_name.find("gfx6") != std::string::npos)
            result = AP_GFX6;
        else if (arch_name.find("gfx7") != std::string::npos)
            result = AP_GFX7;
        else if (arch_name.find("gfx8") != std::string::npos)
            result = AP_GFX8;
        else if (arch_name.find("gfx9") != std::string::npos)
            result = AP_GFX9;
        else
            result = AP_None;

        return result;
    }
    //-----------------------------------------------------------------------------
    inline EBinaryFormat getBinaryFormatFromName(const std::string& asm_kernel_format_name)
    {
        EBinaryFormat result;
        if (asm_kernel_format_name.find("amdcl2") != std::string::npos)
            result = BF_AMDCL2;
        else if (asm_kernel_format_name.find("ROCm") != std::string::npos)
            result = BF_ROCm;
        else
            result = BF_None;

        return result;
    }
    //-----------------------------------------------------------------------------
    //! Create an OpenCL program from file.
    inline int readFile(unsigned char **output, size_t *size, const char *file_name, const char* mode)
    {
        FILE* fp = fopen_utf8(file_name, mode);
        if (!fp)
        {
            return -1;
        }

        fseek(fp, 0, SEEK_END);
        *size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        *output = (unsigned char *)malloc(*size);
        if (!*output)
        {
            fclose(fp);
            return -1;
        }
        memset(*output, 0, *size);

        fread(*output, *size, 1, fp);
        fclose(fp);
        return 0;
    }
    //-----------------------------------------------------------------------------
    //! Reads text file. See "readFile()"
    inline int readTextFile(unsigned char **output, size_t *size, const char *file_name)
    {
        return readFile(output, size, file_name, "r");
    }
    //-----------------------------------------------------------------------------
    //! Reads binary file. See "readFile()"
    inline int readBinaryFile(unsigned char **output, size_t *size, const char *file_name)
    {
        return readFile(output, size, file_name, "rb");
    }
    //-----------------------------------------------------------------------------
    //! Creates program from text file(kernel source code).
    inline cl_program cluCreateProgramFromFile(cl_context context, cl_device_id device_id,
                                               const char* build_options, const char* file_name)
    {
        cl_int errorCode = CL_SUCCESS;
        cl_program program;

        // open kernel file
        unsigned char* sprogram_file = NULL;
        size_t sprogram_size = 0;
        if (readTextFile(&sprogram_file, &sprogram_size, file_name) < 0)
        {
            printf("Failed to read file: %s\n", file_name);
            fflush(stdout);
        }

        // create an OpenCL program
        program = clCreateProgramWithSource(context, 1, (const char**)&sprogram_file, &sprogram_size, &errorCode);
        if (errorCode != CL_SUCCESS)
        {
            printf("Failed to create an OpenCL program from source.\n");
            fflush(stdout);

            free(sprogram_file);

            return NULL;
        }

        free(sprogram_file);

        // build
        errorCode = clBuildProgram(program, 1, &device_id, build_options, NULL, NULL);

        // get build logs if there are any.
        size_t logSize = 0;
        errorCode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

        // hack for NVIDIA OpenCL runtime: returns empty logs with size 2 with some build_options
        if (logSize > 2)
        {
            char *buildLog = (char*)malloc(logSize);
            memset(buildLog, '\0', logSize);
            errorCode = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);

            puts(buildLog);
            fflush(stdout);

            free(buildLog);
        }

        if (errorCode != CL_SUCCESS)
        {
            clReleaseProgram(program);
            return NULL;
        }

        return program;
    }

    //-----------------------------------------------------------------------------
    inline cl_program cluCreateProgramWithBinaryFromFile(cl_context context, cl_device_id device, const char* file_name)
    {
        unsigned char* sprogram_file = NULL;
        size_t sprogram_size = 0;
        if (readBinaryFile(&sprogram_file, &sprogram_size, file_name) < 0)
        {
            printf("Failed to read file: %s\n", file_name);
            fflush(stdout);
        }

        cl_int errorCode;
        cl_program program;

        program = clCreateProgramWithBinary(context, 1, &device, &sprogram_size, (const unsigned char **)&sprogram_file, NULL, &errorCode);
        errorCode = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        free(sprogram_file);
        if (errorCode != CL_SUCCESS)
            return NULL;
        else
            return program;
    } 

    //-----------------------------------------------------------------------------

} // end namespace vh

#endif // !CLUtils_INCLUDE_ONCE
