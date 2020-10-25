
/*
 * Copyright 2018-2019 CryptoGraphics
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

#include <CL/opencl.h>

#include <iostream> // cerr
#include <fstream> // ifstream
#include <sstream> // ostringstream

//-----------------------------------------------------------------------------
// cl_amd_device_attribute_query can be undefined inside standard OpenCL headers
#ifndef CL_DEVICE_TOPOLOGY_AMD

#define CL_DEVICE_TOPOLOGY_AMD                      0x4037
#define CL_DEVICE_BOARD_NAME_AMD                    0x4038
typedef union
{
    struct { cl_uint type; cl_uint data[5]; } raw;
    struct { cl_uint type; cl_char unused[17]; cl_char bus; cl_char device; cl_char function; } pcie;
} cl_device_topology_amd;
#define CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD            1

#endif
//-----------------------------------------------------------------------------

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
        cl_platform_id clPlatformId;
        cl_device_id clId;
        int32_t pcieBusId;
        int32_t platformIndex;
        EAsmProgram asmProgram;
        EBinaryFormat binaryFormat;
        EVendor vendor;
    };
    //-----------------------------------------------------------------------------
    //! Compare cl devices by PCIe Bus Id.
    inline bool compareLogicalDevices(const cldevice_t & d1, const cldevice_t& d2)
    {
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
    inline cl_program cluCreateProgramFromFile(cl_context context, cl_device_id cldevice, const char* build_options, const char* file_name)
    {
        cl_int errNum;
        cl_program program;

        std::ifstream kernelFile(file_name, std::ios::in);
        if (!kernelFile.is_open())
        {
            std::cerr << "Failed to open file for reading: " << file_name << std::endl;
            return NULL;
        }

        std::ostringstream oss;
        oss << kernelFile.rdbuf();

        std::string srcStdStr = oss.str();
        const char *srcStr = srcStdStr.c_str();
        program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, nullptr, nullptr);
        if (program == nullptr)
        {
            std::cerr << "Failed to create CL program from source." << std::endl;
            return NULL;
        }

        errNum = clBuildProgram(program, 1, &cldevice, build_options, NULL, NULL);
        if (errNum != CL_SUCCESS)
        {
            // Determine the reason for the error
            char buildLog[16384];
            clGetProgramBuildInfo(program, cldevice, CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), buildLog, NULL);

            std::cerr << "Error in kernel: " << std::endl;
            std::cerr << buildLog;
            clReleaseProgram(program);
            return NULL;
        }

        return program;
    }
    //-----------------------------------------------------------------------------
    inline int readFile(unsigned char **output, size_t *size, const char *name)
    {
        FILE* fp = fopen(name, "rb");
        if (!fp)
        {
            return -1;
        }

        fseek(fp, 0, SEEK_END);
        *size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        *output = (unsigned char*)malloc(*size);
        if (!(*output))
        {
            fclose(fp);
            return -1;
        }

        size_t readRes = fread(*output, 1, *size, fp);
        if (readRes != (*size))
        {
            free(*output);
            fclose(fp);
            return -1;
        }

        fclose(fp);

        return 0;
    }
    //-----------------------------------------------------------------------------
    inline cl_program cluCreateProgramWithBinaryFromFile(cl_context context, cl_device_id cldevice, const char* file_name)
    {
        unsigned char* sprogram_file = NULL;
        size_t sprogram_size = 0;
        if (readFile(&sprogram_file, &sprogram_size, file_name) < 0)
        {
            printf("Failed to read file: %s\n", file_name);
            fflush(stdout);
            return NULL;
        }

        cl_int errorCode;
        cl_program program;

        program = clCreateProgramWithBinary(context, 1, &cldevice, &sprogram_size, (const unsigned char **)&sprogram_file, NULL, &errorCode);
        // TODO: add check for createWithBinary
        errorCode = clBuildProgram(program, 1, &cldevice, NULL, NULL, NULL);
        free(sprogram_file);
        if (errorCode != CL_SUCCESS)
        {
            return NULL;
        }
        else
        {
            return program;
        }
    }
    //-----------------------------------------------------------------------------

} // end namespace vh

#endif // !CLUtils_INCLUDE_ONCE
