
/*
 * Copyright 2018-2021 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#ifndef ConfigGenerator_INCLUDE_ONCE
#define ConfigGenerator_INCLUDE_ONCE

#include "DeviceUtils.h"
#include <vector>
#include <string>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif


namespace vh
{
    // default work batch time for all devices
    static const uint32_t defaultBatchTimeMs = 500; // must be above 0
    static const int defaultDeviceMonitor = 1; // 1 means enabled
    static const uint32_t defaultWorkSize = 0; // 0 means automatic
    static const int defaultGPUTemperatureLimit = 79; // 0 means ignore
    static const uint32_t defaultOccupancyPct = 100; // 100%

/******************************************************************************
*  Generates configuration text data for selected OpenCL devices.
* Input
*  - clplatformIds - All avaliable OpenCL platforms.
*  - cldevices     - All/Filtered OpenCL devices.
* 
*  - rawDeviceList - true:  Generates configuration in raw device list format,
*                           All devices will be selected selected by index.
*                  - false: Generates configuration in PCIe bus device list format.
*                           All devices will be selected selected by PCIe bus Id.
* 
* Input&Output
*  - configText    - Configuration text data.
*******************************************************************************/
static void generateCLDeviceConfig(const std::vector<cl_platform_id>& clplatformIds,
                                   const std::vector<cldevice_t>& cldevices,
                                   std::string& configText)
{
    configText += "#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
                  "# OpenCL device config:\n"
                  "#\n"
                  "# Available platforms:\n"
                  "#\n";

    //-------------------------------------
    // OpenCL devices

    // generate platform list
    std::string platformListText;
    size_t infoSize = 0;
    std::string infoString(infoSize, ' ');
    for (size_t i = 0; i < clplatformIds.size(); ++i)
    {
        infoString.clear();
        clGetPlatformInfo(clplatformIds[i], CL_PLATFORM_VENDOR, 0, nullptr, &infoSize);
        infoString.resize(infoSize);
        clGetPlatformInfo(clplatformIds[i], CL_PLATFORM_VENDOR, infoSize, (void*)infoString.data(), nullptr);
        infoString.pop_back();

        platformListText += "# ";
        platformListText += std::to_string(i+1);
        platformListText += ". Platform name: ";
        platformListText += infoString;
        platformListText += "\n#    Index: ";
        platformListText += std::to_string(i);
        platformListText += "\n";
    }

    const std::string defaultWorkSizeString(std::to_string(defaultWorkSize));
    const std::string defaultBatchTimeMsString(std::to_string(defaultBatchTimeMs));
    const std::string defaultDeviceMonitorString(std::to_string(defaultDeviceMonitor));
    const std::string defaultGPUTemperatureLimitString(std::to_string(defaultGPUTemperatureLimit));
    const std::string defaultOccupancyPctString(std::to_string(defaultOccupancyPct));
    std::string deviceConfText;
    std::string deviceListText;
    std::string deviceName;
    std::string asmProgramName;

    for (size_t i = 0; i < cldevices.size(); ++i)
    {
        const cldevice_t& cldevice = cldevices[i];

        // get device info
        deviceListText += "\n#\n# DeviceIndex: ";
        deviceListText += std::to_string(i);

        deviceListText += "\n#    Name: ";
        // get device name
        size_t infoSize = 0;
        deviceName.clear();
        clGetDeviceInfo(cldevice.clId, CL_DEVICE_NAME, 0, NULL, &infoSize);
        deviceName.resize(infoSize);
        clGetDeviceInfo(cldevice.clId, CL_DEVICE_NAME, infoSize, (void *)deviceName.data(), NULL);
        deviceName.pop_back();

        deviceListText += deviceName;

        deviceListText += "\n#    PCIeId: ";

        if (cldevice.pcieBusId == -1)
        {
            deviceListText += "not avilable";
        }
        else
        {
            char pcieStr[9] = { };
            snprintf (pcieStr, 8, "%02x:%02x:%01x",
                      cldevice.pcieBusId, cldevice.pcieDeviceId, cldevice.pcieFunctionId);
            deviceListText += std::string(pcieStr);
        }

        deviceListText += "\n#    Platform index: ";
        deviceListText += std::to_string(cldevice.platformIndex);

        //  get device config
        deviceConfText += "<CL_Device";
        deviceConfText += std::to_string(i);

        deviceConfText += " DeviceIndex = \"";
        deviceConfText += std::to_string(i);
        deviceConfText += "\"";

        deviceConfText += " BinaryFormat = \"";
        deviceConfText += "auto";
        deviceConfText += "\"";

        deviceConfText += " AsmProgram = \"";
        vh::getAsmProgramNameFromDeviceName(deviceName, asmProgramName);
        deviceConfText += asmProgramName;
        deviceConfText += "\"";
        
        deviceConfText += " WorkSize = \"";
        deviceConfText += defaultWorkSizeString;
        deviceConfText += "\"";

        deviceConfText += " BatchTimeMs = \"";
        deviceConfText += defaultBatchTimeMsString;
        deviceConfText += "\"";

        deviceConfText += " OccupancyPct = \"";
        deviceConfText += defaultOccupancyPctString;
        deviceConfText += "\"";

        deviceConfText += " DeviceMonitor = \"";
        deviceConfText += defaultDeviceMonitorString;
        deviceConfText += "\"";

        deviceConfText += " GPUTemperatureLimit = \"";
        deviceConfText += defaultGPUTemperatureLimitString;
        deviceConfText += "\">\n";
    }

    // final composition
    configText += platformListText;
    configText += "#\n# Available devices:";
    configText += deviceListText;
    configText += "\n#\n#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n\n";
    configText += deviceConfText;
}

#ifdef HAVE_CUDA
/******************************************************************************
*  Generates configuration text data for selected OpenCL devices.
* Input
*  - cudevices     - All/Filtered CUDA devices.
* 
*  - rawDeviceList - true:  Generates configuration in raw device list format,
*                           All devices will be selected selected by index.
*                  - false: Generates configuration in PCIe bus device list format.
*                           All devices will be selected selected by PCIe bus Id.
* 
* Input&Output
*  - configText    - Configuration text data.
*******************************************************************************/

// TODO: PCIeBus id list is not supported
static void generateCUDADeviceConfig(const std::vector<cudevice_t>& cudevices,
                                     std::string& configText)
{
    const std::string defaultWorkSizeString(std::to_string(defaultWorkSize));
    const std::string defaultBatchTimeMsString(std::to_string(defaultBatchTimeMs));
    const std::string defaultDeviceMonitorString(std::to_string(defaultDeviceMonitor));
    const std::string defaultGPUTemperatureLimitString(std::to_string(defaultGPUTemperatureLimit));
    const std::string defaultOccupancyPctString(std::to_string(defaultOccupancyPct));
    std::string deviceConfText;
    std::string deviceListText;
    std::string deviceName;

    //-------------------------------------
    // CUDA devices
    configText += "#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n"
                  "# CUDA Device config:\n"
                  "#\n"
                  "# Available devices:";

    deviceListText.clear();
    deviceConfText.clear();
    for (size_t i = 0; i < cudevices.size(); ++i)
    {
        const vh::cudevice_t& cudevice = cudevices[i];

        cudaDeviceProp cudeviceProp;
        cudaGetDeviceProperties(&cudeviceProp, cudevice.cudeviceHandle);

        // get device info
        deviceListText += "\n#\n# DeviceIndex: ";
        deviceListText += std::to_string(i);

        deviceListText += "\n#    Name: ";
        // get device name
        size_t infoSize = 0;
        deviceName.clear();
        deviceName.assign(cudeviceProp.name);
        deviceListText += deviceName;

        deviceListText += "\n#    PCIeId: ";

        if (cudevice.pcieBusId == -1)
        {
            deviceListText += "not avilable";
        }
        else
        {
            char pcieStr[9] = { };
            snprintf (pcieStr, 8, "%02x:%02x:0",
                      cudevice.pcieBusId, cudevice.pcieDeviceId);
            deviceListText += std::string(pcieStr);
        }

        //  get device config
        deviceConfText += "<CU_Device";
        deviceConfText += std::to_string(i);

        deviceConfText += " DeviceIndex = \"";
        deviceConfText += std::to_string(i);
        deviceConfText += "\"";

        deviceConfText += " WorkSize = \"";
        deviceConfText += defaultWorkSizeString;
        deviceConfText += "\"";

        deviceConfText += " BatchTimeMs = \"";
        deviceConfText += defaultBatchTimeMsString;
        deviceConfText += "\"";

        deviceConfText += " OccupancyPct = \"";
        deviceConfText += defaultOccupancyPctString;
        deviceConfText += "\"";

        deviceConfText += " DeviceMonitor = \"";
        deviceConfText += defaultDeviceMonitorString;
        deviceConfText += "\"";

        deviceConfText += " GPUTemperatureLimit = \"";
        deviceConfText += defaultGPUTemperatureLimitString;
        deviceConfText += "\">\n";
    }

    configText += deviceListText;
    configText += "\n#\n#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n\n";
    configText += deviceConfText;
}
#endif

} // end namespace vh

#endif // !ConfigGenerator_INCLUDE_ONCE

