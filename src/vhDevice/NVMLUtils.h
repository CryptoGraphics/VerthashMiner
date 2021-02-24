/*
 * Copyright 2021 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#ifndef NVML_INCLUDE_ONCE
#define NVML_INCLUDE_ONCE

//#include <nvml.h>
#include "../external/nvml/nvml.h"

#ifdef _WIN32
#define NVMLAPI __stdcall
#else
#define NVMLAPI
#endif

#ifdef __cplusplus
extern "C"
{
#endif

int nvmlInitApi(void);
void* nvmlGetProcAddress(const char* proc);

typedef nvmlReturn_t (NVMLAPI * PFNNVMLINITPROC)(void);
typedef nvmlReturn_t (NVMLAPI * PFNNVMLINITWITHFLAGSPROC)(unsigned int flags);
typedef nvmlReturn_t (NVMLAPI * PFNNVMLSHUTDOWNPROC)(void);
typedef const char* (NVMLAPI * PFNNVMLERRORSTRINGPROC)(nvmlReturn_t result);
typedef nvmlReturn_t (NVMLAPI * PFNNVMLDEVICEGETCOUNTPROC)(unsigned int *deviceCount);
typedef nvmlReturn_t (NVMLAPI * PFNNVMLDEVICEGETHANDLEBYINDEXPROC)(unsigned int index, nvmlDevice_t *device);
typedef nvmlReturn_t (NVMLAPI * PFNNVMLDEVICEGETNAMEPROC)(nvmlDevice_t device, char *name, unsigned int length);
typedef nvmlReturn_t (NVMLAPI * PFNNVMLDEVICEGETPCIINFOPROC)(nvmlDevice_t device, nvmlPciInfo_t *pci);
typedef nvmlReturn_t (NVMLAPI * PFNNVMLDEVICEGETUUIDPROC)(nvmlDevice_t device, char *uuid, unsigned int length);
typedef nvmlReturn_t (NVMLAPI * PFNNVMLDEVICEGETTEMPERATUREPROC)(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int *temp);
typedef nvmlReturn_t (NVMLAPI * PFNNVMLDEVICEGETPOWERUSAGEPROC)(nvmlDevice_t device, unsigned int *power);
typedef nvmlReturn_t (NVMLAPI * PFNNVMLDEVICEGETFANSPEEDPROC)(nvmlDevice_t device, unsigned int *speed);

extern PFNNVMLINITPROC mlInit;
extern PFNNVMLINITWITHFLAGSPROC mlInitWithFlags;
extern PFNNVMLSHUTDOWNPROC mlShutdown;
extern PFNNVMLERRORSTRINGPROC mlErrorString;
extern PFNNVMLDEVICEGETCOUNTPROC mlDeviceGetCount;
extern PFNNVMLDEVICEGETHANDLEBYINDEXPROC mlDeviceGetHandleByIndex;
extern PFNNVMLDEVICEGETNAMEPROC mlDeviceGetName;
extern PFNNVMLDEVICEGETPCIINFOPROC mlDeviceGetPciInfo;
extern PFNNVMLDEVICEGETUUIDPROC mlDeviceGetUUID;
extern PFNNVMLDEVICEGETTEMPERATUREPROC mlDeviceGetTemperature;
extern PFNNVMLDEVICEGETPOWERUSAGEPROC mlDeviceGetPowerUsage;
extern PFNNVMLDEVICEGETFANSPEEDPROC mlDeviceGetFanSpeed;

#define nvmlInit_v2 mlInit
#define nvmlInitWithFlags mlInitWithFlags
#define nvmlShutdown mlShutdown
#define nvmlErrorString mlErrorString
#define nvmlDeviceGetCount_v2 mlDeviceGetCount
#define nvmlDeviceGetHandleByIndex_v2 mlDeviceGetHandleByIndex
#define nvmlDeviceGetName mlDeviceGetName
#define nvmlDeviceGetPciInfo_v3 mlDeviceGetPciInfo
#define nvmlDeviceGetUUID mlDeviceGetUUID
#define nvmlDeviceGetTemperature mlDeviceGetTemperature
#define nvmlDeviceGetPowerUsage mlDeviceGetPowerUsage
#define nvmlDeviceGetFanSpeed mlDeviceGetFanSpeed


#ifdef __cplusplus
}
#endif

#endif // !NVML_INCLUDE_ONCE
