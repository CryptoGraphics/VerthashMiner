/*
 * Copyright 2021 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#include "NVMLUtils.h"
#include <stdlib.h> // atexit

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static HMODULE libnvml = NULL;

static void open_libnvml(void)
{
    libnvml = LoadLibraryW(L"nvml.dll");
    if (libnvml == NULL)
    {
        // library was not found, try to get data from registry...
    	HKEY hKey = 0;
    	LONG errorCode = RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\NVIDIA Corporation\\Global\\NVSMI", 0, KEY_QUERY_VALUE, &hKey);

    	if (errorCode == ERROR_SUCCESS)
    	{
    		DWORD valueSizeBytes = 0;
    		DWORD valueType = REG_SZ; // A null-terminated string
    		RegQueryValueExW(hKey, L"NVSMIPATH", NULL, &valueType, NULL, &valueSizeBytes);

    		// library name
    		const wchar_t* libName = L"\\nvml.dll";
    		const size_t libNameLen = wcslen(libName);

    		// According to the documentation:
    		// The string may not have been stored with the proper terminating null characters,
    		// so add +1 to the "valueSizeBytes"
    		const size_t fullNameLen = (size_t)(valueSizeBytes >> sizeof(wchar_t)-1) + 1 + libNameLen;
    		wchar_t* fullName = (wchar_t*)calloc(fullNameLen, sizeof(wchar_t));
    		if (fullName != NULL)
    		{
    			errorCode = RegQueryValueExW(hKey, L"NVSMIPATH", NULL, &valueType, (LPBYTE)fullName, &valueSizeBytes);
    			if (errorCode == ERROR_SUCCESS)
    			{
                    // append libName to the end
    				wcscat_s(fullName, fullNameLen, libName);
    				// final attempt(NULL check is done inside InitApi)
    				libnvml = LoadLibraryW(fullName);
    			}
    		}

    		free(fullName);

    		RegCloseKey(hKey);
    	}
    }

    // hardcoded paths are not reliable
    //const char* libnvidia_ml = "%PROGRAMFILES%\\NVIDIA Corporation\\NVSMI\\nvml.dll";
    //char tmp[512];
    //ExpandEnvironmentStrings(libnvidia_ml, tmp, sizeof(tmp));
    //libnvml = LoadLibrary(tmp);
}

static void close_libnvml(void)
{
    FreeLibrary(libnvml);
    libnvml = NULL;
}

static void* get_proc(const char* proc)
{
    void *res = (void*)GetProcAddress(libnvml, proc);
    return res;
}

#else

#include <dlfcn.h>

static void *libnvml;

static void open_libnvml(void)
{
    // RTLD_NOW instead of RTLD_LAZY
    libnvml = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_GLOBAL);
}

static void close_libnvml(void)
{
    dlclose(libnvml);
    libnvml = NULL;
}

static void* get_proc(const char* proc)
{
    void* res = dlsym(libnvml, proc);
    return res;
}

#endif

static void mlExit(void)
{
    if (libnvml != NULL)
        close_libnvml();
}

void* mlGetProcAddress(const char* proc)
{
    return get_proc(proc);
}

//-----------------------------------------------------------------------------
PFNNVMLINITPROC mlInit = NULL;
PFNNVMLINITWITHFLAGSPROC mlInitWithFlags = NULL;
PFNNVMLSHUTDOWNPROC mlShutdown = NULL;
PFNNVMLERRORSTRINGPROC mlErrorString = NULL;
PFNNVMLDEVICEGETCOUNTPROC mlDeviceGetCount = NULL;
PFNNVMLDEVICEGETHANDLEBYINDEXPROC mlDeviceGetHandleByIndex = NULL;
PFNNVMLDEVICEGETNAMEPROC mlDeviceGetName = NULL;
PFNNVMLDEVICEGETPCIINFOPROC mlDeviceGetPciInfo = NULL;
PFNNVMLDEVICEGETUUIDPROC mlDeviceGetUUID = NULL;
PFNNVMLDEVICEGETTEMPERATUREPROC mlDeviceGetTemperature = NULL;
PFNNVMLDEVICEGETPOWERUSAGEPROC mlDeviceGetPowerUsage = NULL;
PFNNVMLDEVICEGETFANSPEEDPROC mlDeviceGetFanSpeed = NULL;


//-----------------------------------------------------------------------------
int nvmlInitApi(void)
{
    // Check if already initialized
    if (libnvml != NULL)
        return 0; // SUCCESS;

    // Load library
    open_libnvml();
    if (libnvml == NULL)
        return -1; // ERROR_OPEN_FAILED;

    // Set unloading
    int errorCode = atexit(mlExit);
    if (errorCode)
    {
        // Failure queuing atexit, shutdown with error
        close_libnvml();
        return -2; // ERROR_ATEXIT_FAILED;
    }

    mlInit = (PFNNVMLINITPROC)get_proc("nvmlInit_v2");
    mlInitWithFlags = (PFNNVMLINITWITHFLAGSPROC)get_proc("nvmlInitWithFlags");
    mlShutdown = (PFNNVMLSHUTDOWNPROC)get_proc("nvmlShutdown");
    mlErrorString = (PFNNVMLERRORSTRINGPROC)get_proc("nvmlErrorString");
    mlDeviceGetCount = (PFNNVMLDEVICEGETCOUNTPROC)get_proc("nvmlDeviceGetCount_v2");
    mlDeviceGetHandleByIndex = (PFNNVMLDEVICEGETHANDLEBYINDEXPROC)get_proc("nvmlDeviceGetHandleByIndex_v2");
    mlDeviceGetName = (PFNNVMLDEVICEGETNAMEPROC)get_proc("nvmlDeviceGetName");
    mlDeviceGetPciInfo = (PFNNVMLDEVICEGETPCIINFOPROC)get_proc("nvmlDeviceGetPciInfo_v3");
    mlDeviceGetUUID = (PFNNVMLDEVICEGETUUIDPROC)get_proc("nvmlDeviceGetUUID");
    mlDeviceGetTemperature = (PFNNVMLDEVICEGETTEMPERATUREPROC)get_proc("nvmlDeviceGetTemperature");
    mlDeviceGetPowerUsage = (PFNNVMLDEVICEGETPOWERUSAGEPROC)get_proc("nvmlDeviceGetPowerUsage");
    mlDeviceGetFanSpeed = (PFNNVMLDEVICEGETFANSPEEDPROC)get_proc("nvmlDeviceGetFanSpeed");

    // Validate function pointers
    if (NULL == mlInit ||
        NULL == mlInitWithFlags ||
        NULL == mlErrorString ||
        NULL == mlDeviceGetCount ||
        NULL == mlDeviceGetHandleByIndex ||
        NULL == mlDeviceGetName ||
        NULL == mlDeviceGetPciInfo ||
        NULL == mlDeviceGetUUID ||
        NULL == mlDeviceGetTemperature ||
        NULL == mlDeviceGetPowerUsage ||
        NULL == mlDeviceGetFanSpeed)
    {
        return -3; // ERROR_FAILED_TO_RETRIEVE_FUNCTIONS;
    }

    return 0;
}
//-----------------------------------------------------------------------------
