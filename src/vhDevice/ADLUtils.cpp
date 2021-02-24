/*
 * Copyright 2021 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#include "ADLUtils.h"
#include <stdlib.h> // atexit

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static HMODULE libadl = NULL;

static void open_libadl(void)
{
    // atiadlxx.dll(64 bit), atiadlxy.dll(32 bit)
    libadl = LoadLibraryW(L"atiadlxx.dll");
}

static void close_libadl(void)
{
    FreeLibrary(libadl);
    libadl = NULL;
}

static void* get_proc(const char* proc)
{
    void *res = (void*)GetProcAddress(libadl, proc);
    return res;
}

#else

#include <dlfcn.h>

static void *libadl;

static void open_libadl(void)
{
    libadl = dlopen("libatiadlxx.so", RTLD_LAZY | RTLD_GLOBAL);
}

static void close_libadl(void)
{
    dlclose(libadl);
    libadl = NULL;
}

static void* get_proc(const char* proc)
{
    void* res = dlsym(libadl, proc);
    return res;
}

#endif

static void adlExit(void)
{
    if (libadl != NULL)
    {
        close_libadl();
    }
}

void* adlGetProcAddress(const char* proc)
{
    return get_proc(proc);
}

//-----------------------------------------------------------------------------
PFNADL2MAINCONTROLCREATEPROC pADL2_Main_Control_Create = NULL;
PFNADL2MAINCONTROLDESTROYPROC pADL2_Main_Control_Destroy = NULL;
PFNADL2ADAPTERNUMBEROFADAPTERSGETPROC pADL2_Adapter_NumberOfAdapters_Get = NULL;
PFNADL2ADAPTERADAPTERINFOGETPROC pADL2_Adapter_AdapterInfo_Get = NULL;
PFNADL2ADAPTERIDGETPROC pADL2_Adapter_ID_Get = NULL;
PFNADL2OVERDRIVECAPSPROC pADL2_Overdrive_Caps = NULL;
PFNADL2OVERDRIVE5TEMPERATUREGETPROC pADL2_Overdrive5_Temperature_Get = NULL;
PFNADL2OVERDRIVE6TEMPERATUREGETPROC pADL2_Overdrive6_Temperature_Get = NULL;
PFNADL2OVERDRIVENTEMPERATUREGETPROC pADL2_OverdriveN_Temperature_Get = NULL;
PFNADL2NEWQUERYPMLOGDATAGETPROC pADL2_New_QueryPMLogData_Get = NULL;


//-----------------------------------------------------------------------------
int adlInitApi(void)
{
    // Check if already initialized
    if (libadl != NULL)
        return 0; // SUCCESS;

    // Load library
    open_libadl();
    if (libadl == NULL)
        return -1; // ERROR_OPEN_FAILED;

    // Set unloading
    int errorCode = atexit(adlExit);
    if (errorCode)
    {
        //  Failure queuing atexit, shutdown with error
        close_libadl();
        return -2; // ERROR_ATEXIT_FAILED;
    }

    pADL2_Main_Control_Create = (PFNADL2MAINCONTROLCREATEPROC)get_proc("ADL2_Main_Control_Create");
    pADL2_Main_Control_Destroy = (PFNADL2MAINCONTROLDESTROYPROC)get_proc("ADL2_Main_Control_Destroy");
    pADL2_Adapter_NumberOfAdapters_Get = (PFNADL2ADAPTERNUMBEROFADAPTERSGETPROC)get_proc("ADL2_Adapter_NumberOfAdapters_Get");
    pADL2_Adapter_AdapterInfo_Get = (PFNADL2ADAPTERADAPTERINFOGETPROC)get_proc("ADL2_Adapter_AdapterInfo_Get");
    pADL2_Adapter_ID_Get = (PFNADL2ADAPTERIDGETPROC)get_proc("ADL2_Adapter_ID_Get");
    pADL2_Overdrive_Caps = (PFNADL2OVERDRIVECAPSPROC)get_proc("ADL2_Overdrive_Caps");
    pADL2_Overdrive5_Temperature_Get = (PFNADL2OVERDRIVE5TEMPERATUREGETPROC)get_proc("ADL2_Overdrive5_Temperature_Get");
    pADL2_Overdrive6_Temperature_Get = (PFNADL2OVERDRIVE6TEMPERATUREGETPROC)get_proc("ADL2_Overdrive6_Temperature_Get");
    pADL2_OverdriveN_Temperature_Get = (PFNADL2OVERDRIVENTEMPERATUREGETPROC)get_proc("ADL2_OverdriveN_Temperature_Get");
    pADL2_New_QueryPMLogData_Get = (PFNADL2NEWQUERYPMLOGDATAGETPROC)get_proc("ADL2_New_QueryPMLogData_Get");

    // Validate function pointers
    if (NULL == pADL2_Main_Control_Create ||
        NULL == pADL2_Main_Control_Destroy ||
        NULL == pADL2_Adapter_NumberOfAdapters_Get ||
        NULL == pADL2_Adapter_AdapterInfo_Get ||
        NULL == pADL2_Adapter_ID_Get ||
        NULL == pADL2_Overdrive_Caps ||
        NULL == pADL2_Overdrive5_Temperature_Get ||
        NULL == pADL2_Overdrive6_Temperature_Get ||
        NULL == pADL2_OverdriveN_Temperature_Get ||
        NULL == pADL2_New_QueryPMLogData_Get)
    {
        return -3; // ERROR_FAILED_TO_RETRIEVE_FUNCTIONS;
    }


    return 0;
}

//-----------------------------------------------------------------------------
void* ADLAPI ADL_Main_Memory_Alloc(int iSize)
{
    void* lpBuffer = malloc(iSize);
    return lpBuffer;
}

//-----------------------------------------------------------------------------
// Optional Memory de-allocation function
void ADLAPI ADL_Main_Memory_Free(void** lpBuffer)
{
    if (*lpBuffer != NULL)
    {
        free(*lpBuffer);
        *lpBuffer = NULL;
    }
}

//-----------------------------------------------------------------------------
int ALD2_Overdrive_Temperature_Get(ADL_CONTEXT_HANDLE context, int iAdapterIndex, int oVersion, int *iTemperature)
{
    int temperature = 0;
    int adlRC = ADL_OK;

    if (oVersion == 8)
    {
        // Overdrive8 GPU (Radeon VII, RX 5000 series, RX 6000 series)
        ADLPMLogDataOutput adlPMLogDataOutput = {};
        adlRC = ADL2_New_QueryPMLogData_Get(context, iAdapterIndex, &adlPMLogDataOutput);
        if (adlRC == ADL_OK)
        {
            if (adlPMLogDataOutput.sensors[PMLOG_TEMPERATURE_EDGE].supported == 1)
            {
                temperature = adlPMLogDataOutput.sensors[PMLOG_TEMPERATURE_EDGE].value;
            }
        }
    }
    else if (oVersion == 7)
    {
        // OverdriveN (290, 290x, 380, 380x, 390, 390x, Fury, Fury X, Nano, 4xx, 5xx, Vega 56, Vega 64)
        adlRC = ADL2_OverdriveN_Temperature_Get(context, iAdapterIndex, 1, &temperature);
    }
    else if (oVersion == 6)
    {
        // GCN 1.0
        adlRC = ADL2_Overdrive6_Temperature_Get(context, iAdapterIndex, &temperature);
    }
    else if (oVersion == 5)
    {
        // Pre GCN
        ADLTemperature adlTemperature = {};
        adlRC = ADL2_Overdrive5_Temperature_Get(context, iAdapterIndex, 0, &adlTemperature);
        if (adlRC == ADL_OK)
        {
            temperature = adlTemperature.iTemperature;
        }
    }

    *iTemperature = temperature/1000;

    return adlRC;
}
//-----------------------------------------------------------------------------
