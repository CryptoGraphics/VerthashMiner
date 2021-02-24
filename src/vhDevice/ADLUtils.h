/*
 * Copyright 2021 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#ifndef ADLLoader_INCLUDE_ONCE
#define ADLLoader_INCLUDE_ONCE

#include "../external/adl/adl_sdk.h"

#ifdef _WIN32
#define ADLAPI __stdcall
#else
#define ADLAPI
#endif

#ifdef __cplusplus
extern "C"
{
#endif

//-----------------------------------------------------------------------------
// nvcu api
int adlInitApi(void);
void* adlGetProcAddress(const char* proc);

// Memory allocation function
void* ADLAPI ADL_Main_Memory_Alloc(int iSize);
// Optional Memory de-allocation function
void ADLAPI ADL_Main_Memory_Free(void** lpBuffer);
// ALD2_Overdrive_Temperature_Get
int ALD2_Overdrive_Temperature_Get(ADL_CONTEXT_HANDLE context, int iAdapterIndex, int oVersion, int *iTemperature);

//-----------------------------------------------------------------------------
typedef int (ADLAPI * PFNADL2MAINCONTROLCREATEPROC)(ADL_MAIN_MALLOC_CALLBACK callback, int iEnumConnectedAdapters, ADL_CONTEXT_HANDLE *context);
typedef int (ADLAPI * PFNADL2MAINCONTROLDESTROYPROC)(ADL_CONTEXT_HANDLE context);
typedef int (ADLAPI * PFNADL2ADAPTERNUMBEROFADAPTERSGETPROC)(ADL_CONTEXT_HANDLE context, int *lpNumAdapters);
typedef int (ADLAPI * PFNADL2ADAPTERADAPTERINFOGETPROC)(ADL_CONTEXT_HANDLE context, LPAdapterInfo lpInfo, int iInputSize); 
typedef int (ADLAPI * PFNADL2ADAPTERIDGETPROC)(ADL_CONTEXT_HANDLE context, int iAdapterIndex, int *lpAdapterID);
typedef int (ADLAPI * PFNADL2OVERDRIVECAPSPROC)(ADL_CONTEXT_HANDLE context, int iAdapterIndex, int* iSupported, int *iEnabled, int *iVersion);
typedef int (ADLAPI * PFNADL2OVERDRIVE5TEMPERATUREGETPROC)(ADL_CONTEXT_HANDLE context, int iAdapterIndex, int iThermalControllerIndex, ADLTemperature *lpTemperature);
typedef int (ADLAPI * PFNADL2OVERDRIVE6TEMPERATUREGETPROC)(ADL_CONTEXT_HANDLE context, int iAdapterIndex, int* lpTemperature);
typedef int (ADLAPI * PFNADL2OVERDRIVENTEMPERATUREGETPROC)(ADL_CONTEXT_HANDLE context, int iAdapterIndex, int iTemperatureType, int * iTemperature);
typedef int (ADLAPI * PFNADL2NEWQUERYPMLOGDATAGETPROC)(ADL_CONTEXT_HANDLE context, int iAdapterIndex, ADLPMLogDataOutput *lpDataOutput);

int ADL2_Main_Control_Create(ADL_MAIN_MALLOC_CALLBACK callback, int iEnumConnectedAdapters, ADL_CONTEXT_HANDLE *context);
int ADL2_Main_Control_Destroy(ADL_CONTEXT_HANDLE context);
int ADL2_Adapter_NumberOfAdapters_Get(ADL_CONTEXT_HANDLE context, int *lpNumAdapters);
int ADL2_Adapter_AdapterInfo_Get(ADL_CONTEXT_HANDLE context, LPAdapterInfo lpInfo, int iInputSize);   
int ADL2_Adapter_ID_Get(ADL_CONTEXT_HANDLE context, int iAdapterIndex, int* lpAdapterID);
int ADL2_Overdrive_Caps(ADL_CONTEXT_HANDLE context, int iAdapterIndex, int* iSupported, int *iEnabled, int *iVersion);
int ADL2_Overdrive5_Temperature_Get(ADL_CONTEXT_HANDLE context, int iAdapterIndex, int iThermalControllerIndex, ADLTemperature *lpTemperature);
int ADL2_Overdrive6_Temperature_Get(ADL_CONTEXT_HANDLE context, int iAdapterIndex, int *lpTemperature);
int ADL2_OverdriveN_Temperature_Get(ADL_CONTEXT_HANDLE context, int iAdapterIndex, int iTemperatureType, int *iTemperature);
int ADL2_New_QueryPMLogData_Get(ADL_CONTEXT_HANDLE context, int iAdapterIndex, ADLPMLogDataOutput *lpDataOutput);

//-----------------------------------------------------------------------------
extern PFNADL2MAINCONTROLCREATEPROC pADL2_Main_Control_Create;
extern PFNADL2MAINCONTROLDESTROYPROC pADL2_Main_Control_Destroy;
extern PFNADL2ADAPTERNUMBEROFADAPTERSGETPROC pADL2_Adapter_NumberOfAdapters_Get;
extern PFNADL2ADAPTERADAPTERINFOGETPROC pADL2_Adapter_AdapterInfo_Get;
extern PFNADL2ADAPTERIDGETPROC pADL2_Adapter_ID_Get;
extern PFNADL2OVERDRIVECAPSPROC pADL2_Overdrive_Caps;
extern PFNADL2OVERDRIVE5TEMPERATUREGETPROC pADL2_Overdrive5_Temperature_Get;
extern PFNADL2OVERDRIVE6TEMPERATUREGETPROC pADL2_Overdrive6_Temperature_Get;
extern PFNADL2OVERDRIVENTEMPERATUREGETPROC pADL2_OverdriveN_Temperature_Get;
extern PFNADL2NEWQUERYPMLOGDATAGETPROC pADL2_New_QueryPMLogData_Get;

//-----------------------------------------------------------------------------
#define ADL2_Main_Control_Create pADL2_Main_Control_Create
#define ADL2_Main_Control_Destroy pADL2_Main_Control_Destroy
#define ADL2_Adapter_NumberOfAdapters_Get pADL2_Adapter_NumberOfAdapters_Get
#define ADL2_Adapter_AdapterInfo_Get pADL2_Adapter_AdapterInfo_Get
#define ADL2_Adapter_ID_Get pADL2_Adapter_ID_Get
#define ADL2_Overdrive_Caps pADL2_Overdrive_Caps
#define ADL2_Overdrive5_Temperature_Get pADL2_Overdrive5_Temperature_Get
#define ADL2_Overdrive6_Temperature_Get pADL2_Overdrive6_Temperature_Get
#define ADL2_OverdriveN_Temperature_Get pADL2_OverdriveN_Temperature_Get
#define ADL2_New_QueryPMLogData_Get pADL2_New_QueryPMLogData_Get

#ifdef __cplusplus
}
#endif

#endif // !ADLLoader_INCLUDE_ONCE
