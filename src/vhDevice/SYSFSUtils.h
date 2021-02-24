/*
 * Copyright 2021 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#ifndef SYSFSUtils_INCLUDE_ONCE
#define SYSFSUtils_INCLUDE_ONCE

#ifdef __linux__

//-----------------------------------------------------------------------------
// Check if SYSFS is available
bool SYSFS_init();

//-----------------------------------------------------------------------------
// Utility functions to get paths for monitoring.
// Every function allocates and returns a path, which needs to be freed manually.
char* SYSFS_get_syspath_device(int pcie_bus, int pcie_device, int pcie_function);
char* SYSFS_get_syspath_hwmon(const char* syspath_device);
char* SYSFS_get_syspath_pwm1(const char* syspath_hwmon);
char* SYSFS_get_syspath_pwm1_max(const char* syspath_hwmon);
char* SYSFS_get_syspath_power1_average(const char* syspath_hwmon);
char* SYSFS_get_syspath_temp1_input(const char* syspath_hwmon);

//-----------------------------------------------------------------------------
// Monitoring funtions.

//! Retrieves the current temperature readings for the device, in degrees C.
int SYSFS_get_temperature(const char* syspath_temp1_input, int* out_val);

//! Retrieves the intended operating speed(percentage) of the device's fan.
int SYSFS_get_fan_speed(const char* syspath_pwm1, const char* syspath_pwm1_max, int* out_val);

//! Retrieves power usage for this GPU in watts.
int SYSFS_get_power_usage(const char* syspath_power1_average, int* out_val);

//-----------------------------------------------------------------------------

#endif // __linux__

#endif // !SYSFSUtils_INCLUDE_ONCE
