/*
 * Copyright 2021 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#ifdef __linux__

#include "SYSFSUtils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dirent.h>
#include <unistd.h>

#define HWMON_PATH_MAX_LEN 0x1000
static const char SYSFS_BUS_PCI_DEVICES[] = "/sys/bus/pci/devices";

//-----------------------------------------------------------------------------
static char *first_file_in_directory(const char *path)
{
    DIR *d = NULL;
    if ((d = opendir(path)) != NULL)
    {
        char *first_file = NULL;

        struct dirent *de;
        while ((de = readdir(d)) != NULL)
        {
            if (de->d_name[0] == '.')
            {
                continue;
            }

            first_file = strdup(de->d_name);
            break;
        }

        closedir(d);
        return first_file;
    }

    return NULL;
}

//-----------------------------------------------------------------------------
bool SYSFS_init()
{
    if (access(SYSFS_BUS_PCI_DEVICES, R_OK) == -1)
    {
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
char* SYSFS_get_syspath_device(int pcie_bus, int pcie_device, int pcie_function)
{
    char* syspath = NULL;
    asprintf(&syspath, "%s/0000:%02x:%02x.%01x", SYSFS_BUS_PCI_DEVICES, pcie_bus, pcie_device, pcie_function);
    return syspath;
}

//-----------------------------------------------------------------------------
char* SYSFS_get_syspath_hwmon(const char* syspath_device)
{
    if (syspath_device == NULL)
    {
        return NULL;
    }

    char *hwmon = (char*)malloc(HWMON_PATH_MAX_LEN);
    snprintf (hwmon, HWMON_PATH_MAX_LEN, "%s/hwmon", syspath_device);

    char *hwmonN = first_file_in_directory(hwmon);
    if (hwmonN == NULL)
    {
        printf("First_file_in_directory() failed!\n");
        fflush(stdout);

        free (hwmon);
        free (hwmonN);

        return NULL;
    }

    snprintf (hwmon, HWMON_PATH_MAX_LEN, "%s/hwmon/%s", syspath_device, hwmonN);
    free (hwmonN);

    return hwmon;
}

//-----------------------------------------------------------------------------
char* SYSFS_get_syspath_pwm1(const char* syspath_hwmon)
{
    char* syspath = NULL;
    asprintf (&syspath, "%s/pwm1", syspath_hwmon);
    return syspath;
}

//-----------------------------------------------------------------------------
char* SYSFS_get_syspath_pwm1_max(const char* syspath_hwmon)
{
    char* syspath = NULL;
    asprintf (&syspath, "%s/pwm1_max", syspath_hwmon);
    return syspath;
}

//-----------------------------------------------------------------------------
char* SYSFS_get_syspath_power1_average(const char* syspath_hwmon)
{
    char* syspath = NULL;
    asprintf (&syspath, "%s/power1_average", syspath_hwmon);
    return syspath;
}

//-----------------------------------------------------------------------------
char* SYSFS_get_syspath_temp1_input(const char* syspath_hwmon)
{
    char* syspath = NULL;
    asprintf (&syspath, "%s/temp1_input", syspath_hwmon);
    return syspath;
}

//-----------------------------------------------------------------------------
int SYSFS_get_temperature(const char* syspath_temp1_input, int* out_val)
{
    if (syspath_temp1_input == NULL)
    {
        return -1;
    }

    FILE* fp = fopen(syspath_temp1_input, "r");
    if(fp == NULL)
    {
        printf("Failed to read path: %s\n", syspath_temp1_input);
        fflush(stdout);

        return -1;
    }

    int temperature = 0;
    if(fscanf(fp, "%d", &temperature) != 1)
    {
        fclose(fp);

        printf("%s: unexpected data.", syspath_temp1_input);
        fflush(stdout);

        return -1;
    }

    fclose(fp);

    *out_val = temperature / 1000;

    return 0;
}

//-----------------------------------------------------------------------------
int SYSFS_get_power_usage(const char* syspath_power1_average, int* out_val)
{
    if (syspath_power1_average == NULL)
    {
        return -1;
    }

    FILE* fp = fopen(syspath_power1_average, "r");
    if (fp == NULL)
    {
        printf("Failed to read path: %s\n", syspath_power1_average);
        fflush(stdout);

        return -1;
    }

    int power = 0;
    if(fscanf(fp, "%d", &power) != 1)
    {
        fclose(fp);

        printf("%s: unexpected data.", syspath_power1_average);
        fflush(stdout);

        return -1;
    }

    fclose(fp);

    *out_val = power / 1000000;

    return 0;
}

//-----------------------------------------------------------------------------
int SYSFS_get_fan_speed(const char* syspath_pwm1, const char* syspath_pwm1_max, int* out_val)
{
    if ((syspath_pwm1 == NULL) || (syspath_pwm1_max == NULL))
    {
        return -1;
    }

    FILE* fp_cur = fopen(syspath_pwm1, "r");
    if (fp_cur == NULL)
    {
        printf("Failed to read path: %s\n", syspath_pwm1);
        fflush(stdout);

        return -1;
    }

    int pwm1_cur = 0;
    if (fscanf(fp_cur, "%d", &pwm1_cur) != 1)
    {
        fclose(fp_cur);
        printf("%s: unexpected data.", syspath_pwm1);

        return -1;
    }

    fclose (fp_cur);

    FILE* fp_max = fopen(syspath_pwm1_max, "r");
    if (fp_max == NULL)
    {
        printf("Failed to read path: %s\n", syspath_pwm1_max);
        fflush(stdout);

        return -1;
    }

    int pwm1_max = 0;
    if (fscanf(fp_max, "%d", &pwm1_max) != 1)
    {
        fclose(fp_max);

        printf("%s: unexpected data.", syspath_pwm1_max);
        fflush(stdout);

        return -1;
    }

    fclose(fp_max);

    if (pwm1_max == 0)
    {
        printf("%s: pwm1_max cannot be 0.", syspath_pwm1_max);
        fflush(stdout);

        return -1;
    }

    const float p1 = (float) pwm1_max / 100.0F;
    const float pwm1_percent = (float) pwm1_cur / p1;
    *out_val = (int) pwm1_percent;

    return 0;
}

//-----------------------------------------------------------------------------

#endif // __linux__

