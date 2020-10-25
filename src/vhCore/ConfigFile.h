/*
 * Copyright 2018-2020 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */

#ifndef ConfigFile_INCLUDE_ONCE
#define ConfigFile_INCLUDE_ONCE

#include <string>
#include <map>
#include <stdarg.h>

namespace vh
{
    struct ConfigSetting
    {
        std::string AsString;
        bool AsBool;
        int AsInt;
        float AsFloat;
    };

    typedef std::map<uint32_t, ConfigSetting> ConfigBlock;

    class ConfigFile
    {
    public:
        ConfigFile();
        ~ConfigFile();

        bool setSource(const char *file, bool ignorecase = true);
        ConfigSetting * getSetting(const char * Block, const char * Setting);

        bool getString(const char * block, const char* name, std::string *value);
        std::string getStringDefault(const char* block, const char* name, const char* def);
        std::string getStringVA(const char* block, const char* def, const char* name, ...);
        bool getString(const char * block, char* buffer, const char* name, const char* def, uint32_t len);

        bool getBool(const char* block, const char* name, bool *value);
        bool getBoolDefault(const char* block, const char* name, const bool def);

        bool getInt(const char* block, const char* name, int *value);
        int getIntDefault(const char* block, const char* name, const int def);
        int getIntVA(const char* block, int def, const char* name, ...);

        bool getFloat(const char* block, const char* name, float *value);
        float getFloatDefault(const char* block, const char* name, const float def);
        float getFloatVA(const char* block, float def, const char* name, ...);

    private:
        std::map<uint32_t, ConfigBlock> m_settings;
    };
}

#endif // !ConfigFile_INCLUDE_ONCE

