
/*
 * Copyright 2018-2020 CryptoGraphics
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version. See LICENSE for more details.
 */


#include <cassert>
#include <iostream>
#include <cstring> 
#include "ConfigFile.h"

using namespace vh;

// MSVC compatibility
#ifdef _MSC_VER 
#define strncasecmp(x,y,z) _strnicmp(x,y,z)
#define strcasecmp(x,y) _stricmp(x,y)
#endif

//-----------------------------------------------------------------------------
// ConfigFile class implementation.
//-----------------------------------------------------------------------------
ConfigFile::ConfigFile() { }
//-----------------------------------------------------------------------------
ConfigFile::~ConfigFile() { }
//-----------------------------------------------------------------------------
void remove_spaces(std::string& str)
{
    while(str.size() && (*str.begin() == ' ' || *str.begin() == '\t'))
    {
        str.erase(str.begin());
    }
}
//-----------------------------------------------------------------------------
void remove_all_spaces(std::string& str)
{
    std::string::size_type off = str.find(" ");
    while(off != std::string::npos)
    {
        str.erase(off, 1);
        off = str.find(" ");
    }
}
//-----------------------------------------------------------------------------
bool is_comment(std::string& str, bool * in_multiline_quote)
{
    std::string stemp = str;
    remove_spaces(stemp);
    if(stemp.length() == 0)
    {
        return false;
    }

    if(stemp[0] == '/')
    {
        if(stemp.length() < 2)
            return false;

        if(stemp[1] == '*')
        {
            *in_multiline_quote = true;
            return true;
        }
        else if(stemp[1] == '/')
        {
            return true;
        }
    }

    if(stemp[0] == '#')
    {
        return true;
    }

    return false;
}
//-----------------------------------------------------------------------------
void apply_setting(std::string & str, ConfigSetting & setting)
{
    setting.AsString = str;
    setting.AsInt = atoi(str.c_str());
    setting.AsBool = (setting.AsInt > 0);
    setting.AsFloat = (float)atof(str.c_str());

    if(str.length() > 1)
    {
        // strncasecmp
        if(str.size() >= 4 && !strncasecmp("true", str.c_str(), 4))
        {
            setting.AsBool = true;
            setting.AsInt = 1;
        }
        else if(str.size() >= 5 && !strncasecmp("false", str.c_str(), 5))
        {
            setting.AsBool = false;
            setting.AsInt = 0;
        }
    }
}
//-----------------------------------------------------------------------------
uint32_t ahash(const char* str)
{
    register size_t len = strlen(str);
    register uint32_t ret = 0;
    register size_t i = 0;
    for(; i < len; ++i)
    {
        ret += 5 * ret + (tolower(str[i]));
    }

    return ret;
}
//-----------------------------------------------------------------------------
uint32_t ahash(std::string& str)
{
    return ahash(str.c_str());
}
//-----------------------------------------------------------------------------
bool ConfigFile::setSource(const char *file, bool ignorecase)
{
    m_settings.clear();

    // open the file
    if(file != 0)
    {
        FILE * f = fopen(file, "r");
        char * buf;
        size_t length;
        if(!f)
            return false;

        // get the length of the file
        fseek(f, 0, SEEK_END);
        length = ftell(f);
        // prevent skipping last line
        buf = new char[length + 2];
        fseek(f, 0, SEEK_SET);

        fread(buf, 1, length, f);
        
        // prevent skipping last line
        buf[length] = '\n';
        buf[length+1] = '\0';
        std::string buffer = std::string(buf);
        delete [] buf;

        fclose(f);

        // start parcing.
        std::string line;
        std::string::size_type end;
        std::string::size_type offset;
        bool in_multiline_comment = false;
        bool in_multiline_quote = false;
        bool in_block = false;
        std::string current_setting = "";
        std::string current_variable = "";
        std::string current_block = "";
        ConfigBlock current_block_map;
        ConfigSetting current_setting_struct;

        for(;;)
        {
            // grab a line.
            end = buffer.find("\n");
            if(end == std::string::npos)
                break;

            line = buffer.substr(0, end);
            buffer.erase(0, end+1);
            goto parse;

parse:
            if(!line.size())
                continue;

            // are we a comment?
            if(!in_multiline_comment && is_comment(line, &in_multiline_comment))
            {
                // our line is a comment.
                if(!in_multiline_comment)
                {
                    // the entire line is a comment, skip it.
                    continue;
                }
            }

            // handle our cases
            if(in_multiline_comment)
            {
                offset = line.find("*/", 0);

                // skip this entire line
                if(offset == std::string::npos)
                    continue;

                // remove up to the end of the comment block.
                line.erase(0, offset + 2);
                in_multiline_comment = false;
            }

            if(in_block)
            {
                // handle settings across multiple lines
                if(in_multiline_quote)
                {
                    // attempt to find the end of the quote block.
                    offset = line.find("\"");

                    if(offset == std::string::npos)
                    {
                        // append the whole line to the quote.
                        current_setting += line;
                        current_setting += "\n";
                        continue;
                    }

                    // only append part of the line to the setting.
                    current_setting.append(line.c_str(), offset+1);
                    line.erase(0, offset + 1);

                    // append the setting to the config block.
                    if(current_block == "" || current_variable == "")
                    {
                        std::cerr << "Error: Quote without variable." << std::endl;
                        return false;
                    }

                    // apply the setting
                    apply_setting(current_setting, current_setting_struct);

                    // the setting is done, append it to the current block.
                    current_block_map[ahash(current_variable)] = current_setting_struct;
#ifndef NDEBUG
                    printf("Block: '%s', Setting: '%s', Value: '%s'\n", current_block.c_str(), current_variable.c_str(), current_setting_struct.AsString.c_str());
#endif
                    // no longer doing this setting, or in a quote.
                    current_setting = "";
                    current_variable = "";
                    in_multiline_quote = false;                 
                }

                // remove any leading spaces
                remove_spaces(line);

                if(!line.size())
                    continue;

                // our target is a *setting*. look for an '=' sign, this is our seperator.
                offset = line.find("=");
                if(offset != std::string::npos)
                {
                    assert(current_variable == "");
                    current_variable = line.substr(0, offset);

                    // remove any spaces from the end of the setting
                    remove_all_spaces(current_variable);

                    // remove the directive *and* the = from the line
                    line.erase(0, offset + 1);
                }

                // look for the opening quote. this signifies the start of a setting.
                offset = line.find("\"");
                if(offset != std::string::npos)
                {
                    assert(current_setting == "");
                    assert(current_variable != "");

                    // try and find the ending quote
                    end = line.find("\"", offset + 1);
                    if(end != std::string::npos)
                    {
                        // the closing quote is on the same line.
                        current_setting = line.substr(offset+1, end-offset-1);

                        // erase up to the end
                        line.erase(0, end + 1);

                        // apply the setting
                        apply_setting(current_setting, current_setting_struct);

                        // the setting is done, append it to the current block.
                        current_block_map[ahash(current_variable)] = current_setting_struct;

#ifndef NDEBUG
                        printf("Block: '%s', Setting: '%s', Value: '%s'\n", current_block.c_str(), current_variable.c_str(), current_setting_struct.AsString.c_str());
#endif
                        // no longer doing this setting, or in a quote.
                        current_setting = "";
                        current_variable = "";
                        in_multiline_quote = false;

                        // attempt to grab more settings from the same line.
                        goto parse;
                    }
                    else
                    {
                        // the closing quote is not on the same line. means we'll try and find it on the next.
                        current_setting.append(line.c_str(), offset);

                        // skip to the next line. (after setting our condition first)
                        in_multiline_quote = true;
                        continue;
                    }
                }

                // are we at the end of the block yet?
                offset = line.find(">");
                if(offset != std::string::npos)
                {
                    line.erase(0, offset+1);

                    // free
                    in_block = false;

                    // assign this block to the main "big" map.
                    m_settings[ahash(current_block)] = current_block_map;

                    // erase all data for this so it doesn't seep through
                    current_block_map.clear();
                    current_setting = "";
                    current_variable = "";
                    current_block = "";
                }
            }
            else
            {
                // we're not in a block. look for the start of one.
                offset = line.find("<");

                if(offset != std::string::npos)
                {
                    in_block = true;

                    line.erase(0, offset + 1);

                    // find the name of the block first.
                    offset = line.find(" ");
                    if(offset != std::string::npos)
                    {
                        current_block = line.substr(0, offset);
                        line.erase(0, offset + 1);
                    }
                    else
                    {
                        std::cerr << "Error: Block without name." << std::endl;
                        return false;
                    }

                    // skip back
                    goto parse;
                }
            }
        }

        // handle any errors
        if(in_block)
        {
            std::cerr << "Error: Unterminated block." << std::endl;
            return false;
        }

        if(in_multiline_comment)
        {
            std::cerr << "Error: Unterminated comment." << std::endl;
            return false;
        }

        if(in_multiline_quote)
        {
            std::cerr << "Error: Unterminated quote." << std::endl;
            return false;
        }

        return true;
    }

    return false;
}
//-----------------------------------------------------------------------------
ConfigSetting * ConfigFile::getSetting(const char * Block, const char * Setting)
{
    uint32_t block_hash = ahash(Block);
    uint32_t setting_hash = ahash(Setting);

    // find it in the big map
    std::map<uint32_t, ConfigBlock>::iterator itr = m_settings.find(block_hash);
    if(itr != m_settings.end())
    {
        ConfigBlock::iterator it2 = itr->second.find(setting_hash);
        if(it2 != itr->second.end())
            return &(it2->second);

        return 0;
    }

    return 0;
}
//-----------------------------------------------------------------------------
bool ConfigFile::getString(const char * block, const char* name, std::string *value)
{
    ConfigSetting * Setting = getSetting(block, name);
    if(Setting == 0)
    {
        return false;
    }

    *value = Setting->AsString;
    return true;
}
//-----------------------------------------------------------------------------
std::string ConfigFile::getStringDefault(const char * block, const char* name, const char* def)
{
    std::string ret;
    return getString(block, name, &ret) ? ret : def;
}
//-----------------------------------------------------------------------------
bool ConfigFile::getBool(const char * block, const char* name, bool *value)
{
    ConfigSetting * Setting = getSetting(block, name);
    if(Setting == 0)
    {
        return false;
    }

    *value = Setting->AsBool;
    return true;
}
//-----------------------------------------------------------------------------
bool ConfigFile::getBoolDefault(const char * block, const char* name, const bool def /* = false */)
{
    bool val;
    return getBool(block, name, &val) ? val : def;
}
//-----------------------------------------------------------------------------
bool ConfigFile::getInt(const char * block, const char* name, int *value)
{
    ConfigSetting * Setting = getSetting(block, name);
    if(Setting == 0)
    {
        return false;
    }

    *value = Setting->AsInt;
    return true;
}
//-----------------------------------------------------------------------------
bool ConfigFile::getFloat(const char * block, const char* name, float *value)
{
    ConfigSetting * Setting = getSetting(block, name);
    if(Setting == 0)
    {
        return false;
    }

    *value = Setting->AsFloat;
    return true;
}
//-----------------------------------------------------------------------------
int ConfigFile::getIntDefault(const char * block, const char* name, const int def)
{
    int val;
    return getInt(block, name, &val) ? val : def;
}
//-----------------------------------------------------------------------------
float ConfigFile::getFloatDefault(const char * block, const char* name, const float def)
{
    float val;
    return (getFloat(block, name, &val) ? val : def);
}
//-----------------------------------------------------------------------------
int ConfigFile::getIntVA(const char * block, int def, const char* name, ...)
{
    va_list ap;
    va_start(ap, name);
    char str[150];
    vsnprintf(str, 150, name, ap);
    va_end(ap);
    int val;
    return getInt(str, block, &val) ? val : def;
}
//-----------------------------------------------------------------------------
float ConfigFile::getFloatVA(const char * block, float def, const char* name, ...)
{
    va_list ap;
    va_start(ap, name);
    char str[150];
    vsnprintf(str, 150, name, ap);
    va_end(ap);
    float val;
    return getFloat(str, block, &val) ? val : def;
}
//-----------------------------------------------------------------------------
std::string ConfigFile::getStringVA(const char * block, const char* def, const char * name, ...)
{
    va_list ap;
    va_start(ap, name);
    char str[150];
    vsnprintf(str, 150, name, ap);
    va_end(ap);

    return getStringDefault(str, block, def);
}
//-----------------------------------------------------------------------------
bool ConfigFile::getString(const char * block, char * buffer, const char * name, const char * def, uint32_t len)
{
    std::string val = getStringDefault(block, name, def);
    size_t blen = val.length();
    if(blen > len)
    {
        blen = len;
    }

    memcpy(buffer, val.c_str(), blen);
    buffer[blen] = 0;

    return true;
}
//-----------------------------------------------------------------------------
