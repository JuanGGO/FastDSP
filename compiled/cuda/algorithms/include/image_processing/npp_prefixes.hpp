//
// Created by juan on 5/13/20.
//


#pragma once
#include<string>
#include<map>

namespace fdsp

    template<typename T>
    std::string GetTypePrefix();

    template<bool hasRegion, bool hasMask>
    std::string GetNChannelsPrefix(int nChannels);
}

template<typename T>
std::string fdsp::GetTypePrefix()
{
    if(typeid(T) == typeid(unsigned char))
        return "8u";
    else if(typeid(T) == typeid(unsigned short))
        return "16u";
    else if(typeid(T) == typeid(short))
        return "16s";
    else if(typeid(T) == typeid(float))
        return "32f";
    else
        return "None";
}

template<bool hasRegion, bool hasMask>
std::string fdsp::GetNChannelsPrefix(int nChannels)
{
    std::string out;
    if(nChannels == 1)
        out = "C1";
    else if(nChannels == 3)
        out = "C3";
    else if(nChannels == 4)
        out = "C4";
    else
        out = "None";

    if(hasMask)
        out += "M";
    if(hasRegion)
        out += "R";

    return out;
}

