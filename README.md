VerthashMiner
===============

VerthashMiner is a high performance GPU miner for the Verthash algorithm.

**Developer:** CryptoGraphics

Stratum, WorkIO and GBT implementations are partly based on [cpuminer-multi](https://github.com/tpruvot/cpuminer-multi)
and [lyclMiner](https://github.com/CryptoGraphics/lyclMiner) 

This open source release was made possible thanks to [Vertcoin project](https://vertcoin.org) and its community.

## Supported hardware
- AMD GPU GCN 1.0 or later.  
- NVIDIA GPU with compute capability 3.0 or later.  
  (Some compute versions may require different miner builds for the CUDA backend. e.g 8.0 requires CUDA 11.0, which removes support for 3.0.)

Additionally miner requires GPU with 2GB VRAM or higher. (depends on the `WorkSize` parameter).

## Supported platforms
- AMD: OpenCL 1.2+ from AMD Radeon Software driver on Windows. AMDGPU-Pro and ROCm on Linux.
- NVIDIA: Both OpenCL 1.2+ and CUDA are supported through the proprietary driver.

Mesa Gallium Compute and macOS are not supported.

## Download
* Binary releases: https://github.com/CryptoGraphics/VerthashMiner/releases
* Clone with `git clone https://github.com/CryptoGraphics/VerthashMiner.git`
* Follow [Building VerthashMiner](#building-verthashminer).

## Quick start guide
Miner can be configured through the command line, configuration file and a mix of both. All options are documented inside.
Most parameters are optional and will be auto-configured to their default values, while some of them are mandatory.
Miner includes a configuration validator. All errors and warnings will be listed.
Both solo(getblocktemplate) and pooled mining(Stratum) are supported.

### Configuration examples:
___
#### Command line only:
Run `./VerthashMiner` to get a full list of possible options.

Solo mining using GBT(getblocktemplate):  
`./VerthashMiner -u user -p password -o http://127.0.0.1:port --coinbase-addr core_wallet_address --verthash-data your_path/verthash.dat --all-cl-devices --all-cu-devices`

Pooled mining using Stratum:  
`./VerthashMiner -u user -p password -o stratum+tcp://example.com:port --verthash-data your_path/verthash.dat --all-cl-devices --all-cu-devices`

___
#### Configuration file:
All miner settings can also be managed through the configuration file. Similar to [lyclMiner](https://github.com/CryptoGraphics/lyclMiner)

1. **Generating a configuration file.**
   - Config file can be generated using the following command inside cmd/terminal:  
`./VerthashMiner --g your_config_file.conf`

   - Alternative (Windows).  
   Create a file `GenerateConfig.bat` in the same folder as `VerthashMiner.exe` with the following content:  
`./VerthashMiner -g your_config_file.conf`
   - **Additional notes:**
     - Configuration file is generated specifically for your GPU and driver setup.
     - Configuration file must be re-generated every time you add/remove a new Device to/from the PCIe slot.
	 - If you want to use NVIDIA GPUs with OpenCL backend when CUDA is available, then configuration file must be generated with `--no-restrict-cuda` option.  
	 	example: `./VerthashMiner -g your_config_file.conf --no-restrict-cuda`

2. **Configuring a miner.**
Open `your_config_file.conf` using any text editor and edit `"Url"`, `"Username"`, `"Password"` and `"CoinbaseAddress"`(Solo mining only) fields inside a `"Connection"` block.
Additional notes:  
   - It is recommended to adjust `WorkSize` parameter for each `Device` to get better performance.
   - Every single option inside the Configuration file is self documented along with examples.

3. **Use** `VerthashMiner -c your_config_file.conf` **to start mining.**
	- Alternative (Windows).  
   	Create a file `Run.bat` in the same folder as `VerthashMiner.exe` with the following content:  
	`VerthashMiner -c your_config_file.conf`
	- **Additional notes:**
	  - To use NVIDIA GPUs with OpenCL backend when CUDA is available: `VerthashMiner -c your_config_file.conf --no-restrict-cuda`.
	    Note that in this case `your_config_file.conf` must be generated with `--no-restrict-cuda` too.

#### Both command line and configuration file:
For example you may want to configure the miner using a configuration file partly or completely.
Command line options have higher priority than the file and it is possible to overwrite almost every option.  
`VerthashMiner -c your_config_file.conf -u user -p password`  
In this case miner will use a configuration file while Username and Password options will be overwritten by command line.


## There is more

### Selecting specific devices

CUDA and OpenCL devices are configured separately.
By default, all devices are being used. However it is possible to select specific ones using a `PCIeBusId` option.
___
#### OpenCL device management
```
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# Device config:
#
# Available platforms:
#
# 1. Platform name: Advanced Micro Devices, Inc.
#    Index: 0
#
# 2. Platform name: NVIDIA Corporation
#    Index: 1
#
# Available devices:
#
# 1. Device: gfx900
#    PCIe bus ID: 1
#    Available platforms indices: 0
#
# 2. Device: Ellesmere
#    PCIe bus ID: 3
#    Available platforms indices: 0
#
# 3. Device: Ellesmere
#    PCIe bus ID: 5
#    Available platforms indices: 0
#
# 4. Device: GeForce RTX 2080 Ti
#    PCIe bus ID: 7
#    Available platforms indices: 1
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

<CL_Device0 PCIeBusId = "1" PlatformIndex = "0" BinaryFormat = "auto" AsmProgram = "none" WorkSize = "131072">
<CL_Device1 PCIeBusId = "3" PlatformIndex = "0" BinaryFormat = "auto" AsmProgram = "none" WorkSize = "131072">
<CL_Device2 PCIeBusId = "5" PlatformIndex = "0" BinaryFormat = "auto" AsmProgram = "none" WorkSize = "131072">
<CL_Device3 PCIeBusId = "7" PlatformIndex = "1" BinaryFormat = "auto" AsmProgram = "none" WorkSize = "131072">
```

For example: We want to use devices only with `PCIeBusId` 3 and 7.  
Comment/backup the original list.
```
/*
<CL_Device0 PCIeBusId = "1" PlatformIndex = "0" BinaryFormat = "auto" AsmProgram = "none" WorkSize = "131072">
<CL_Device1 PCIeBusId = "3" PlatformIndex = "0" BinaryFormat = "auto" AsmProgram = "none" WorkSize = "131072">
<CL_Device2 PCIeBusId = "5" PlatformIndex = "0" BinaryFormat = "auto" AsmProgram = "none" WorkSize = "131072">
<CL_Device3 PCIeBusId = "7" PlatformIndex = "1" BinaryFormat = "auto" AsmProgram = "none" WorkSize = "131072">
*/
```

Copy selected device configurations and rename blocks, so they will start from 0. e.g `<CL_Device0>`, `<CL_Device1>` ...
```
<CL_Device0 PCIeBusId = "3" PlatformIndex = "0" BinaryFormat = "auto" AsmProgram = "none" WorkSize = "131072">
<CL_Device1 PCIeBusId = "7" PlatformIndex = "1" BinaryFormat = "auto" AsmProgram = "none" WorkSize = "131072">
```
___
#### CUDA device management
```
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# CUDA Device config:
#
# Available devices:
#
# DeviceIndex: 0
#    Name: GeForce RTX 2080 Ti
#
# DeviceIndex: 1
#    Name: GeForce GTX 1080
#
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

<CU_Device0 DeviceIndex = "0" WorkSize = "131072">
<CU_Device1 DeviceIndex = "1" WorkSize = "131072">
```

### Pool connection setup:
- **Connection password**  
  - If the pool doesn't require this parameter, leave it as `x`  

### Comments inside a config file

- Comments can be in C format, e.g. `/* some stuff */`, with a `//` at the start of the line, or in shell format (`#`).

### Raw device list configuration file format:
This option affects OpenCL device configuration only.  
There can be a case when all OpenCL devices return the same PCIeBusId and it will be impossible to distinguish between them.  
If there will be duplicate PCIeBusIds on the same platform, then miner will automatically switch to the `raw device list format`  
All possible Device/Platform configurations will be listed. A `DeviceIndex` option will used instead of `PCIeBusId` and `PlatformIndex`.  
If Device has more than one platform available, all duplicates must be handled manually.  
**Note:** The order in which devices are listed is platform implementation defined. Prefer to use PCIeBusId version whenever possible.

- **DeviceIndex**  
Specifies a Device/Platform combination from the list.

- **Generating a configuration file with raw device list**  
`VerthashMiner -G your_config_file.conf`



## Building VerthashMiner

Make sure that OpenCL drivers are installed. See [Supported platforms](#supported-platforms).
lyclMiner uses [CMake](https://cmake.org/) to build platform specific projects.

### Dependencies:
1. OpenCL
2. Jansson(https://github.com/akheron/jansson)
3. CURL(https://curl.haxx.se/libcurl/)
4. OpenSSL(optional on Windows)
5. CUDA(optional) Otherwise OpenCL will be used. Both versions are optimized and performance will be the same.
   With CUDA you can avoid 100% CPU usage during mining using NVIDIA GPUs.

### Compiling from the source code
1. Make sure that all dependencies are installed
2. Install the latest version of [CMake](https://cmake.org/). 3.18 or above is required.
https://cmake.org/
3. Open CMake and in `Where is the source code` select miner root directory with `CMakeLists.txt`
4. Choose the path "Where to build the binaries" for cache.
5. Press `Configure` and select `Generator`. Note, that CUDA is not supported when using MinGW compiler on Windows platform.
   Recommended generators: `Visual Studio`(select installed version) on Windows and `Unix Makefiles` on Linux.
6. Make sure that `Optional platform for generator` is `x64` and press `Finish`
7. Build system will configure everything automatically and use precompiled dependencies on Windows if possible. You can always specify your own.
8. Modify `CMAKE_INSTALL_PREFIX` option and set the miner install path.
9. Some build systems have `CMAKE_BUILD_TYPE` option set to empty. Make sure it is set to `Release` for the final use.
10. Use `Generate` and navigate to `Where to build the binaries` directory.
11. Compile miner(depends on the selected compiler and generator)
- Navigate to `Where to build the binaries` directory
- On Linux and Windows MinGW
    * Open Terminal/Windows PowerShell in this directory.
    * Linux: `make`, Windows MinGW: `mingw32-make`
    * Wait for the compilation to finish
    * Linux: `make install`, Windows MinGW: `mingw32-make install`
- On Windows(Microsoft Visual Studio)
    * Open `VerthashMiner.sln` using Microsoft Visual Studio
    * Right click on the `ALL_BUILD` solution inside the Solution Explorer window and select `Build`
    * Wait for the compilation to finish
    * Right click on the `INSTALL` solution inside the Solution Explorer window and select `Build`
12. Miner binaries will be stored inside the `CMAKE_INSTALL_PREFIX` directory.

## Additional Notes
- `LONGPOLL pushed new work` spam may happen during GMT solo mining if network was stale for a long time. (e.g. testnet)  
   In this case miner should be run with either `--no-longpoll` or `LongPoll` option set to `false` inside the configuration file.
- To enable file logger use `--log-file` command option.
- All miner "devices" are virtual. By default miner assigns 1 virtual GPU per physical one. Thus 1 thread per GPU.
It is possible to emulate any devices you want by putting duplicates in the list. You can even use multiple CUDA and OpenCL devices at the same time while having only 1 physical NVIDIA GPU.
There are 3 ways to do it:
  1. Using Command Line.  
Instead of `--all-cl-devices` and/or `--all-cu-devices` use:  
`--cl-devices ...`(-d) and `--cu-devices ...(-D)` respectively.  
To get all physical devices available to the miner use:  
`-l` or `--device-list`  
To create 2 virtual devices for one physical device, specify the same device twice.  
`--cl-devices 0:w131072,0:w131072`  
131072 is a work size(default value). You can try specify your own(e.g 32768, 65536, 262144, 524288 etc) and check performance/power consumption.

  2. and 3. Using a Configuration File.  
OpenCL devices have 2 configuration file formats `PCIeBusID` and `Raw device list`. CUDA devices have only 1(`Raw device list`).  
Duplicate device configuration block.  
For example: There will be only 1 `<CL_Device0 ...>` block with 1 physical GPU.
Duplicate it and rename a new one to `<CL_Device1 ...>`.

* When using 2 or more devices for a single physical GPU, their hash-rate will probably be the same.  
You can try to specify a different `WorkSize` for each of them and compare multiple `WorkSize` values at the same time. Not sure about accuracy though. It may vary between different GPUs, drivers, OS and other apps running in the background.
