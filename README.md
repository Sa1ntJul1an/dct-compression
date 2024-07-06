# dct-compression
DCT Image Compression algorithm

## Build and Run
Download OpenCV, move to C:\\
Add following system variables: 
    Variable: `OpenCV_DIR`
    Value: `C:\opencv\build`

Download CMake, add environment variables:
    Variable: `PATH`
    Values:
        `C:\Program Files\CMake\bin`
        `C:\opencv\build\x64\vc16\bin`
        `C:\opencv\build\x64\vc16\lib`

Build: 
    `cmake --build .\build\`

Run:
    `.\build\Debug\DCT-Compression.exe`