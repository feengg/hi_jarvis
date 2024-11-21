mkdir build
cd whisper.cpp\ggml\src
expand ggml-vulkan-shaders.hp_ ggml-vulkan-shaders.hpp

cd ..\..\..\build
set PATH=%VK_SDK_PATH%;%PATH%
set PATH=%VK_SDK_PATH%\bin;%PATH%

cmake -G Ninja -DGGML_VULKAN=1 -DGGML_STATIC=ON -DBUILD_SHARED_LIBS=ON -DGGML_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release ..\whisper.cpp

cmake --build .
