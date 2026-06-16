cmake_minimum_required(VERSION 3.12)

set(MY_PROJECT "pt")
set(MY_EXECUTABLE "pt")
set(CLI_EXECUTABLE "pt_cli") # [新增] CLI 執行檔名稱

project("pt"
    VERSION 0.1.0
    LANGUAGES CXX C CUDA
    DESCRIPTION "pt"
)

# 宣告兩個執行檔
add_executable(${MY_EXECUTABLE})
add_executable(${CLI_EXECUTABLE})

if(WIN32)
    message(STATUS "Detected System: Windows")
    target_compile_definitions(${MY_EXECUTABLE} PRIVATE WINDOWS)
    target_compile_definitions(${CLI_EXECUTABLE} PRIVATE WINDOWS)
elseif(APPLE)
    message(STATUS "Detected System: macOS")
    target_compile_definitions(${MY_EXECUTABLE} PRIVATE MACOS)
    target_compile_definitions(${CLI_EXECUTABLE} PRIVATE MACOS)
elseif(UNIX AND NOT APPLE)
    message(STATUS "Detected System: Linux")
    target_compile_definitions(${MY_EXECUTABLE} PRIVATE LINUX)
    target_compile_definitions(${CLI_EXECUTABLE} PRIVATE LINUX)
endif()

# ==========================================
# 強制關閉所有編譯器 Warning
# ==========================================
if(MSVC)
    target_compile_options(${MY_EXECUTABLE} PRIVATE $<$<COMPILE_LANGUAGE:C>:/W0> $<$<COMPILE_LANGUAGE:CXX>:/W0>)
    target_compile_options(${CLI_EXECUTABLE} PRIVATE $<$<COMPILE_LANGUAGE:C>:/W0> $<$<COMPILE_LANGUAGE:CXX>:/W0>)
else()
    target_compile_options(${MY_EXECUTABLE} PRIVATE $<$<COMPILE_LANGUAGE:C>:-w> $<$<COMPILE_LANGUAGE:CXX>:-w>)
    target_compile_options(${CLI_EXECUTABLE} PRIVATE $<$<COMPILE_LANGUAGE:C>:-w> $<$<COMPILE_LANGUAGE:CXX>:-w>)
endif()

target_compile_options(${MY_EXECUTABLE} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-w>)
target_compile_options(${CLI_EXECUTABLE} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-w>)

# 設定 C++ / CUDA 標準
set_target_properties(${MY_EXECUTABLE} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION ON CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON C_STANDARD 11)
set_target_properties(${CLI_EXECUTABLE} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION ON CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON C_STANDARD 11)

# Find external libraries
find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(glm REQUIRED)
find_package(imgui REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories("include\\")

# [關鍵] 抓取所有源碼，但把 main 排除掉
file(GLOB COMMON_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.cc"
    "src/*.c"
    "src/*.cu"
)
list(REMOVE_ITEM COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM COMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main_cli.cpp")

# 1. 綁定 GUI 版本
target_sources(${MY_EXECUTABLE} PRIVATE src/main.cpp ${COMMON_SOURCES})
target_include_directories(${MY_EXECUTABLE} PRIVATE "include" ${STB_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${MY_EXECUTABLE} PRIVATE
    glfw
    GLEW::GLEW
    OpenGL::GL
    OpenGL::GLU
    GLUT::GLUT
    glad::glad
    glm::glm
    imgui::imgui
    Eigen3::Eigen
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

# 2. 綁定 CLI 版本 (移除 GUI 相關庫)
target_sources(${CLI_EXECUTABLE} PRIVATE src/main_cli.cpp ${COMMON_SOURCES})
target_include_directories(${CLI_EXECUTABLE} PRIVATE "include" ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${CLI_EXECUTABLE} PRIVATE
    glm::glm
    glad::glad 
    Eigen3::Eigen
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

add_custom_command(TARGET ${MY_EXECUTABLE} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
    "${CMAKE_CURRENT_SOURCE_DIR}/shader"
    "$<TARGET_FILE_DIR:${MY_EXECUTABLE}>/shader"
    VERBATIM
)