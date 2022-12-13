cmake_minimum_required(VERSION 3.13)
project(TFTRT_Benchmark_Runner)

#-------------------------------------------------------------
# Configuration
#-------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(TF_INSTALL_DIR "/usr/local" CACHE PATH "Path to Tensorflow install directory")

#-------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------
find_library(tf_shared_lib2 NAMES libtensorflow_cc.so.2 HINTS ${TF_INSTALL_DIR}/lib/tensorflow)
find_library(tf_framework_shared_lib2 NAMES libtensorflow_framework.so.2 HINTS ${TF_INSTALL_DIR}/lib/python3.8/dist-packages/tensorflow)
find_path(trt_include_path NAME NvInfer.h HINTS)
get_filename_component(tf_dir ${tf_shared_lib2} DIRECTORY)
get_filename_component(tf_python_dir ${tf_framework_shared_lib2} DIRECTORY)

set(tf_shared_lib ${tf_dir}/libtensorflow_cc.so)
set(tf_framework_shared_lib ${tf_python_dir}/libtensorflow_framework.so)

add_custom_command(OUTPUT ${tf_framework_shared_lib} ${tf_shared_lib}
  COMMAND ln -s ${tf_shared_lib2} ${tf_shared_lib}
  COMMAND ln -s ${tf_framework_shared_lib2} ${tf_framework_shared_lib}
  COMMENT "Generating legacy symbolic link")

add_custom_target(tf_symlinks DEPENDS ${tf_framework_shared_lib} ${tf_shared_lib})

#-----------------------------------------------------------
# Simple Runner Targets
#-----------------------------------------------------------
add_executable(simple_cpp_example main.cc)

target_link_libraries(simple_cpp_example tensorflow_cc)
target_link_libraries(simple_cpp_example tensorflow_framework)

target_compile_options(simple_cpp_example PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1 -DGOOGLE_CUDA -DGOOGLE_TENSORRT)

target_link_directories(simple_cpp_example PRIVATE ${tf_python_dir})
target_link_directories(simple_cpp_example PRIVATE ${tf_dir})

target_compile_options(simple_cpp_example PRIVATE -O2 -Wl,-rpath=${tf_python_dir})

target_include_directories(simple_cpp_example PRIVATE ${tf_python_dir}/include)
target_include_directories(simple_cpp_example PRIVATE ${trt_include_path})

add_dependencies(simple_cpp_example tf_symlinks)
