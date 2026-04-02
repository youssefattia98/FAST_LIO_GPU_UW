cmake_minimum_required(VERSION 3.8)
project(fast_lio)

option(FASTLIO_USE_CUDA "Enable CUDA accelerated kernels" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

 ADD_COMPILE_OPTIONS(-std=c++17) 
 ADD_COMPILE_OPTIONS(-std=c++17) 
 set(CMAKE_CXX_FLAGS "-std=c++17 -O3") 
  
 add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\") 
  
 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions") 
 set(CMAKE_CXX_STANDARD 17) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(FASTLIO_USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "72;75;87" CACHE STRING "CUDA architectures" FORCE)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_definitions(-DFASTLIO_USE_CUDA)
endif()

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")

include(ProcessorCount)
ProcessorCount(N)
if(NOT N)
  set(N 1)
endif()
message("Processer number:  ${N}")

set(FASTLIO_MP_THREADS 1)
if(N GREATER 4)
  set(FASTLIO_MP_THREADS 3)
elseif(N GREATER 2)
  set(FASTLIO_MP_THREADS 2)
endif()

if(FASTLIO_MP_THREADS GREATER 1)
  add_definitions(-DMP_EN)
endif()
add_definitions(-DMP_PROC_NUM=${FASTLIO_MP_THREADS})
message("core for MP: ${FASTLIO_MP_THREADS}")

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

# ROS dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(auv_core_helper REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCONFIGPP REQUIRED libconfig++)
find_library(UNDERWATER_VEHICLE_MODEL NAMES underwater_vehicle_model REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(GTSAM REQUIRED QUIET)
option(FASTLIO_REQUIRE_LIVOX "Fail configuration if livox_ros_driver2 is missing" OFF)
find_package(livox_ros_driver2 QUIET)
find_package(rosidl_default_generators REQUIRED)

set(dependencies
  rclcpp
  rclcpp_components
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
  pcl_ros
  pcl_conversions
  auv_core_helper
  ament_index_cpp
)

set(FASTLIO_HAS_LIVOX FALSE)
if(livox_ros_driver2_FOUND)
  list(APPEND dependencies livox_ros_driver2)
  set(FASTLIO_HAS_LIVOX TRUE)
  message(STATUS "livox_ros_driver2 detected: Livox LiDAR support enabled")
elseif(FASTLIO_REQUIRE_LIVOX)
  message(FATAL_ERROR "livox_ros_driver2 not found but FASTLIO_REQUIRE_LIVOX is ON. Install the driver or disable the option.")
else()
  message(WARNING "livox_ros_driver2 not found: Livox-specific LiDAR pipelines will be disabled.")
endif()

# Thirdparty libraries
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters kdtree registration)

message(Eigen: ${EIGEN3_INCLUDE_DIR})
message(STATUS "PCL: ${PCL_INCLUDE_DIRS}")

set(msg_files
  "msg/Pose6D.msg"
  "msg/FrontendFrame.msg"
  "msg/OptimizedKeyframes.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES geometry_msgs sensor_msgs std_msgs
)
ament_export_dependencies(rosidl_default_runtime)

set(FASTLIO_SOURCES
  src/laserMapping.cpp
  include/ikd-Tree/ikd_Tree.cpp
  src/preprocess.cpp
  src/dynamics_bridge.cpp)

if(FASTLIO_USE_CUDA)
  list(APPEND FASTLIO_SOURCES
    src/gpu/preprocess_gpu.cu
    src/gpu/gpu_voxel_map.cu
    src/gpu/point_transform_gpu.cu)
endif()

add_executable(fastlio_mapping ${FASTLIO_SOURCES})
set_target_properties(fastlio_mapping PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON)
target_include_directories(fastlio_mapping PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${LIBCONFIGPP_INCLUDE_DIRS}
)
target_link_libraries(fastlio_mapping ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} Eigen3::Eigen ${UNDERWATER_VEHICLE_MODEL} ${LIBCONFIGPP_LIBRARIES})
if(FASTLIO_USE_CUDA)
  target_link_libraries(fastlio_mapping CUDA::cudart CUDA::cublas CUDA::cusolver)
  target_include_directories(fastlio_mapping PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()
target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})
if(FASTLIO_HAS_LIVOX)
  target_compile_definitions(fastlio_mapping PRIVATE FASTLIO_HAS_LIVOX)
endif()

add_executable(loop_closure_backend_node
  src/loop_closure_backend_node.cpp
)
set_target_properties(loop_closure_backend_node PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)
target_include_directories(loop_closure_backend_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(loop_closure_backend_node
  ${PCL_LIBRARIES}
  Eigen3::Eigen
  gtsam
)

list(APPEND EOL_LIST "foxy" "galactic" "eloquent" "dashing" "crystal")

if($ENV{ROS_DISTRO} IN_LIST EOL_LIST)
  # Custommsg to support foxy & galactic
  rosidl_target_interfaces(fastlio_mapping
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
  rosidl_target_interfaces(loop_closure_backend_node
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
else()
  rosidl_get_typesupport_target(cpp_typesupport_target
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
  target_link_libraries(fastlio_mapping ${cpp_typesupport_target})
  target_link_libraries(loop_closure_backend_node ${cpp_typesupport_target})
endif()

ament_target_dependencies(fastlio_mapping ${dependencies})
ament_target_dependencies(loop_closure_backend_node
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  pcl_conversions
)

# ---------------- Install --------------- #
install(TARGETS fastlio_mapping loop_closure_backend_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/fastlio_path_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)


ament_package()
