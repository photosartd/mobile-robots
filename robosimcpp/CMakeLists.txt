cmake_minimum_required(VERSION 3.10)
project(robosimcpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(${CMAKE_HOME_DIRECTORY}/thirdparty/pybind11) #need some functions to install
include_directories(${CMAKE_HOME_DIRECTORY}/thirdparty/eigen) #just need cmakelists and headerfiles inside

# --- 3) Collect all C++ source files in src/
file(GLOB LOCALISE_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
# EXCLUDE bindings.cpp if it is in the same folder:
list(REMOVE_ITEM LOCALISE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp")

# --- 4) Build a static (or shared) library from those .cpp files
add_library(localise SHARED ${LOCALISE_SRC})
target_include_directories(localise PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    # If you plan to install headers, this second path is standard practice:
    "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(localise PUBLIC pybind11::pybind11)

# --- 5) Create a Python extension module with pybind11
# Suppose you have a 'bindings.cpp' in src/ for all your pybind11 code:
pybind11_add_module(pylocalise
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
)

# Link the 'pylocalise' module against your localise library
target_link_libraries(pylocalise PRIVATE localise)

# Also make sure the extension module sees your headers:
target_include_directories(pylocalise PRIVATE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)

# --- 6) (Optional) Install steps if you want to 'make install'
install(TARGETS localise pylocalise
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

# --- Add a test executable
# Assume your tests are in test/test_robosimcpp.cpp
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
add_executable(robosimcpp_tests test/test_robosimcpp.cpp)
target_link_libraries(robosimcpp_tests PRIVATE localise gtest gtest_main Python3::Python)

# Discover and add tests automatically.
include(GoogleTest)
gtest_discover_tests(robosimcpp_tests)
