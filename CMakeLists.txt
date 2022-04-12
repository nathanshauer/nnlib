cmake_minimum_required(VERSION 3.14)

# ----- Basic properties -----
project(NeuralNet LANGUAGES CXX)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----- Linking with accelerate lib if on macos -----
IF(APPLE)
   add_definitions(-DMACOSX)
   set(MACOSX MACOSX)
   FIND_LIBRARY(ACCELERATE_LIB Accelerate)
   link_libraries(${ACCELERATE_LIB})
ENDIF(APPLE)

# As all the files are not in the parent folder, CMake needs to know
# in which folders he should look for header files.
include_directories(
    "${PROJECT_SOURCE_DIR}/nn" # ${PROJECT_SOURCE_DIR} is an automatic variable of CMake and gives the path to the project folder
)


# Next, cmake needs to know where he should look for folders that
# also have CMakelists.txt files.
add_subdirectory(nn) # folder where the source files are
add_subdirectory(lib) # folder where the lib is specified
add_subdirectory(driver) # folder where the executable is created and linked with the lib

