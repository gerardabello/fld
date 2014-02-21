ADD_DEFINITIONS(
    -std=c++0x # Or -std=c++0x
    # Other flags
)

# driver source files
SET(sources fld.cpp)
# application header files
SET(headers fld.h)
# locate the necessary dependencies
find_package(OpenCV REQUIRED )
FIND_PACKAGE(MRPT REQUIRED slam;vision;gui;topography)
# add the necessary include directories
INCLUDE_DIRECTORIES(.)
# create the shared library
ADD_LIBRARY(fld SHARED ${sources})
# link necessary libraries
INSTALL(TARGETS fld
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib/iridrivers
        ARCHIVE DESTINATION lib/iridrivers)
INSTALL(FILES ${headers} DESTINATION include/iridrivers)
INSTALL(FILES ../Findfld.cmake DESTINATION ${CMAKE_ROOT}/Modules/)
target_link_libraries(fld ${OpenCV_LIBS} )
ADD_SUBDIRECTORY(examples)
