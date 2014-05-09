# create an example application
ADD_EXECUTABLE(fld_test fld_test.cpp)

INCLUDE_DIRECTORIES(.
                    /home/gabello/Code/pfc_gas/include
)
# link necessary libraries

TARGET_LINK_LIBRARIES(fld_test 
                        fld 
                        /home/gabello/Code/pfc_gas/lib/libSfMToyLibrary.a
                       
                        /home/gabello/Code/pfc_gas/lib/SFML/libsfml-audio.so
                        /home/gabello/Code/pfc_gas/lib/SFML/libsfml-graphics.so
                        /home/gabello/Code/pfc_gas/lib/SFML/libsfml-network.so
                        /home/gabello/Code/pfc_gas/lib/SFML/libsfml-system.so
                        /home/gabello/Code/pfc_gas/lib/SFML/libsfml-window.so
)


ADD_EXECUTABLE(calib calibration.cpp)
find_package(OpenCV 2.4.8 EXACT REQUIRED )
#FIND_PACKAGE(MRPT REQUIRED slam;vision;gui;topography)
# add the necessary include directories
INCLUDE_DIRECTORIES(.
                    /usr/local/include/opencv2
                    #/home/gabello/Code/pfc_gas/src/Eigen
    )

target_link_libraries(calib ${OpenCV_LIBS} )
