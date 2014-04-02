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

