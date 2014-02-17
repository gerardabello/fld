#edit the following line to add the librarie's header files
FIND_PATH(fld_INCLUDE_DIR fld.h /usr/include/iridrivers /usr/local/include/iridrivers)

FIND_LIBRARY(fld_LIBRARY
    NAMES fld
    PATHS /usr/lib /usr/local/lib /usr/local/lib/iridrivers) 

IF (fld_INCLUDE_DIR AND fld_LIBRARY)
   SET(fld_FOUND TRUE)
ENDIF (fld_INCLUDE_DIR AND fld_LIBRARY)

IF (fld_FOUND)
   IF (NOT fld_FIND_QUIETLY)
      MESSAGE(STATUS "Found fld: ${fld_LIBRARY}")
   ENDIF (NOT fld_FIND_QUIETLY)
ELSE (fld_FOUND)
   IF (fld_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find fld")
   ENDIF (fld_FIND_REQUIRED)
ENDIF (fld_FOUND)

