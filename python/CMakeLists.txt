set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(pybind11_catkin REQUIRED)

# Python extension
add_library ( py_im_enhance
	MODULE
		binding.cpp
		cv_conversion.cpp )
		
target_include_directories( py_im_enhance
	BEFORE
	PRIVATE 
		${pybind11_catkin_INCLUDE_DIRS} )

set_target_properties (py_im_enhance PROPERTIES 
	PREFIX ""
	LIBRARY_OUTPUT_NAME "im_enhance"
	LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION} )
	
target_link_libraries(py_im_enhance
	PRIVATE
		im_enhance
		${PYTHON_LIBRARIES}
		${OpenCV_LIBS}
)

install(
	TARGETS py_im_enhance
		LIBRARY DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION}
)

