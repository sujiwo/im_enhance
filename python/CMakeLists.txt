find_package(pybind11_catkin REQUIRED)

# Python extension
pybind11_add_module (py_im_enhance
	MODULE 
		binding2.cpp )
set_target_properties (py_im_enhance PROPERTIES 
	PREFIX ""
	LIBRARY_OUTPUT_NAME "im_enhance"
	LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION} )
install(
	TARGETS py_im_enhance
		LIBRARY DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION}
)

