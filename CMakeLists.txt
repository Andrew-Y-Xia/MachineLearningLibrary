cmake_minimum_required(VERSION 3.20)
project(MachineLearningLibrary)

set(CMAKE_CXX_STANDARD 14)

add_compile_options(
	-Wall -Werror -Wno-sign-compare
        $<$<CONFIG:RELEASE>:-O3> $<$<CONFIG:RELEASE>:-flto> $<$<CONFIG:RELEASE>:-march=native>
        $<$<CONFIG:DEBUG>:-O1>
        $<$<CONFIG:DEBUG>:-ggdb3>
)
add_compile_definitions(
        $<$<CONFIG:RELEASE>:NDEBUG>
        $<$<CONFIG:RELEASE>:BOOST_DISABLE_ASSERTS>
)

add_executable(MachineLearningLibrary main.cpp Matrix.h Model.cpp Model.h Parser.cpp Parser.h)
