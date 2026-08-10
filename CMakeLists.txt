cmake_minimum_required(VERSION 3.20)

project(Weft LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# The previous compiler is isolated under materials/ and is intentionally not
# part of this build. New targets begin here only after their canonical IR and
# dependency direction satisfy doc/WEFT_FINAL_SPEC.md.
message(STATUS "Weft clean reconstruction root: no compiler targets yet")
