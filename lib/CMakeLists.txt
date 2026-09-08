add_library(WeftRISCVContracts STATIC
  Target/IME/FragmentContracts.cpp
)

add_subdirectory(Dialect)
add_subdirectory(Target)
