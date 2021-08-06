
add_library(blur_model STATIC)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/include")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/xgboost")


target_include_directories(blur_model PRIVATE "xgboost/include")
target_link_directories(blur_model PRIVATE "xgboost/src")

target_link_libraries(blur_model PRIVATE xgboost)

