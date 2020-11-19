set(ONNX_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include")
message("配置ONNX include: ${ONNX_INCLUDE_DIRS}")
include_directories(${ONNX_INCLUDE_DIRS})

set(ONNX_LIBS libonnxruntime)
add_library(${ONNX_LIBS} SHARED IMPORTED)
set_target_properties(${ONNX_LIBS} PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../onnx/${ANDROID_ABI}/libonnxruntime.so)

set(ONNX_FOUND TRUE)