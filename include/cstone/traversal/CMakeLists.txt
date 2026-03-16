if(CMAKE_HIP_COMPILER)
    set_source_files_properties(collisions_gpu.cu groups_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

set(TRAVERSAL_GPU_HEADERS traversal_gpu.cuh queue_gpu.cuh)
set_source_files_properties(${TRAVERSAL_GPU_HEADERS} PROPERTIES HEADER_FILE_ONLY ON)

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(traversal_obj OBJECT collisions_gpu.cu groups_gpu.cu ${TRAVERSAL_GPU_HEADERS})
    target_include_directories(traversal_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(traversal_obj PRIVATE ${CSTONE_THRUST_LIBS})
endif()
