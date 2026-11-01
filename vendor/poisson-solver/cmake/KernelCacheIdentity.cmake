# The cache key describes the code-generation recipe, never the final shared
# library (which would make its own embedded identity recursive). Installation
# retains the exact recipe so the caller can bind it in its build receipt.
get_target_property(_cache_nvrtc CUDA::nvrtc IMPORTED_LOCATION)
if (NOT _cache_nvrtc OR NOT EXISTS "${_cache_nvrtc}")
    message(FATAL_ERROR "Kernel cache identity requires the selected NVRTC library")
endif()
get_filename_component(_cache_cuda_lib "${_cache_nvrtc}" DIRECTORY)
find_library(_cache_builtins NAMES nvrtc-builtins
    PATHS "${_cache_cuda_lib}" NO_DEFAULT_PATH REQUIRED)
file(REAL_PATH "${_cache_builtins}" POISSON_SOLVER_CACHE_NVRTC_BUILTINS)
file(SHA256 "${POISSON_SOLVER_CACHE_NVRTC_BUILTINS}" _cache_builtins_sha256)

file(GLOB_RECURSE _cache_source_inputs CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/*" "${PROJECT_SOURCE_DIR}/src/*"
    "${PROJECT_SOURCE_DIR}/cmake/*")
list(APPEND _cache_source_inputs "${PROJECT_SOURCE_DIR}/CMakeLists.txt"
    "${PROJECT_SOURCE_DIR}/dependencies.lock.json")
file(GLOB_RECURSE _cache_vkfft_inputs CONFIGURE_DEPENDS "${vkfft_SOURCE_DIR}/vkFFT/*.h")
set(_cache_inputs ${_cache_source_inputs} ${_cache_vkfft_inputs}
    "${_cache_nvrtc}" "${POISSON_SOLVER_CACHE_NVRTC_BUILTINS}"
    "${CMAKE_CXX_COMPILER}" "${CMAKE_CUDA_COMPILER}" "${OPENSSL_CRYPTO_LIBRARY}")
foreach(_cache_include IN LISTS CUDAToolkit_INCLUDE_DIRS)
    file(GLOB_RECURSE _cache_headers CONFIGURE_DEPENDS "${_cache_include}/*")
    list(APPEND _cache_inputs ${_cache_headers})
endforeach()
file(GLOB_RECURSE _cache_crypto_headers CONFIGURE_DEPENDS "${OPENSSL_INCLUDE_DIR}/openssl/*")
list(APPEND _cache_inputs ${_cache_crypto_headers})
# Record the host compiler executable used behind the driver as well.
execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-prog-name=cc1plus
    OUTPUT_VARIABLE _cache_cc1plus OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _cache_cc1plus_status TIMEOUT 15)
if (_cache_cc1plus_status EQUAL 0 AND EXISTS "${_cache_cc1plus}")
    list(APPEND _cache_inputs "${_cache_cc1plus}")
endif()
list(REMOVE_DUPLICATES _cache_inputs)
list(SORT _cache_inputs)
set(_cache_recipe "poisson-nvrtc-cubin-build-recipe-v1\n")
# Length framing also preserves flags containing separators or newlines.
function(_poisson_cache_part value)
    string(LENGTH "${value}" _cache_length)
    string(APPEND _cache_recipe "${_cache_length}:${value}\n")
    set(_cache_recipe "${_cache_recipe}" PARENT_SCOPE)
endfunction()
get_cmake_property(_cache_variables VARIABLES)
list(REMOVE_DUPLICATES _cache_variables)
list(SORT _cache_variables)
foreach(_cache_variable IN LISTS _cache_variables)
    if (_cache_variable MATCHES "^CMAKE_(BUILD_TYPE|SYSTEM_NAME|SYSTEM_PROCESSOR|CXX_COMPILER(_ID|_VERSION|_FRONTEND_VARIANT)?|CUDA_COMPILER(_ID|_VERSION)?|Fortran_COMPILER(_ID|_VERSION)?|CUDA_HOST_COMPILER|CXX_STANDARD|CXX_EXTENSIONS|CUDA_ARCHITECTURES)$"
            OR (_cache_variable MATCHES "^CMAKE_(CXX|CUDA|Fortran|SHARED_LINKER)_FLAGS(_[A-Za-z0-9_]+)?$"
                AND NOT _cache_variable MATCHES "_INIT$")
            OR _cache_variable MATCHES "^POISSON_SOLVER_(USE_MPI|ENABLE_BENCHMARKS)$")
        _poisson_cache_part("${_cache_variable}=${${_cache_variable}}")
    endif()
endforeach()
foreach(_cache_input IN LISTS _cache_inputs)
    if (IS_DIRECTORY "${_cache_input}")
        continue()
    endif()
    file(REAL_PATH "${_cache_input}" _cache_real_input)
    file(SHA256 "${_cache_real_input}" _cache_hash)
    _poisson_cache_part("${_cache_real_input}")
    _poisson_cache_part("${_cache_hash}")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_cache_inputs})
string(SHA256 POISSON_SOLVER_CACHE_BUILD_KEY "${_cache_recipe}")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/generated")
file(WRITE "${PROJECT_BINARY_DIR}/generated/kernel-cache-build-recipe.txt" "${_cache_recipe}")
# Reject unsupported literal characters rather than generating an ambiguous
# C++ string or JSON path. This is only the selected system library location.
if (POISSON_SOLVER_CACHE_NVRTC_BUILTINS MATCHES "[\"\\\\\n\r]")
    message(FATAL_ERROR "Unsupported characters in NVRTC builtins path")
endif()
configure_file("${CMAKE_CURRENT_LIST_DIR}/kernel_cache_build_identity.hpp.in"
    "${PROJECT_BINARY_DIR}/generated/kernel_cache_build_identity.hpp" @ONLY)
file(WRITE "${PROJECT_BINARY_DIR}/generated/kernel-cache-identity.json"
"{\n  \"schema_version\": 1,\n  \"kind\": \"poisson-nvrtc-cubin-cache\",\n  \"cache_format\": \"POISSON_CUBIN_1\",\n  \"build_key\": \"${POISSON_SOLVER_CACHE_BUILD_KEY}\",\n  \"environment_variable\": \"POISSON_SOLVER_KERNEL_CACHE_DIR\",\n  \"max_image_bytes\": 16777216,\n  \"max_cache_bytes\": 67108864,\n  \"nvrtc_builtins\": {\"path\": \"${POISSON_SOLVER_CACHE_NVRTC_BUILTINS}\", \"sha256\": \"${_cache_builtins_sha256}\"},\n  \"recipe_file\": \"kernel-cache-build-recipe.txt\"\n}\n")
