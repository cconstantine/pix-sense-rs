#ifndef TRT_WRAPPER_H
#define TRT_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TrtEngine TrtEngine;

/// Build or load a TensorRT engine from an ONNX model.
/// If cache_path exists, loads the serialized engine (fast).
/// Otherwise, parses onnx_path, builds the engine, and saves to cache_path.
/// Set fp16=1 to enable FP16 precision (recommended on Jetson).
/// Returns NULL on failure.
TrtEngine* trt_engine_create(const char* onnx_path, const char* cache_path, int fp16);

/// Destroy the engine and free all GPU resources.
void trt_engine_destroy(TrtEngine* engine);

/// Get the number of I/O tensors.
int32_t trt_engine_num_io(TrtEngine* engine);

/// Get the name of the i-th I/O tensor. The returned pointer is valid until engine is destroyed.
const char* trt_engine_tensor_name(TrtEngine* engine, int32_t index);

/// Returns 1 if the named tensor is an input, 0 if output.
int32_t trt_engine_tensor_is_input(TrtEngine* engine, const char* name);

/// Get the shape of a tensor. Writes up to 8 dims and sets nb_dims.
void trt_engine_tensor_shape(TrtEngine* engine, const char* name, int64_t* dims, int32_t* nb_dims);

/// Run inference synchronously.
/// input_data: host pointer to float input, input_bytes: size in bytes.
/// output_ptrs: array of host float pointers (pre-allocated by caller), one per output.
/// output_sizes: array of sizes in bytes for each output buffer.
/// num_outputs: number of output tensors.
/// Returns 0 on success, non-zero on failure.
int32_t trt_engine_infer(TrtEngine* engine,
                         const float* input_data, size_t input_bytes,
                         float** output_ptrs, size_t* output_sizes,
                         int32_t num_outputs);

#ifdef __cplusplus
}
#endif

#endif // TRT_WRAPPER_H
