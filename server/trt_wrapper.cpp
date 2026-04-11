#include "trt_wrapper.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace nvinfer1;

// Simple logger that writes to stderr
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            const char* level = "UNKNOWN";
            switch (severity) {
                case Severity::kINTERNAL_ERROR: level = "INTERNAL_ERROR"; break;
                case Severity::kERROR:          level = "ERROR"; break;
                case Severity::kWARNING:        level = "WARNING"; break;
                default: break;
            }
            fprintf(stderr, "[TensorRT %s] %s\n", level, msg);
        }
    }
};

static Logger gLogger;

struct TrtEngine {
    std::unique_ptr<IRuntime, void(*)(IRuntime*)> runtime;
    std::unique_ptr<ICudaEngine, void(*)(ICudaEngine*)> engine;
    std::unique_ptr<IExecutionContext, void(*)(IExecutionContext*)> context;
    cudaStream_t stream = nullptr;

    // Pre-allocated GPU buffers for each I/O tensor
    struct TensorInfo {
        std::string name;
        bool is_input;
        Dims shape;
        size_t size_bytes;
        void* device_ptr;
    };
    std::vector<TensorInfo> tensors;

    TrtEngine()
        : runtime(nullptr, [](IRuntime* r){ delete r; })
        , engine(nullptr, [](ICudaEngine* e){ delete e; })
        , context(nullptr, [](IExecutionContext* c){ delete c; })
    {}

    ~TrtEngine() {
        for (auto& t : tensors) {
            if (t.device_ptr) cudaFree(t.device_ptr);
        }
        if (stream) cudaStreamDestroy(stream);
    }
};

static size_t dims_volume(const Dims& dims) {
    size_t vol = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        vol *= static_cast<size_t>(dims.d[i]);
    }
    return vol;
}

static std::vector<uint8_t> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return {};
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

static bool write_file(const char* path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.write(reinterpret_cast<const char*>(data), size);
    return f.good();
}

extern "C" TrtEngine* trt_engine_create(const char* onnx_path, const char* cache_path, int fp16, int input_size) {
    auto eng = std::make_unique<TrtEngine>();

    IHostMemory* serialized = nullptr;

    // Try to load cached engine
    auto cached = read_file(cache_path);
    if (!cached.empty()) {
        fprintf(stderr, "[TensorRT] Loading cached engine from %s\n", cache_path);
        eng->runtime.reset(createInferRuntime(gLogger));
        if (!eng->runtime) {
            fprintf(stderr, "[TensorRT] Failed to create runtime\n");
            return nullptr;
        }
        eng->engine.reset(eng->runtime->deserializeCudaEngine(cached.data(), cached.size()));
        if (!eng->engine) {
            fprintf(stderr, "[TensorRT] Failed to deserialize cached engine, rebuilding...\n");
            // Fall through to build from ONNX
        }
    }

    // Build from ONNX if no cached engine
    if (!eng->engine) {
        fprintf(stderr, "[TensorRT] Building engine from %s (this may take a while)...\n", onnx_path);

        std::unique_ptr<IBuilder, void(*)(IBuilder*)> builder(
            createInferBuilder(gLogger), [](IBuilder* b){ delete b; });
        if (!builder) {
            fprintf(stderr, "[TensorRT] Failed to create builder\n");
            return nullptr;
        }

        std::unique_ptr<INetworkDefinition, void(*)(INetworkDefinition*)> network(
            builder->createNetworkV2(0), [](INetworkDefinition* n){ delete n; });
        if (!network) {
            fprintf(stderr, "[TensorRT] Failed to create network\n");
            return nullptr;
        }

        std::unique_ptr<nvonnxparser::IParser, void(*)(nvonnxparser::IParser*)> parser(
            nvonnxparser::createParser(*network, gLogger), [](nvonnxparser::IParser* p){ delete p; });
        if (!parser) {
            fprintf(stderr, "[TensorRT] Failed to create ONNX parser\n");
            return nullptr;
        }

        if (!parser->parseFromFile(onnx_path, static_cast<int>(ILogger::Severity::kWARNING))) {
            fprintf(stderr, "[TensorRT] Failed to parse ONNX model %s\n", onnx_path);
            for (int i = 0; i < parser->getNbErrors(); i++) {
                fprintf(stderr, "[TensorRT]   %s\n", parser->getError(i)->desc());
            }
            return nullptr;
        }

        std::unique_ptr<IBuilderConfig, void(*)(IBuilderConfig*)> config(
            builder->createBuilderConfig(), [](IBuilderConfig* c){ delete c; });
        if (!config) {
            fprintf(stderr, "[TensorRT] Failed to create builder config\n");
            return nullptr;
        }

        if (fp16 && builder->platformHasFastFp16()) {
            config->setFlag(BuilderFlag::kFP16);
            fprintf(stderr, "[TensorRT] FP16 enabled\n");
        }

        // Set optimization profile for dynamic input shapes
        auto profile = builder->createOptimizationProfile();
        for (int i = 0; i < network->getNbInputs(); i++) {
            auto input = network->getInput(i);
            auto name = input->getName();
            auto dims = input->getDimensions();

            // Check if any dimension is dynamic (-1)
            bool has_dynamic = false;
            for (int d = 0; d < dims.nbDims; d++) {
                if (dims.d[d] == -1) { has_dynamic = true; break; }
            }

            if (has_dynamic) {
                // For models with dynamic spatial dims, fix to requested input_size
                Dims fixed = dims;
                for (int d = 0; d < fixed.nbDims; d++) {
                    if (fixed.d[d] == -1) {
                        fixed.d[d] = input_size;
                    }
                }
                profile->setDimensions(name, OptProfileSelector::kMIN, fixed);
                profile->setDimensions(name, OptProfileSelector::kOPT, fixed);
                profile->setDimensions(name, OptProfileSelector::kMAX, fixed);
                fprintf(stderr, "[TensorRT] Set optimization profile for '%s': [", name);
                for (int d = 0; d < fixed.nbDims; d++)
                    fprintf(stderr, "%ld%s", fixed.d[d], d < fixed.nbDims - 1 ? "," : "");
                fprintf(stderr, "]\n");
            }
        }
        config->addOptimizationProfile(profile);

        serialized = builder->buildSerializedNetwork(*network, *config);
        if (!serialized) {
            fprintf(stderr, "[TensorRT] Failed to build serialized network\n");
            return nullptr;
        }

        // Cache the engine
        if (cache_path && strlen(cache_path) > 0) {
            if (write_file(cache_path, serialized->data(), serialized->size())) {
                fprintf(stderr, "[TensorRT] Engine cached to %s\n", cache_path);
            }
        }

        // Deserialize
        eng->runtime.reset(createInferRuntime(gLogger));
        if (!eng->runtime) {
            delete serialized;
            return nullptr;
        }
        eng->engine.reset(eng->runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
        delete serialized;
        if (!eng->engine) {
            fprintf(stderr, "[TensorRT] Failed to deserialize engine\n");
            return nullptr;
        }
    }

    // Create execution context
    eng->context.reset(eng->engine->createExecutionContext());
    if (!eng->context) {
        fprintf(stderr, "[TensorRT] Failed to create execution context\n");
        return nullptr;
    }

    // Create CUDA stream
    if (cudaStreamCreate(&eng->stream) != cudaSuccess) {
        fprintf(stderr, "[TensorRT] Failed to create CUDA stream\n");
        return nullptr;
    }

    // First pass: set input shapes on the context so output shapes can be inferred
    int32_t nb = eng->engine->getNbIOTensors();
    eng->tensors.resize(nb);
    for (int32_t i = 0; i < nb; i++) {
        auto& t = eng->tensors[i];
        t.name = eng->engine->getIOTensorName(i);
        t.is_input = (eng->engine->getTensorIOMode(t.name.c_str()) == TensorIOMode::kINPUT);
        t.device_ptr = nullptr;

        if (t.is_input) {
            // For dynamic inputs, set the concrete shape (e.g. [1, 3, 640, 640])
            Dims shape = eng->engine->getTensorShape(t.name.c_str());
            for (int d = 0; d < shape.nbDims; d++) {
                if (shape.d[d] == -1) shape.d[d] = 640;
            }
            eng->context->setInputShape(t.name.c_str(), shape);
            t.shape = shape;
        }
    }

    // Second pass: query actual shapes (now resolved) and allocate GPU buffers
    for (int32_t i = 0; i < nb; i++) {
        auto& t = eng->tensors[i];
        if (!t.is_input) {
            t.shape = eng->context->getTensorShape(t.name.c_str());
        }
        t.size_bytes = dims_volume(t.shape) * sizeof(float);
        if (cudaMalloc(&t.device_ptr, t.size_bytes) != cudaSuccess) {
            fprintf(stderr, "[TensorRT] Failed to allocate %zu bytes for tensor %s\n", t.size_bytes, t.name.c_str());
            return nullptr;
        }
        eng->context->setTensorAddress(t.name.c_str(), t.device_ptr);
        fprintf(stderr, "[TensorRT] Tensor '%s' %s shape=[", t.name.c_str(), t.is_input ? "INPUT" : "OUTPUT");
        for (int d = 0; d < t.shape.nbDims; d++) {
            fprintf(stderr, "%ld%s", t.shape.d[d], d < t.shape.nbDims - 1 ? "," : "");
        }
        fprintf(stderr, "] %zu bytes\n", t.size_bytes);
    }

    fprintf(stderr, "[TensorRT] Engine ready with %d I/O tensors\n", nb);
    return eng.release();
}

extern "C" void trt_engine_destroy(TrtEngine* engine) {
    delete engine;
}

extern "C" int32_t trt_engine_num_io(TrtEngine* engine) {
    return static_cast<int32_t>(engine->tensors.size());
}

extern "C" const char* trt_engine_tensor_name(TrtEngine* engine, int32_t index) {
    if (index < 0 || index >= static_cast<int32_t>(engine->tensors.size())) return nullptr;
    return engine->tensors[index].name.c_str();
}

extern "C" int32_t trt_engine_tensor_is_input(TrtEngine* engine, const char* name) {
    for (auto& t : engine->tensors) {
        if (t.name == name) return t.is_input ? 1 : 0;
    }
    return -1;
}

extern "C" void trt_engine_tensor_shape(TrtEngine* engine, const char* name, int64_t* dims, int32_t* nb_dims) {
    for (auto& t : engine->tensors) {
        if (t.name == name) {
            *nb_dims = t.shape.nbDims;
            for (int i = 0; i < t.shape.nbDims; i++) {
                dims[i] = t.shape.d[i];
            }
            return;
        }
    }
    *nb_dims = 0;
}

extern "C" int32_t trt_engine_infer(TrtEngine* engine,
                                     const float* input_data, size_t input_bytes,
                                     float** output_ptrs, size_t* output_sizes,
                                     int32_t num_outputs) {
    // Find the input tensor and copy data to GPU
    for (auto& t : engine->tensors) {
        if (t.is_input) {
            size_t copy_bytes = (input_bytes < t.size_bytes) ? input_bytes : t.size_bytes;
            if (cudaMemcpyAsync(t.device_ptr, input_data, copy_bytes,
                               cudaMemcpyHostToDevice, engine->stream) != cudaSuccess) {
                return -1;
            }
            break;
        }
    }

    // Run inference
    if (!engine->context->enqueueV3(engine->stream)) {
        return -2;
    }

    // Copy outputs back to host
    int32_t out_idx = 0;
    for (auto& t : engine->tensors) {
        if (!t.is_input && out_idx < num_outputs) {
            size_t copy_bytes = (output_sizes[out_idx] < t.size_bytes) ? output_sizes[out_idx] : t.size_bytes;
            if (cudaMemcpyAsync(output_ptrs[out_idx], t.device_ptr, copy_bytes,
                               cudaMemcpyDeviceToHost, engine->stream) != cudaSuccess) {
                return -3;
            }
            out_idx++;
        }
    }

    // Synchronize
    if (cudaStreamSynchronize(engine->stream) != cudaSuccess) {
        return -4;
    }

    return 0;
}
