#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vad {

enum class ModelFormat { kOnnx, kTflite };

// Sniffs the file content: TFLite flatbuffers carry the "TFL3" identifier at byte 4,
// ONNX files are protobufs starting with the ir_version field (tag 0x08).
// Throws VadError for anything else.
ModelFormat DetectFormat(const uint8_t* data, size_t size);
const char* FormatName(ModelFormat f);

enum class DType { kFloat32, kInt64 };

struct InputTensor {
    std::string name;
    DType type = DType::kFloat32;
    std::vector<int64_t> shape;  // empty = scalar
    const void* data = nullptr;
};

struct OutputTensor {
    std::vector<int64_t> shape;
    std::vector<float> data;  // quantized outputs are dequantized
};

// Minimal runtime-agnostic session used by the VAD runners. Tensors are matched by
// name first; when a model uses different names (e.g. the TF-Hub YAMNet), the n-th
// requested tensor falls back to the n-th tensor of the model.
class InferenceSession {
public:
    virtual ~InferenceSession() = default;
    virtual ModelFormat format() const = 0;
    virtual bool HasInput(const std::string& name) const = 0;
    // Shape of an input as declared by the model; -1 marks a dynamic dimension.
    virtual std::vector<int64_t> InputShape(size_t index) const = 0;
    virtual std::vector<OutputTensor> Run(const std::vector<InputTensor>& inputs,
                                          const std::vector<std::string>& outputs) = 0;
};

// Picks ONNX Runtime or TFLite from the bytes. The session keeps `bytes` alive
// (TFLite reads weights from the buffer for its whole lifetime).
std::unique_ptr<InferenceSession> CreateSession(std::vector<uint8_t> bytes, int num_threads);

}  // namespace vad
