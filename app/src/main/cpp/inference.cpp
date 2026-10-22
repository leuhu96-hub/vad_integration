#include "inference.h"

#include <onnxruntime_cxx_api.h>
#include <tflite/c/c_api.h>
#include <tflite/c/common.h>

#include <algorithm>
#include <cstring>

#include "common.h"

namespace vad {

ModelFormat DetectFormat(const uint8_t* data, size_t size) {
    if (size >= 8 && std::memcmp(data + 4, "TFL3", 4) == 0) return ModelFormat::kTflite;
    if (size >= 2 && data[0] == 0x08) return ModelFormat::kOnnx;
    throw VadError("Unrecognised model format (expected .onnx or .tflite)");
}

const char* FormatName(ModelFormat f) {
    return f == ModelFormat::kTflite ? "TFLite" : "ONNX";
}

namespace {

// Maps each requested name to an index among `available`; unknown names take the
// position they were requested at.
std::vector<size_t> ResolveNames(const std::vector<std::string>& requested,
                                 const std::vector<std::string>& available, const char* what) {
    std::vector<size_t> idx;
    for (size_t i = 0; i < requested.size(); ++i) {
        auto it = std::find(available.begin(), available.end(), requested[i]);
        if (it != available.end()) {
            idx.push_back(static_cast<size_t>(it - available.begin()));
        } else if (i < available.size()) {
            idx.push_back(i);
        } else {
            throw VadError(std::string("Model has no ") + what + " '" + requested[i] + "'");
        }
    }
    return idx;
}

// ------------------------------------------------------------------ ONNX Runtime

Ort::Env& OrtEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vadnative");
    return env;
}

class OrtSession : public InferenceSession {
public:
    OrtSession(const std::vector<uint8_t>& bytes, int num_threads) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(std::max(1, num_threads));
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // ORT copies what it needs, the bytes may be released afterwards.
        session_ = Ort::Session(OrtEnv(), bytes.data(), bytes.size(), opts);
        Ort::AllocatorWithDefaultOptions alloc;
        for (size_t i = 0; i < session_.GetInputCount(); ++i) {
            in_names_.push_back(session_.GetInputNameAllocated(i, alloc).get());
            in_shapes_.push_back(session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
            out_names_.push_back(session_.GetOutputNameAllocated(i, alloc).get());
        }
    }

    ModelFormat format() const override { return ModelFormat::kOnnx; }

    bool HasInput(const std::string& name) const override {
        return std::find(in_names_.begin(), in_names_.end(), name) != in_names_.end();
    }

    std::vector<int64_t> InputShape(size_t index) const override { return in_shapes_.at(index); }

    std::vector<OutputTensor> Run(const std::vector<InputTensor>& inputs,
                                  const std::vector<std::string>& outputs) override {
        std::vector<std::string> req_in;
        for (const auto& t : inputs) req_in.push_back(t.name);
        const auto in_idx = ResolveNames(req_in, in_names_, "input");
        const auto out_idx = ResolveNames(outputs, out_names_, "output");

        std::vector<Ort::Value> values;
        std::vector<const char*> in_ptrs, out_ptrs;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& t = inputs[i];
            size_t count = 1;
            for (int64_t d : t.shape) count *= static_cast<size_t>(d);
            if (t.type == DType::kFloat32) {
                values.push_back(Ort::Value::CreateTensor<float>(
                        mem_, static_cast<float*>(const_cast<void*>(t.data)), count,
                        t.shape.data(), t.shape.size()));
            } else {
                values.push_back(Ort::Value::CreateTensor<int64_t>(
                        mem_, static_cast<int64_t*>(const_cast<void*>(t.data)), count,
                        t.shape.data(), t.shape.size()));
            }
            in_ptrs.push_back(in_names_[in_idx[i]].c_str());
        }
        for (size_t i : out_idx) out_ptrs.push_back(out_names_[i].c_str());

        auto out = session_.Run(Ort::RunOptions{nullptr}, in_ptrs.data(), values.data(),
                                values.size(), out_ptrs.data(), out_ptrs.size());
        std::vector<OutputTensor> result(out.size());
        for (size_t i = 0; i < out.size(); ++i) {
            const auto info = out[i].GetTensorTypeAndShapeInfo();
            result[i].shape = info.GetShape();
            const float* p = out[i].GetTensorData<float>();
            result[i].data.assign(p, p + info.GetElementCount());
        }
        return result;
    }

private:
    Ort::Session session_{nullptr};
    Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::vector<std::string> in_names_, out_names_;
    std::vector<std::vector<int64_t>> in_shapes_;
};

// ------------------------------------------------------------------ TFLite

class TfliteSession : public InferenceSession {
public:
    TfliteSession(std::vector<uint8_t> bytes, int num_threads) : bytes_(std::move(bytes)) {
        model_ = TfLiteModelCreate(bytes_.data(), bytes_.size());
        if (!model_) throw VadError("TFLite: cannot parse model");
        options_ = TfLiteInterpreterOptionsCreate();
        TfLiteInterpreterOptionsSetNumThreads(options_, std::max(1, num_threads));
        interp_ = TfLiteInterpreterCreate(model_, options_);
        if (!interp_) throw VadError("TFLite: cannot create interpreter (unsupported ops?)");
        if (TfLiteInterpreterAllocateTensors(interp_) != kTfLiteOk) {
            throw VadError("TFLite: AllocateTensors failed");
        }
        for (int i = 0; i < TfLiteInterpreterGetInputTensorCount(interp_); ++i) {
            const TfLiteTensor* t = TfLiteInterpreterGetInputTensor(interp_, i);
            in_names_.push_back(TfLiteTensorName(t));
            std::vector<int64_t> shape;
            const TfLiteIntArray* sig = TfLiteTensorGetDimsSignature(t);
            if (sig && sig->size > 0) {
                for (int d = 0; d < sig->size; ++d) shape.push_back(sig->data[d]);
            } else {
                for (int d = 0; d < TfLiteTensorNumDims(t); ++d) shape.push_back(TfLiteTensorDim(t, d));
            }
            in_shapes_.push_back(shape);
        }
        for (int i = 0; i < TfLiteInterpreterGetOutputTensorCount(interp_); ++i) {
            out_names_.push_back(TfLiteTensorName(TfLiteInterpreterGetOutputTensor(interp_, i)));
        }
    }

    ~TfliteSession() override {
        if (interp_) TfLiteInterpreterDelete(interp_);
        if (options_) TfLiteInterpreterOptionsDelete(options_);
        if (model_) TfLiteModelDelete(model_);
    }

    ModelFormat format() const override { return ModelFormat::kTflite; }

    bool HasInput(const std::string& name) const override {
        return std::find(in_names_.begin(), in_names_.end(), name) != in_names_.end();
    }

    std::vector<int64_t> InputShape(size_t index) const override { return in_shapes_.at(index); }

    std::vector<OutputTensor> Run(const std::vector<InputTensor>& inputs,
                                  const std::vector<std::string>& outputs) override {
        std::vector<std::string> req_in;
        for (const auto& t : inputs) req_in.push_back(t.name);
        const auto in_idx = ResolveNames(req_in, in_names_, "input");
        const auto out_idx = ResolveNames(outputs, out_names_, "output");

        // Resize only when a shape actually changes; reallocation is not free.
        bool resized = false;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const TfLiteTensor* t = TfLiteInterpreterGetInputTensor(interp_, static_cast<int>(in_idx[i]));
            const auto& want = inputs[i].shape;
            bool same = TfLiteTensorNumDims(t) == static_cast<int>(want.size());
            for (size_t d = 0; same && d < want.size(); ++d) {
                same = TfLiteTensorDim(t, static_cast<int>(d)) == want[d];
            }
            if (!same) {
                std::vector<int> dims(want.begin(), want.end());
                if (TfLiteInterpreterResizeInputTensor(interp_, static_cast<int>(in_idx[i]),
                                                       dims.data(), static_cast<int>(dims.size())) != kTfLiteOk) {
                    throw VadError("TFLite: cannot resize input '" + in_names_[in_idx[i]] + "'");
                }
                resized = true;
            }
        }
        if (resized && TfLiteInterpreterAllocateTensors(interp_) != kTfLiteOk) {
            throw VadError("TFLite: AllocateTensors failed after resize");
        }

        for (size_t i = 0; i < inputs.size(); ++i) {
            TfLiteTensor* t = TfLiteInterpreterGetInputTensor(interp_, static_cast<int>(in_idx[i]));
            CopyIn(inputs[i], t);
        }
        if (TfLiteInterpreterInvoke(interp_) != kTfLiteOk) throw VadError("TFLite: Invoke failed");

        std::vector<OutputTensor> result;
        for (size_t i : out_idx) {
            const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(interp_, static_cast<int>(i));
            result.push_back(CopyOut(t));
        }
        return result;
    }

private:
    static void CopyIn(const InputTensor& in, TfLiteTensor* t) {
        size_t count = 1;
        for (int64_t d : in.shape) count *= static_cast<size_t>(d);
        void* dst = TfLiteTensorData(t);
        const TfLiteType type = TfLiteTensorType(t);
        if (in.type == DType::kFloat32 && type == kTfLiteFloat32) {
            std::memcpy(dst, in.data, count * sizeof(float));
        } else if (in.type == DType::kInt64 && type == kTfLiteInt64) {
            std::memcpy(dst, in.data, count * sizeof(int64_t));
        } else if (in.type == DType::kInt64 && type == kTfLiteInt32) {
            const auto* src = static_cast<const int64_t*>(in.data);
            auto* out = static_cast<int32_t*>(dst);
            for (size_t k = 0; k < count; ++k) out[k] = static_cast<int32_t>(src[k]);
        } else if (in.type == DType::kInt64 && type == kTfLiteFloat32) {
            const auto* src = static_cast<const int64_t*>(in.data);
            auto* out = static_cast<float*>(dst);
            for (size_t k = 0; k < count; ++k) out[k] = static_cast<float>(src[k]);
        } else {
            throw VadError(std::string("TFLite: unsupported input type for '") + TfLiteTensorName(t) + "'");
        }
    }

    static OutputTensor CopyOut(const TfLiteTensor* t) {
        OutputTensor out;
        size_t count = 1;
        for (int d = 0; d < TfLiteTensorNumDims(t); ++d) {
            out.shape.push_back(TfLiteTensorDim(t, d));
            count *= static_cast<size_t>(TfLiteTensorDim(t, d));
        }
        out.data.resize(count);
        const void* src = TfLiteTensorData(t);
        const TfLiteQuantizationParams q = TfLiteTensorQuantizationParams(t);
        switch (TfLiteTensorType(t)) {
            case kTfLiteFloat32:
                std::memcpy(out.data.data(), src, count * sizeof(float));
                break;
            case kTfLiteUInt8:
                for (size_t k = 0; k < count; ++k) {
                    out.data[k] = q.scale * (static_cast<const uint8_t*>(src)[k] - q.zero_point);
                }
                break;
            case kTfLiteInt8:
                for (size_t k = 0; k < count; ++k) {
                    out.data[k] = q.scale * (static_cast<const int8_t*>(src)[k] - q.zero_point);
                }
                break;
            default:
                throw VadError(std::string("TFLite: unsupported output type for '") + TfLiteTensorName(t) + "'");
        }
        return out;
    }

    std::vector<uint8_t> bytes_;
    TfLiteModel* model_ = nullptr;
    TfLiteInterpreterOptions* options_ = nullptr;
    TfLiteInterpreter* interp_ = nullptr;
    std::vector<std::string> in_names_, out_names_;
    std::vector<std::vector<int64_t>> in_shapes_;
};

}  // namespace

std::unique_ptr<InferenceSession> CreateSession(std::vector<uint8_t> bytes, int num_threads) {
    switch (DetectFormat(bytes.data(), bytes.size())) {
        case ModelFormat::kTflite:
            return std::make_unique<TfliteSession>(std::move(bytes), num_threads);
        case ModelFormat::kOnnx:
            return std::make_unique<OrtSession>(bytes, num_threads);
    }
    throw VadError("Unknown model format");
}

}  // namespace vad
