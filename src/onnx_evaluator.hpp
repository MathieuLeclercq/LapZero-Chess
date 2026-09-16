#pragma once
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "evaluator.hpp"

class ONNXEvaluator : public Evaluator {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;

public:

    ONNXEvaluator(const std::string& model_path, bool use_gpu = false);

    // NOUVEAU : Inférence Batchée (pour Self-Play Manager / GPU)
    void evaluate_batch(
        const std::vector<float>& input_tensor, 
        std::vector<float>& policies, 
        std::vector<float>& values, 
        int batch_size) override;
};
