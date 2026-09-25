#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <onnxruntime_cxx_api.h>
#include "evaluator.hpp"

class ONNXEvaluator : public Evaluator {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    bool m_timing_enabled = false;
    EvaluatorTiming m_last_timing;
    EvaluatorTotals m_totals;

public:

    ONNXEvaluator(const std::string& model_path, bool use_gpu = false);

    // NOUVEAU : Inférence Batchée (pour Self-Play Manager / GPU)
    void evaluate_batch(
        const std::vector<float>& input_tensor, 
        std::vector<float>& policies, 
        std::vector<float>& values, 
        int batch_size) override;

    void set_timing_enabled(bool enabled) override {
        m_timing_enabled = enabled;
    }
    EvaluatorTiming get_last_timing() const { return m_last_timing; }
    EvaluatorTotals diagnostic_totals() const override { return m_totals; }
};
