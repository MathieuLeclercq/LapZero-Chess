#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <onnxruntime_cxx_api.h>
#include "evaluator.hpp"

// Chronometrage interne optionnel, desactive par defaut. Sert au banc hors
// arbre pour separer session->Run du softmax C++.
struct EvaluatorTiming {
    std::uint64_t run_ns = 0;
    std::uint64_t softmax_ns = 0;
};

class ONNXEvaluator : public Evaluator {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    bool m_timing_enabled = false;
    EvaluatorTiming m_last_timing;

public:

    ONNXEvaluator(const std::string& model_path, bool use_gpu = false);

    // NOUVEAU : Inférence Batchée (pour Self-Play Manager / GPU)
    void evaluate_batch(
        const std::vector<float>& input_tensor, 
        std::vector<float>& policies, 
        std::vector<float>& values, 
        int batch_size) override;

    void set_timing_enabled(bool enabled) { m_timing_enabled = enabled; }
    EvaluatorTiming get_last_timing() const { return m_last_timing; }
};
