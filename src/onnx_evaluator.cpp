#include "onnx_evaluator.hpp"
#include "softmax.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

ONNXEvaluator::ONNXEvaluator(const std::string& model_path, bool use_gpu)
    : env(ORT_LOGGING_LEVEL_ERROR, "AlphaZeroMCTS")
{
    if (use_gpu) {
        // Activation de la carte graphique (Nvidia CUDA)
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0; // Utilise le GPU 0
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    // Optimisations CPU classiques
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::wstring w_model_path(model_path.begin(), model_path.end());
    session = std::make_unique<Ort::Session>(env, w_model_path.c_str(), session_options);
}

void ONNXEvaluator::evaluate_batch(
    const std::vector<float>& input_tensor, 
    std::vector<float>& policies, 
    std::vector<float>& values, 
    int batch_size) {

    std::array<int64_t, 4> input_shape = { batch_size, 119, 8, 8 };
    const char* input_names[] = { "input" };
    const char* output_names[] = { "policy", "value" };

    // Comptes toujours tenus : ils ne dependent pas des horloges.
    m_totals.run_calls++;
    m_totals.evaluated_rows += static_cast<std::uint64_t>(batch_size);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Le nombre EXACT d'éléments attendus pour ce batch
    size_t expected_elements = batch_size * 119 * 8 * 8;

    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(input_tensor.data()), expected_elements,
        input_shape.data(), input_shape.size()
    );

    //Ort::Value input_ort = Ort::Value::CreateTensor<float>(
    //    memory_info, const_cast<float*>(input_tensor.data()), input_tensor.size(),
    //    input_shape.data(), input_shape.size()
    //);

    std::vector<Ort::Value> output_tensors;
    const auto run_start = m_timing_enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    try {
        output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_names, &input_ort, 1, output_names, 2);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "\n[ERREUR FATALE ONNX RUNTIME] : " << e.what() << std::endl;
        throw std::runtime_error(e.what());
    }
    if (m_timing_enabled) {
        m_last_timing.run_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - run_start).count());
        m_totals.timing.run_ns += m_last_timing.run_ns;
    }

    const float* policy_data = output_tensors[0].GetTensorData<float>();
    const float* value_data = output_tensors[1].GetTensorData<float>();

    policies.resize(batch_size * 4672);
    values.resize(batch_size);

    const auto softmax_start = m_timing_enabled
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};

    // Softmax indépendant pour CHAQUE position du batch. Le mode parallèle
    // reste disponible dans softmax.hpp, mais il est désactivé ici : en
    // contexte self-play chargé, sa région OpenMP par appel coûte plus cher
    // que le calcul séquentiel (mesure du 2026-09-25).
    softmax_rows(policy_data, policies.data(), batch_size, 4672, false);
    for (int b = 0; b < batch_size; ++b) {
        values[b] = value_data[b];
    }

    if (m_timing_enabled) {
        m_last_timing.softmax_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - softmax_start).count());
        m_totals.timing.softmax_ns += m_last_timing.softmax_ns;
    }
}
