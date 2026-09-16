#pragma once

#include "evaluator.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

class ControlledEvaluator final : public Evaluator {
public:
    std::vector<int> batch_sizes;
    int fail_on_call = 0;
    float output_value = 0.0f;
    std::function<void()> before_evaluate;

    void evaluate_batch(const std::vector<float>& input,
                        std::vector<float>& policies,
                        std::vector<float>& values,
                        int batch_size) override {
        if (batch_size <= 0) {
            throw std::invalid_argument("batch_size doit etre positif");
        }
        const std::size_t expected =
            static_cast<std::size_t>(batch_size) * 119 * 64;
        if (input.size() != expected) {
            throw std::invalid_argument("taille du tenseur inattendue");
        }

        batch_sizes.push_back(batch_size);
        if (before_evaluate) {
            before_evaluate();
        }
        if (fail_on_call > 0
            && static_cast<int>(batch_sizes.size()) == fail_on_call) {
            throw std::runtime_error("echec controle de l evaluateur");
        }

        policies.assign(static_cast<std::size_t>(batch_size) * 4672,
                        1.0f / 4672.0f);
        values.assign(batch_size, output_value);
    }
};
