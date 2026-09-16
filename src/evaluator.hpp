#pragma once

#include <vector>

class Evaluator {
public:
    virtual ~Evaluator() = default;

    virtual void evaluate_batch(const std::vector<float>& input,
                                std::vector<float>& policies,
                                std::vector<float>& values,
                                int batch_size) = 0;

    void evaluate(const std::vector<float>& input,
                  std::vector<float>& policy,
                  float& value) {
        std::vector<float> values(1);
        evaluate_batch(input, policy, values, 1);
        value = values[0];
    }
};
