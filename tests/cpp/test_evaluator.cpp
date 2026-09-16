#include "controlled_evaluator.hpp"
#include "mcts.hpp"
#include "test_support.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void test_single_evaluation_uses_batch_contract() {
    ControlledEvaluator evaluator;
    evaluator.output_value = 0.375f;
    bool hook_called = false;
    evaluator.before_evaluate = [&]() { hook_called = true; };

    std::vector<float> input(119 * 64, 0.0f);
    std::vector<float> policy;
    float value = 0.0f;
    evaluator.evaluate(input, policy, value);

    require_test(evaluator.batch_sizes == std::vector<int>{1},
                 "single evaluation did not use a batch of one");
    require_test(hook_called, "evaluation hook was not called");
    require_test(policy.size() == 4672, "single policy has wrong size");
    require_test(value == 0.375f, "single value was not forwarded");
}

void test_mcts_accepts_an_injected_evaluator() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board;
    board.setStartupPieces();

    mcts.step_analysis(board, 17, 1.4f, 8);
    const TreeReport report = mcts.inspect_tree();

    require_test(report.violations == 0, "invalid reference tree");
    require_test(!evaluator.batch_sizes.empty(), "network was not called");
}

void test_controlled_failure_is_observable() {
    ControlledEvaluator evaluator;
    evaluator.fail_on_call = 1;
    std::vector<float> input(119 * 64, 0.0f);
    std::vector<float> policy;
    float value = 0.0f;

    bool threw = false;
    try {
        evaluator.evaluate(input, policy, value);
    }
    catch (const std::runtime_error&) {
        threw = true;
    }
    require_test(threw, "controlled failure did not propagate");
}

}  // namespace

int main() {
    try {
        test_single_evaluation_uses_batch_contract();
        test_mcts_accepts_an_injected_evaluator();
        test_controlled_failure_is_observable();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
