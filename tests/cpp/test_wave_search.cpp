#include "controlled_evaluator.hpp"
#include "mcts.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

class EvaluationGate {
public:
    void block() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_blocked = true;
        m_cv.notify_all();
        m_cv.wait(lock, [&]() { return m_released; });
    }

    void wait_until_blocked() {
        std::unique_lock<std::mutex> lock(m_mutex);
        require_test(m_cv.wait_for(lock, 2s, [&]() { return m_blocked; }),
                     "evaluator did not block");
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_released = true;
        }
        m_cv.notify_all();
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_blocked = false;
    bool m_released = false;
};

Chessboard startup_board() {
    Chessboard board;
    board.setStartupPieces();
    return board;
}

void require_quiescent(const TreeReport& report) {
    require_test(report.en_vol == 0, "tree retained in-flight nodes");
    require_test(report.pending == 0, "tree retained Pending nodes");
    require_test(report.violations == 0, "tree invariant failed");
}

void require_best_move_is_legal(MCTS& mcts, Chessboard& board) {
    const std::vector<MoveStats> stats = mcts.get_analysis_results();
    if (stats.empty()) return;
    const std::vector<int> legal = board.getLegalMoveIndices();
    require_test(std::find(legal.begin(), legal.end(), stats.front().move_idx)
                     != legal.end(),
                 "best move is illegal on root board");
}

void test_budgets_and_invariants_across_configurations() {
    constexpr std::array<int, 3> WORKERS{2, 4, 8};
    constexpr std::array<int, 2> BATCHES{1, 8};
    constexpr std::array<int, 6> BUDGETS{0, 1, 3, 8, 17, 100};

    for (int workers : WORKERS) {
        for (int batch : BATCHES) {
            ControlledEvaluator evaluator;
            MCTS mcts(&evaluator, 8192, 0);
            Chessboard board = startup_board();
            const std::string fen_before = board.toFEN();

            for (int budget : BUDGETS) {
                const TreeReport before = mcts.inspect_tree();
                const SearchCounters counters_before = mcts.get_counters();
                mcts.step_analysis(
                    board, budget, 1.4f, batch, workers);
                const TreeReport after = mcts.inspect_tree();
                const SearchCounters counters_after = mcts.get_counters();

                require_test(
                    after.root_visits - before.root_visits
                        == static_cast<std::uint64_t>(budget),
                    "multicore search completed the wrong visit budget");
                require_test(
                    counters_after.completed_simulations
                        - counters_before.completed_simulations
                        == static_cast<std::uint64_t>(budget),
                    "completed_simulations does not match budget");
                require_quiescent(after);
                require_test(board.toFEN() == fen_before,
                             "multicore search changed input board");
                if (budget > 0) require_best_move_is_legal(mcts, board);
            }
        }
    }
}

void test_terminal_root_uses_root_visits() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board;
    board.loadFEN(
        "r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4");

    mcts.step_analysis(board, 17, 1.4f, 8, 4);
    const TreeReport report = mcts.inspect_tree();
    const SearchCounters counters = mcts.get_counters();

    require_test(report.root_visits == 17,
                 "terminal root lost completed simulations");
    require_test(report.nodes == 1, "terminal root grew children");
    require_quiescent(report);
    require_test(counters.terminal_hits == 17,
                 "known terminal hits were not counted");
    require_test(counters.completed_simulations == 17,
                 "terminal simulations were not completed");
}

void test_evaluator_failure_cleans_session_and_allows_recovery() {
    ControlledEvaluator evaluator;
    evaluator.fail_on_call = 2;
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = startup_board();

    bool failed = false;
    try {
        mcts.step_analysis(board, 8, 1.4f, 8, 4);
    }
    catch (const std::runtime_error&) {
        failed = true;
    }
    require_test(failed, "controlled evaluator failure was swallowed");
    require_quiescent(mcts.inspect_tree());

    evaluator.fail_on_call = 0;
    mcts.step_analysis(board, 3, 1.4f, 3, 2);
    const TreeReport recovered = mcts.inspect_tree();
    require_quiescent(recovered);
    require_test(recovered.root_visits == 3,
                 "search did not recover after evaluator failure");
}

void test_invalid_evaluator_outputs_clean_session() {
    for (bool truncate_policy : {true, false}) {
        ControlledEvaluator evaluator;
        if (truncate_policy) evaluator.truncate_policy_on_call = 2;
        else evaluator.truncate_values_on_call = 2;
        MCTS mcts(&evaluator, 8192, 0);
        Chessboard board = startup_board();

        bool failed = false;
        try {
            mcts.step_analysis(board, 8, 1.4f, 8, 4);
        }
        catch (const std::runtime_error&) {
            failed = true;
        }
        require_test(failed, "invalid evaluator dimensions were accepted");
        require_quiescent(mcts.inspect_tree());
    }

    ControlledEvaluator evaluator;
    evaluator.output_value = std::numeric_limits<float>::quiet_NaN();
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = startup_board();
    bool failed = false;
    try {
        mcts.step_analysis(board, 8, 1.4f, 8, 4);
    }
    catch (const std::runtime_error&) {
        failed = true;
    }
    require_test(failed, "non-finite evaluator value was accepted");
    require_quiescent(mcts.inspect_tree());
}

void test_fixed_batch_pads_wave_calls_and_keeps_budget() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    require_test(!mcts.fixed_batch(),
                 "le lot fixe doit etre desactive par defaut");
    mcts.set_fixed_batch(true);
    require_test(mcts.fixed_batch(), "le lot fixe n'a pas ete active");

    Chessboard board = startup_board();
    mcts.step_analysis(board, 17, 1.4f, 8, 4);

    require_test(!evaluator.batch_sizes.empty(), "evaluateur jamais appele");
    require_test(evaluator.batch_sizes.front() == 1,
                 "l'expansion de racine doit rester a batch 1");
    for (std::size_t i = 1; i < evaluator.batch_sizes.size(); ++i) {
        require_test(evaluator.batch_sizes[i] == 8,
                     "un appel de vague n'a pas la forme fixe");
    }

    const TreeReport report = mcts.inspect_tree();
    const SearchCounters counters = mcts.get_counters();
    require_test(report.root_visits == 17, "budget de vagues incorrect");
    require_test(counters.completed_simulations == 17,
                 "simulations terminees incorrectes");
    require_quiescent(report);
    require_test(board.toFEN() == startup_board().toFEN(),
                 "le lot fixe a modifie le plateau d'entree");
}

void test_fixed_batch_pads_mono_batch() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 8192, 0);
    mcts.set_fixed_batch(true);
    Chessboard board = startup_board();
    mcts.mcts_search(board, 17, 1.4f, false, 8, 1);

    require_test(!evaluator.batch_sizes.empty(), "evaluateur jamais appele");
    require_test(evaluator.batch_sizes.front() == 1,
                 "l'expansion de racine doit rester a batch 1");
    for (std::size_t i = 1; i < evaluator.batch_sizes.size(); ++i) {
        require_test(evaluator.batch_sizes[i] == 8,
                     "un lot mono n'a pas la forme fixe");
    }
}

void test_gpu_phase_is_quiet_and_update_root_waits_for_session() {
    ControlledEvaluator evaluator;
    EvaluationGate gate;
    evaluator.before_evaluate = [&]() {
        if (evaluator.batch_sizes.size() == 2) gate.block();
    };
    MCTS mcts(&evaluator, 8192, 0);
    Chessboard board = startup_board();
    const int legal_move = board.getLegalMoveIndices().front();
    std::exception_ptr search_error;

    std::thread search([&]() {
        try {
            mcts.step_analysis(board, 8, 1.4f, 8, 4);
        }
        catch (...) {
            search_error = std::current_exception();
        }
    });
    gate.wait_until_blocked();

    const SearchCounters blocked = mcts.get_counters();
    for (int i = 0; i < 1000; ++i) std::this_thread::yield();
    const SearchCounters still_blocked = mcts.get_counters();
    require_test(blocked.waves == still_blocked.waves
                     && blocked.leaf_collisions
                         == still_blocked.leaf_collisions
                     && blocked.completed_simulations
                         == still_blocked.completed_simulations,
                 "CPU collection continued during evaluator call");

    auto update = std::async(std::launch::async, [&]() {
        mcts.update_root(legal_move);
        return true;
    });
    require_test(update.wait_for(0ms) == std::future_status::timeout,
                 "update_root entered while search held raw node pointers");

    gate.release();
    search.join();
    if (search_error) std::rethrow_exception(search_error);
    require_test(update.wait_for(2s) == std::future_status::ready,
                 "update_root did not resume after search");
    require_test(update.get(), "update_root future failed");
    require_quiescent(mcts.inspect_tree());
}

}  // namespace

int main() {
    try {
        test_budgets_and_invariants_across_configurations();
        test_terminal_root_uses_root_visits();
        test_evaluator_failure_cleans_session_and_allows_recovery();
        test_invalid_evaluator_outputs_clean_session();
        test_fixed_batch_pads_wave_calls_and_keeps_budget();
        test_fixed_batch_pads_mono_batch();
        test_gpu_phase_is_quiet_and_update_root_waits_for_session();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
