#include "controlled_evaluator.hpp"
#include "mcts.hpp"
#include "selfplay_manager.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

constexpr int GAME_COUNT = 8;
constexpr int INPUT_SIZE = 119 * 64;
constexpr int POLICY_SIZE = 4672;

void test_openmp_selfplay_core_with_shared_cache() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 3, 0);
    std::vector<Chessboard> boards(GAME_COUNT);
    std::vector<std::unique_ptr<MCTSNode>> roots;
    roots.reserve(GAME_COUNT);
    for (int game = 0; game < GAME_COUNT; ++game) {
        boards[game].setStartupPieces();
        roots.push_back(std::make_unique<MCTSNode>(0.0f));
        mcts.expand_node_single(roots.back().get(), boards[game]);
    }

    std::vector<MCTSNode*> leaves(GAME_COUNT);
    std::vector<int> moves_played(GAME_COUNT);
    std::vector<PathReservation> reservations(GAME_COUNT);
    std::vector<std::vector<float>> tensors(
        GAME_COUNT, std::vector<float>(INPUT_SIZE));
    std::vector<float> input;
    std::vector<float> policies;
    std::vector<float> values;

    for (int round = 0; round < 64; ++round) {
        std::fill(leaves.begin(), leaves.end(), nullptr);
        std::fill(moves_played.begin(), moves_played.end(), 0);

#pragma omp parallel for num_threads(GAME_COUNT) schedule(static)
        for (int game = 0; game < GAME_COUNT; ++game) {
            leaves[game] = mcts.advance_to_leaf(
                roots[game].get(), boards[game], 1.4f,
                moves_played[game], reservations[game]);
            if (leaves[game] != nullptr) {
                boards[game].getAlphaZeroTensor(tensors[game]);
            }
        }

        input.clear();
        std::vector<int> active;
        for (int game = 0; game < GAME_COUNT; ++game) {
            if (leaves[game] == nullptr) continue;
            active.push_back(game);
            input.insert(input.end(), tensors[game].begin(),
                         tensors[game].end());
        }
        if (!active.empty()) {
            evaluator.evaluate_batch(
                input, policies, values, static_cast<int>(active.size()));
            for (std::size_t index = 0; index < active.size(); ++index) {
                const int game = active[index];
                mcts.expand_and_backup(
                    leaves[game], boards[game],
                    policies.data() + index * POLICY_SIZE,
                    values[index], reservations[game]);
                for (int move = 0; move < moves_played[game]; ++move) {
                    boards[game].undoMove();
                }
            }
        }
    }

    for (const auto& root : roots) {
        const TreeReport report = mcts.inspect_tree(root.get());
        require_test(report.root_visits == 64,
                     "shared self-play root lost visits");
        require_test(report.en_vol == 0 && report.pending == 0,
                     "shared self-play root retained a reservation");
        require_test(report.violations == 0,
                     "shared self-play root is invalid");
    }
}

void test_selfplay_manager_with_controlled_evaluator() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);
    const std::vector<GameResult> games = manager.generate_games(2);

    require_test(games.size() == 2, "self-play returned the wrong game count");
    for (const GameResult& game : games) {
        require_test(game.move_count >= 0, "negative training move count");
        require_test(game.total_real_moves > 0
                         && game.total_real_moves <= 300,
                     "self-play game length is outside its bounds");
        require_test(game.flat_states.size()
                         == static_cast<std::size_t>(game.move_count)
                             * INPUT_SIZE,
                     "self-play state dimensions are inconsistent");
        require_test(game.flat_policies.size()
                         == static_cast<std::size_t>(game.move_count)
                             * POLICY_SIZE,
                     "self-play policy dimensions are inconsistent");
        require_test(std::all_of(
            game.flat_states.begin(), game.flat_states.end(),
            [](float value) { return std::isfinite(value); }),
            "self-play emitted a non-finite state");
        require_test(std::all_of(
            game.flat_policies.begin(), game.flat_policies.end(),
            [](float value) { return std::isfinite(value); }),
            "self-play emitted a non-finite policy");
        require_test(std::isfinite(game.final_outcome),
                     "self-play emitted a non-finite outcome");
    }
}

}  // namespace

int main() {
    try {
        test_openmp_selfplay_core_with_shared_cache();
        test_selfplay_manager_with_controlled_evaluator();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
