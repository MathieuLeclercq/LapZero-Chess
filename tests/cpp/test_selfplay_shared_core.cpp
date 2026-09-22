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

class SelfPlayTestAccess {
public:
    static MCTSNode* root(SelfPlayManager& manager, int game_idx) {
        return manager.m_roots[static_cast<std::size_t>(game_idx)].get();
    }

    static float pending_epsilon(const SelfPlayManager& manager,
                                 int game_idx) {
        return manager.m_pending_epsilon[static_cast<std::size_t>(game_idx)];
    }

    static void demarrer_slot(SelfPlayManager& manager, int game_idx) {
        manager.demarrer_slot(game_idx);
    }
};

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

void test_selfplay_rejects_invalid_batch_and_cleans_up() {
    ControlledEvaluator evaluator;
    // Deux expansions de racine a la construction, puis le premier lot GPU.
    evaluator.corrupt_call = 3;
    evaluator.values_corruption =
        ControlledEvaluator::Corruption::NonFinite;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);

    bool failed = false;
    try {
        (void)manager.generate_games(2);
    }
    catch (const std::runtime_error&) {
        failed = true;
    }
    require_test(failed, "self-play accepted an invalid batch output");

    // Les reservations et les plateaux en attente doivent avoir ete nettoyes :
    // le gestionnaire reste utilisable apres le rejet.
    evaluator.corrupt_call = 0;
    const std::vector<GameResult> games = manager.generate_games(2);
    require_test(games.size() == 2, "self-play did not recover");
}

void test_selfplay_roots_are_expanded_and_noised_exactly_once() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);
    // Le constructeur ne demarre plus de partie : le quota n'est connu que dans
    // generate_games. On demarre deux places pour controler les racines.
    SelfPlayTestAccess::demarrer_slot(manager, 0);
    SelfPlayTestAccess::demarrer_slot(manager, 1);

    // La deuxieme partie partage la position de depart de la premiere : sa
    // racine est servie par la table et doit tout de meme avoir ses enfants et
    // son bruit de Dirichlet.
    for (int game = 0; game < 2; ++game) {
        MCTSNode* root = SelfPlayTestAccess::root(manager, game);
        require_test(root != nullptr, "missing self-play root");
        require_test(root->state.load() == NodeState::Expanded,
                     "self-play root was left unexpanded");
        require_test(!root->children.empty(),
                     "self-play root has no children");
        require_test(SelfPlayTestAccess::pending_epsilon(manager, game) == 0.0f,
                     "self-play root kept a pending noise");
    }
}

void test_selfplay_generation_is_finite_and_counted() {
    struct Cas {
        int places;
        int quota;
    };
    const Cas cas[] = {{2, 3}, {4, 0}, {4, 1}, {4, 3}, {4, 4}, {4, 7}};

    for (const Cas& c : cas) {
        ControlledEvaluator evaluator;
        SelfPlayManager manager(&evaluator, c.places, 2, 1, 0.5f, 8192);
        const std::vector<GameResult> games = manager.generate_games(c.quota);
        const SelfPlayStats stats = manager.get_stats();

        require_test(games.size() == static_cast<std::size_t>(c.quota),
                     "the generation did not return exactly N games");
        require_test(stats.games_started == static_cast<std::uint64_t>(c.quota),
                     "the number of starts does not match the quota");
        require_test(stats.games_completed == static_cast<std::uint64_t>(c.quota),
                     "the number of completions does not match the quota");
        require_test(stats.active_slots == 0,
                     "a place stayed active after the generation");
        require_test(stats.replayed_plies == 0,
                     "replayed plies were counted without puzzles");
        if (c.quota == 0) {
            require_test(stats.new_plies == 0, "N=0 played a move");
            require_test(evaluator.batch_sizes.empty(),
                         "N=0 called the evaluator");
        }
        else {
            require_test(stats.new_plies > 0, "no new ply was counted");
        }
    }
}

void test_successive_selfplay_generations_are_independent() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);

    const std::vector<GameResult> premier = manager.generate_games(2);
    const SelfPlayStats stats_premier = manager.get_stats();
    const std::vector<GameResult> second = manager.generate_games(3);
    const SelfPlayStats stats_second = manager.get_stats();

    require_test(premier.size() == 2, "the first generation is incomplete");
    require_test(second.size() == 3, "the second generation is incomplete");
    require_test(stats_premier.games_completed == 2,
                 "the first counters were changed by the second call");
    require_test(stats_second.games_started == 3
                     && stats_second.games_completed == 3,
                 "the second generation did not restart its counters");
}

}  // namespace

int main() {
    try {
        test_openmp_selfplay_core_with_shared_cache();
        test_selfplay_manager_with_controlled_evaluator();
        test_selfplay_rejects_invalid_batch_and_cleans_up();
        test_selfplay_roots_are_expanded_and_noised_exactly_once();
        test_selfplay_generation_is_finite_and_counted();
        test_successive_selfplay_generations_are_independent();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
