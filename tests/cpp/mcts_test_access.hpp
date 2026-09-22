#pragma once

#include "mcts.hpp"
#include "mcts_wave.hpp"
#include "search_executor.hpp"

#include <atomic>
#include <vector>

class MCTSTestAccess {
public:
    static LeafWork collect_leaf(
            MCTS& mcts, MCTSNode* root, WorkerContext& context,
            float c_puct, const std::atomic<bool>& cancelled,
            const WaveTestHooks* hooks = nullptr) {
        return mcts.collect_wave_leaf(
            root, context, c_puct, cancelled, hooks);
    }

    static std::vector<LeafWork> collect_wave(
            MCTS& mcts, MCTSNode* root, float c_puct,
            std::size_t slots, std::size_t worker_count,
            SearchExecutor& executor,
            std::vector<WorkerContext>& contexts,
            const std::atomic<bool>& cancelled,
            const WaveTestHooks* hooks = nullptr) {
        return mcts.collect_wave(
            root, c_puct, slots, worker_count, executor, contexts,
            cancelled, hooks);
    }

    static void set_wave_hooks(MCTS& mcts, const WaveTestHooks* hooks) {
        mcts.m_wave_test_hooks = hooks;
    }

    // Preuve directe que la collecte n'a laisse aucun resultat proprietaire
    // dans les contextes persistants, meme sans arbre a inspecter.
    static std::size_t pending_results(MCTS& mcts) {
        std::size_t total = 0;
        for (const WorkerContext& context : mcts.m_worker_contexts) {
            total += context.results.size();
        }
        return total;
    }

    static void store(
            MCTS& mcts, const Chessboard& board,
            const std::vector<int>& legal_moves,
            const std::vector<float>& policy, float value) {
        mcts.store_tt(
            mcts.make_cache_key(board), legal_moves, policy.data(), value);
    }
};
