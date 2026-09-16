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
            const std::atomic<bool>& cancelled) {
        return mcts.collect_wave(
            root, c_puct, slots, worker_count, executor, contexts,
            cancelled);
    }

    static void store(
            MCTS& mcts, const Chessboard& board,
            const std::vector<int>& legal_moves,
            const std::vector<float>& policy, float value) {
        mcts.store_tt(
            mcts.make_cache_key(board), legal_moves, policy.data(), value);
    }
};
