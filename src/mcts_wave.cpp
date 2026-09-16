#include "mcts.hpp"

#include "search_executor.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

using NodeChildren =
    std::vector<std::pair<int, std::unique_ptr<MCTSNode>>>;

class BoardRollback {
public:
    explicit BoardRollback(Chessboard& board) : m_board(board) {}

    BoardRollback(const BoardRollback&) = delete;
    BoardRollback& operator=(const BoardRollback&) = delete;

    ~BoardRollback() {
        while (m_moves_played > 0) {
            m_board.undoMove();
            --m_moves_played;
        }
    }

    void move_played() noexcept { ++m_moves_played; }

private:
    Chessboard& m_board;
    int m_moves_played = 0;
};

bool is_rule_terminal(const Chessboard& board) {
    return board.checkThreefoldRepetition()
        || board.getHalfMoveClock() >= 100
        || board.checkInsufficientMaterial();
}

float terminal_value(Chessboard& board) {
    if (is_rule_terminal(board)) return 0.0f;
    return board.isInCheck() ? -1.0f : 0.0f;
}

NodeChildren make_children_from_probe(MCTSNode* parent,
                                      const TTProbe& probe) {
    if (probe.policy_size <= 0) {
        throw std::logic_error("hit TT sans politique legale");
    }

    float sum_legal = 0.0f;
    for (int i = 0; i < probe.policy_size; ++i) {
        sum_legal += probe.legal_policy[i].second;
    }

    NodeChildren children;
    children.reserve(static_cast<std::size_t>(probe.policy_size));
    const float uniform = 1.0f / static_cast<float>(probe.policy_size);
    for (int i = 0; i < probe.policy_size; ++i) {
        const int move_index = probe.legal_policy[i].first;
        const float prior = sum_legal > 0.0f
            ? probe.legal_policy[i].second / sum_legal
            : uniform;
        children.emplace_back(
            move_index,
            std::make_unique<MCTSNode>(prior, move_index, parent));
    }
    return children;
}

bool acquire_slot(std::atomic<std::size_t>& available) {
    std::size_t current = available.load(std::memory_order_relaxed);
    while (current > 0) {
        if (available.compare_exchange_weak(
                current, current - 1,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

}  // namespace

LeafWork MCTS::collect_wave_leaf(
        MCTSNode* root, WorkerContext& context, float c_puct,
        const std::atomic<bool>& cancelled,
        const WaveTestHooks* hooks) {
    if (root == nullptr) {
        throw std::invalid_argument("collect_wave_leaf : racine nulle");
    }

    LeafWork work;
    work.node = root;
    work.reservation.reserve(root);
    BoardRollback rollback(context.board);
    MCTSNode* node = root;

    const auto collision = [&]() -> LeafWork {
        work.kind = LeafKind::Collision;
        work.reservation.release();
        return std::move(work);
    };

    while (true) {
        if (cancelled.load(std::memory_order_relaxed)) {
            return collision();
        }

        NodeState state = node->state.load(std::memory_order_acquire);

        // Les nulles de regle ne consultent jamais la table. Le noeud est
        // propre a ce chemin d'arbre, donc un etat Expanded serait incoherent.
        if (is_rule_terminal(context.board)) {
            if (state == NodeState::Pending) return collision();
            if (state == NodeState::Unexpanded) {
                if (hooks && hooks->before_claim) hooks->before_claim(node);
                if (!work.reservation.try_claim(node)) return collision();
                work.reservation.publish(NodeState::Terminal);
            }
            else if (state != NodeState::Terminal) {
                throw std::logic_error(
                    "noeud de nulle deja publie Expanded");
            }
            work.kind = LeafKind::Terminal;
            work.node = node;
            work.terminal_value = 0.0f;
            return work;
        }

        if (state == NodeState::Terminal) {
            work.kind = LeafKind::Terminal;
            work.node = node;
            work.terminal_value = terminal_value(context.board);
            return work;
        }
        if (state == NodeState::Pending) {
            return collision();
        }

        if (state == NodeState::Unexpanded) {
            if (hooks && hooks->before_claim) hooks->before_claim(node);
            if (!work.reservation.try_claim(node)) {
                return collision();
            }

            std::vector<int> legal_moves;
            {
                PhaseTimer timer(&context.timing, SearchPhase::TensorKey);
                legal_moves = context.board.getLegalMoveIndices();
            }
            if (legal_moves.empty()) {
                work.reservation.publish(NodeState::Terminal);
                work.kind = LeafKind::Terminal;
                work.node = node;
                work.terminal_value = terminal_value(context.board);
                return work;
            }

            EvaluationCacheKey key;
            {
                PhaseTimer timer(&context.timing, SearchPhase::TensorKey);
                key = make_cache_key(context.board);
            }
            const TTProbe probe = probe_tt(key, &context.timing);
            record_tt_probe(probe.status);

            if (probe.status == TTProbeStatus::HIT) {
                NodeChildren children = make_children_from_probe(node, probe);
                if (hooks && hooks->before_tt_publish) {
                    hooks->before_tt_publish(node);
                }
                {
                    PhaseTimer timer(
                        &context.timing, SearchPhase::Expansion);
                    node->children.swap(children);
                    work.reservation.publish(NodeState::Expanded);
                }
                state = NodeState::Expanded;
            }
            else {
                work.kind = LeafKind::Network;
                work.node = node;
                work.key = key;
                work.legal_moves = std::move(legal_moves);
                {
                    PhaseTimer timer(
                        &context.timing, SearchPhase::TensorKey);
                    context.board.getAlphaZeroTensor(work.tensor);
                }
                return work;
            }
        }

        if (state != NodeState::Expanded || node->children.empty()) {
            throw std::logic_error(
                "collect_wave_leaf : noeud publie sans enfants");
        }

        PhaseTimer selection_timer(
            &context.timing, SearchPhase::Selection);
        float visited_policy_sum = 0.0f;
        for (const auto& child : node->children) {
            if (child.second->visit_count > 0) {
                visited_policy_sum += child.second->prior;
            }
        }
        const float fpu_reduction =
            0.30f * std::sqrt(visited_policy_sum);
        const float parent_q = node->q_value();
        const float exploration_factor =
            c_puct * std::sqrt(static_cast<float>(node->visit_count));

        float best_score = -std::numeric_limits<float>::infinity();
        int best_move = -1;
        MCTSNode* best_child = nullptr;
        for (const auto& child : node->children) {
            const float score = child.second->ucb_score(
                exploration_factor, parent_q, fpu_reduction);
            if (score > best_score) {
                best_score = score;
                best_move = child.first;
                best_child = child.second.get();
            }
        }
        if (best_child == nullptr) {
            throw std::logic_error(
                "collect_wave_leaf : aucun enfant selectionnable");
        }

        // Le virtual loss devient visible avant de jouer le coup. Deux
        // workers peuvent encore choisir simultanement le meme enfant, le CAS
        // Pending reste donc la garantie de correction.
        work.reservation.reserve(best_child);
        if (!apply_move_by_index(context.board, best_move)) {
            throw std::runtime_error(
                "collect_wave_leaf : application du coup impossible");
        }
        rollback.move_played();
        node = best_child;
        work.node = node;
    }
}

std::vector<LeafWork> MCTS::collect_wave(
        MCTSNode* root, float c_puct, std::size_t slots,
        std::size_t worker_count, SearchExecutor& executor,
        std::vector<WorkerContext>& contexts,
        const std::atomic<bool>& cancelled,
        const WaveTestHooks* hooks) {
    if (root == nullptr) {
        throw std::invalid_argument("collect_wave : racine nulle");
    }
    if (slots == 0 || worker_count == 0) {
        throw std::invalid_argument(
            "collect_wave : slots et workers doivent etre positifs");
    }
    if (contexts.size() < worker_count) {
        throw std::invalid_argument(
            "collect_wave : contextes workers insuffisants");
    }
    if (slots > (std::numeric_limits<std::size_t>::max() - worker_count) / 4) {
        throw std::overflow_error("collect_wave : budget trop grand");
    }

    for (std::size_t id = 0; id < worker_count; ++id) {
        contexts[id].results.clear();
    }

    std::atomic<std::size_t> available{slots};
    std::atomic<std::size_t> attempts{0};
    const std::size_t max_attempts = 4 * slots + worker_count;

    try {
        executor.run(worker_count, [&](std::size_t worker_id) {
            WorkerContext& context = contexts[worker_id];
            while (!cancelled.load(std::memory_order_relaxed)) {
                if (!acquire_slot(available)) return;

                const std::size_t attempt =
                    attempts.fetch_add(1, std::memory_order_relaxed);
                if (attempt >= max_attempts) {
                    available.fetch_add(1, std::memory_order_relaxed);
                    return;
                }

                LeafWork work = collect_wave_leaf(
                    root, context, c_puct, cancelled, hooks);
                if (work.kind == LeafKind::Collision) {
                    available.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                context.results.push_back(std::move(work));
            }
        });
    }
    catch (...) {
        for (std::size_t id = 0; id < worker_count; ++id) {
            contexts[id].results.clear();
        }
        throw;
    }

    std::vector<LeafWork> merged;
    merged.reserve(slots - available.load(std::memory_order_relaxed));
    for (std::size_t id = 0; id < worker_count; ++id) {
        for (LeafWork& work : contexts[id].results) {
            merged.push_back(std::move(work));
        }
        contexts[id].results.clear();
    }

    if (merged.empty() && !cancelled.load(std::memory_order_relaxed)) {
        throw std::logic_error(
            "collect_wave : aucune progression apres les collisions");
    }
    return merged;
}
