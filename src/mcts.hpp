#pragma once

#include "chessboard.hpp"
#include <vector>
#include <memory>
#include <utility>
#include <cstdint>
#include <mutex>
#include <array>
#include <random>
#include <atomic>
#include <string>

#include "evaluator.hpp"
#include "search_timing.hpp"

struct MCTSNode {
    int visit_count;
    int move_idx;
    bool is_terminal;
    float prior;
    float total_value;

    // Virtual loss, approche LC0 : nombre de descentes en cours passant par ce
    // noeud. N'entre QUE dans le denominateur du terme U de ucb_score, jamais
    // dans q_value(), sans quoi Q se diluerait vers zero et avantagerait les
    // noeuds perdants.
    uint32_t n_in_flight;

    MCTSNode* parent;
    std::vector<std::pair<int, std::unique_ptr<MCTSNode>>> children;

    MCTSNode(float prior, int move_idx = -1, MCTSNode* parent = nullptr);
    float q_value() const;
    float ucb_score(float exploration_factor, float parent_q, float fpu_reduction) const;
    MCTSNode* find_child(int idx) const;
    bool has_child(int idx) const;
    std::unique_ptr<MCTSNode> extract_child(int idx);
};


static constexpr int TT_MAX_MOVES = 128;
static constexpr int LEGACY_CACHE_HISTORY_DEPTH = -1;
static constexpr int DEFAULT_CACHE_HISTORY_DEPTH = 0;

enum class TTProbeStatus {
    HIT,
    MISS,
    RULE50_REJECT,
    CONTEXT_REJECT,
    HISTORY_REJECT,
};

struct TTEntry {
    uint64_t hash = 0;
    uint64_t evaluation_hash = 0;
    uint64_t current_context_hash = 0;
    uint64_t history_hash = 0;
    uint16_t half_move_clock = 0;
    float value = 0.0f;
    int policy_size = 0;
    std::array<std::pair<int, float>, TT_MAX_MOVES> legal_policy;
};

struct TTProbe {
    const TTEntry* entry = nullptr;
    TTProbeStatus status = TTProbeStatus::MISS;
};


struct MoveStats {
    int move_idx;
    int visits;
    float q_value;
    float prior;
};


// Instrumentation de la recherche. nn_calls compte les positions evaluees,
// nn_batches les appels physiques a l'evaluateur. Leur ratio donne le
// remplissage moyen des lots.
struct SearchCounters {
    uint64_t nn_calls = 0;
    uint64_t nn_batches = 0;
    uint64_t tt_hits = 0;
    uint64_t tt_misses = 0;
    uint64_t tt_position_matches = 0;
    uint64_t tt_rule50_rejects = 0;
    uint64_t tt_context_rejects = 0;
    uint64_t tt_history_rejects = 0;
    uint64_t terminal_hits = 0;
};


// Rapport de parcours de l'arbre, sur le modele de PerftReport.
struct TreeReport {
    uint64_t nodes = 0;
    uint64_t max_depth = 0;
    uint64_t violations = 0;
    uint64_t en_vol = 0;   // noeuds dont n_in_flight != 0 apres la recherche
    std::vector<std::string> messages;
};


class MCTS {
private:
    std::vector<TTEntry> transposition_table;
    static constexpr size_t DEFAULT_TT_SIZE = 2097143;
    size_t m_tt_size;
    int m_cache_history_depth;
    std::vector<float> m_eval_tensor;
    std::vector<float> m_eval_policy;
    std::unique_ptr<MCTSNode> m_analysis_root;
    mutable std::mutex m_mutex;
    Evaluator* m_evaluator;
    std::mt19937 m_noise_rng;
    bool m_timing_enabled = false;
    SearchTiming m_last_timing;

    // Atomiques parce que le self-play appelle advance_to_leaf depuis une region
    // OpenMP a 8 fils sur une instance de MCTS partagee
    // (selfplay_manager.cpp:376-387). L'ordre relache suffit : on ne lit ces
    // compteurs qu'apres la recherche, et un incremente relache coute quelques
    // dizaines de cycles contre 2,7 ms d'inference.
    std::atomic<uint64_t> m_nn_calls{ 0 };
    std::atomic<uint64_t> m_nn_batches{ 0 };
    std::atomic<uint64_t> m_tt_hits{ 0 };
    std::atomic<uint64_t> m_tt_misses{ 0 };
    std::atomic<uint64_t> m_tt_position_matches{ 0 };
    std::atomic<uint64_t> m_tt_rule50_rejects{ 0 };
    std::atomic<uint64_t> m_tt_context_rejects{ 0 };
    std::atomic<uint64_t> m_tt_history_rejects{ 0 };
    std::atomic<uint64_t> m_terminal_hits{ 0 };

public:
    MCTS(Evaluator* evaluator, size_t tt_size = DEFAULT_TT_SIZE,
         int cache_history_depth = DEFAULT_CACHE_HISTORY_DEPTH);

    void step_analysis(Chessboard& board, int num_simulations, float c_puct,
                       int batch_size = 0);
    void reset_analysis();
    void update_root(int move_idx);
    float get_root_q() const;
    std::vector<MoveStats> get_analysis_results() const;
    std::vector<float> mcts_search(Chessboard& board, int num_simulations, float c_puct,
                                   bool add_dirichlet, int batch_size = 0);
    float expand_node_single(MCTSNode* node, Chessboard& board,
                             SearchTiming* timing = nullptr);
    bool apply_move_by_index(Chessboard& board, int idx);
    void add_dirichlet_noise(MCTSNode* root, float epsilon);

    // Observabilite, definie dans mcts_observe.cpp.
    SearchCounters get_counters() const;
    void reset_counters();
    TreeReport inspect_tree() const;
    void set_timing_enabled(bool enabled);
    SearchTiming get_last_timing() const;

    // recherche mcts asynchrone
    MCTSNode* advance_to_leaf(MCTSNode* root, Chessboard& board, float c_puct, int& moves_played);
    void expand_and_backup(MCTSNode* leaf_node, Chessboard& board, const float* policy, float value);

private:
    void backup(MCTSNode* node, float value, SearchTiming* timing = nullptr);
    EvaluationCacheKey make_cache_key(const Chessboard& board) const;
    TTProbe probe_tt(const EvaluationCacheKey& key,
                     SearchTiming* timing = nullptr) const;
    void record_tt_probe(TTProbeStatus status);
    void store_tt(const EvaluationCacheKey& key,
                  const std::vector<int>& legal_indices,
                  const float* policy, float value,
                  SearchTiming* timing = nullptr);
    std::pair<MCTSNode*, int> select_leaf(MCTSNode* root, Chessboard& board,
                                         float c_puct,
                                         SearchTiming* timing = nullptr);
    void expand_and_backup_prepared(MCTSNode* leaf_node,
                                    const std::vector<int>& legal_indices,
                                    const EvaluationCacheKey& key,
                                    const float* policy, float value,
                                    SearchTiming* timing = nullptr);

    // Noyau unique de recherche, defini dans mcts_batch.cpp.
    // batch_size == 0 : boucle sequentielle historique, conservee telle quelle.
    // batch_size >= 1 : boucle batchee avec virtual loss.
    void run_search(MCTSNode* root, Chessboard& board, int simulations,
                    float c_puct, int batch_size,
                    SearchTiming* timing = nullptr);

};
