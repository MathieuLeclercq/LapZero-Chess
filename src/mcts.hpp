#pragma once

#include "chessboard.hpp"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>
#include <utility>
#include <cstdint>
#include <mutex>
#include <array>
#include <random>
#include <atomic>
#include <string>

#include "onnx_evaluator.hpp"

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

struct TTEntry {
    uint64_t hash = 0;
    float value = 0.0f;
    int policy_size = 0;
    std::array<std::pair<int, float>, TT_MAX_MOVES> legal_policy;
};


struct MoveStats {
    int move_idx;
    int visits;
    float q_value;
    float prior;
};


// Instrumentation de la recherche. Sans elle, le nombre d'inferences et le taux
// de succes de la table ne sont pas observables de l'exterieur.
struct SearchCounters {
    uint64_t nn_calls = 0;
    uint64_t tt_hits = 0;
    uint64_t tt_misses = 0;
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
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<TTEntry> transposition_table;
    static constexpr size_t DEFAULT_TT_SIZE = 2097143;
    size_t m_tt_size;
    std::vector<float> m_eval_tensor;
    std::vector<float> m_eval_policy;
    std::unique_ptr<MCTSNode> m_analysis_root;
    std::mutex m_mutex;
    ONNXEvaluator* m_evaluator;
    std::mt19937 m_noise_rng;

    // Atomiques parce que le self-play appelle advance_to_leaf depuis une region
    // OpenMP a 8 fils sur une instance de MCTS partagee
    // (selfplay_manager.cpp:376-387). L'ordre relache suffit : on ne lit ces
    // compteurs qu'apres la recherche, et un incremente relache coute quelques
    // dizaines de cycles contre 2,7 ms d'inference.
    std::atomic<uint64_t> m_nn_calls{ 0 };
    std::atomic<uint64_t> m_tt_hits{ 0 };
    std::atomic<uint64_t> m_tt_misses{ 0 };
    std::atomic<uint64_t> m_terminal_hits{ 0 };

public:
    MCTS(ONNXEvaluator* evaluator, size_t tt_size = DEFAULT_TT_SIZE);

    void step_analysis(Chessboard& board, int num_simulations, float c_puct);
    void reset_analysis();
    void update_root(int move_idx);
    float get_root_q() const;
    std::vector<MoveStats> get_analysis_results() const;
    std::vector<float> mcts_search(Chessboard& board, int num_simulations, float c_puct, bool add_dirichlet);
    float expand_node_single(MCTSNode* node, Chessboard& board);
    bool apply_move_by_index(Chessboard& board, int idx);
    void add_dirichlet_noise(MCTSNode* root, float epsilon);

    // Observabilite, definie dans mcts_observe.cpp.
    SearchCounters get_counters() const;
    void reset_counters();
    TreeReport inspect_tree() const;

    // recherche mcts asynchrone
    MCTSNode* advance_to_leaf(MCTSNode* root, Chessboard& board, float c_puct, int& moves_played);
    void expand_and_backup(MCTSNode* leaf_node, Chessboard& board, const float* policy, float value);

private:
    void backup(MCTSNode* node, float value);
    std::pair<MCTSNode*, int> select_leaf(MCTSNode* root, Chessboard& board, float c_puct);

};