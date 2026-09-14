#include "mcts.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <cstdint>


// ============================================================
//                     MCTSNode
// ============================================================

MCTSNode::MCTSNode(float prior, int move_idx, MCTSNode* parent)
    : prior(prior), move_idx(move_idx), parent(parent),
    visit_count(0), total_value(0.0f), is_terminal(false), n_in_flight(0) {
}

float MCTSNode::ucb_score(float exploration_factor, float parent_q, float fpu_reduction) const {
    // Implémentation du FPU de LeelaChess0
    // Si noeud pas visité, on ne met pas sa Q value à 0,
    // mais on utilise celle du parent.
    float u = exploration_factor * prior / (1.0f + visit_count + n_in_flight);
    float exploitation = (visit_count == 0) ? (parent_q - fpu_reduction) : -q_value();
    return exploitation + u;
}

float MCTSNode::q_value() const {
    if (visit_count == 0) return 0.0f;
    return total_value / visit_count;
}

MCTSNode* MCTSNode::find_child(int idx) const {
    for (const auto& pair : children) {
        if (pair.first == idx) return pair.second.get();
    }
    return nullptr;
}

bool MCTSNode::has_child(int idx) const {
    return find_child(idx) != nullptr;
}

std::unique_ptr<MCTSNode> MCTSNode::extract_child(int idx) {
    for (auto it = children.begin(); it != children.end(); ++it) {
        if (it->first == idx) {
            auto result = std::move(it->second);
            children.erase(it);
            return result;
        }
    }
    return nullptr;
}


// ============================================================
//                     MCTS
// ============================================================
MCTS::MCTS(ONNXEvaluator* evaluator, size_t tt_size) : 
        m_evaluator(evaluator), 
        m_tt_size(tt_size),
        m_noise_rng(std::random_device{}()) {
    transposition_table.resize(m_tt_size);
    m_eval_tensor.reserve(119 * 64);
    m_eval_policy.reserve(4672);
}

void MCTS::backup(MCTSNode* node, float value) {
    while (node != nullptr) {
        node->visit_count += 1;
        node->total_value += value;
        value = -value;
        node = node->parent;
    }
}

std::pair<MCTSNode*, int> MCTS::select_leaf(MCTSNode* root, Chessboard& board, float c_puct) {
    MCTSNode* node = root;
    int moves_played = 0;

    while (!node->is_terminal) {

        // --- 1. EXPANSION PARESSEUSE ---
        if (node->children.empty()) {
            // Collision : cette feuille est deja collectee par une descente
            // precedente du meme lot. Sans cette garde, l'expansion paresseuse
            // ci-dessous lui creerait des enfants, et le backup du lot lui en
            // creerait un second jeu. L'arbre serait corrompu en silence.
            if (node->n_in_flight > 0) break;

            uint64_t hash = board.getZobristHash();
            size_t tt_idx = hash % m_tt_size;
            const TTEntry& entry = transposition_table[tt_idx];

            if (entry.hash == hash && entry.policy_size > 0) {
                m_tt_hits.fetch_add(1, std::memory_order_relaxed);
                int size = entry.policy_size;
                float sum_legal = 0.0f;
                for (int k = 0; k < size; ++k) sum_legal += entry.legal_policy[k].second;

                node->children.reserve(size);
                for (int k = 0; k < size; ++k) {
                    int idx = entry.legal_policy[k].first;
                    float prob = (sum_legal > 0.0f)
                        ? (entry.legal_policy[k].second / sum_legal)
                        : (1.0f / (float)size);
                    node->children.emplace_back(idx, std::make_unique<MCTSNode>(prob, idx, node));
                }
            }
            else {
                break; // Vraie feuille, besoin du GPU
            }
        }

        // --- 2. FPU ---
        float visited_policy_sum = 0.0f;
        for (const auto& pair : node->children) {
            if (pair.second->visit_count > 0) {
                visited_policy_sum += pair.second->prior;
            }
        }
        float fpu_reduction = 0.30f * std::sqrt(visited_policy_sum);

        // --- 3. SÉLECTION UCB ---
        float max_ucb = -1e9f;
        int best_move_idx = -1;
        float parent_q = node->q_value();
        float exploration_factor = c_puct * std::sqrt(static_cast<float>(node->visit_count));

        MCTSNode* best_child = nullptr;
        for (const auto& pair : node->children) {
            float score = pair.second->ucb_score(exploration_factor, parent_q, fpu_reduction);
            if (score > max_ucb) {
                max_ucb = score;
                best_move_idx = pair.first;
                best_child = pair.second.get(); // Descente
            }
        }

        if (best_move_idx == -1) break;

        if (!apply_move_by_index(board, best_move_idx)) {
            throw std::runtime_error("Problème lors de l'application du coup dans select_leaf");
        }

        node = best_child;
        moves_played++;

        if (board.checkThreefoldRepetition() ||
            board.getHalfMoveClock() >= 100 ||
            board.checkInsufficientMaterial()) {
            node->is_terminal = true;
            break;
        }
    }

    return { node, moves_played };
}

float MCTS::expand_node_single(MCTSNode* node, Chessboard& board) {

    if (board.checkThreefoldRepetition() || board.getHalfMoveClock() >= 100 || board.checkInsufficientMaterial()) {
        node->is_terminal = true;
        return 0.0f;
    }

    uint64_t hash = board.getZobristHash();
    size_t tt_idx = hash % m_tt_size;

    // Cache hit
    if (transposition_table[tt_idx].hash == hash && transposition_table[tt_idx].policy_size > 0) {
        m_tt_hits.fetch_add(1, std::memory_order_relaxed);
        return transposition_table[tt_idx].value;
    }

    // Cache miss
    m_tt_misses.fetch_add(1, std::memory_order_relaxed);
    std::vector<int> legal_indices = board.getLegalMoveIndices();
    if (legal_indices.empty()) {
        node->is_terminal = true;
        return board.isInCheck() ? -1.0f : 0.0f;
    }

    board.getAlphaZeroTensor(m_eval_tensor);
    float value;
    m_nn_calls.fetch_add(1, std::memory_order_relaxed);
    m_evaluator->evaluate(m_eval_tensor, m_eval_policy, value);

    // Stockage dans la TT (taille fixe, pas d'allocation)
    TTEntry& tt = transposition_table[tt_idx];
    tt.hash = hash;
    tt.value = value;
    tt.policy_size = std::min((int)legal_indices.size(), TT_MAX_MOVES);

    float sum_legal = 0.0f;
    for (int k = 0; k < tt.policy_size; ++k) {
        int idx = legal_indices[k];
        float prob = m_eval_policy[idx];
        tt.legal_policy[k] = { idx, prob };
        sum_legal += prob;
    }

    // Création des enfants
    node->children.reserve(tt.policy_size);
    if (sum_legal > 0.0f) {
        for (int k = 0; k < tt.policy_size; ++k) {
            node->children.emplace_back(
                tt.legal_policy[k].first,
                std::make_unique<MCTSNode>(tt.legal_policy[k].second / sum_legal, tt.legal_policy[k].first, node));
        }
    }
    else {
        float uniform_prob = 1.0f / tt.policy_size;
        for (int k = 0; k < tt.policy_size; ++k) {
            node->children.emplace_back(
                tt.legal_policy[k].first,
                std::make_unique<MCTSNode>(uniform_prob, tt.legal_policy[k].first, node));
        }
    }

    return value;
}

void MCTS::add_dirichlet_noise(MCTSNode* root, float epsilon) {
    if (root->children.empty()) return;
    std::gamma_distribution<float> gamma(0.3f, 1.0f);

    float sum_noise = 0.0f;
    std::vector<float> noise(root->children.size());
    for (size_t i = 0; i < root->children.size(); i++) {
        noise[i] = gamma(m_noise_rng);
        sum_noise += noise[i];
    }

    int i = 0;
    for (auto& pair : root->children) {
        float dirichlet = noise[i++] / sum_noise;
        pair.second->prior = (1.0f - epsilon) * pair.second->prior + epsilon * dirichlet;
    }
}

std::vector<float> MCTS::mcts_search(Chessboard& board, int num_simulations, float c_puct,
                                     bool add_dirichlet, int batch_size) {

    //if (board.getZobristHash() != board.computeZobristFromScratch()) {
    //    throw std::runtime_error("Erreur fatale : Desynchronisation du Zobrist Hash detectee !");
    //}

    std::unique_ptr<MCTSNode> root = std::make_unique<MCTSNode>(0.0f);
    expand_node_single(root.get(), board);

    if (add_dirichlet) {
        add_dirichlet_noise(root.get(), 0.12f);
    }

    run_search(root.get(), board, num_simulations, c_puct, batch_size);

    std::vector<float> pi(4672, 0.0f);
    float sum_visits = 0.0f;
    for (const auto& pair : root->children) {
        pi[pair.first] = static_cast<float>(pair.second->visit_count);
        sum_visits += pi[pair.first];
    }

    if (sum_visits > 0.0f) {
        for (float& prob : pi) {
            prob /= sum_visits;
        }
    }

    return pi;
}

bool MCTS::apply_move_by_index(Chessboard& board, int index) {
    bool is_black = (board.getTurn() == BLACK);
    int plane = index / 64;
    int remainder = index % 64;
    int orig_r = remainder / 8;
    int orig_f = remainder % 8;

    int df = 0, dr = 0;
    PieceType promotion = NONE;

    if (plane < 56) {
        int dir_idx = plane / 7;
        int dist = (plane % 7) + 1;
        int dirs[8][2] = { {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1} };
        df = dirs[dir_idx][0] * dist;
        dr = dirs[dir_idx][1] * dist;
    }
    else if (plane < 64) {
        int knight_idx = plane - 56;
        int knight_moves[8][2] = { {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2} };
        df = knight_moves[knight_idx][0];
        dr = knight_moves[knight_idx][1];
    }
    else {
        int sub_idx = plane - 64;
        int dir_idx = sub_idx / 3;
        int p_idx = sub_idx % 3;
        df = dir_idx - 1;
        dr = 1;

        if (p_idx == 0) promotion = KNIGHT;
        else if (p_idx == 1) promotion = BISHOP;
        else promotion = ROOK;
    }

    int dest_f = orig_f + df;
    int dest_r = orig_r + dr;

    if (is_black) {
        orig_r = 7 - orig_r;
        dest_r = 7 - dest_r;
    }

    if (board.getSquare(orig_f, orig_r).getPiece().getType() == PAWN) {
        if ((!is_black && dest_r == 7) || (is_black && dest_r == 0)) {
            if (promotion == NONE) {
                promotion = QUEEN;
            }
        }
    }

    return board.movePiece(orig_f, orig_r, dest_f, dest_r, promotion, false);
}


// ============================================================
//                     ANALYSE CONTINUE
// ============================================================

void MCTS::reset_analysis() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_analysis_root.reset();
}

void MCTS::update_root(int move_idx) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_analysis_root && m_analysis_root->has_child(move_idx)) {
        m_analysis_root = m_analysis_root->extract_child(move_idx);
        m_analysis_root->parent = nullptr;
    }
    else {
        m_analysis_root.reset();
    }
}

float MCTS::get_root_q() const {
    if (!m_analysis_root) return 0.0f;
    return m_analysis_root->q_value();
}

void MCTS::step_analysis(Chessboard& board, int num_simulations, float c_puct,
                         int batch_size) {
    // Le verrou couvre desormais toute la duree de l'appel, au lieu d'etre pris
    // et relache a chaque simulation. C'est necessaire pour la boucle batchee,
    // qui detiendra des MCTSNode* bruts pendant l'inference : relacher le verrou
    // en cours de lot laisserait update_root detruire l'arbre sous ces
    // pointeurs. Sans consequence pratique depuis que parse_position appelle
    // stop_search() avant toute modification de l'arbre (uci.py).
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_analysis_root) {
        m_analysis_root = std::make_unique<MCTSNode>(0.0f);
        expand_node_single(m_analysis_root.get(), board);
    }

    run_search(m_analysis_root.get(), board, num_simulations, c_puct, batch_size);
}

std::vector<MoveStats> MCTS::get_analysis_results() const {
    std::lock_guard<std::mutex> lock(const_cast<MCTS&>(*this).m_mutex);

    std::vector<MoveStats> results;
    if (!m_analysis_root) return results;

    for (const auto& pair : m_analysis_root->children) {
        MCTSNode* child = pair.second.get();
        if (child->visit_count > 0) {
            results.push_back({
                pair.first,
                child->visit_count,
                -child->q_value(),
                child->prior
                });
        }
    }

    std::sort(results.begin(), results.end(), [](const MoveStats& a, const MoveStats& b) {
        return a.visits > b.visits;
        });

    return results;
}

MCTSNode* MCTS::advance_to_leaf(MCTSNode* root, Chessboard& board, float c_puct, int& moves_played) {
    auto [node, moves] = select_leaf(root, board, c_puct);
    moves_played = moves;

    if (node->is_terminal) {
        float value = 0.0f;
        if (board.checkThreefoldRepetition() ||
            board.getHalfMoveClock() >= 100 ||
            board.checkInsufficientMaterial()) {
            value = 0.0f;
        }
        else {
            value = board.isInCheck() ? -1.0f : 0.0f;
        }
        backup(node, value);
        for (int i = 0; i < moves_played; i++) board.undoMove();
        return nullptr;
    }

    // Vérification TT avant GPU
    uint64_t hash = board.getZobristHash();
    size_t tt_idx = hash % m_tt_size;

    if (transposition_table[tt_idx].hash == hash && transposition_table[tt_idx].policy_size > 0) {
        m_tt_hits.fetch_add(1, std::memory_order_relaxed);
        backup(node, transposition_table[tt_idx].value);
        for (int i = 0; i < moves_played; i++) board.undoMove();
        return nullptr;
    }

    m_tt_misses.fetch_add(1, std::memory_order_relaxed);
    return node;
}

void MCTS::expand_and_backup(MCTSNode* leaf_node, Chessboard& board, const float* policy, float value) {

    std::vector<int> legal_indices = board.getLegalMoveIndices();
    if (legal_indices.empty()) {
        leaf_node->is_terminal = true;
        backup(leaf_node, board.isInCheck() ? -1.0f : 0.0f);
        return;
    }

    expand_and_backup_prepared(leaf_node, legal_indices, board.getZobristHash(), policy, value);
}

void MCTS::expand_and_backup_prepared(MCTSNode* leaf_node,
                                      const std::vector<int>& legal_indices,
                                      uint64_t hash, const float* policy, float value) {
    // Les feuilles terminales sont traitees pendant la descente, jamais ici :
    // legal_indices est donc non vide et le plateau n'est pas necessaire.
    size_t tt_idx = hash % m_tt_size;
    TTEntry& tt = transposition_table[tt_idx];
    tt.hash = hash;
    tt.value = value;
    tt.policy_size = std::min((int)legal_indices.size(), TT_MAX_MOVES);

    float sum_legal = 0.0f;
    for (int k = 0; k < tt.policy_size; ++k) {
        int idx = legal_indices[k];
        float prob = policy[idx];
        tt.legal_policy[k] = { idx, prob };
        sum_legal += prob;
    }

    // Création des enfants
    leaf_node->children.reserve(tt.policy_size);
    if (sum_legal > 0.0f) {
        for (int k = 0; k < tt.policy_size; ++k) {
            leaf_node->children.emplace_back(
                tt.legal_policy[k].first,
                std::make_unique<MCTSNode>(tt.legal_policy[k].second / sum_legal, tt.legal_policy[k].first, leaf_node));
        }
    }
    else {
        float uniform_prob = 1.0f / tt.policy_size;
        for (int k = 0; k < tt.policy_size; ++k) {
            leaf_node->children.emplace_back(
                tt.legal_policy[k].first,
                std::make_unique<MCTSNode>(uniform_prob, tt.legal_policy[k].first, leaf_node));
        }
    }

    backup(leaf_node, value);
}
