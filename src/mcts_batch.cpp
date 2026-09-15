// Noyau de recherche : boucle sequentielle historique et boucle batchee.
//
// Separe de mcts.cpp, qui depasse 520 lignes, sur le modele de
// mcts_observe.cpp. Voir docs/superpowers/specs/2026-09-11-mcts-batching-design.md

#include "mcts.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

// Le virtual loss de LC0 est un compteur, pas une valeur. backup() alterne le
// signe en remontant et ne peut donc pas etre reutilise ici.
void poser_virtual_loss(MCTSNode* node) {
    for (MCTSNode* current = node; current != nullptr; current = current->parent) {
        current->n_in_flight += 1;
    }
}

void annuler_virtual_loss(MCTSNode* node) {
    for (MCTSNode* current = node; current != nullptr; current = current->parent) {
        current->n_in_flight -= 1;
    }
}

// Capture prise pendant que le plateau est sur la feuille. Le backup et
// l'expansion n'ont ensuite besoin que de cette capture et des pointeurs parent.
struct FeuilleCollectee {
    MCTSNode* node = nullptr;
    std::vector<int> legal_moves;
    EvaluationCacheKey key;
};

}  // namespace

// Boucle sequentielle : une inference par simulation. Deplacee telle quelle
// depuis mcts_search, sans modification de l'ordre des operations, parce
// qu'elle sert de reference au test d'equivalence de la boucle batchee.

void MCTS::run_search(MCTSNode* root, Chessboard& board, int simulations,
                      float c_puct, int batch_size) {
    if (batch_size < 0) {
        throw std::invalid_argument("run_search : batch_size doit etre positif");
    }

    if (batch_size == 0) {
        for (int sim = 0; sim < simulations; sim++) {
            auto [node, moves_played] = select_leaf(root, board, c_puct);

            if (node->is_terminal) {
                m_terminal_hits.fetch_add(1, std::memory_order_relaxed);
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
                continue;
            }

            if (node->children.empty()) {
                float value = expand_node_single(node, board);
                backup(node, value);
            }

            for (int i = 0; i < moves_played; i++) {
                board.undoMove();
            }
        }
        return;
    }

    std::vector<FeuilleCollectee> batch;
    std::vector<float> tensors;
    std::vector<float> policies;
    std::vector<float> values;
    std::vector<float> current_tensor;
    batch.reserve(static_cast<size_t>(batch_size));
    tensors.reserve(static_cast<size_t>(batch_size) * 119 * 64);
    current_tensor.reserve(119 * 64);

    int completed = 0;
    while (completed < simulations) {
        batch.clear();
        tensors.clear();

        while (static_cast<int>(batch.size()) < batch_size &&
               completed + static_cast<int>(batch.size()) < simulations) {
            auto [node, moves_played] = select_leaf(root, board, c_puct);

            if (node->is_terminal) {
                m_terminal_hits.fetch_add(1, std::memory_order_relaxed);
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
                completed++;
                continue;
            }

            // select_leaf s'arrete ici lorsqu'une feuille deja collectee est
            // retrouvee. On evalue alors le lot partiel, comme LC0 apres
            // TryStartScoreUpdate, plutot que de forcer une nouvelle descente.
            if (node->children.empty() && node->n_in_flight > 0) {
                for (int i = 0; i < moves_played; i++) board.undoMove();
                break;
            }

            // Une entree de table vient d'etre materialisee par select_leaf,
            // qui a continue la descente. expand_node_single rend alors la
            // valeur en cache sans inferer.
            if (!node->children.empty()) {
                float value = expand_node_single(node, board);
                backup(node, value);
                for (int i = 0; i < moves_played; i++) board.undoMove();
                completed++;
                continue;
            }

            const EvaluationCacheKey key = make_cache_key(board);
            const TTProbe probe = probe_tt(key);
            if (probe.status == TTProbeStatus::HIT) {
                throw std::logic_error(
                    "run_search : select_leaf a ignore un hit de TT");
            }
            record_tt_probe(probe.status);

            FeuilleCollectee leaf;
            leaf.node = node;
            leaf.legal_moves = board.getLegalMoveIndices();
            leaf.key = key;

            if (leaf.legal_moves.empty()) {
                node->is_terminal = true;
                backup(node, board.isInCheck() ? -1.0f : 0.0f);
                for (int i = 0; i < moves_played; i++) board.undoMove();
                completed++;
                continue;
            }

            board.getAlphaZeroTensor(current_tensor);
            tensors.insert(tensors.end(), current_tensor.begin(), current_tensor.end());
            poser_virtual_loss(node);
            batch.push_back(std::move(leaf));

            for (int i = 0; i < moves_played; i++) board.undoMove();
        }

        if (batch.empty()) continue;

        const int batch_count = static_cast<int>(batch.size());
        m_nn_calls.fetch_add(batch_count, std::memory_order_relaxed);
        m_nn_batches.fetch_add(1, std::memory_order_relaxed);
        m_evaluator->evaluate_batch(tensors, policies, values, batch_count);

        for (int i = 0; i < batch_count; ++i) {
            annuler_virtual_loss(batch[i].node);
            expand_and_backup_prepared(batch[i].node, batch[i].legal_moves,
                                       batch[i].key,
                                       policies.data() + static_cast<size_t>(i) * 4672,
                                       values[i]);
            completed++;
        }
    }
}
