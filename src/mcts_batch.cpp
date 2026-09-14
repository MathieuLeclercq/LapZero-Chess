// Noyau de recherche : boucle sequentielle historique et boucle batchee.
//
// Separe de mcts.cpp, qui depasse 520 lignes, sur le modele de
// mcts_observe.cpp. Voir docs/superpowers/specs/2026-09-11-mcts-batching-design.md

#include "mcts.hpp"

#include <stdexcept>

// Boucle sequentielle : une inference par simulation. Deplacee telle quelle
// depuis mcts_search, sans modification de l'ordre des operations, parce
// qu'elle sert de reference au test d'equivalence de la boucle batchee.

void MCTS::run_search(MCTSNode* root, Chessboard& board, int simulations,
                      float c_puct, int batch_size) {
    if (batch_size != 0) {
        throw std::runtime_error(
            "run_search : seul batch_size = 0 est implemente pour l'instant");
    }

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
}
