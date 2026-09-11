// Observabilite de la recherche : compteurs et parcours d'arbre.
//
// Separe de mcts.cpp, qui fait deja 512 lignes, et sur le modele de perft.cpp :
// un parcours de diagnostic a sa propre responsabilite et son propre fichier.

#include "mcts.hpp"

SearchCounters MCTS::get_counters() const {
    SearchCounters c;
    c.nn_calls = m_nn_calls.load(std::memory_order_relaxed);
    c.tt_hits = m_tt_hits.load(std::memory_order_relaxed);
    c.tt_misses = m_tt_misses.load(std::memory_order_relaxed);
    c.terminal_hits = m_terminal_hits.load(std::memory_order_relaxed);
    return c;
}

void MCTS::reset_counters() {
    m_nn_calls.store(0, std::memory_order_relaxed);
    m_tt_hits.store(0, std::memory_order_relaxed);
    m_tt_misses.store(0, std::memory_order_relaxed);
    m_terminal_hits.store(0, std::memory_order_relaxed);
}
