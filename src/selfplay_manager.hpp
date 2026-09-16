#pragma once
#include <vector>
#include <memory>
#include <random>
#include <string>
#include "chessboard.hpp"
#include "mcts.hpp"
#include "onnx_evaluator.hpp"


struct GameResult {
    std::vector<float> flat_states; // Taille : [NbCoups * 119 * 64]
    std::vector<float> flat_policies; // Taille : [NbCoups * 4672]
    float final_outcome;
    int move_count; // Pour savoir comment découper le vecteur plat
    int total_real_moves;
    int end_reason;
};

struct ThreadLocalBuffer {
    std::vector<MCTSNode*> leaves;
    std::vector<int> game_indices;
    std::vector<int> moves_played;
    std::vector<PathReservation> reservations;
    std::vector<float> tensors;
    std::vector<float> tensor_scratch;

    void clear() {
        leaves.clear();
        game_indices.clear();
        moves_played.clear();
        reservations.clear();
    }
};

class SelfPlayManager {
private:

    // constantes de config
    static constexpr float NORMAL_EPSILON = 0.12f;
    static constexpr float TACTICAL_EPSILON = 0.30f;
    static constexpr int TACTICAL_FIRST_MOVE_SIMS = 4000;
    static constexpr int MAX_PLIES_BEFORE_FORCED_DRAW = 300;

    int m_num_concurrent_games;
    int m_slow_sims;
    int m_fast_sims;
    float m_slow_ratio;
    std::vector<int> m_sims_target;  // sims à faire pour le coup en cours
    std::vector<char> m_is_slow_move;

    ONNXEvaluator* m_evaluator;

    // L'état complet des parties en cours
    std::vector<Chessboard> m_boards;
    std::vector<std::unique_ptr<MCTSNode>> m_roots;
    std::unique_ptr<MCTS> m_shared_mcts;

    // Suivi de l'avancement de chaque arbre (combien de simulations terminées pour ce coup)
    std::vector<int> m_sims_completed;

    // Données accumulées pour l'entraînement final
    std::vector<GameResult> m_finished_games;
    std::vector<std::vector<std::vector<float>>> m_game_states;
    std::vector<std::vector<std::vector<float>>> m_game_policies;

    // Buffers pour le GPU
    std::vector<float> m_batch_input;
    std::vector<float> m_batch_policies;
    std::vector<float> m_batch_values;

    // Suivi de l'état de la boucle asynchrone
    std::vector<MCTSNode*> m_waiting_leaves;
    std::vector<int> m_waiting_game_indices;
    std::vector<int> m_waiting_moves_played;
    std::vector<PathReservation> m_waiting_reservations;
    std::vector<char> m_is_waiting;

    std::mt19937 m_rng{ std::random_device{}() };

    // pour entrainement tactique sur puzzles lichess
    // Renfort tactique du PREMIER coup d'une partie amorcee par un puzzle :
    // TACTICAL_FIRST_MOVE_SIMS simulations et TACTICAL_EPSILON de bruit. Le
    // drapeau est consomme dans play_best_move une fois ce coup joue, apres
    // quoi la partie redevient une partie normale, budget de recherche comme
    // bruit de Dirichlet.
    std::vector<char> m_tactical_boost;
    struct TacticalPuzzle {
        std::string start_fen;
        std::vector<std::string> moves; // UCI, jusqu'à la position du puzzle incluse
    };
    std::vector<TacticalPuzzle> m_tactical_puzzles;


public:
    SelfPlayManager(ONNXEvaluator* evaluator, int num_concurrent_games, 
                    int slow_sims, int fast_sims, float slow_ratio,
                    size_t tt_size = 2097143);
    std::vector<GameResult> generate_games(int total_games_to_play);

private:
    void reset_game(int game_idx);
    void play_best_move(int game_idx);
    void roll_next_move(int game_idx);
    void execute_gpu_batch();
    void load_tactical_puzzles(const std::string& filepath);
};
