#include "controlled_evaluator.hpp"
#include "discriminating_evaluator.hpp"
#include "mcts.hpp"
#include "mcts_test_access.hpp"
#include "selfplay_manager.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class SelfPlayTestAccess {
public:
    static MCTSNode* root(SelfPlayManager& manager, int game_idx) {
        return manager.m_roots[static_cast<std::size_t>(game_idx)].get();
    }

    static float pending_epsilon(const SelfPlayManager& manager,
                                 int game_idx) {
        return manager.m_pending_epsilon[static_cast<std::size_t>(game_idx)];
    }

    static void demarrer_slot(SelfPlayManager& manager, int game_idx) {
        manager.demarrer_slot(game_idx);
    }

    // Impose la fin d'une partie apres un nombre de plies connu. Le levier est
    // a zero par defaut et ne sert qu'aux tests de comptabilite.
    static void forcer_fin_apres(SelfPlayManager& manager, int game_idx,
                                 int plies) {
        manager.m_forced_end_plies[static_cast<std::size_t>(game_idx)] = plies;
    }

    static void seed_rng(SelfPlayManager& manager, std::uint32_t graine) {
        manager.m_rng.seed(graine);
    }

    static void reset_game(SelfPlayManager& manager, int game_idx) {
        manager.reset_game(game_idx);
    }

    static void play_best_move(SelfPlayManager& manager, int game_idx) {
        manager.play_best_move(game_idx);
    }

    static int sims_target(const SelfPlayManager& manager, int game_idx) {
        return manager.m_sims_target[static_cast<std::size_t>(game_idx)];
    }

    static bool is_slow_move(const SelfPlayManager& manager, int game_idx) {
        return manager.m_is_slow_move[static_cast<std::size_t>(game_idx)] != 0;
    }

    static bool tactical_boost(const SelfPlayManager& manager, int game_idx) {
        return manager.m_tactical_boost[static_cast<std::size_t>(game_idx)]
            != 0;
    }

    static MCTS& mcts(SelfPlayManager& manager) {
        return *manager.m_shared_mcts;
    }

    static std::string fen(SelfPlayManager& manager, int game_idx) {
        return manager.m_boards[static_cast<std::size_t>(game_idx)].toFEN();
    }

    // Prepare une position distincte par partie, puis une racine deja
    // developpee, sans passer par le scheduler.
    static void preparer_position(SelfPlayManager& manager, int game_idx,
                                  const std::string& uci) {
        const std::size_t index = static_cast<std::size_t>(game_idx);
        manager.m_boards[index].setStartupPieces();
        if (!uci.empty()) {
            if (!manager.m_boards[index].movePieceUCI(uci)) {
                throw std::runtime_error("preparer_position : coup invalide");
            }
        }
        manager.m_roots[index] = std::make_unique<MCTSNode>(0.0f);
        manager.m_shared_mcts->expand_node_single(
            manager.m_roots[index].get(), manager.m_boards[index]);
    }

    struct ResultatLot {
        std::vector<MCTSNode*> feuilles;
        std::vector<std::vector<float>> tenseurs;
        std::vector<std::vector<int>> coups_legaux;
    };

    // Monte un lot a partir de feuilles de parties distinctes, comme le font
    // les phases de generation, puis execute la consommation de production.
    static ResultatLot executer_lot_par_partie(
            SelfPlayManager& manager, const std::vector<int>& parties) {
        ResultatLot resultat;
        resultat.feuilles.assign(parties.size(), nullptr);
        resultat.tenseurs.resize(parties.size());
        resultat.coups_legaux.resize(parties.size());

        for (std::size_t j = 0; j < parties.size(); ++j) {
            const int i = parties[j];
            int moves_played = 0;
            PathReservation reservation;
            MCTSNode* leaf = manager.m_shared_mcts->advance_to_leaf(
                manager.m_roots[static_cast<std::size_t>(i)].get(),
                manager.m_boards[static_cast<std::size_t>(i)], 1.4f,
                moves_played, reservation);
            if (leaf == nullptr) continue;

            resultat.feuilles[j] = leaf;
            resultat.coups_legaux[j] =
                manager.m_boards[static_cast<std::size_t>(i)]
                    .getLegalMoveIndices();
            manager.m_boards[static_cast<std::size_t>(i)]
                .getAlphaZeroTensor(resultat.tenseurs[j]);

            manager.m_waiting_leaves.push_back(leaf);
            manager.m_waiting_game_indices.push_back(i);
            manager.m_waiting_moves_played.push_back(moves_played);
            manager.m_waiting_reservations.push_back(std::move(reservation));
            manager.m_is_waiting[i] = true;

            const std::size_t offset = manager.m_waiting_leaves.size() - 1;
            std::copy(resultat.tenseurs[j].begin(),
                      resultat.tenseurs[j].end(),
                      manager.m_batch_input.begin() + offset * 119 * 64);
        }

        manager.execute_gpu_batch();
        return resultat;
    }
};

namespace {

constexpr int GAME_COUNT = 8;
constexpr int INPUT_SIZE = 119 * 64;
constexpr int POLICY_SIZE = 4672;

void test_openmp_selfplay_core_with_shared_cache() {
    ControlledEvaluator evaluator;
    MCTS mcts(&evaluator, 3, 0);
    std::vector<Chessboard> boards(GAME_COUNT);
    std::vector<std::unique_ptr<MCTSNode>> roots;
    roots.reserve(GAME_COUNT);
    for (int game = 0; game < GAME_COUNT; ++game) {
        boards[game].setStartupPieces();
        roots.push_back(std::make_unique<MCTSNode>(0.0f));
        mcts.expand_node_single(roots.back().get(), boards[game]);
    }

    std::vector<MCTSNode*> leaves(GAME_COUNT);
    std::vector<int> moves_played(GAME_COUNT);
    std::vector<PathReservation> reservations(GAME_COUNT);
    std::vector<std::vector<float>> tensors(
        GAME_COUNT, std::vector<float>(INPUT_SIZE));
    std::vector<float> input;
    std::vector<float> policies;
    std::vector<float> values;

    for (int round = 0; round < 64; ++round) {
        std::fill(leaves.begin(), leaves.end(), nullptr);
        std::fill(moves_played.begin(), moves_played.end(), 0);

#pragma omp parallel for num_threads(GAME_COUNT) schedule(static)
        for (int game = 0; game < GAME_COUNT; ++game) {
            leaves[game] = mcts.advance_to_leaf(
                roots[game].get(), boards[game], 1.4f,
                moves_played[game], reservations[game]);
            if (leaves[game] != nullptr) {
                boards[game].getAlphaZeroTensor(tensors[game]);
            }
        }

        input.clear();
        std::vector<int> active;
        for (int game = 0; game < GAME_COUNT; ++game) {
            if (leaves[game] == nullptr) continue;
            active.push_back(game);
            input.insert(input.end(), tensors[game].begin(),
                         tensors[game].end());
        }
        if (!active.empty()) {
            evaluator.evaluate_batch(
                input, policies, values, static_cast<int>(active.size()));
            for (std::size_t index = 0; index < active.size(); ++index) {
                const int game = active[index];
                mcts.expand_and_backup(
                    leaves[game], boards[game],
                    policies.data() + index * POLICY_SIZE,
                    values[index], reservations[game]);
                for (int move = 0; move < moves_played[game]; ++move) {
                    boards[game].undoMove();
                }
            }
        }
    }

    for (const auto& root : roots) {
        const TreeReport report = mcts.inspect_tree(root.get());
        require_test(report.root_visits == 64,
                     "shared self-play root lost visits");
        require_test(report.en_vol == 0 && report.pending == 0,
                     "shared self-play root retained a reservation");
        require_test(report.violations == 0,
                     "shared self-play root is invalid");
    }
}

void test_selfplay_manager_with_controlled_evaluator() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);
    const std::vector<GameResult> games = manager.generate_games(2);

    require_test(games.size() == 2, "self-play returned the wrong game count");
    for (const GameResult& game : games) {
        require_test(game.move_count >= 0, "negative training move count");
        require_test(game.total_real_moves > 0
                         && game.total_real_moves <= 300,
                     "self-play game length is outside its bounds");
        require_test(game.flat_states.size()
                         == static_cast<std::size_t>(game.move_count)
                             * INPUT_SIZE,
                     "self-play state dimensions are inconsistent");
        require_test(game.flat_policies.size()
                         == static_cast<std::size_t>(game.move_count)
                             * POLICY_SIZE,
                     "self-play policy dimensions are inconsistent");
        require_test(std::all_of(
            game.flat_states.begin(), game.flat_states.end(),
            [](float value) { return std::isfinite(value); }),
            "self-play emitted a non-finite state");
        require_test(std::all_of(
            game.flat_policies.begin(), game.flat_policies.end(),
            [](float value) { return std::isfinite(value); }),
            "self-play emitted a non-finite policy");
        require_test(std::isfinite(game.final_outcome),
                     "self-play emitted a non-finite outcome");
        require_test(game.end_reason >= 0 && game.end_reason <= 5,
                     "end reason is out of range");
        if (game.end_reason == 0) {
            require_test(game.final_outcome == 1.0f
                             || game.final_outcome == -1.0f,
                         "checkmate without a decisive outcome");
        }
        else {
            require_test(game.final_outcome == 0.0f,
                         "non-checkmate game with a decisive outcome");
        }
    }
}

void test_selfplay_rejects_invalid_batch_and_cleans_up() {
    ControlledEvaluator evaluator;
    // Deux expansions de racine a la construction, puis le premier lot GPU.
    evaluator.corrupt_call = 3;
    evaluator.values_corruption =
        ControlledEvaluator::Corruption::NonFinite;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);

    bool failed = false;
    try {
        (void)manager.generate_games(2);
    }
    catch (const std::runtime_error&) {
        failed = true;
    }
    require_test(failed, "self-play accepted an invalid batch output");

    // Les reservations et les plateaux en attente doivent avoir ete nettoyes :
    // le gestionnaire reste utilisable apres le rejet.
    evaluator.corrupt_call = 0;
    const std::vector<GameResult> games = manager.generate_games(2);
    require_test(games.size() == 2, "self-play did not recover");
}

void test_selfplay_roots_are_expanded_and_noised_exactly_once() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);
    // Le constructeur ne demarre plus de partie : le quota n'est connu que dans
    // generate_games. On demarre deux places pour controler les racines.
    SelfPlayTestAccess::demarrer_slot(manager, 0);
    SelfPlayTestAccess::demarrer_slot(manager, 1);

    // La deuxieme partie partage la position de depart de la premiere : sa
    // racine est servie par la table et doit tout de meme avoir ses enfants et
    // son bruit de Dirichlet.
    for (int game = 0; game < 2; ++game) {
        MCTSNode* root = SelfPlayTestAccess::root(manager, game);
        require_test(root != nullptr, "missing self-play root");
        require_test(root->state.load() == NodeState::Expanded,
                     "self-play root was left unexpanded");
        require_test(!root->children.empty(),
                     "self-play root has no children");
        require_test(SelfPlayTestAccess::pending_epsilon(manager, game) == 0.0f,
                     "self-play root kept a pending noise");
    }
}

void test_selfplay_generation_is_finite_and_counted() {
    struct Cas {
        int places;
        int quota;
    };
    const Cas cas[] = {{2, 3}, {4, 0}, {4, 1}, {4, 3}, {4, 4}, {4, 5},
                       {4, 7}};

    for (const Cas& c : cas) {
        ControlledEvaluator evaluator;
        SelfPlayManager manager(&evaluator, c.places, 2, 1, 0.5f, 8192);
        const std::vector<GameResult> games = manager.generate_games(c.quota);
        const SelfPlayStats stats = manager.get_stats();

        require_test(games.size() == static_cast<std::size_t>(c.quota),
                     "the generation did not return exactly N games");
        require_test(stats.games_started == static_cast<std::uint64_t>(c.quota),
                     "the number of starts does not match the quota");
        require_test(stats.games_completed == static_cast<std::uint64_t>(c.quota),
                     "the number of completions does not match the quota");
        require_test(stats.active_slots == 0,
                     "a place stayed active after the generation");
        require_test(stats.replayed_plies == 0,
                     "replayed plies were counted without puzzles");
        if (c.quota == 0) {
            require_test(stats.new_plies == 0, "N=0 played a move");
            require_test(evaluator.batch_sizes.empty(),
                         "N=0 called the evaluator");
        }
        else {
            require_test(stats.new_plies > 0, "no new ply was counted");
        }
    }
}

void test_successive_selfplay_generations_are_independent() {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);

    const std::vector<GameResult> premier = manager.generate_games(2);
    const SelfPlayStats stats_premier = manager.get_stats();
    const std::vector<GameResult> second = manager.generate_games(3);
    const SelfPlayStats stats_second = manager.get_stats();

    require_test(premier.size() == 2, "the first generation is incomplete");
    require_test(second.size() == 3, "the second generation is incomplete");
    require_test(stats_premier.games_completed == 2,
                 "the first counters were changed by the second call");
    require_test(stats_second.games_started == 3
                     && stats_second.games_completed == 3,
                 "the second generation did not restart its counters");
}

void test_selfplay_batch_associates_each_game_with_its_own_tensor() {
    DiscriminatingEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 1, 1, 0.5f, 8192);

    // Deux positions distinctes : les tenseurs et les reponses different, un
    // decalage entre lignes du lot et parties devient visible.
    SelfPlayTestAccess::preparer_position(manager, 0, "e2e4");
    SelfPlayTestAccess::preparer_position(manager, 1, "d2d4");
    const std::string fen0 = SelfPlayTestAccess::fen(manager, 0);
    const std::string fen1 = SelfPlayTestAccess::fen(manager, 1);

    const SelfPlayTestAccess::ResultatLot lot =
        SelfPlayTestAccess::executer_lot_par_partie(manager, {0, 1});

    for (std::size_t j = 0; j < 2; ++j) {
        require_test(lot.feuilles[j] != nullptr,
                     "a game produced no network leaf");
        const auto& reponse = evaluator.reponse_pour(lot.tenseurs[j]);
        require_test(
            std::fabs(lot.feuilles[j]->network_value - reponse.value) < 1e-6f,
            "a game received another game's value");
        require_test(lot.feuilles[j]->visit_count == 1,
                     "the leaf did not receive exactly one backup");

        float somme = 0.0f;
        for (int idx : lot.coups_legaux[j]) {
            somme += reponse.policy[static_cast<std::size_t>(idx)];
        }
        require_test(somme > 0.0f, "the response has no legal mass");
        require_test(lot.feuilles[j]->children.size()
                         == lot.coups_legaux[j].size(),
                     "the leaf did not expose its legal children");
        for (const auto& child : lot.feuilles[j]->children) {
            const float attendu =
                reponse.policy[static_cast<std::size_t>(child.first)] / somme;
            require_test(std::fabs(child.second->prior - attendu) < 1e-5f,
                         "a game received another game's policy");
        }

        MCTSNode* racine =
            SelfPlayTestAccess::root(manager, static_cast<int>(j));
        int profondeur = 0;
        for (MCTSNode* n = lot.feuilles[j]; n->parent != nullptr;
             n = n->parent) {
            ++profondeur;
        }
        const float signe = (profondeur % 2 == 0) ? 1.0f : -1.0f;
        require_test(
            std::fabs(racine->total_value - signe * reponse.value) < 1e-6f,
            "the backup sign is wrong at the root");
    }

    require_test(SelfPlayTestAccess::fen(manager, 0) == fen0,
                 "the first board was not restored");
    require_test(SelfPlayTestAccess::fen(manager, 1) == fen1,
                 "the second board was not restored");
}

void test_one_active_slot_finishes_with_partial_batches() {
    // Une place seule ne peut jamais remplir un lot de deux : la fin de
    // generation doit avancer par lots partiels, sans attendre un lot plein et
    // sans toucher la place inactive restee sans racine.
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);

    const std::vector<GameResult> games = manager.generate_games(1);
    const SelfPlayStats stats = manager.get_stats();

    require_test(games.size() == 1, "the single game was not collected");
    require_test(stats.games_started == 1 && stats.games_completed == 1,
                 "the single active slot did not start and finish once");
    require_test(stats.active_slots == 0, "a place stayed active");
    require_test(!evaluator.batch_sizes.empty(),
                 "the single game never reached the evaluator");
    for (int taille : evaluator.batch_sizes) {
        require_test(taille <= 1, "the drain waited for a fuller batch");
    }
    require_test(SelfPlayTestAccess::root(manager, 1) == nullptr,
                 "the inactive slot received a root");
}

void test_imposed_ends_start_and_collect_each_game_once() {
    // Une place finit tot, l'autre tard, et les fins sont imposees a des
    // moments connus : aucune attente d'horloge, aucun terminal aleatoire.
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);
    SelfPlayTestAccess::forcer_fin_apres(manager, 0, 2);
    SelfPlayTestAccess::forcer_fin_apres(manager, 1, 8);

    const std::vector<GameResult> games = manager.generate_games(3);
    const SelfPlayStats stats = manager.get_stats();

    require_test(games.size() == 3,
                 "the generation did not collect exactly three games");
    require_test(stats.games_started == 3 && stats.games_completed == 3,
                 "the starts and completions do not match the quota");
    require_test(stats.active_slots == 0,
                 "a place stayed active after the generation");

    // Identites : la place 0 a joue deux parties de 2 plies, la place 1 une
    // partie de 8. Aucune partie fantome, aucune collecte en double.
    std::vector<int> longueurs;
    for (const GameResult& game : games) {
        longueurs.push_back(game.total_real_moves);
    }
    std::sort(longueurs.begin(), longueurs.end());
    require_test(longueurs == std::vector<int>({2, 2, 8}),
                 "the collected games do not match the imposed ends");
}

std::string ecrire_fixture_puzzle() {
    const std::string chemin = "puzzles_test_fixture.txt";
    std::ofstream fichier(chemin);
    fichier << "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
               "|e2e4 e7e5|e7e5|1500|fork\n";
    return chemin;
}

bool chercher_graine_injectee(SelfPlayManager& manager) {
    // L'injection est tiree au hasard : on cherche une graine qui la
    // declenche, sans attendre ni dependre du temps.
    for (std::uint32_t graine = 1; graine <= 2000; ++graine) {
        SelfPlayTestAccess::seed_rng(manager, graine);
        SelfPlayTestAccess::reset_game(manager, 0);
        if (SelfPlayTestAccess::tactical_boost(manager, 0)) {
            return true;
        }
    }
    return false;
}

void test_puzzle_first_move_boost_then_normal_move() {
    const std::string fixture = ecrire_fixture_puzzle();
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192, fixture);
    std::remove(fixture.c_str());

    require_test(chercher_graine_injectee(manager),
                 "no seed injected the puzzle");
    require_test(SelfPlayTestAccess::sims_target(manager, 0) == 4000,
                 "the puzzle first move lost its tactical budget");
    require_test(SelfPlayTestAccess::is_slow_move(manager, 0),
                 "the puzzle first move is not a slow move");
    require_test(SelfPlayTestAccess::pending_epsilon(manager, 0) == 0.0f,
                 "the tactical noise was not consumed");
    require_test(SelfPlayTestAccess::root(manager, 0) != nullptr,
                 "the puzzle root is missing");

    // Le coup suivant retrouve un budget normal et abandonne l'epsilon
    // tactique : le bruit normal est soit deja applique, soit en attente.
    SelfPlayTestAccess::play_best_move(manager, 0);
    require_test(!SelfPlayTestAccess::tactical_boost(manager, 0),
                 "the tactical boost was not consumed");
    const int cible = SelfPlayTestAccess::sims_target(manager, 0);
    require_test(cible == 1 || cible == 2,
                 "the next move did not return to a normal budget");
    require_test(SelfPlayTestAccess::pending_epsilon(manager, 0) <= 0.12f,
                 "the tactical epsilon survived the first move");
}

void test_puzzle_first_move_served_by_the_table_keeps_the_boost() {
    const std::string fixture = ecrire_fixture_puzzle();
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 1, 2, 1, 0.5f, 8192, fixture);
    std::remove(fixture.c_str());

    // Prechauffe la table avec la position du puzzle.
    Chessboard position;
    position.loadFEN(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    require_test(position.movePieceUCI("e2e4"), "e2e4 failed");
    require_test(position.movePieceUCI("e7e5"), "e7e5 failed");
    const std::vector<int> legal = position.getLegalMoveIndices();
    std::vector<float> policy(POLICY_SIZE, 0.0f);
    for (int idx : legal) policy[static_cast<std::size_t>(idx)] = 1.0f;
    MCTSTestAccess::store(SelfPlayTestAccess::mcts(manager), position, legal,
                          policy, 0.1f);

    require_test(chercher_graine_injectee(manager),
                 "no seed injected the puzzle");
    MCTSNode* root = SelfPlayTestAccess::root(manager, 0);
    require_test(root != nullptr, "the puzzle root is missing");
    require_test(root->state.load() == NodeState::Expanded,
                 "a TT hit left the puzzle root undeveloped");
    require_test(!root->children.empty(),
                 "a TT hit lost the puzzle root children");
    require_test(SelfPlayTestAccess::pending_epsilon(manager, 0) == 0.0f,
                 "the tactical noise was not consumed on a TT hit");
    require_test(SelfPlayTestAccess::sims_target(manager, 0) == 4000,
                 "a TT hit lost the tactical budget");
}

void test_slow_puzzle_game_is_not_lost_behind_a_fast_one() {
    const std::string fixture = ecrire_fixture_puzzle();
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192, fixture);
    std::remove(fixture.c_str());

    // Une place joue un puzzle, premier coup a 4000 simulations, et doit etre
    // recuperee malgre la fin rapide de l'autre. On cherche une graine ou la
    // place 0 est injectee et la place 1 ne l'est pas.
    bool trouve = false;
    for (std::uint32_t graine = 1; graine <= 4000 && !trouve; ++graine) {
        SelfPlayTestAccess::seed_rng(manager, graine);
        SelfPlayTestAccess::reset_game(manager, 0);
        SelfPlayTestAccess::reset_game(manager, 1);
        trouve = SelfPlayTestAccess::tactical_boost(manager, 0)
            && !SelfPlayTestAccess::tactical_boost(manager, 1);
    }
    require_test(trouve, "no seed gave one puzzle and one normal game");
    require_test(SelfPlayTestAccess::sims_target(manager, 0) == 4000,
                 "the puzzle lost its tactical budget");

    SelfPlayTestAccess::forcer_fin_apres(manager, 0, 4);
    SelfPlayTestAccess::forcer_fin_apres(manager, 1, 2);

    const std::vector<GameResult> games = manager.generate_games(2);
    const SelfPlayStats stats = manager.get_stats();

    require_test(games.size() == 2, "the slow game was not collected");
    require_test(stats.games_started == 2 && stats.games_completed == 2,
                 "the starts and completions do not match the quota");
    require_test(stats.active_slots == 0, "a place stayed active");
    std::vector<int> longueurs;
    for (const GameResult& game : games) {
        longueurs.push_back(game.total_real_moves);
    }
    std::sort(longueurs.begin(), longueurs.end());
    require_test(longueurs == std::vector<int>({2, 4}),
                 "the slow puzzle game and the fast game were not both kept");
}

void test_game_conclusion_signs_and_reasons() {
    Chessboard board;

    board.loadFEN("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
    GameConclusion c = conclure_partie(board, false);
    require_test(c.end_reason == 0 && c.final_outcome == 1.0f,
                 "a mated black side should be a win for White");

    board.loadFEN("8/8/8/8/8/5k2/6q1/7K w - - 0 1");
    c = conclure_partie(board, false);
    require_test(c.end_reason == 0 && c.final_outcome == -1.0f,
                 "a mated white side should be a loss for White");

    board.loadFEN("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    c = conclure_partie(board, false);
    require_test(c.end_reason == 1 && c.final_outcome == 0.0f,
                 "stalemate should be a draw");

    board.loadFEN(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    c = conclure_partie(board, true);
    require_test(c.end_reason == 5 && c.final_outcome == 0.0f,
                 "the maximum length should be a draw");

    board.setStartupPieces();
    const char* moves[] = {"g1f3", "g8f6", "f3g1", "f6g8",
                           "g1f3", "g8f6", "f3g1", "f6g8"};
    for (const char* uci : moves) {
        require_test(board.movePieceUCI(uci), "repetition move failed");
    }
    c = conclure_partie(board, false);
    require_test(c.end_reason == 2 && c.final_outcome == 0.0f,
                 "repetition should be a draw");

    board.loadFEN(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 1");
    c = conclure_partie(board, false);
    require_test(c.end_reason == 3 && c.final_outcome == 0.0f,
                 "the fifty-move rule should be a draw");

    board.loadFEN("8/8/8/8/8/8/8/K6k w - - 0 1");
    c = conclure_partie(board, false);
    require_test(c.end_reason == 4 && c.final_outcome == 0.0f,
                 "insufficient material should be a draw");
}

std::vector<GameResult> generer_avec_reglages(int virtual_loss,
                                              std::uint32_t graine) {
    ControlledEvaluator evaluator;
    SelfPlayManager manager(&evaluator, 2, 2, 1, 0.5f, 8192);
    SelfPlayTestAccess::seed_rng(manager, graine);
    MCTSTestAccess::seed_noise(SelfPlayTestAccess::mcts(manager), graine);
    SelfPlayTestAccess::mcts(manager).set_tuning(
        SearchTuning{virtual_loss, 0.30f, 4});
    return manager.generate_games(2);
}

void test_virtual_loss_is_inert_in_self_play() {
    // Chaque arbre n'est descendu qu'une fois par vague : n_in_flight n'a pas
    // de lecteur entre la reservation et sa liberation. Deux generations de
    // meme graine doivent donc etre identiques, quelle que soit l'amplitude.
    const std::vector<GameResult> reference = generer_avec_reglages(1, 12345);
    const std::vector<GameResult> candidat = generer_avec_reglages(2, 12345);

    require_test(reference.size() == candidat.size(),
                 "the two generations do not have the same game count");
    for (std::size_t i = 0; i < reference.size(); ++i) {
        require_test(reference[i].flat_states == candidat[i].flat_states,
                     "the amplitude changed the self-play states");
        require_test(reference[i].flat_policies == candidat[i].flat_policies,
                     "the amplitude changed the self-play policies");
        require_test(reference[i].final_outcome == candidat[i].final_outcome
                         && reference[i].end_reason == candidat[i].end_reason,
                     "the amplitude changed the self-play outcome");
    }
}

}  // namespace

int main() {
    try {
        test_openmp_selfplay_core_with_shared_cache();
        test_selfplay_manager_with_controlled_evaluator();
        test_selfplay_rejects_invalid_batch_and_cleans_up();
        test_selfplay_roots_are_expanded_and_noised_exactly_once();
        test_selfplay_generation_is_finite_and_counted();
        test_successive_selfplay_generations_are_independent();
        test_selfplay_batch_associates_each_game_with_its_own_tensor();
        test_one_active_slot_finishes_with_partial_batches();
        test_imposed_ends_start_and_collect_each_game_once();
        test_puzzle_first_move_boost_then_normal_move();
        test_puzzle_first_move_served_by_the_table_keeps_the_boost();
        test_slow_puzzle_game_is_not_lost_behind_a_fast_one();
        test_game_conclusion_signs_and_reasons();
        test_virtual_loss_is_inert_in_self_play();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
