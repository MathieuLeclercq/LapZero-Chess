#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "chessboard.hpp"
#include "piece.hpp"
#include "move.hpp"
#include "square.hpp"
#include "mcts.hpp"
#include "onnx_evaluator.hpp"
#include "selfplay_manager.hpp"
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(chess_engine, m) {
    m.doc() = "Moteur d'échecs C++ bindé pour Python";

    // --- Enums ---
    py::enum_<Color>(m, "Color")
        .value("WHITE", WHITE)
        .value("BLACK", BLACK)
        .value("NO_COLOR", NO_COLOR)
        .export_values();

    py::enum_<PieceType>(m, "PieceType")
        .value("NONE", NONE)
        .value("PAWN", PAWN)
        .value("KNIGHT", KNIGHT)
        .value("BISHOP", BISHOP)
        .value("ROOK", ROOK)
        .value("QUEEN", QUEEN)
        .value("KING", KING)
        .export_values();

    py::enum_<GameState>(m, "GameState")
        .value("ONGOING", ONGOING)
        .value("CHECKMATE", CHECKMATE)
        .value("STALEMATE", STALEMATE)
        .value("DRAW_REPETITION", DRAW_REPETITION)
        .value("DRAW_50_MOVES", DRAW_50_MOVES)
        .value("DRAW_INSUFF_MATERIAL", DRAW_INSUFF_MATERIAL)
        .export_values();

    // --- Classes ---
    py::class_<EvaluationCacheKey>(m, "EvaluationCacheKey")
        .def_readonly("position_hash", &EvaluationCacheKey::position_hash)
        .def_readonly("current_context_hash", &EvaluationCacheKey::current_context_hash)
        .def_readonly("history_hash", &EvaluationCacheKey::history_hash)
        .def_readonly("combined_hash", &EvaluationCacheKey::combined_hash)
        .def_readonly("half_move_clock", &EvaluationCacheKey::half_move_clock)
        .def_readonly("repetition_category", &EvaluationCacheKey::repetition_category)
        .def_readonly("total_moves_bucket", &EvaluationCacheKey::total_moves_bucket);

    py::class_<Piece>(m, "Piece")
        .def(py::init<>())
        .def(py::init<Color, PieceType>())
        .def("get_type", &Piece::getType)
        .def("get_color", static_cast<const Color & (Piece::*)() const>(&Piece::getColor));

    py::class_<Square>(m, "Square")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def("get_file", &Square::getFile)
        .def("get_rank", &Square::getRank)
        .def("get_piece", static_cast<const Piece & (Square::*)() const>(&Square::getPiece))
        .def("is_occupied", &Square::checkOccupied)
        .def("get_name", &Square::getName);

    py::class_<Move>(m, "Move")
        .def("get_dest_square", static_cast<const Square & (Move::*)() const>(&Move::getDestSquare))
        .def("get_orig_square", static_cast<const Square & (Move::*)() const>(&Move::getOrigSquare))
        .def("get_promotion", &Move::getPromotion);

    py::class_<Chessboard>(m, "Chessboard")
        .def(py::init<>())
        .def("clear", &Chessboard::clear)
        .def("set_startup_pieces", &Chessboard::setStartupPieces)
        .def("load_fen", &Chessboard::loadFEN, py::arg("fen"))
        .def("set_kiwipete", &Chessboard::setKiwipete)
        .def("decode_move_index", [](const Chessboard& board, int index) {
            const Move move = board.decodeMoveIndex(index);
            return py::make_tuple(
                move.getOrigSquare().getFile(),
                move.getOrigSquare().getRank(),
                move.getDestSquare().getFile(),
                move.getDestSquare().getRank(),
                move.getPromotion());
        }, py::arg("index"),
           "Decode un index de policy (0 a 4671) en coordonnees absolues et "
           "promotion, orientation du trait incluse.")
        .def("get_square", static_cast<const Square & (Chessboard::*)(int, int) const>(&Chessboard::getSquare))
        .def("get_legal_moves", [](Chessboard& cb, int file, int rank)
            {
                std::vector<Move> result;
                result.reserve(100);
                std::vector<Move> pseudo_buffer;
                pseudo_buffer.reserve(27);

                cb.getLegalMovesForSquare(file, rank, result, pseudo_buffer);
                return result;
            }, py::arg("file"), py::arg("rank"))
        .def("move_piece", static_cast<bool (Chessboard::*)(int, int, int, int, PieceType, bool)>(&Chessboard::movePiece),
            py::arg("orig_file"), 
            py::arg("orig_rank"), 
            py::arg("file"), 
            py::arg("rank"), 
            py::arg("promotion") = NONE, 
            py::arg("check_game_end") = true)
        .def("has_any_legal_move", &Chessboard::hasAnyLegalMove)
        .def("is_in_check", &Chessboard::isInCheck)
        .def("undo_move", &Chessboard::undoMove)
        .def_property_readonly("turn", &Chessboard::getTurn)
        .def_property_readonly("game_state", &Chessboard::getGameState)
        .def_property_readonly("half_move_clock", &Chessboard::getHalfMoveClock)
        .def("get_alphazero_tensor", [](const Chessboard& cb)
            {
                std::vector<float> tensor;
                cb.getAlphaZeroTensor(tensor);
                py::array_t<float> result({ 119, 8, 8 });
                std::memcpy(result.mutable_data(), tensor.data(), tensor.size() * sizeof(float));

                return result;
            })
        .def("move_piece_san", &Chessboard::movePieceSAN)
        .def("move_piece_uci", &Chessboard::movePieceUCI)
        .def("get_all_legal_moves", &Chessboard::getAllLegalMoves)
        .def("to_fen", &Chessboard::toFEN)
        .def("get_evaluation_cache_key", &Chessboard::getEvaluationCacheKey,
             py::arg("history_depth"))
        .def("set_amnesia_mode", &Chessboard::setAmnesiaMode,
             py::arg("amnesia"))
        .def("get_legal_move_indices", &Chessboard::getLegalMoveIndices)
        .def("get_board_history", static_cast<const std::vector<std::array<Square, 64>>&(Chessboard::*)() const>(&Chessboard::getBoardHistory))
        .def("get_last_move_data", [](const Chessboard& cb)
            {
                if (cb.getMoveHistory().empty()) return py::make_tuple(-1, -1, -1, -1, NONE);
                const Move& last_move = cb.getMoveHistory().back();
                return py::make_tuple(
                    last_move.getOrigSquare().getFile(),
                    last_move.getOrigSquare().getRank(),
                    last_move.getDestSquare().getFile(),
                    last_move.getDestSquare().getRank(),
                    last_move.getPromotion()
                );
            });

    // --- Structure MoveStats ---
    py::class_<MoveStats>(m, "MoveStats")
        .def_readonly("move_idx", &MoveStats::move_idx)
        .def_readonly("visits", &MoveStats::visits)
        .def_readonly("q_value", &MoveStats::q_value)
        .def_readonly("prior", &MoveStats::prior)
        .def_readonly("v_value", &MoveStats::v_value);

    // --- Instrumentation de la recherche ---
    py::class_<SearchCounters>(m, "SearchCounters")
        .def_readonly("nn_calls", &SearchCounters::nn_calls)
        .def_readonly("nn_batches", &SearchCounters::nn_batches)
        .def_readonly("tt_hits", &SearchCounters::tt_hits)
        .def_readonly("tt_misses", &SearchCounters::tt_misses)
        .def_readonly("tt_position_matches", &SearchCounters::tt_position_matches)
        .def_readonly("tt_rule50_rejects", &SearchCounters::tt_rule50_rejects)
        .def_readonly("tt_context_rejects", &SearchCounters::tt_context_rejects)
        .def_readonly("tt_history_rejects", &SearchCounters::tt_history_rejects)
        .def_readonly("terminal_hits", &SearchCounters::terminal_hits)
        .def_readonly("waves", &SearchCounters::waves)
        .def_readonly("leaf_collisions", &SearchCounters::leaf_collisions)
        .def_readonly("completed_simulations",
                      &SearchCounters::completed_simulations);

    py::class_<TreeReport>(m, "TreeReport")
        .def_readonly("nodes", &TreeReport::nodes)
        .def_readonly("max_depth", &TreeReport::max_depth)
        .def_readonly("violations", &TreeReport::violations)
        .def_readonly("en_vol", &TreeReport::en_vol)
        .def_readonly("pending", &TreeReport::pending)
        .def_readonly("root_visits", &TreeReport::root_visits)
        .def_readonly("messages", &TreeReport::messages);

    py::class_<SearchTiming>(m, "SearchTiming")
        .def_readonly("enabled", &SearchTiming::enabled)
        .def_readonly("wall_ns", &SearchTiming::wall_ns)
        .def_property_readonly("selection_ns", [](const SearchTiming& t) {
            return t.phase_ns(SearchPhase::Selection);
        })
        .def_property_readonly("tensor_key_ns", [](const SearchTiming& t) {
            return t.phase_ns(SearchPhase::TensorKey);
        })
        .def_property_readonly("tt_probe_store_ns", [](const SearchTiming& t) {
            return t.phase_ns(SearchPhase::TTProbeStore);
        })
        .def_property_readonly("tt_wait_ns", [](const SearchTiming& t) {
            return t.phase_ns(SearchPhase::TTWait);
        })
        .def_property_readonly("board_copy_ns", [](const SearchTiming& t) {
            return t.phase_ns(SearchPhase::BoardCopy);
        })
        .def_property_readonly("batch_assembly_ns", [](const SearchTiming& t) {
            return t.phase_ns(SearchPhase::BatchAssembly);
        })
        .def_property_readonly("evaluator_ns", [](const SearchTiming& t) {
            return t.phase_ns(SearchPhase::Evaluator);
        })
        .def_property_readonly("expansion_ns", [](const SearchTiming& t) {
            return t.phase_ns(SearchPhase::Expansion);
        })
        .def_property_readonly("backup_ns", [](const SearchTiming& t) {
            return t.phase_ns(SearchPhase::Backup);
        })
        .def_property_readonly("worker_wait_ns", [](const SearchTiming& t) {
            return t.phase_ns(SearchPhase::WorkerWait);
        });

    py::class_<ONNXEvaluator>(m, "ONNXEvaluator")
        .def(py::init<const std::string&, bool>(), py::arg("model_path"), py::arg("use_gpu") = false);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init([](ONNXEvaluator* evaluator, size_t tt_size,
                        int cache_history_depth) {
                return std::make_unique<MCTS>(
                    evaluator, tt_size, cache_history_depth);
            }),
            py::arg("evaluator"), py::arg("tt_size") = 2097143,
            py::arg("cache_history_depth") = DEFAULT_CACHE_HISTORY_DEPTH,
            py::keep_alive<1, 2>())
        .def("mcts_search", &MCTS::mcts_search,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("board"), py::arg("num_simulations"), py::arg("c_puct") = 1.4f,
            py::arg("add_dirichlet") = false, py::arg("batch_size") = 0,
            py::arg("worker_count") = 1)

        // --- BINDINGS D'ANALYSE ---
        .def("step_analysis", &MCTS::step_analysis,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("board"), py::arg("num_simulations"), py::arg("c_puct") = 1.4f,
            py::arg("batch_size") = 0, py::arg("worker_count") = 1)
        .def("reset_analysis", &MCTS::reset_analysis,
             py::call_guard<py::gil_scoped_release>())
        .def("clear_evaluation_cache", &MCTS::clear_evaluation_cache,
             py::call_guard<py::gil_scoped_release>(),
             "Remet la table de transposition a froid, au repos seulement")
        .def("update_root", &MCTS::update_root,
             py::call_guard<py::gil_scoped_release>(),
             "Déplace la racine de l'arbre vers un coup spécifique")
        .def("get_root_q", &MCTS::get_root_q,
             py::call_guard<py::gil_scoped_release>())
        .def("get_analysis_results", &MCTS::get_analysis_results,
             py::call_guard<py::gil_scoped_release>())
        .def("get_counters", &MCTS::get_counters)
        .def("reset_counters", &MCTS::reset_counters)
        .def("set_timing_enabled", &MCTS::set_timing_enabled,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("enabled"))
        .def("set_fixed_batch", &MCTS::set_fixed_batch,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("enabled"),
             "Padde les lots d evaluation a une forme fixe")
        .def("fixed_batch", &MCTS::fixed_batch,
             py::call_guard<py::gil_scoped_release>())
        .def("set_tuning",
             [](MCTS& mcts, int virtual_loss, float fpu_reduction,
                int collision_attempt_factor) {
                 mcts.set_tuning(SearchTuning{
                     virtual_loss, fpu_reduction, collision_attempt_factor});
             },
             py::call_guard<py::gil_scoped_release>(),
             py::arg("virtual_loss") = 1,
             py::arg("fpu_reduction") = 0.30f,
             py::arg("collision_attempt_factor") = 4,
             "Reglages de divergence de la collecte")
        .def("get_tuning", [](const MCTS& mcts) {
                const SearchTuning tuning = mcts.get_tuning();
                return py::make_tuple(tuning.virtual_loss,
                                      tuning.fpu_reduction,
                                      tuning.collision_attempt_factor);
            })
        .def("get_last_timing", &MCTS::get_last_timing,
             py::call_guard<py::gil_scoped_release>())
        .def("inspect_tree",
             py::overload_cast<>(&MCTS::inspect_tree, py::const_),
             py::call_guard<py::gil_scoped_release>());

    m.def("encode_move", &encodeMoveIndex,
          py::arg("orig_f"), py::arg("orig_r"), py::arg("dest_f"),
          py::arg("dest_r"), py::arg("promotion"), py::arg("is_black"),
          "Encode un coup en index de policy (0 a 4671), -1 si le coup n'est "
          "pas encodable.");

    py::class_<GameResult>(m, "GameResult")
        .def_property_readonly("state_tensors", [](py::object& self) {
        auto& res = self.cast<GameResult&>();
        return py::array_t<float>(
            { res.move_count, 119, 8, 8 }, // Shape
            res.flat_states.data(),         // Pointer
            self                           // Base (lifetime tracker)
        );
            })
        .def_property_readonly("policies", [](py::object& self) {
        auto& res = self.cast<GameResult&>();
        return py::array_t<float>(
            { res.move_count, 4672 },
            res.flat_policies.data(),
            self
        );
            })
        .def_readonly("total_real_moves", &GameResult::total_real_moves)
        .def_readonly("final_outcome", &GameResult::final_outcome)
        .def_readonly("end_reason", &GameResult::end_reason);

    py::class_<SelfPlayStats>(m, "SelfPlayStats")
        .def_readonly("games_started", &SelfPlayStats::games_started)
        .def_readonly("games_completed", &SelfPlayStats::games_completed)
        .def_readonly("active_slots", &SelfPlayStats::active_slots)
        .def_readonly("new_plies", &SelfPlayStats::new_plies)
        .def_readonly("replayed_plies", &SelfPlayStats::replayed_plies);

    // --- Fonction de génération globale ---
    m.def("generate_self_play_games", [](
        ONNXEvaluator* evaluator,
        int concurrent_games,
        int slow_sims, int fast_sims,
        int total_games, float slow_ratio,
        size_t tt_size = 2097143,
        const std::string& puzzles_path = "../training_data/puzzles_train.txt") {
            SelfPlayManager manager(
                evaluator, concurrent_games, slow_sims, fast_sims, slow_ratio,
                tt_size, puzzles_path);
            return manager.generate_games(total_games);
        },
        py::call_guard<py::gil_scoped_release>(),
        py::arg("evaluator"),
        py::arg("concurrent_games"),
        py::arg("slow_sims"),
        py::arg("fast_sims"),
        py::arg("total_games"),
        py::arg("slow_ratio") = 0.25f,
        py::arg("tt_size") = 2097143,
        py::arg("puzzles_path") = "../training_data/puzzles_train.txt",
        "Génère un dataset de parties en self-play en utilisant un batching GPU massif.");

    // Variante de diagnostic : meme generation, plus les compteurs. La fonction
    // de production garde son type de retour.
    m.def("generate_self_play_games_with_stats", [](
        ONNXEvaluator* evaluator,
        int concurrent_games,
        int slow_sims, int fast_sims,
        int total_games, float slow_ratio,
        size_t tt_size = 2097143,
        const std::string& puzzles_path = "../training_data/puzzles_train.txt") {
            SelfPlayManager manager(
                evaluator, concurrent_games, slow_sims, fast_sims, slow_ratio,
                tt_size, puzzles_path);
            auto parties = manager.generate_games(total_games);
            return py::make_tuple(std::move(parties), manager.get_stats());
        },
        py::call_guard<py::gil_scoped_release>(),
        py::arg("evaluator"),
        py::arg("concurrent_games"),
        py::arg("slow_sims"),
        py::arg("fast_sims"),
        py::arg("total_games"),
        py::arg("slow_ratio") = 0.25f,
        py::arg("tt_size") = 2097143,
        py::arg("puzzles_path") = "../training_data/puzzles_train.txt",
        "Genere des parties de self-play et renvoie aussi les compteurs.");
}
