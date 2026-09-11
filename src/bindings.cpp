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
        .def_readonly("prior", &MoveStats::prior);

    // --- Instrumentation de la recherche ---
    py::class_<SearchCounters>(m, "SearchCounters")
        .def_readonly("nn_calls", &SearchCounters::nn_calls)
        .def_readonly("tt_hits", &SearchCounters::tt_hits)
        .def_readonly("tt_misses", &SearchCounters::tt_misses)
        .def_readonly("terminal_hits", &SearchCounters::terminal_hits);

    py::class_<TreeReport>(m, "TreeReport")
        .def_readonly("nodes", &TreeReport::nodes)
        .def_readonly("max_depth", &TreeReport::max_depth)
        .def_readonly("violations", &TreeReport::violations)
        .def_readonly("en_vol", &TreeReport::en_vol)
        .def_readonly("messages", &TreeReport::messages);

    py::class_<ONNXEvaluator>(m, "ONNXEvaluator")
        .def(py::init<const std::string&, bool>(), py::arg("model_path"), py::arg("use_gpu") = false);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<ONNXEvaluator*, size_t>(),
            py::arg("evaluator"), py::arg("tt_size") = 2097143)
        .def("mcts_search", &MCTS::mcts_search,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("board"), py::arg("num_simulations"), py::arg("c_puct") = 1.4f, py::arg("add_dirichlet") = false)

        // --- BINDINGS D'ANALYSE ---
        .def("step_analysis", &MCTS::step_analysis,
            py::call_guard<py::gil_scoped_release>(),
            py::arg("board"), py::arg("num_simulations"), py::arg("c_puct") = 1.4f)
        .def("reset_analysis", &MCTS::reset_analysis)
        .def("update_root", &MCTS::update_root, "Déplace la racine de l'arbre vers un coup spécifique")
        .def("get_root_q", &MCTS::get_root_q)
        .def("get_analysis_results", &MCTS::get_analysis_results)
        .def("get_counters", &MCTS::get_counters)
        .def("reset_counters", &MCTS::reset_counters)
        .def("inspect_tree", &MCTS::inspect_tree);

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

    // --- Fonction de génération globale ---
    m.def("generate_self_play_games", [](
        ONNXEvaluator* evaluator,
        int concurrent_games,
        int slow_sims, int fast_sims,
        int total_games, float slow_ratio,
        size_t tt_size = 2097143) {
            SelfPlayManager manager(
                evaluator, concurrent_games, slow_sims, fast_sims, slow_ratio, tt_size);
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
        "Génère un dataset de parties en self-play en utilisant un batching GPU massif.");
}
