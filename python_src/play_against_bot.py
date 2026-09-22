import warnings
import threading
import pygame
import sys
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

import chess_engine
from lib_gui import (
    SQUARE_SIZE,
    pygame_init,
    load_images,
    rendu
)
from lib import (move_to_san, print_pgn, decode_move_index, chose_move_idx)

# ============================================================
#                     CONFIGURATION
# ============================================================

HUMAN_COLOR = chess_engine.Color.BLACK
CHECKPOINT_PATH = "checkpoints/2026_03_21_10h36_iter136_unsupervised.onnx"

# Nombre de positions evaluees par inference, grace au virtual loss.
MCTS_BATCH_SIZE = 8

# Virtual loss 2 : +9 a +20 % de debit mesures, qualite non-inferieure sur
# 2500 puzzles. Voir docs/superpowers/specs/2026-09-22-rebalayage-virtual-loss-resultats.md.
MCTS_VIRTUAL_LOSS = 2

MCTS_PARAMS = {
    "num_sim": 1200,
    "tau_first_move": 2,
    "tau_opening": 1,
    "tau_endgame": 0.1,
    "tau_threshold": 8
}

# ============================================================
#                     BOUCLE PRINCIPALE
# ============================================================


class ChessGame:
    def __init__(
            self,
            board: chess_engine.Chessboard,
            mcts_engine: chess_engine.MCTS,
            mcts_params: dict,
            human_color: chess_engine.Color = chess_engine.Color.WHITE,
    ):
        self.board: chess_engine.Chessboard = board
        self.mcts_engine: chess_engine.MCTS = mcts_engine
        self.mcts_params: dict = mcts_params
        self.human_color = human_color

        self.selected_filerank: tuple[int, int] | None = None
        self.san_moves: list = []
        self.ai_result_container: list = []
        self.current_legal_moves: list[chess_engine.Move] = []
        self.drag_pos: tuple[int, int] = (0, 0)

        self.red_squares: set = set()

        # states
        self.is_human_turn: bool = False
        self._is_running: bool = False  # game process
        self._game_over: bool = False  # chess game
        self._is_dragging: bool = False
        self._is_ai_thinking: bool = False
        self._just_got_selected = False

        self.screen, self.clock = pygame_init()
        load_images()

    def start(self):
        if self._is_running:
            raise Exception("Game is already running!")
        self._is_running = True

    def stop(self):
        if not self._is_running:
            raise Exception("Process is not Currently running!")
        self._game_over = True
        self._is_running = False

    def endGame(self):
        if self._game_over:
            raise Exception("Game has already stopped!")
        self._game_over = True

    def isRunning(self):
        return self._is_running

    def gameEnded(self):
        return self._game_over

    def isAiThinking(self):
        return self._is_ai_thinking

    def AiStartThinking(self):
        if self._is_ai_thinking:
            raise Exception("AI is already thinking!")
        self._is_ai_thinking = True

    def AiStopThinking(self):
        if not self._is_ai_thinking:
            raise Exception("AI is NOT currently thinking!")
        self._is_ai_thinking = False

    def isDragging(self):
        return self._is_dragging

    def startDragging(self, clicked_sq, mouse_x, mouse_y):
        # self.clearActions()
        if self._is_dragging:
            raise Exception("Already currently dragging!")
        if clicked_sq != self.selected_filerank:
            self._just_got_selected = True
        else:
            self._just_got_selected = False
        self.selected_filerank = clicked_sq
        self._is_dragging = True
        self.drag_pos = (mouse_x, mouse_y)

    def stopDragging(self):
        if not self._is_dragging:
            raise Exception("Not currently dragging!")
        self._is_dragging = False

    def computeClickedSquare(self):
        x, y = pygame.mouse.get_pos()
        if self.human_color == chess_engine.Color.WHITE:
            clicked_file = x // SQUARE_SIZE
            clicked_rank = 7 - (y // SQUARE_SIZE)
        else:
            clicked_file = 7 - (x // SQUARE_SIZE)
            clicked_rank = y // SQUARE_SIZE

        clicked_sq = (clicked_file, clicked_rank)
        return clicked_sq, x, y

    def clearActions(self):
        self.selected_filerank = None
        self.current_legal_moves = []
        self.red_squares.clear()
        self._just_got_selected = False

    def checkIfValidMove(self, clicked_file, clicked_rank):
        valid_move = None
        promotion_type = chess_engine.PieceType.NONE

        for move in self.current_legal_moves:
            if (move.get_dest_square().get_file() == clicked_file and
                    move.get_dest_square().get_rank() == clicked_rank):
                valid_move = move
                if move.get_promotion() == chess_engine.PieceType.QUEEN:
                    promotion_type = chess_engine.PieceType.QUEEN
                    break

        return valid_move, promotion_type

    def makeMove(self, clicked_file, clicked_rank, promotion_type):
        orig_f, orig_r = self.selected_filerank
        san = move_to_san(self.board, orig_f, orig_r, clicked_file, clicked_rank,
                          promotion_type)
        success = self.board.move_piece(orig_f, orig_r, clicked_file, clicked_rank,
                                        promotion_type)

        if success:
            self.clearActions()
            if self.board.game_state == chess_engine.GameState.CHECKMATE:
                san += "#"
            elif self.board.is_in_check():
                san += "+"
            self.san_moves.append(san)
            if self.board.game_state != chess_engine.GameState.ONGOING:
                self.endGame()
        else:
            print(f'board.move_piece() a renvoyé "False. Coup : {san}"')

    def eventLeftClickDown(self, clicked_sq, mouse_x, mouse_y):
        clicked_file, clicked_rank = clicked_sq
        self.red_squares.clear()
        sq = self.board.get_square(clicked_file, clicked_rank)

        if self.selected_filerank is None:
            # Cas 1 : Aucune pièce sélectionnée
            if sq.is_occupied():
                self.startDragging(clicked_sq, mouse_x, mouse_y)

                # SECURITE : On ne tape dans le C++ que si c'est notre tour
                if sq.get_piece().get_color() == self.human_color and self.is_human_turn:
                    self.current_legal_moves = self.board.get_legal_moves(clicked_file,
                                                                          clicked_rank)
                else:
                    self.current_legal_moves = []
        else:
            # Cas 2 : Une pièce est déjà sélectionnée
            valid_move, promotion_type = self.checkIfValidMove(clicked_file, clicked_rank)

            if valid_move is not None and self.is_human_turn and not self.gameEnded():
                self.makeMove(clicked_file, clicked_rank, promotion_type)
            else:
                sq = self.board.get_square(clicked_file, clicked_rank)
                if sq.is_occupied():
                    self.startDragging(clicked_sq, mouse_x, mouse_y)
                    if sq.get_piece().get_color() == self.human_color and self.is_human_turn:
                        self.current_legal_moves = self.board.get_legal_moves(clicked_file,
                                                                              clicked_rank)
                    else:
                        self.current_legal_moves = []
                else:
                    self.current_legal_moves = []

    def eventLeftClickUp(self, clicked_file, clicked_rank):
        if self.isDragging():
            self.stopDragging()
            if ((clicked_file, clicked_rank) == self.selected_filerank and
                    not self._just_got_selected):
                self.clearActions()  # On déselectionne si on reclique sur la même case
            else:
                valid_move, promotion_type = self.checkIfValidMove(clicked_file, clicked_rank)

                if valid_move is not None:
                    self.makeMove(clicked_file, clicked_rank, promotion_type)

    def mcts_worker(self):
        """
        Exécute le MCTS en arrière-plan avec logique de température.
        """
        # 1. Recréation de l'état du plateau
        temp_board = chess_engine.Chessboard()
        temp_board.set_startup_pieces()
        for move in self.san_moves:
            temp_board.move_piece_san(move)

        # 2. Lancement de la recherche C++ / ONNX
        pi_raw = self.mcts_engine.mcts_search(
            temp_board, self.mcts_params.get("num_sim", 1200),
            batch_size=MCTS_BATCH_SIZE)
        pi = np.array(pi_raw, dtype=np.float32)

        # 3. Logique de température
        move_count = len(self.san_moves)
        if move_count < 2:
            current_tau = self.mcts_params.get("tau_first_move", 2)
        elif move_count < self.mcts_params.get("tau_threshold", 8):
            current_tau = self.mcts_params.get("tau_opening", 1)
        else:
            current_tau = self.mcts_params.get("tau_endgame", 0.1)

        best_idx = chose_move_idx(pi, current_tau)

        # 5. Décodage
        is_black = (temp_board.turn == chess_engine.Color.BLACK)
        orig_f, orig_r, dest_f, dest_r, promo = decode_move_index(temp_board, best_idx, is_black)

        # Affichage console
        san = move_to_san(temp_board, orig_f, orig_r, dest_f, dest_r, promo)
        # On affiche la probabilité réelle de l'index choisi pour voir l'influence de Tau
        print(f"\n[AI] Coup choisi : {san} (Tau: {current_tau})")

        self.ai_result_container.append((orig_f, orig_r, dest_f, dest_r, promo))

    def AiTurn(self):
        if not self.isAiThinking():
            self.AiStartThinking()
            self.ai_result_container.clear()
            thread = threading.Thread(
                target=self.mcts_worker,
                args=()
            )
            thread.start()

        elif len(self.ai_result_container) > 0:
            result = self.ai_result_container.pop()
            if result is not None:
                orig_f, orig_r, dest_f, dest_r, promo = result

                san = move_to_san(self.board, orig_f, orig_r, dest_f, dest_r, promo)
                success = self.board.move_piece(orig_f, orig_r, dest_f, dest_r, promo)

                if success:
                    if self.board.game_state == chess_engine.GameState.CHECKMATE:
                        san += "#"
                    elif self.board.is_in_check():
                        san += "+"
                    self.san_moves.append(san)
                    if self.board.game_state != chess_engine.GameState.ONGOING:
                        self.endGame()
                else:
                    print(f"FATAL ERROR: Le moteur a rejeté le coup de l'IA {san}.")
                    self.endGame()

            self.AiStopThinking()

    def loop(self):
        self.is_human_turn = (self.board.turn == self.human_color)
        for event in pygame.event.get():
            clicked_sq, mouse_x, mouse_y = self.computeClickedSquare()
            clicked_file, clicked_rank = clicked_sq

            if event.type == pygame.QUIT:
                self.stop()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if clicked_sq in self.red_squares:
                    self.red_squares.remove(clicked_sq)
                else:
                    self.red_squares.add(clicked_sq)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.eventLeftClickDown(clicked_sq, mouse_x, mouse_y)

            elif event.type == pygame.MOUSEMOTION:
                if self.isDragging():
                    self.drag_pos = event.pos

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.eventLeftClickUp(clicked_file, clicked_rank)

        # Tour de l'IA
        if not self.is_human_turn and not self.gameEnded():
            self.AiTurn()

        # 3. Appel de rendu mis à jour
        self.clock = rendu(
            self.screen,
            self.board,
            self.selected_filerank,
            self.current_legal_moves,
            self.clock,
            self.isDragging(),
            self.drag_pos,
            self.red_squares,
            perspective=self.human_color
        )


def main():
    print(f"Chargement du moteur MCTS avec {CHECKPOINT_PATH}...")

    # Processus unique et interactif : le GPU est un gain net ici, contrairement
    # au tournoi qui fait tourner huit parties en parallele. Repli sur le CPU
    # si le provider CUDA manque.
    try:
        evaluator = chess_engine.ONNXEvaluator(CHECKPOINT_PATH, True)
    except Exception as e:
        print(f"GPU indisponible, repli sur le CPU : {e}")
        evaluator = chess_engine.ONNXEvaluator(CHECKPOINT_PATH, False)
    mcts_engine = chess_engine.MCTS(evaluator)
    mcts_engine.set_tuning(MCTS_VIRTUAL_LOSS)

    board = chess_engine.Chessboard()
    board.set_startup_pieces()
    game = ChessGame(
        board=board,
        mcts_engine=mcts_engine,
        mcts_params=MCTS_PARAMS,
        human_color=HUMAN_COLOR)
    game.start()

    while game.isRunning():
        game.loop()
    print_pgn(game.board, game.san_moves)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
