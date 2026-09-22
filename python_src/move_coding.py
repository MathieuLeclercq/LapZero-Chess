"""Encodage et decodage des coups pour la policy AlphaZero, 4672 index.

Extrait de lib.py pour etre importable sans torch : le banc de puzzles fait
tourner 16 processus travailleurs, et torch plus onnx coutent 475 Mio a
l'import contre 2 Mio pour chess_engine seul. Les corps de ces fonctions n'ont
pas change, lib les reexporte pour ne casser aucun appelant.
"""

import chess_engine


def encode_move(orig_f, orig_r, dest_f, dest_r, promotion_type, is_black_turn):
    """Convertit un coup en un index plat (0 à 4671). Retourne -1 en cas d'erreur de parsing."""
    if is_black_turn:
        orig_r = 7 - orig_r
        dest_r = 7 - dest_r

    df = dest_f - orig_f
    dr = dest_r - orig_r
    plane = -1

    try:
        if promotion_type in [chess_engine.PieceType.KNIGHT, chess_engine.PieceType.BISHOP,
                              chess_engine.PieceType.ROOK]:
            dir_idx = df + 1
            if promotion_type == chess_engine.PieceType.KNIGHT:
                p_idx = 0
            elif promotion_type == chess_engine.PieceType.BISHOP:
                p_idx = 1
            elif promotion_type == chess_engine.PieceType.ROOK:
                p_idx = 2
            plane = 64 + dir_idx * 3 + p_idx

        elif (abs(df) == 2 and abs(dr) == 1) or (abs(df) == 1 and abs(dr) == 2):
            knight_moves = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
            plane = 56 + knight_moves.index((df, dr))

        else:
            dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
            dist = max(abs(df), abs(dr))
            dir_tuple = (df // dist, dr // dist)
            dir_idx = dirs.index(dir_tuple)
            plane = dir_idx * 7 + (dist - 1)

        return plane * 64 + orig_r * 8 + orig_f
    except ValueError:
        return -1


def decode_move_index(board, index, is_black=None):
    """Inverse de encode_move : delegue au decodeur C++ de Chessboard.

    Le parametre is_black est conserve pour les appelants historiques, mais le
    plateau fait foi : une incoherence est signalee plutot que corrigee en
    silence. Le decodeur C++ inclut l'orientation du camp au trait et
    l'inference de promotion dame.
    """
    if is_black is not None:
        attendu = (board.turn == chess_engine.Color.BLACK)
        if bool(is_black) != attendu:
            raise ValueError("is_black ne correspond pas au trait du plateau")
    return board.decode_move_index(index)


def gestion_promo_dame(board, orig_f, orig_r, dest_r, promo):
    # Promotion dame implicite (convention AlphaZero)
    piece = board.get_square(orig_f, orig_r).get_piece()
    if (piece.get_type() == chess_engine.PieceType.PAWN
            and promo == chess_engine.PieceType.NONE
            and (dest_r == 0 or dest_r == 7)):
        promo = chess_engine.PieceType.QUEEN
    return promo


def parse_uci_to_coords(uci_str):
    """
    Transforme 'e2e4' ou 'a7a8q' en (orig_f, orig_r, dest_f, dest_r, promotion)
    """
    # 1. Coordonnées de base (a-h -> 0-7, 1-8 -> 0-7)
    orig_f = ord(uci_str[0]) - ord('a')
    orig_r = int(uci_str[1]) - 1
    dest_f = ord(uci_str[2]) - ord('a')
    dest_r = int(uci_str[3]) - 1

    # 2. Gestion de la promotion (si la chaîne fait 5 caractères)
    promotion = chess_engine.PieceType.NONE
    if len(uci_str) == 5:
        promo_char = uci_str[4].lower()
        mapping = {
            'q': chess_engine.PieceType.QUEEN,
            'r': chess_engine.PieceType.ROOK,
            'b': chess_engine.PieceType.BISHOP,
            'n': chess_engine.PieceType.KNIGHT
        }
        promotion = mapping.get(promo_char, chess_engine.PieceType.NONE)

    return orig_f, orig_r, dest_f, dest_r, promotion


def coords_to_uci(orig_f, orig_r, dest_f, dest_r, promotion):
    """
    Transforme les coordonnées et le type de promotion en string UCI (ex: 'e7e8q')
    """
    files = "abcdefgh"
    # Les rangs dans ton moteur sont 0-indexed, en UCI ils sont 1-8
    move_uci = f"{files[orig_f]}{orig_r + 1}{files[dest_f]}{dest_r + 1}"

    # Ajout du suffixe de promotion si nécessaire
    if promotion != chess_engine.PieceType.NONE:
        mapping = {
            chess_engine.PieceType.QUEEN: 'q',
            chess_engine.PieceType.ROOK: 'r',
            chess_engine.PieceType.BISHOP: 'b',
            chess_engine.PieceType.KNIGHT: 'n'
        }
        move_uci += mapping.get(promotion, '')

    return move_uci
