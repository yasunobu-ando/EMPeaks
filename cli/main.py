# Copyright © 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# License: BSD-3-clause
# empeaks CLI エントリーポイント

import argparse
import subprocess
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        prog="empeaks",
        description="EMPeaks - スペクトルピークフィッティングツール",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- deck サブコマンド ---
    deck_parser = subparsers.add_parser("deck", help="Streamlit GUIを起動する")
    deck_parser.add_argument(
        "--port", type=int, default=8501,
        help="ポート番号 (デフォルト: 8501)"
    )

    args = parser.parse_args()

    if args.command == "deck":
        _launch_deck(args)
    else:
        parser.print_help()


def _launch_deck(args):
    # gui パッケージの __file__ からパスを解決
    # editable install / 通常インストール の両方で動作する
    import gui as _gui_pkg
    app_path = os.path.join(os.path.dirname(_gui_pkg.__file__), "app.py")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", str(args.port),
    ])
