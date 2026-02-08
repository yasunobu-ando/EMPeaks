# License: BSD-3-clause
# Copyright © 2023 National Institute of Advanced Industrial Science and Technology (AIST)
# Author: Yasunobu ANDO

"""EMCore custom exceptions."""


class EMCoreError(Exception):
    """EMCore関連のエラー基底クラス"""
    pass


class ParameterError(EMCoreError):
    """パラメータ設定エラー"""
    pass


class ConvergenceError(EMCoreError):
    """収束エラー"""
    pass


class BackgroundTypeError(EMCoreError):
    """未対応の背景タイプエラー"""
    pass
