# License: BSD-3-clause
# Copyright © 2023 National Institute of Advanced Industrial Science and Technology (AIST)
# Author: Yasunobu ANDO

"""背景モデル生成を担当するファクトリクラス"""

from typing import List, Optional
import numpy as np

from EMPeaks.Background import UniformModel, SquareRootModel, LinearModel, TriangleModel, RampModel
from EMPeaks.EMCore.exceptions import BackgroundTypeError


class BackgroundFactory:
    """背景モデルを生成するファクトリクラス
    
    背景モデルの種類に応じて適切なモデルインスタンスを生成します。
    
    Attributes:
        x_min: X軸の最小値
        x_max: X軸の最大値
        k_ramp: RampSumモデルのセグメント数
    """
    
    # 対応する背景タイプ
    SUPPORTED_TYPES = ('none', 'uniform', 'squareroot', 'linear', 'ramp_sum')
    
    def __init__(self, x_min: float, x_max: float, k_ramp: int = 5):
        """BackgroundFactoryを初期化
        
        Args:
            x_min: X軸の最小値
            x_max: X軸の最大値
            k_ramp: RampSumモデルのセグメント数 (デフォルト: 5)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.k_ramp = k_ramp
        self._ramp_node: Optional[np.ndarray] = None
    
    @property
    def ramp_node(self) -> np.ndarray:
        """RampSum用のノード位置を取得（遅延初期化）"""
        if self._ramp_node is None:
            self._ramp_node = np.linspace(self.x_min, self.x_max, self.k_ramp + 1, endpoint=False)
        return self._ramp_node
    
    def update_range(self, x_min: float, x_max: float) -> None:
        """X軸範囲を更新
        
        Args:
            x_min: 新しいX軸の最小値
            x_max: 新しいX軸の最大値
        """
        self.x_min = x_min
        self.x_max = x_max
        self._ramp_node = None  # ノードをリセット
    
    def create(self, background_type: str, **kwargs) -> List:
        """背景モデルを生成
        
        Args:
            background_type: 背景タイプ ('none', 'uniform', 'squareroot', 'linear', 'ramp_sum')
            **kwargs: 追加パラメータ (例: s_tri for linear)
        
        Returns:
            生成された背景モデルのリスト
        
        Raises:
            BackgroundTypeError: 未対応の背景タイプが指定された場合
        """
        if background_type == 'none':
            return []
        elif background_type == 'uniform':
            return [UniformModel(self.x_min, self.x_max)]
        elif background_type == 'squareroot':
            return [SquareRootModel(self.x_min, self.x_max)]
        elif background_type == 'linear':
            s_tri = kwargs.get('s_tri', None)
            if s_tri is not None and 0 <= s_tri <= 1.0:
                return [LinearModel(self.x_min, self.x_max, s_tri=s_tri)]
            else:
                return [LinearModel(self.x_min, self.x_max)]
        elif background_type == 'ramp_sum':
            return self.create_ramp_sum()
        else:
            raise BackgroundTypeError(f"Unknown background type: {background_type}")
    
    def create_ramp_sum(self) -> List:
        """RampSum用の複合背景モデルを生成
        
        Returns:
            UniformModel + RampModels + TriangleModel のリスト
        """
        models = [UniformModel(self.x_min, self.x_max)]
        for k in range(self.k_ramp):
            models.append(RampModel(self.ramp_node[k], self.ramp_node[k + 1], self.x_max))
        models.append(TriangleModel(self.ramp_node[-1], self.x_max))
        return models
    
    def get_num_models(self, background_type: str) -> int:
        """指定した背景タイプのモデル数を取得
        
        Args:
            background_type: 背景タイプ
        
        Returns:
            モデル数
        """
        if background_type == 'none':
            return 0
        elif background_type in ('uniform', 'squareroot', 'linear'):
            return 1
        elif background_type == 'ramp_sum':
            return self.k_ramp + 2  # uniform + k_ramp ramps + triangle
        else:
            return 0
