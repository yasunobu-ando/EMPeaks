# License: BSD-3-clause
# Copyright © 2023 National Institute of Advanced Industrial Science and Technology (AIST)
# Author: Yasunobu ANDO

"""可視化を担当するクラス"""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    """フィッティング結果の可視化を担当するクラス
    
    EMCoreのplotメソッドの機能を独立クラスとして実装。
    """
    
    DEFAULT_FIGSIZE = (8, 3)
    
    @staticmethod
    def plot(
        x_data: np.ndarray,
        intensity: np.ndarray,
        model: List,
        N: List[float],
        N_tot: float,
        predict_func,
        x_min: float,
        x_max: float,
        dx: float,
        K: int,
        background: str = 'none',
        figsize: tuple = None,
        show: bool = True
    ):
        """フィッティング結果をプロット
        
        Args:
            x_data: 元データのX座標
            intensity: 元データの強度
            model: モデルのリスト
            N: 各モデルの正規化係数
            N_tot: 全体の正規化係数
            predict_func: 予測関数 (callable)
            x_min: X軸最小値
            x_max: X軸最大値
            dx: X軸刻み幅
            K: ピーク数
            background: 背景タイプ
            k_ramp: RampSumのセグメント数
            figsize: 図のサイズ (デフォルト: (8, 3))
            show: Trueならplt.show()を呼び出す
        
        Returns:
            matplotlib.figure.Figure: 作成した図
        """
        if figsize is None:
            figsize = Visualizer.DEFAULT_FIGSIZE
        
        fig = plt.figure(figsize=figsize)
        x = np.arange(x_min, x_max, dx)
        ax = fig.add_subplot(1, 1, 1)
        
        # Peak models
        for k in range(K):
            ax.plot(x, model[k].predict(x) * N[k], label='model_' + str(k))
        
        # Background models
        if background == 'sharley':
            y = model[-1].predict(x) * N[-1] + model[-2].predict(x) * N[-2]
            ax.plot(x, y, label=background)
        elif background in ('ramp_sum', 'b_spline'):
            n_bg = len(model) - K
            y = np.sum([model[K+k].predict(x) * N[K+k] for k in range(n_bg)], axis=0)
            ax.plot(x, y, label=background)
        elif background != 'none':
            ax.plot(x, model[-1].predict(x) * N[-1], label=background)
        
        # Full model and data
        ax.plot(x, predict_func(x) * N_tot, 'black', linewidth=3, ls='--', label='full_model')
        ax.scatter(x_data, intensity, label='data')
        ax.set_xlabel('Energy [eV]')
        ax.set_ylabel('Intensity')
        ax.legend()
        
        if show:
            plt.show()
        
        return fig
