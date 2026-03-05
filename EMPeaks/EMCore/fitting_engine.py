# License: BSD-3-clause
# Copyright © 2023 National Institute of Advanced Industrial Science and Technology (AIST)
# Author: Yasunobu ANDO

"""フィッティングアルゴリズムを担当するクラス"""

import time
from typing import Dict, List, Callable, Any

import numpy as np


class FittingEngine:
    """EMアルゴリズムのフィッティングを担当するクラス
    
    EMCoreから呼び出されるヘルパークラスとして機能し、
    フィッティングアルゴリズムのコアロジックを提供します。
    
    Note:
        このクラスはEMCoreの状態更新メソッド（e_step, m_step等）を
        コールバックとして受け取り、アルゴリズムのフローを制御します。
    """
    
    @staticmethod
    def run_adapted_em(
        max_iter: int,
        r_eps: float,
        stdout: bool,
        log_likelihood_func: Callable,
        e_step_func: Callable,
        m_step_func: Callable,
        init_param_func: Callable,
        x: np.ndarray,
        intensity: np.ndarray
    ) -> Dict[str, Any]:
        """Adapted EMアルゴリズムを実行
        
        Args:
            max_iter: 最大イテレーション数
            r_eps: 収束判定の閾値
            stdout: 標準出力に進捗を表示するか
            log_likelihood_func: 対数尤度を計算する関数
            e_step_func: E-stepを実行する関数
            m_step_func: M-stepを実行する関数
            init_param_func: パラメータを初期化する関数
            x: X座標データ
            intensity: 強度データ
        
        Returns:
            Dict containing:
                - total_iter: 総イテレーション数
                - total_time: 総実行時間
                - time_per_iter: イテレーションあたりの時間
                - LL: 最終対数尤度
                - LL_hist: 対数尤度の履歴
                - LL_residual: 最終残差
                - LL_residual_hist: 残差の履歴
                - converged: 収束したかどうか
        """
        start = time.time()
        it_tot = 0
        t_tot = 0
        LL_hist = []
        res_hist = []
        ll_0 = log_likelihood_func(x, intensity)
        converged = False
        ll = 0.0
        residual = 0.0

        print("<< Start fitting via Adapted EM Algorithm. >>")
        t1 = time.time()

        for it in range(max_iter):
            e_step_func(x)
            m_step_func(x, intensity)

            ll = log_likelihood_func(x, intensity)
            residual = (ll - ll_0) / np.abs(ll_0)
            t2 = time.time()

            if stdout:
                if it % 10 == 0:
                    print("> iteration #{:3d}, LL={:10.8e}, residual={:4.3e}, elapsed time: {:5.2f} s"
                          .format(it, ll, residual, t2 - t1))
                    t1 = time.time()

            LL_hist.append(ll)
            res_hist.append(residual)

            if residual < 0.0:
                print("Warning!!!! residual is negative!!!  Parameters are initialized again.")
                init_param_func()
            elif residual < r_eps:
                t_tot = time.time() - start
                it_tot = it

                print('Convergence is achieved at iter. {:3d}, elapsed time {:5.2f} s'
                      .format(it, t_tot))
                print('   LogLikelihood:     {:12.8e}\n'
                      '        residual:      {:12.8e}'.format(ll, residual))
                converged = True
                break
            ll_0 = ll

        if not converged:
            t_tot = time.time() - start
            it_tot = max_iter
            print('>>> Convergence is not achieved within the iteration of {:3d}, elapsed time: {:5.2f} s'
                  .format(max_iter, t_tot))
            print('   LogLikelihood:      {:12.8e}\n'
                  '        residual:       {:12.8e}'.format(ll, residual))

        return {
            'total_iter': (it_tot + 1),
            'total_time': t_tot,
            'time_per_iter': t_tot / (it_tot + 1) if it_tot > 0 else 0,
            'LL': ll,
            'LL_hist': LL_hist,
            'LL_residual': residual,
            'LL_residual_hist': res_hist,
            'converged': converged
        }
