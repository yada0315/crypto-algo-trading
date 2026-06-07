#!/usr/bin/env python3
"""
production/run_donchian_hl.py

X1 Donchian(LS) 戦略の Hyperliquid perp 本番稼働スクリプト。

【戦略概要】Candidate A (提案2: Bull/Bear 条件付きショート + FR Z-score ブレンド)
  - シグナル: Donchianチャネル内ポジション
      score = (close - m_low日安値) / (n_high日高値 - m_low日安値) - 0.5
  - ロング(Bull時のみ): score 上位 TOP_PCT_L=15% (SMA100 & 出来高フィルタ & BTC Regime)
  - ショート:
      Bull 時: score_short = score - CARRY_WEIGHT_BULL * FR_zscore
               vol_ok フィルタ候補から下位 TOP_PCT_S_BULL=15% を選択
      Bear 時: pure Donchian score で下位 TOP_PCT_S_BEAR=20% 分位以下のみ候補
               そこから下位 TOP_PCT_S_BULL=15% を選択 (集中ショート)
  - サイジング: ATRベース (rf_l / ATR, rf_s / ATR)
  - リバランス: 毎週フルリバランス（全4トランシェ do_reb=True）

【追加フィルタ】
  [A] BTC Regime フィルタ: BTC < SMA200 (Bear) のときロングを全停止
  [B] 出来高フィルタ: vol(5日MA) / vol(20日MA) >= 0.7 のみ Long/Short エントリー
  [C] FR Z-score ブレンド (Bull時ショートのみ):
      FR z-score = (daily_FR - rolling_mean(30d)) / rolling_std(30d), clip(±3)
      クロスセクション正規化後、score_short = score - 0.5 * FR_zscore
      → 高いファンディングレートの銘柄ほどショート優先

【バックテスト結果 (Candidate A, IS: ~2024-09, OOS: 2024-10~)】
  IS  Sharpe=+2.02  APR=+69.6%  MaxDD=-28.2%  Calmar=2.47
  OOS Sharpe=+5.64  APR=+335.5% MaxDD=-8.8%   Calmar=37.98

【実行方法】
  # ドライラン（デフォルト）
  python production/run_donchian_hl.py

  # 本番実行（実際に注文する）
  python production/run_donchian_hl.py --live

  # スケジュール実行（cron）: n=4 トランシェ方式 — 各曜日に独立してリバランス
  #   サーバータイムゾーン: JST (Asia/Tokyo, UTC+9)
  #   0 23 * * 3 cd /path && python production/run_donchian_hl.py --live  # tranche 0: 水曜 23:00 JST = 14:00 UTC
  #   0 23 * * 4 cd /path && python production/run_donchian_hl.py --live  # tranche 1: 木曜 23:00 JST = 14:00 UTC
  #   0 23 * * 6 cd /path && python production/run_donchian_hl.py --live  # tranche 2: 土曜 23:00 JST = 14:00 UTC
  #   0 23 * * 1 cd /path && python production/run_donchian_hl.py --live  # tranche 3: 月曜 23:00 JST = 14:00 UTC
  #   ※ 実行曜日からトランシェ ID を自動検出。--tranche INT で手動指定も可能。

  # 週番号リセット（初回・再起動時）
  python production/run_donchian_hl.py --reset-state --force-reb

【必要な .env 設定】
  PRIVATE_KEY=0x...     # Hyperliquid ウォレット秘密鍵
  PUBLIC_ADDRESS=0x...  # ウォレットアドレス（省略可）

【依存パッケージ】
  pip install hyperliquid-python-sdk eth-account python-dotenv requests pandas numpy
"""

# ──────────────────────────────────────────────────────────────────────────────
# 標準ライブラリ
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import time
import fcntl
import logging
import argparse
import traceback
from dataclasses import dataclass, field
from typing import Literal
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# サードパーティ
# ──────────────────────────────────────────────────────────────────────────────
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# シグナルハンドラ (SIGTERM / SIGINT による graceful shutdown)
# ──────────────────────────────────────────────────────────────────────────────
import signal as _signal

_shutdown_requested = False

def _handle_shutdown(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True

_signal.signal(_signal.SIGTERM, _handle_shutdown)
_signal.signal(_signal.SIGINT,  _handle_shutdown)

# ──────────────────────────────────────────────────────────────────────────────
# パス設定
# ──────────────────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent
_PROD_DIR   = Path(__file__).resolve().parent
_ENV_PATH   = _PROD_DIR / '.env'
_STATE_PATH = _PROD_DIR / 'donchian_state.json'
_LOG_PATH   = _PROD_DIR / 'donchian.log'
_LOCK_PATH  = _PROD_DIR / 'donchian.lock'
_EXEC_LOG_PATH = _ROOT / 'data' / 'live' / 'execution_log.csv'

# ─── 実行ログ ─────────────────────────────────────────────────────────────────
_EXEC_LOG_FIELDS = [
    'exec_datetime', 'eval_date', 'tranche_id',
    'coin', 'side', 'order_type', 'reason',
    'target_qty', 'target_px', 'notional_usd',
    'method',          # ALO_filled / ALO_resting / ALO_timeout_IOC / ALO_rejected_IOC / IOC_direct
    'final_status',    # filled / partial / unfilled / error
    'filled_qty',
    'slippage_pct',    # (limit_px - target_px) / target_px (IOC の場合)
]

def _write_exec_rows(rows: list) -> None:
    """execution_log.csv に行を追記する（ファイルなければ新規作成）"""
    if not rows:
        return
    import csv
    _EXEC_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _EXEC_LOG_PATH.exists()
    with open(_EXEC_LOG_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_EXEC_LOG_FIELDS, extrasaction='ignore')
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

# ──────────────────────────────────────────────────────────────────────────────
# ===== 戦略パラメータ（ロバスト性検証に基づく推奨パラメータ 2026-04-06） =====
# 旧値: N_HIGH=126, M_LOW=20, RF_L=0.020, tpl=20%, tps=30% → IS Sh=1.610
# 中間: N_HIGH=168, M_LOW=15, RF_L=0.015, tpl=20%, tps=30% → IS Sh=2.188, OOS Sh=2.138
# 前値: Candidate A (+提案2 FR ブレンド) RF_L=0.015, cw=0.7, tpl=15%, tps=15%
#        → IS Sh=2.02, OOS Sh=5.64, OOS MaxDD=-8.8%
# v2前: ロバスト性検証 (Step1-4 + WFO) 推奨パラメータ
#        RF_L=0.008, cw=0.5, tpl=10%, tps=10%, M_LOW=15, MEQ_LEV_LOW=0.7
#        → Full Sh=2.703, Bootstrap P10=1.246, TC感度低下率=7.5%, WFO最小=-0.876
# 現値: v2グリッドサーチ + ロバスト性検証 総合スコア#1 (2026-04-06)
#        M_LOW=10, cw=0.3, tpl=15%, tps=10%, MEQ_LEV_LOW=0.5, RF_S_BEAR=0.005
#        → Full Sh=3.262, OOS Sh=3.222, Bootstrap P10=1.604, MC p=0.002
#           近傍最小Sharpe=2.103 (感度低), 全期間Bootstrap Positive率=100%
# ──────────────────────────────────────────────────────────────────────────────
# シグナル
N_HIGH          = 168      # Donchian 上限バンド期間 (日)  close.rolling(N_HIGH).max()
M_LOW           = 10       # Donchian 下限バンド期間 (日)  low.rolling(M_LOW).min()  (旧: 15)

# インディケータ
SMA_PERIOD      = 100      # SMA フィルタ期間 (日)
ATR_L           = 20       # ロング ATR 期間 (日)
ATR_S           = 10       # ショート ATR 期間 (日)
REG_L           = 120      # ロング max_daily_return 期間 (日)
REG_S           = 10       # ショート max_daily_return 期間 (日)
MAX_DAILY_MOVE  = 0.40     # 日次変動フィルタ (40% 超は除外)

# ポジション選択
TOP_PCT_L       = 0.15     # score 上位 15% をロング候補 (旧: 0.10)
TOP_PCT_S_BULL  = 0.10     # Bull時ショート選択率 10%
TOP_PCT_S_BEAR  = 0.20     # Bear時ショート事前フィルタ分位点 (下位20%のみ候補)

# [C] FR Z-score ブレンド
CARRY_WEIGHT_BULL = 0.3    # Bull時: score_short = score - 0.3 * FR_zscore  (旧: 0.5)
CARRY_WINDOW      = 30     # FR z-score のローリング窓 (日)
FR_FETCH_DAYS     = 20     # HL API から取得するファンディングレートの日数
                           # ※ HL fundingHistory API は 1リクエスト最大 500 件を返す。
                           #   BTC等の1時間FR銘柄: 24件/日 × 20日 = 480件 < 500件上限 → 全期間取得可能
                           #   20日を超えると古いデータから切り取られ直近データが欠損する。
                           #   CARRY_WINDOW=30 より短いが rolling(min_periods=1) で対応。

# サイジング
RF_L            = 0.008    # ロング: equity × RF_L / ATR = target_qty  (旧: 0.015)
RF_S            = 0.005    # ショート (Bear時): equity × RF_S / ATR = target_qty  (旧: 0.003)
RF_S_BULL       = 0.003    # ショート (Bull時): equity × RF_S_BULL / ATR  (旧: 0.002)

# リスク上限
MAX_GROSS_LEV          = 1.0   # グロスレバレッジ上限 (DYNLEV_ENABLED=False 時のフォールバック)
MIN_ORDER_USD          = 11.0  # 最小注文額 USD (HL 最小 $10 を余裕で上回る)
HYPE_COLLATERAL_HAIRCUT = 0.95  # Spot HYPE を証拠金として使う際のヘアカット率 (HL 基準)

# ── 動的レバレッジ EWM Sharpe (案2: 指数加重移動平均 Sharpe-adaptive + Rolling-DD Guard) ────
# v3 グリッドサーチ最良パラメータ (案2 EWM Sharpe):
#   DDpen=5.50, WFO中央値=7.43, WFO最小=0.95, 全窓正率=100%
#   ewm_span=3, ts=0.3, dd=0.10, roll=6, lev_max=3.5 (Bear期もショート継続)
#   ※ Phase A (単純3週平均) から変更: 暴落時も過去の蓄積Sharpeが残るため stuck-at-0 を回避
DYNLEV_ENABLED       = False  # True: EWM動的レバレッジ ON / False: MAX_GROSS_LEV固定
DYNLEV_EWM_SPAN      = 3      # EWM Sharpe の span (α = 1 - 2/(span+1) = 0.5)
DYNLEV_TARGET_SHARPE = 0.3    # 目標年率Sharpe: ewm_sharpe / target で sharpe_lev をスケール
DYNLEV_DD_THRESH     = 0.10   # DDガード閾値: rolling_high から 10% 下落で leverage → 0
DYNLEV_ROLL_W        = 6      # rolling high の窓 (週)
DYNLEV_LEV_MAX       = 3.5    # グロスレバレッジ絶対上限
DYNLEV_LEV_MIN       = 0.0    # グロスレバレッジ下限

# ── M_eq (Equity Trend Filter) ──────────────────────────────────────────────
# ペーパーエクイティ(1x固定ベースライン)のSMAトレンドでレバレッジを切り替える
# グリッドサーチ最良: sma=13w, lh=2.5x, ll=0.7x → 近傍安定性分析による更新 (2026-03-28)
MEQ_ENABLED    = True   # True: M_eq ON / False: レバレッジ変更なし
MEQ_SMA_WINDOW = 13     # SMA窓 (週)
MEQ_LEV_HIGH   = 2.5    # ペーパーeq > SMA の時のレバレッジ倍率
MEQ_LEV_LOW    = 0.5    # ペーパーeq <= SMA の時のレバレッジ倍率  (旧: 0.7)
MEQ_MIN_WEEKS  = 7      # 最小履歴週数 (不足時はデフォルト 1.0x)

# ── 追加フィルタ ────────────────────────────────────────────────────────────
# [A] BTC Regime フィルタ: Bear (BTC < SMA200) 時はロングポジションを取らない
BTC_SMA_PERIOD  = 200      # BTC SMA期間 (日)  close.rolling(200).mean()

# [B] 出来高フィルタ: 直近出来高が低下している銘柄はエントリーしない
#     条件: vol(VOL_SHORT日MA) / vol(VOL_LONG日MA) >= VOL_THRESH
VOL_SHORT       = 5        # 直近出来高平均窓 (日)
VOL_LONG        = 20       # ベース出来高平均窓 (日)
VOL_THRESH      = 0.7      # 閾値: この値以上のみエントリー許可 (Long+Short 両方に適用)

# データ
DATA_DAYS       = 250      # 過去何日の日足を取得するか (N_HIGH=168 + BTC_SMA_PERIOD=200 に十分な量)
# LISTING_MIN_DAYS は廃止: 上場期間フィルタなし
# データ不足銘柄は fetch_ohlcv_all で自動スキップ (N_HIGH=168日未満はシグナル計算不可)

# ── ブラックリスト ────────────────────────────────────────────────────────────
# 取引停止・OI上限・その他の理由で HL での注文が恒常的に失敗する銘柄を除外する
# 追加理由:
#   FXS: "Trading is halted." が繰り返し発生 (2026-03-02 ログ確認 4回)
COIN_BLACKLIST: set = {
    'FXS',
}

# 注文設定 — Strategy C: ATR 連動 ALO 指値 → リプライス → IOC taker
# ─────────────────────────────────────────────────────────────────────
# 【執行アルゴリズム: Strategy C】
#   OPEN / CLOSE ともに ALO (Post-Only) 指値を出し maker を狙う。
#   CLOSE は OPEN より mid に近いオフセット・短いタイムアウトで管理する。
#
#   initial_offset = clip(ATR/close × ATR_RATIO, MIN_OFFSET, MAX_OFFSET)
#   BUY:  limit = mid × (1 − offset)   offset は 0 に向かって線形縮小
#   SELL: limit = mid × (1 + offset)
#   タイムアウト到達 or offset≈0 で IOC (taker) にフォールバック。
#
#   [OPEN バックテスト結果]
#   Sharpe:  1.474 (IOC only) → 1.773 (Strategy C ALO)
#   APR:    +107% → +130%、1注文コスト +0.149% → −0.049%

# OPEN 執行パラメータ
C_ATR_RATIO     = 0.20     # offset = ATR/close × 0.20
C_MIN_OFFSET    = 0.002    # 最小オフセット: 0.2%
C_MAX_OFFSET    = 0.010    # 最大オフセット: 1.0%
C_TIMEOUT_MIN   = 120      # X1: タイムアウト上限 (分) — 流動性大なら短縮
C_MIN_TIMEOUT   = 30       # X1: タイムアウト下限 (分)
C_CHECK_EVERY   = 30       # 確認・リプライス間隔 (分)
C_TWAP_SPLITS   = 3        # X3: TWAP 分割数 (未使用)
C_TWAP_INTERVAL = 2        # X3: TWAP 分割間隔 (分)

# CLOSE 執行パラメータ (OPEN より mid に近く、タイムアウトを短くする)
CLOSE_ATR_RATIO   = 0.10   # OPEN の半分のオフセット
CLOSE_MIN_OFFSET  = 0.001  # 最小オフセット: 0.1%
CLOSE_MAX_OFFSET  = 0.005  # 最大オフセット: 0.5%
CLOSE_TIMEOUT_MIN = 15     # 15分で IOC に切り替え (exit の遅延を抑制)
CLOSE_CHECK_EVERY = 5      # 5分ごとに確認・リプライス

# ── n=4 トランシェ設定 ─────────────────────────────────────────────────────
# ポートフォリオを TRANCHE_N 等分し、各トランシェを異なる曜日にリバランス
# バックテスト結果: n=1 Sh=1.48 → n=4 Sh=2.27 (+53%), MaxDD -59.6% → -50.8%
TRANCHE_N         = 4          # トランシェ数
TRANCHE_EVAL_DAYS = [2, 3, 5, 0]  # 各トランシェの評価曜日 (0=月,...,6=日)
                                    # tranche 0=水(2), 1=木(3), 2=土(5), 3=月(0)

# X1: 出来高ティアと注文方向に基づく動的タイムアウト (時間)
# 時間出来高 USD ティア → base timeout (h):
#   > $10M/h → 1.0h (流動性大)  |  > $1M/h → 1.5h  |  その他 → 2.0h
# 方向: BUY = base−0.5h (短め, モメンタム追随), SELL = base (長め, 売り急がない)
X1_VOL_TIERS      = [(10_000_000, 1.0), (1_000_000, 1.5)]  # (USD/h threshold, base_h)
X1_VOL_DEFAULT_H  = 2.0        # 低流動性デフォルト (時間)
X1_BUY_SUBTRACT_H = 0.5        # BUY のタイムアウト短縮量 (時間)
LIMIT_SLIPPAGE  = 0.020    # taker フォールバック用 IOC スリッページ (2.0%)

# API
HL_INFO_URL     = "https://api.hyperliquid.xyz/info"
REQUEST_DELAY   = 0.5      # API 呼び出し間隔 (秒) — 0.25→0.5 に変更 (429対策)

# ──────────────────────────────────────────────────────────────────────────────
# ロギング設定
# ──────────────────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    """ファイル + コンソール 両方に出力するロガーを作成する。"""
    logger = logging.getLogger('donchian_bot')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # root logger への伝播を防ぎ、SDK が basicConfig を呼んでも2重出力にならないようにする

    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)-5s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # コンソールハンドラ (INFO 以上)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ファイルハンドラ (DEBUG 以上) — 10MB × 10世代でローテーション
    log_path.parent.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(
        str(log_path), encoding='utf-8',
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────────────────────────────────
# State 管理（週番号 / do_reb 判定）
# ──────────────────────────────────────────────────────────────────────────────
def _default_state() -> dict:
    """デフォルト（初回）state を返す。"""
    return {
        'tranche_targets': {str(i): {} for i in range(TRANCHE_N)},
        'last_run':        None,
        'last_tranche':    None,
        'equity_history':  [],   # A_R 動的レバレッジ用: 週次 equity 履歴 (最大52週)
    }


def load_state(state_path: Path) -> dict:
    """
    実行状態を JSON ファイルから読み込む。
    初回実行時またはフォーマット移行時はデフォルト状態を返す。

    ファイルが存在しない場合はデフォルト状態を返す。
    """
    def _try_load(path: Path):
        with open(path, 'r', encoding='utf-8') as f:
            s = json.load(f)
        return s

    if state_path.exists():
        try:
            return _try_load(state_path)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # JSON 破損: バックアップから復旧を試みる
            bak_path = state_path.with_suffix('.bak')
            if bak_path.exists():
                try:
                    state = _try_load(bak_path)
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        f'[State] {state_path} が破損 ({e}) → {bak_path} から復旧'
                    )
                    return state
                except Exception:
                    pass
            import logging as _logging
            _logging.getLogger(__name__).error(
                f'[State] {state_path} が破損かつバックアップなし → デフォルト状態で起動: {e}'
            )
    return _default_state()


def save_state(state_path: Path, state: dict) -> None:
    """
    実行状態を JSON ファイルに原子的に書き込む。

    手順: tmp ファイルに書き出し → flush/fsync → rename
    これにより書き込み途中のプロセス終了で JSON が破損しない。
    前回の正常ファイルは .bak として保持し、破損時の復旧に使用する。
    """
    import os
    tmp_path = state_path.with_suffix('.tmp')
    bak_path = state_path.with_suffix('.bak')
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        # 旧ファイルをバックアップに退避してから tmp をリネーム
        if state_path.exists():
            state_path.rename(bak_path)
        tmp_path.rename(state_path)
    except Exception:
        # tmp が残っていれば削除
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def compute_dynamic_leverage(equity_history: list, logger: logging.Logger,
                              paper_equity_history: list = None) -> float:
    """
    A_R (Sharpe-adaptive + Rolling-DD Guard) 動的レバレッジを計算する。

    equity_history      : 各週実行開始時の実際の equity 値リスト（DD ガード用）
    paper_equity_history: DynLev スケールなし（ベースライン固定レバ）での仮想 equity 値リスト
                          バックテスト案2と同様に「素のリターン」で EWM Sharpe を計算する。
                          None の場合は equity_history にフォールバック（後方互換）。

    Returns
    -------
    eff_lev : float in [DYNLEV_LEV_MIN, DYNLEV_LEV_MAX]
        今週適用するグロスレバレッジ上限
    """
    if not DYNLEV_ENABLED:
        return MAX_GROSS_LEV

    if len(equity_history) < 2:
        logger.info(f'[DynLev] 履歴不足 ({len(equity_history)}週) → デフォルト {MAX_GROSS_LEV:.1f}x')
        return MAX_GROSS_LEV

    # DD ガード用: 常に実際の equity_history を使う
    eq = pd.Series([float(v) for v in equity_history], dtype=float)

    # ── EWM Sharpe ベースレバレッジ (案2: ペーパーポートフォリオ方式) ──────
    # EWM Sharpe の入力には paper_equity_history を優先する。
    # → DynLev=0 で取引停止中でもベースライン戦略の損益が入り stuck-at-0 を防止。
    # → paper_equity_history がない場合は equity_history にフォールバック。
    sharpe_src = (paper_equity_history
                  if (paper_equity_history and len(paper_equity_history) >= 2)
                  else equity_history)
    eq_s = pd.Series([float(v) for v in sharpe_src], dtype=float)
    wr_s = eq_s.pct_change().dropna()
    ewm_mean   = float(wr_s.ewm(span=DYNLEV_EWM_SPAN, adjust=False).mean().iloc[-1])
    ewm_std    = float(wr_s.ewm(span=DYNLEV_EWM_SPAN, adjust=False).std().iloc[-1])
    if ewm_std > 1e-8:
        rolling_sharpe = ewm_mean / ewm_std * np.sqrt(52)
    else:
        rolling_sharpe = 0.0
    sharpe_lev = float(np.clip(rolling_sharpe / DYNLEV_TARGET_SHARPE,
                               DYNLEV_LEV_MIN, DYNLEV_LEV_MAX))

    # ── Rolling DD ガード (実際の equity_history で計算) ─────────────────
    rolling_high = eq.rolling(DYNLEV_ROLL_W, min_periods=1).max()
    dd_current   = float((eq / rolling_high - 1).iloc[-1])
    guard        = float(np.clip(1.0 + dd_current / DYNLEV_DD_THRESH, 0.0, 1.0))

    # ── 有効レバレッジ ────────────────────────────────────────────────────
    eff_lev = float(np.clip(sharpe_lev * guard, DYNLEV_LEV_MIN, DYNLEV_LEV_MAX))

    src_label = f'paper({len(sharpe_src)}週)' if sharpe_src is not equity_history else f'actual({len(equity_history)}週)'
    logger.info(
        f'[DynLev] Sharpe入力={src_label} | '
        f'EWM_Sharpe(span={DYNLEV_EWM_SPAN})={rolling_sharpe:.2f} → sharpe_lev={sharpe_lev:.3f} | '
        f'DD={dd_current:.1%}(actual {len(equity_history)}週) → guard={guard:.3f} | '
        f'eff_lev={eff_lev:.3f}x (上限={DYNLEV_LEV_MAX}x)'
    )
    return eff_lev


def compute_meq_leverage(paper_hist: list, logger: logging.Logger) -> float:
    """
    M_eq (Equity Trend Filter): ペーパーエクイティのSMAトレンドでレバレッジを切り替える。

    paper_hist: list of float — ペーパーエクイティの週次履歴 (最新が末尾)
    Returns: float — 今週適用するレバレッジ倍率 (MEQ_LEV_HIGH or MEQ_LEV_LOW or 1.0)
    """
    if not MEQ_ENABLED:
        return 1.0

    if len(paper_hist) < MEQ_MIN_WEEKS:
        logger.info(
            f'[MEQ] 履歴不足 ({len(paper_hist)}週 < {MEQ_MIN_WEEKS}週) → デフォルト 1.0x'
        )
        return 1.0

    eq_s = pd.Series([float(v) for v in paper_hist], dtype=float)
    # SMA: min_periods を SMA_WINDOW//2+1 として不完全な窓でも計算可
    sma_val = float(eq_s.rolling(MEQ_SMA_WINDOW, min_periods=MEQ_SMA_WINDOW // 2 + 1).mean().iloc[-1])
    current = float(eq_s.iloc[-1])

    if np.isnan(sma_val):
        logger.info(f'[MEQ] SMA計算不可 (履歴={len(paper_hist)}週) → デフォルト 1.0x')
        return 1.0

    if current > sma_val:
        lev = MEQ_LEV_HIGH
        trend = 'Bull (eq > SMA)'
    else:
        lev = MEQ_LEV_LOW
        trend = 'Bear (eq <= SMA)'

    logger.info(
        f'[MEQ] paper_eq={current:,.2f} SMA{MEQ_SMA_WINDOW}={sma_val:,.2f} → {trend} → {lev:.1f}x '
        f'(履歴={len(paper_hist)}週)'
    )
    return lev


# ──────────────────────────────────────────────────────────────────────────────
# Hyperliquid API (認証不要の読み取り)
# ──────────────────────────────────────────────────────────────────────────────
def _hl_post(payload: dict, timeout: int = 30, max_retries: int = 5) -> object:
    """Hyperliquid Info API に POST して JSON を返す。429 時は指数バックオフでリトライ。"""
    for attempt in range(max_retries):
        resp = requests.post(HL_INFO_URL, json=payload, timeout=timeout)
        if resp.status_code == 429:
            wait = 2 ** attempt  # 1, 2, 4, 8, 16 秒
            logging.getLogger('donchian_bot').warning(
                f'API 429 レート制限 (attempt {attempt+1}/{max_retries}) → {wait}秒待機'
            )
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()  # 最終試行失敗時は例外を投げる
    return resp.json()


def get_hl_universe() -> dict:
    """
    HL の全 perp 銘柄メタ情報を返す。

    Returns:
        {coin_name: {'szDecimals': int, 'maxLeverage': int}}
    """
    data = _hl_post({'type': 'metaAndAssetCtxs'})
    result = {}
    for asset in data[0]['universe']:
        result[asset['name']] = {
            'szDecimals':  asset['szDecimals'],
            'maxLeverage': asset['maxLeverage'],
        }
    return result


def fetch_daily_candles(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Hyperliquid から日足 OHLCV を取得する。

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume]
        取得失敗 or データなし → 空 DataFrame
    """
    payload = {
        'type': 'candleSnapshot',
        'req': {
            'coin':      coin,
            'interval':  '1d',
            'startTime': int(start_ms),
            'endTime':   int(end_ms),
        },
    }
    try:
        data = _hl_post(payload)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        # HL フィールド: t=open_time_ms, T=close_time_ms, o/h/l/c/v
        df = df.rename(columns={
            't': 'timestamp', 'o': 'open', 'h': 'high',
            'l': 'low',       'c': 'close', 'v': 'volume',
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.normalize().dt.tz_localize(None)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        import logging as _logging
        _logging.getLogger(__name__).warning(f'fetch_daily_candles({coin}): {type(e).__name__}: {e}')
        return pd.DataFrame()


def fetch_funding_rates_recent(
    coins: list,
    n_days: int,
    logger: logging.Logger,
    workers: int = 3,
) -> pd.DataFrame:
    """
    全銘柄の直近 n_days 日分の 8h ファンディングレートを取得し、
    日次平均 DataFrame (date × coin) を返す。
    変更9: ThreadPoolExecutor(workers=3) で並列取得。

    API: {"type": "fundingHistory", "coin": coin, "startTime": ms}
    取得失敗銘柄はゼロ埋め。全銘柄失敗時は空 DataFrame を返す。
    """
    import concurrent.futures

    now_utc  = datetime.now(timezone.utc)
    start_ms = int((now_utc - timedelta(days=n_days)).timestamp() * 1000)
    end_ms   = int(now_utc.timestamp() * 1000)

    logger.info(f"ファンディングレート取得中: {len(coins)} 銘柄 × 過去 {n_days} 日 (workers={workers}) ...")

    def _fetch_one_fr(coin: str):
        time.sleep(REQUEST_DELAY)  # スレッドごとに待機して 429 を抑制
        payload = {
            'type':      'fundingHistory',
            'coin':      coin,
            'startTime': start_ms,
            'endTime':   end_ms,
        }
        try:
            data = _hl_post(payload)
            if not data:
                return coin, {}, False
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True)
            df['timestamp'] = df['timestamp'].dt.normalize().dt.tz_localize(None)
            df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
            daily = df.groupby('timestamp')['fundingRate'].mean()
            return coin, daily.to_dict(), True
        except Exception as e:
            logger.debug(f"  {coin}: FR 取得失敗 → ゼロ埋め ({e})")
            return coin, {}, False

    daily_dict = {}
    ok_cnt     = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fetch_one_fr, coin) for coin in coins]
        for future in concurrent.futures.as_completed(futures):
            coin, result, success = future.result()
            daily_dict[coin] = result
            if success:
                ok_cnt += 1

    if ok_cnt == 0:
        logger.warning('ファンディングレート取得: 有効データ 0 銘柄 → FR ブレンドを無効化')
        return pd.DataFrame()

    # date index を統一して DataFrame 化
    all_dates = sorted({d for v in daily_dict.values() for d in v.keys()})
    if not all_dates:
        return pd.DataFrame()
    date_idx = pd.DatetimeIndex(all_dates)
    # fill_value は NaN を保持 (0.0 埋めはクロスセクション z-score を歪めるため)
    # z-score 計算時に valid 銘柄のみで平均・分散を算出し、欠損は最後に 0 埋めする
    fr_df = pd.DataFrame(
        {coin: pd.Series(vals, index=pd.DatetimeIndex(list(vals.keys())))
                .reindex(date_idx)  # NaN 保持
         for coin, vals in daily_dict.items()},
    )
    logger.info(
        f"ファンディングレート取得完了: {ok_cnt}/{len(coins)} 銘柄有効, "
        f"{len(fr_df)} 日分 ({fr_df.index[0].date()} ~ {fr_df.index[-1].date()})"
    )
    return fr_df


def compute_fr_zscore_latest(
    fr_daily: pd.DataFrame,
    window: int,
    coins: list,
) -> dict:
    """
    日次 FR DataFrame から最終日のクロスセクション z-score を計算し {coin: float} で返す。

    計算式 (バックテスト backtest_improvements_abcde.py の fr_zscore() と完全一致):
      1. fr_smooth = fr.rolling(window, min_periods=1).mean()   ← 時系列スムージング
      2. mu    = fr_smooth.mean(axis=1)                         ← クロスセクション平均
      3. sigma = fr_smooth.std(axis=1)                          ← クロスセクション標準偏差
      4. z     = (fr_smooth - mu) / sigma                       ← クロスセクション正規化
      5. 最終行を {coin: zscore} として返す

    ※ FRの「絶対水準」が他コインより高いほど高スコア → ショート優先
       (自身の歴史との乖離ではなく、今日の断面的な高さを測定)

    FR データが不足 or 空の場合は {coin: 0.0} を返す。
    """
    if fr_daily.empty or len(fr_daily) < 2:
        return {c: 0.0 for c in coins}

    # NaN を保持したままスムージング → z-score を計算し、欠損銘柄は最後に 0 埋め
    # (先に fillna(0) するとクロスセクション平均・分散が歪むため)
    fr = fr_daily.reindex(columns=coins).infer_objects(copy=False)  # NaN を保持

    # バックテスト fr_zscore() と同一: ローリング平均でスムージング → クロスセクション正規化
    fr_smooth = fr.rolling(window, min_periods=1).mean()
    mu        = fr_smooth.mean(axis=1, skipna=True)             # 欠損銘柄を除いた平均
    sigma     = fr_smooth.std(axis=1, skipna=True).replace(0.0, np.nan)  # 欠損銘柄を除いた標準偏差
    z_cs      = fr_smooth.subtract(mu, axis=0).divide(sigma, axis=0).fillna(0.0)
    z_cs      = z_cs.clip(-3, 3)  # 極端な値を±3に制限
    # fillna(0.0) により欠損銘柄は中立スコア 0 に統一

    last = z_cs.iloc[-1]
    return {c: float(last.get(c, 0.0)) if pd.notna(last.get(c)) else 0.0
            for c in coins}


def get_all_mids() -> dict:
    """全銘柄の現在 Mid 価格を返す。{coin: float}"""
    data = _hl_post({'type': 'allMids'})
    return {k: float(v) for k, v in data.items() if v}


def get_account_state(address: str, logger: logging.Logger = None) -> dict:
    """
    指定アドレスのアカウント状態を返す。
    Unified Account の場合、Perp 残高が 0 でもスポット USDC 残高を equity として使用。

    Returns:
        {'equity': float, 'positions': {coin: {'qty': float, 'entry_px': float}}}
    """
    data = _hl_post({'type': 'clearinghouseState', 'user': address})

    perp_equity = float(data['marginSummary']['accountValue'])

    # Unified Account 対応: Spot USDC + Spot HYPE (証拠金設定時) を equity として使用
    # (Unified Account では Spot 資産がそのまま Perp のコラテラルになるが、
    #  perp_equity (marginSummary.accountValue) は Perp 側の損益のみを示す場合がある)
    spot_data = _hl_post({'type': 'spotClearinghouseState', 'user': address})
    spot_usdc     = 0.0
    spot_hype_qty = 0.0
    for b in spot_data.get('balances', []):
        coin_name = b.get('coin', '')
        if coin_name == 'USDC':
            spot_usdc = float(b.get('total', 0.0))
        elif coin_name == 'HYPE':
            spot_hype_qty = float(b.get('total', 0.0))

    # Spot HYPE を USD 換算 (ヘアカット適用)
    spot_hype_usd = 0.0
    if spot_hype_qty > 0:
        all_mids = _hl_post({'type': 'allMids'})
        hype_price = float(all_mids.get('HYPE', 0.0))
        spot_hype_usd = spot_hype_qty * hype_price * HYPE_COLLATERAL_HAIRCUT

    # Spot 合計 (USDC + HYPE換算)
    spot_total = spot_usdc + spot_hype_usd

    # Unified Account の equity 判定:
    #   spot_total (USDC + HYPE haircut後) と perp_equity の大きい方を採用
    #   ・perp_equity が大きい場合: Perp側に損益が蓄積されている
    #   ・spot_total が大きい場合: Spot資産がコラテラルの大半
    equity = max(perp_equity, spot_total)
    if logger:
        logger.info(
            f'  Unified Account: '
            f'SpotUSDC=${spot_usdc:,.2f} + SpotHYPE={spot_hype_qty:.2f}枚×haircut=${spot_hype_usd:,.2f} '
            f'= SpotTotal=${spot_total:,.2f} | Perp残高=${perp_equity:,.2f} '
            f'→ equity=max(...)=${equity:,.2f}'
        )

    positions = {}
    for p in data.get('assetPositions', []):
        pos  = p['position']
        coin = pos['coin']
        qty  = float(pos['szi'])          # 正=ロング, 負=ショート
        if qty == 0.0:
            continue
        entry_px = float(pos['entryPx']) if pos.get('entryPx') else 0.0
        positions[coin] = {'qty': qty, 'entry_px': entry_px}

    return {'equity': equity, 'positions': positions}



# ──────────────────────────────────────────────────────────────────────────────
# OHLCV 取得
# ──────────────────────────────────────────────────────────────────────────────
def fetch_ohlcv_all(
    coins: list,
    n_days: int,
    logger: logging.Logger,
    workers: int = 3,
) -> dict:
    """
    全対象銘柄の日足 OHLCV を取得し、DataFrame の辞書で返す。
    変更9: ThreadPoolExecutor(workers=3) で並列取得。
    各スレッドは REQUEST_DELAY で個別に待機し、429 対策を維持する。

    Returns:
        {coin: DataFrame(timestamp, open, high, low, close, volume)}
    """
    import concurrent.futures

    now_utc   = datetime.now(timezone.utc)
    # 当日の未確定足を除外するため endTime を本日 00:00 UTC に固定 (確定済み日足のみ使用)
    today_midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    end_ms    = int(today_midnight.timestamp() * 1000)
    start_ms  = int((today_midnight - timedelta(days=n_days)).timestamp() * 1000)

    target_coins = [c for c in coins if c not in COIN_BLACKLIST]
    skip_cnt     = len(coins) - len(target_coins)

    logger.info(
        f"OHLCV 取得中: {len(target_coins)} 銘柄 × 過去 {n_days} 日 "
        f"(〜{today_midnight.strftime('%Y-%m-%d')} 確定足, workers={workers}) ..."
    )

    def _fetch_one(coin: str):
        time.sleep(REQUEST_DELAY)  # スレッドごとに待機して 429 を抑制
        df = fetch_daily_candles(coin, start_ms, end_ms)
        return coin, df

    ohlcv = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, coin): coin for coin in target_coins}
        for future in concurrent.futures.as_completed(futures):
            coin, df = future.result()
            if df.empty or len(df) < max(N_HIGH, SMA_PERIOD):
                logger.debug(f"  {coin:12s} データ不足 ({len(df)} 行) → スキップ")
                skip_cnt += 1
            else:
                ohlcv[coin] = df
                logger.debug(f"  {coin:12s} {len(df)} 行取得")

    logger.info(f"OHLCV 取得完了: {len(ohlcv)} 銘柄有効, {skip_cnt} 銘柄スキップ")
    return ohlcv


# ──────────────────────────────────────────────────────────────────────────────
# シグナル計算（バックテストと同一ロジック）
# ──────────────────────────────────────────────────────────────────────────────
def compute_signals(
    ohlcv: dict,
    logger: logging.Logger,
    fr_zscore_latest: dict = None,
) -> pd.DataFrame:
    """
    全銘柄のシグナル・インディケータを計算し、DataFrame で返す。

    Returns:
        DataFrame with index=coin, columns:
          [score, score_short_bull, atr_long, atr_short, sma100, close,
           vol_ratio, filter_long, filter_short]

    シグナル計算式 (バックテストと完全一致):
      upper  = close.rolling(N_HIGH).max()        # 168日高値 (close ベース)
      lower  = low.rolling(M_LOW).min()           # 15日安値  (low ベース)
      score  = (close - lower) / (upper - lower + ε) - 0.5   # [-0.5, +0.5]

    score_short_bull (Bull 時ショートスコア, Candidate A):
      score_short_bull = score - CARRY_WEIGHT_BULL * FR_zscore
      ※ FR_zscore が高い (ファンディング高い) 銘柄ほど score_short_bull が低くなり
         ショート優先度が上がる

    フィルタ条件:
      filter_long : SMA100 valid & close > SMA100 & max_daily_move(120d) < 0.40
                    & score valid & atr_long valid & vol_ok
      filter_short: SMA100 valid & max_daily_move(10d) < 0.40
                    & score valid & atr_short valid & vol_ok
      vol_ok      : vol(VOL_SHORT日MA) / vol(VOL_LONG日MA) >= VOL_THRESH

    ※ BTCレジームフィルタ (Bear時 filter_long=False) は apply_btc_regime_filter() で適用
    """
    records = {}

    for coin, df in ohlcv.items():
        try:
            close = df['close']
            high  = df['high']
            low   = df['low']

            # ── Donchian スコア ────────────────────────────────────────────────
            upper   = close.rolling(N_HIGH).max()
            lower   = low.rolling(M_LOW).min()
            channel = (upper - lower).clip(lower=1e-8)
            score   = (close - lower) / channel - 0.5

            # ── ATR (Average True Range) ───────────────────────────────────────
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr_long  = tr.rolling(ATR_L).mean()
            atr_short = tr.rolling(ATR_S).mean()

            # ── SMA ────────────────────────────────────────────────────────────
            sma100 = close.rolling(SMA_PERIOD).mean()

            # ── max daily return フィルタ ──────────────────────────────────────
            daily_ret = close.pct_change().abs()
            mar_long  = daily_ret.rolling(REG_L).max()
            mar_short = daily_ret.rolling(REG_S).max()

            # ── 出来高フィルタ ─────────────────────────────────────────────────
            vol_ratio  = (df['volume'].rolling(VOL_SHORT).mean()
                          / df['volume'].rolling(VOL_LONG).mean())
            last_vol_r = float(vol_ratio.iloc[-1])
            last_vol_ok = (not np.isnan(last_vol_r)) and (last_vol_r >= VOL_THRESH)

            # ── 最新値を取得 ──────────────────────────────────────────────────
            last_score     = float(score.iloc[-1])
            last_atr_long  = float(atr_long.iloc[-1])
            last_atr_short = float(atr_short.iloc[-1])
            last_sma100    = float(sma100.iloc[-1])
            last_close     = float(close.iloc[-1])
            last_mar_long  = float(mar_long.iloc[-1])
            last_mar_short = float(mar_short.iloc[-1])

            # ── Bull 時ショートスコア (FR Z-score ブレンド) ────────────────
            frzs = (fr_zscore_latest or {}).get(coin, 0.0)
            last_score_short_bull = last_score - CARRY_WEIGHT_BULL * frzs

            # ── フィルタ評価 ──────────────────────────────────────────────────
            # ロング: SMA有効 & close > SMA & 極端な価格変動なし & score valid & ATR valid & vol_ok
            filter_long = (
                not np.isnan(last_sma100)      and
                last_close > last_sma100       and
                not np.isnan(last_mar_long)    and
                last_mar_long < MAX_DAILY_MOVE and
                not np.isnan(last_score)       and
                not np.isnan(last_atr_long)    and
                last_atr_long > 0              and
                last_vol_ok                        # [B] 出来高フィルタ
            )

            # ショート: SMA有効 & 極端な価格変動なし & score valid & ATR valid & vol_ok
            # NOTE: close > SMA 条件はショートには課さない (バックテストと同一)
            filter_short = (
                not np.isnan(last_sma100)       and
                not np.isnan(last_mar_short)    and
                last_mar_short < MAX_DAILY_MOVE and
                not np.isnan(last_score)        and
                not np.isnan(last_atr_short)    and
                last_atr_short > 0              and
                last_vol_ok                         # [B] 出来高フィルタ
            )

            records[coin] = {
                'score':            last_score,
                'score_short_bull': last_score_short_bull,  # Bull時ショートスコア
                'atr_long':         last_atr_long,
                'atr_short':        last_atr_short,
                'sma100':           last_sma100,
                'close':            last_close,
                'vol_ratio':        last_vol_r,
                'filter_long':      filter_long,
                'filter_short':     filter_short,
                'mar_long':         last_mar_long,
                'mar_short':        last_mar_short,
            }

        except Exception as e:
            logger.warning(f"  {coin}: シグナル計算エラー: {e}")

    if not records:
        return pd.DataFrame()

    sig = pd.DataFrame.from_dict(records, orient='index')
    sig.index.name = 'coin'
    return sig


# ──────────────────────────────────────────────────────────────────────────────
# [A] BTC Regime フィルタ
# ──────────────────────────────────────────────────────────────────────────────
def apply_btc_regime_filter(
    sig: pd.DataFrame,
    ohlcv: dict,
    logger: logging.Logger,
) -> tuple:
    """
    BTC が SMA200 を下回る Bear 相場の場合、全銘柄の filter_long を False にする。

    バックテスト研究結果:
      Bear 相場 (BTC < SMA200): Long 側 Sharpe < 0 → ロングは有害
      Bull 相場 (BTC > SMA200): Long 側 Sharpe > 2.5 → 有効
      +Regime フィルタで OOS Sharpe: 2.18 → 3.67 (全期間ベースライン比)

    Returns:
        (sig, is_bear)
          sig:     filter_long が更新された DataFrame のコピー
          is_bear: True = Bear 相場 (ロング停止 & Bear ショートロジック適用)
    """
    if 'BTC' not in ohlcv:
        logger.warning('[Regime] BTC データなし → 保守的に Bear 扱い (ロング停止)')
        sig = sig.copy()
        sig['filter_long'] = False
        return sig, True

    btc_close   = ohlcv['BTC']['close']
    btc_sma200  = btc_close.rolling(BTC_SMA_PERIOD).mean()
    last_btc    = float(btc_close.iloc[-1])
    last_sma200 = float(btc_sma200.iloc[-1])

    if np.isnan(last_sma200):
        logger.warning(
            f'[Regime] BTC SMA{BTC_SMA_PERIOD} 計算不可 (データ不足) → 保守的に Bear 扱い (ロング停止)'
        )
        sig = sig.copy()
        sig['filter_long'] = False
        return sig, True

    is_bear = last_btc < last_sma200
    regime_str = f'BEAR (BTC={last_btc:,.0f} < SMA{BTC_SMA_PERIOD}={last_sma200:,.0f})' \
                 if is_bear else \
                 f'BULL (BTC={last_btc:,.0f} > SMA{BTC_SMA_PERIOD}={last_sma200:,.0f})'
    logger.info(f'[Regime] BTCレジーム: {regime_str}')

    if is_bear:
        sig = sig.copy()
        sig['filter_long'] = False
        logger.info('[Regime] Bear 相場のためロング候補を全銘柄ゼロにしました')

    return sig, is_bear


# ──────────────────────────────────────────────────────────────────────────────
# ターゲットポジション計算
# ──────────────────────────────────────────────────────────────────────────────
def compute_target_positions(
    sig: pd.DataFrame,
    equity: float,
    mid_prices: dict,
    universe_meta: dict,
    logger: logging.Logger,
    is_bear: bool = False,
    max_gross_lev: float = None,
    rf_l_scale: float = 1.0,
    rf_s_scale: float = 1.0,
) -> dict:
    """
    シグナルから目標ポジションを計算する (Candidate A: Bull/Bear 条件付きショート)。

    ロング: score 上位 TOP_PCT_L=15% かつ filter_long=True (Bear 時は全銘柄 filter_long=False)

    ショート (Bull 時):
      score_short_bull = score - CARRY_WEIGHT_BULL * FR_zscore で昇順ソート
      filter_short=True の候補から下位 TOP_PCT_S_BULL=15% を選択

    ショート (Bear 時):
      純 Donchian score で全銘柄の下位 TOP_PCT_S_BEAR=20% 分位未満のみを候補に絞る
      (集中ショート: 候補が少ないため結果として銘柄が絞られる)
      候補から下位 TOP_PCT_S_BULL=15% を選択

    サイジング (バックテストと同一):
      target_qty_long[coin]  = +equity * RF_L / atr_long[coin]   (コイン単位, 正)
      target_qty_short[coin] = -equity * RF_S_BULL(Bull) or RF_S(Bear) / atr_short[coin]  (コイン単位, 負)

    グロスレバレッジ上限: MAX_GROSS_LEV を超えたら全ポジションを比例縮小

    Returns:
        {coin: target_qty}  (正=ロング, 負=ショート)
    """
    if sig.empty or equity <= 0:
        return {}

    # ── ロング候補 ────────────────────────────────────────────────────────────
    long_cands = sig[sig['filter_long']].sort_values('score', ascending=False)
    n_long     = max(1, int(len(long_cands) * TOP_PCT_L)) if len(long_cands) > 0 else 0
    long_sel   = long_cands.head(n_long)

    # ── ショート候補 (Bull/Bear 分岐) ─────────────────────────────────────────
    base_short_cands = sig[sig['filter_short']]
    if is_bear:
        # Bear 時: filter_short 候補の中での下位 TOP_PCT_S_BEAR 分位未満のみに絞る
        # ※ 全銘柄スコアではなく候補集合内で分位を算出し整合性を保つ
        if len(base_short_cands) > 0:
            bear_thresh = base_short_cands['score'].quantile(TOP_PCT_S_BEAR)
            bear_cands  = base_short_cands[base_short_cands['score'] < bear_thresh]
            # フォールバック: quantile 境界に一致する銘柄が多く bear_cands が空になる場合
            # バックテストと同一: sc_s.iloc[:max(1, int(len(sc_s) * top_pct_s))] で補完
            if len(bear_cands) == 0:
                n_fallback = max(1, int(len(base_short_cands) * TOP_PCT_S_BULL))
                bear_cands = base_short_cands.sort_values('score', ascending=True).iloc[:n_fallback]
                logger.info(
                    f'[Bear] quantile フォールバック: score < {bear_thresh:+.4f} で 0 銘柄 '
                    f'→ 下位 {n_fallback} 銘柄に切り替え'
                )
        else:
            bear_cands = base_short_cands
        short_cands = bear_cands.sort_values('score', ascending=True)
        qt_str = f'{base_short_cands["score"].quantile(TOP_PCT_S_BEAR):+.3f}' \
                 if len(base_short_cands) > 0 else 'N/A'
        mode_str    = f'Bear (score < {TOP_PCT_S_BEAR:.0%} 分位={qt_str} (候補内))'
    else:
        # Bull 時: score_short_bull (FR ブレンド) でソート
        sort_col = 'score_short_bull' if 'score_short_bull' in sig.columns else 'score'
        short_cands = base_short_cands.sort_values(sort_col, ascending=True)
        mode_str    = f'Bull (sort={sort_col})'

    n_short   = max(1, int(len(short_cands) * TOP_PCT_S_BULL)) if len(short_cands) > 0 else 0
    short_sel = short_cands.head(n_short)

    logger.info(
        f"ターゲット計算 [{mode_str}]: "
        f"Long候補 {len(long_cands)} 銘柄 → 上位 {n_long} 銘柄 / "
        f"Short候補 {len(short_cands)} 銘柄 → 下位 {n_short} 銘柄"
    )

    if len(long_cands) == 0:
        logger.warning(
            "Long候補が0銘柄です。市場が下落傾向でSMA100を上回る銘柄がない可能性があります。"
            "ロングポジションは取りません（戦略として正常）。"
        )

    target = {}

    # ── ロング目標数量 ─────────────────────────────────────────────────────────
    for coin, row in long_sel.iterrows():
        price = mid_prices.get(coin)
        if price is None or price <= 0:
            logger.debug(f"  {coin}: mid price なし → スキップ")
            continue
        atr   = row['atr_long']
        tq    = equity * RF_L * rf_l_scale / atr   # コイン単位の目標数量
        notional = tq * price                 # USD 換算
        # HL の maxLeverage チェック (notional / equity = leverage per coin)
        max_lev = universe_meta.get(coin, {}).get('maxLeverage', 1)
        if notional / equity > max_lev:
            tq = equity * max_lev / price
            logger.debug(f"  {coin}: maxLev={max_lev}x でキャップ")
        target[coin] = +tq

    # ── ショート目標数量 ───────────────────────────────────────────────────────
    for coin, row in short_sel.iterrows():
        if coin in target:
            # 同銘柄がロング候補にも入っている場合はスキップ
            logger.debug(f"  {coin}: ロング候補と重複 → ショートスキップ")
            continue
        price = mid_prices.get(coin)
        if price is None or price <= 0:
            logger.debug(f"  {coin}: mid price なし → スキップ")
            continue
        atr      = row['atr_short']
        _rf_s    = RF_S_BULL if not is_bear else RF_S  # Bull時は縮小サイズ (E2)
        tq       = equity * _rf_s * rf_s_scale / atr
        notional = tq * price
        max_lev  = universe_meta.get(coin, {}).get('maxLeverage', 1)
        if notional / equity > max_lev:
            tq = equity * max_lev / price
            logger.debug(f"  {coin}(short): maxLev={max_lev}x でキャップ")
        target[coin] = -tq

    # ── グロスレバレッジ上限チェック ──────────────────────────────────────────
    gross_notional = sum(abs(qty) * mid_prices.get(c, 0) for c, qty in target.items())
    gross_lev      = gross_notional / equity if equity > 0 else 0.0

    # max_gross_lev: 呼び出し元が動的レバレッジ計算済みの値を渡す (None の場合はグローバル定数)
    _max_lev = max_gross_lev if max_gross_lev is not None else MAX_GROSS_LEV

    logger.info(
        f"グロスレバレッジ計算: {gross_lev:.2f}x "
        f"(上限: {_max_lev:.2f}x, 合計 ${gross_notional:,.0f})"
    )

    if _max_lev <= 0:
        logger.warning(f"グロスレバレッジ上限 {_max_lev:.3f}x ≤ 0 → 全ポジションをゼロに")
        return {}

    if gross_lev > _max_lev:
        scale = _max_lev / gross_lev
        target = {c: q * scale for c, q in target.items() if abs(q * scale) > 1e-10}
        logger.warning(
            f"グロスレバレッジ上限超過 → 全ポジションを {scale:.3f}x にスケールダウン"
        )

    # ── 詳細ログ ──────────────────────────────────────────────────────────────
    long_items  = [(c, q) for c, q in target.items() if q > 0]
    short_items = [(c, q) for c, q in target.items() if q < 0]

    logger.info(f"ターゲット: ロング {len(long_items)} / ショート {len(short_items)}")
    for coin, qty in sorted(long_items, key=lambda x: -x[1] * mid_prices.get(x[0], 0)):
        price    = mid_prices.get(coin, 0)
        notional = qty * price
        score    = sig.loc[coin, 'score'] if coin in sig.index else float('nan')
        logger.debug(
            f"  [L] {coin:12s} qty={qty:+.4f} "
            f"(${notional:8.1f}) score={score:+.3f}"
        )
    for coin, qty in sorted(short_items, key=lambda x: x[1] * mid_prices.get(x[0], 0)):
        price    = mid_prices.get(coin, 0)
        notional = abs(qty) * price
        score    = sig.loc[coin, 'score'] if coin in sig.index else float('nan')
        logger.debug(
            f"  [S] {coin:12s} qty={qty:+.4f} "
            f"(${notional:8.1f}) score={score:+.3f}"
        )

    return target


# ──────────────────────────────────────────────────────────────────────────────
# 注文サイズ丸め
# ──────────────────────────────────────────────────────────────────────────────
def round_sz(qty: float, sz_decimals: int, round_up: bool = False) -> float:
    """
    HL の szDecimals に従って注文サイズを丸める。
    round_up=False (default): 切り捨て → 「注文しすぎ」を防ぐ (open 注文用)
    round_up=True           : 切り上げ → ダスト残りを防ぐ (reduce_only クローズ用)
    """
    import math
    factor = 10 ** sz_decimals
    if round_up:
        return float(math.ceil(abs(qty) * factor) / factor)
    return float(int(abs(qty) * factor) / factor)


# ──────────────────────────────────────────────────────────────────────────────
# X1: 動的タイムアウト計算
# ──────────────────────────────────────────────────────────────────────────────
def _compute_timeout_min(coin: str, is_buy: bool, ohlcv: dict) -> int:
    """
    X1: 出来高ティアと注文方向に基づく動的 ALO タイムアウトを返す (分)。

    出来高ティア (過去7日平均時間出来高 USD):
      > $10M/h → base=1.0h  (流動性大: 早く約定)
      > $1M/h  → base=1.5h
      その他   → base=2.0h  (流動性小: 時間をかけてメイカー狙い)

    方向:
      BUY  (モメンタム追随): timeout = max(C_MIN_TIMEOUT, (base − 0.5h) × 60)
      SELL (ポジション整理): timeout = base × 60

    Returns: タイムアウト (分), 計算不可の場合は C_TIMEOUT_MIN
    """
    try:
        df = ohlcv.get(coin)
        if df is None or df.empty or len(df) < 1:
            return C_TIMEOUT_MIN
        last_price = float(df['close'].iloc[-1])
        avg_daily_vol = float(df['volume'].iloc[-7:].mean())
        if last_price <= 0 or np.isnan(avg_daily_vol) or avg_daily_vol <= 0:
            return C_TIMEOUT_MIN
        hourly_vol_usd = avg_daily_vol * last_price / 24.0

        base_h = X1_VOL_DEFAULT_H
        for threshold, h in X1_VOL_TIERS:
            if hourly_vol_usd > threshold:
                base_h = h
                break

        if is_buy:
            return max(C_MIN_TIMEOUT, int((base_h - X1_BUY_SUBTRACT_H) * 60))
        else:
            return int(base_h * 60)
    except Exception:
        return C_TIMEOUT_MIN


# ──────────────────────────────────────────────────────────────────────────────
# 注文リスト構築
# ──────────────────────────────────────────────────────────────────────────────
def build_orders(
    target: dict,
    current: dict,
    mid_prices: dict,
    sig: pd.DataFrame,
    universe_meta: dict,
    do_reb: bool,
    logger: logging.Logger,
) -> list:
    """
    目標ポジション vs 現在ポジションを比較し、必要な注文リストを返す。

    注文の種類:
      'close_short':  ショートクローズ（buying back）
      'close_long':   ロングクローズ（selling）
      'resize':       既存ポジションのサイズ調整 (do_reb のみ)
      'open_long':    新規ロング (do_reb のみ)
      'open_short':   新規ショート (do_reb のみ)

    実行順 (margin 効率のため):
      1. close_short (買い戻し → margin 解放)
      2. close_long  (売り → cash 回収)
      3. resize       (既存調整)
      4. open_long   (新規ロング)
      5. open_short  (新規ショート)

    バックテストのエグジット条件:
      ロング: ターゲットに入っていない OR close <= SMA100
      ショート: ターゲットに入っていない OR close > SMA100

    Returns:
        list of {
          'type': str, 'coin': str,
          'is_buy': bool, 'sz': float, 'sz_raw': float,
          'notional_usd': float, 'reason': str,
        }
    """
    orders = []

    def make_order(order_type, coin, is_buy, sz_raw, reason, reduce_only=False):
        price = mid_prices.get(coin, 0.0)
        # reduce_only（強制クローズ）で mid 価格が不明な場合は entry_px をフォールバックとして使う
        if price <= 0 and reduce_only:
            price = current.get(coin, {}).get('entry_px', 0.0)
            if price > 0:
                logger.debug(f"  {coin}: mid 価格不明 → entry_px={price:.4f} をフォールバック使用")
        if price <= 0:
            logger.warning(f"  {coin}: 価格が取得できないため注文をスキップ (reduce_only={reduce_only})")
            return
        meta      = universe_meta.get(coin, {})
        decimals  = meta.get('szDecimals', 2)
        # reduce_only (クローズ) は切り上げでダスト残りを防ぐ
        sz        = round_sz(sz_raw, decimals, round_up=reduce_only)
        notional  = sz * price
        # reduce_only（強制クローズ）は MIN_ORDER_USD チェックをスキップ（ポジション清算を優先）
        if notional < MIN_ORDER_USD and not reduce_only:
            if notional >= MIN_ORDER_USD * 0.5:
                # MIN_ORDER_USD の半額以上: HL最小注文額を満たすよう切り上げ実行
                # 目標との差分が$5.5〜$11の場合、$11相当に切り上げて発注（翌週目標再計算で修正）
                sz      = round_sz(MIN_ORDER_USD / price, decimals, round_up=True)
                notional = sz * price
                logger.debug(
                    f"  {coin}: 注文額 ${sz_raw * price:.2f} < 最小 ${MIN_ORDER_USD}"
                    f" → ${notional:.2f} に切り上げ実行"
                )
            else:
                logger.debug(
                    f"  {coin}: 注文額 ${notional:.2f} < 最小 ${MIN_ORDER_USD} × 0.5 → スキップ"
                )
                return
        # ATR を注文に含める (Strategy C のオフセット計算に使用)
        if coin in sig.index:
            atr_val = float(sig.at[coin, 'atr_long'] if is_buy
                            else sig.at[coin, 'atr_short'])
        else:
            atr_val = float('nan')
        orders.append({
            'type':         order_type,
            'coin':         coin,
            'is_buy':       is_buy,
            'sz':           sz,
            'sz_raw':       sz_raw,
            'price':        price,
            'notional_usd': notional,
            'reduce_only':  reduce_only,
            'reason':       reason,
            'atr':          atr_val,
        })

    # ── Exit 判定 (毎週実行) ──────────────────────────────────────────────────
    for coin, pos in current.items():
        qty   = pos['qty']
        # mid_prices に無い（上場廃止等）場合は entry_px をフォールバックに使う
        price = mid_prices.get(coin) or pos.get('entry_px', 0.0)
        if price <= 0:
            # ダミー価格での発注は limit 価格が異常になるため行わない
            # → ポジション残留として記録し、次週の mid_prices 更新後に再試行
            logger.error(
                f'  {coin}: 価格完全不明 (mid={mid_prices.get(coin)}, entry={pos.get("entry_px")}) '
                f'→ エグジット注文スキップ (ポジション qty={qty:+.4f} が残留)'
            )
            continue

        if qty > 0:
            # --- ロングエグジット条件 ---
            not_in_target = coin not in target or target[coin] <= 0
            sma100_break  = (
                coin in sig.index and
                not np.isnan(sig.loc[coin, 'sma100']) and
                price <= sig.loc[coin, 'sma100']
            )
            if not_in_target or sma100_break:
                reason = 'target外' if not_in_target else 'SMA100割れ'
                logger.info(f"  EXIT LONG  {coin:12s} qty={qty:+.4f} (${qty*price:.1f}) 理由: {reason}")
                make_order('close_long', coin, is_buy=False, sz_raw=abs(qty),
                           reason=reason, reduce_only=True)

        elif qty < 0:
            # --- ショートエグジット条件 ---
            not_in_target  = coin not in target or target[coin] >= 0
            sma100_recovery = (
                coin in sig.index and
                not np.isnan(sig.loc[coin, 'sma100']) and
                price > sig.loc[coin, 'sma100']
            )
            if not_in_target or sma100_recovery:
                reason = 'target外' if not_in_target else 'SMA100回復'
                logger.info(f"  EXIT SHORT {coin:12s} qty={qty:+.4f} (${abs(qty)*price:.1f}) 理由: {reason}")
                make_order('close_short', coin, is_buy=True, sz_raw=abs(qty),
                           reason=reason, reduce_only=True)

    # exit 済み coin セット (exit 注文があるもの)
    exiting = {o['coin'] for o in orders}

    # ── Resize: 既存ポジション調整 (do_reb のみ) ──────────────────────────────
    if do_reb:
        for coin, pos in current.items():
            if coin in exiting:
                continue
            qty    = pos['qty']
            tgt_q  = target.get(coin, 0.0)
            if abs(tgt_q) < 1e-12:
                continue   # target=0 は exit 側で処理済みのはず
            delta  = tgt_q - qty
            price  = mid_prices.get(coin, 0.0)
            if abs(delta * price) < MIN_ORDER_USD:
                logger.debug(f"  {coin}: resize delta ${abs(delta)*price:.2f} < min → スキップ")
                continue
            is_buy = delta > 0
            logger.info(
                f"  RESIZE {coin:12s} qty={qty:+.4f}→{tgt_q:+.4f} Δ={delta:+.4f}"
                f" (${abs(delta)*price:.1f})"
            )
            make_order('resize', coin, is_buy=is_buy, sz_raw=abs(delta), reason='resize')
    else:
        logger.info("  偶数週のためリサイズなし（エグジット + 新規エントリーのみ）")

    # New LONG entries (score 降順)
    long_targets = {c: q for c, q in target.items() if q > 0}
    for coin in sorted(long_targets, key=lambda c: -sig.loc[c, 'score']
                        if c in sig.index else 0):
        if coin in current and coin not in exiting:
            continue  # 既存ポジション（resize 済み）
        tgt_q = long_targets[coin]
        price = mid_prices.get(coin, 0.0)
        logger.info(
            f"  OPEN LONG  {coin:12s} tgt_qty={tgt_q:+.4f}"
            f" (${tgt_q*price:.1f})"
            f" score={sig.loc[coin, 'score']:+.3f}"
            if coin in sig.index else
            f"  OPEN LONG  {coin:12s} tgt_qty={tgt_q:+.4f}"
        )
        make_order('open_long', coin, is_buy=True, sz_raw=tgt_q, reason='new_long')

    # New SHORT entries (score 昇順)
    short_targets = {c: q for c, q in target.items() if q < 0}
    for coin in sorted(short_targets, key=lambda c: sig.loc[c, 'score']
                        if c in sig.index else 0):
        if coin in current and coin not in exiting:
            continue
        tgt_q = short_targets[coin]
        price = mid_prices.get(coin, 0.0)
        logger.info(
            f"  OPEN SHORT {coin:12s} tgt_qty={tgt_q:+.4f}"
            f" (${abs(tgt_q)*price:.1f})"
            f" score={sig.loc[coin, 'score']:+.3f}"
            if coin in sig.index else
            f"  OPEN SHORT {coin:12s} tgt_qty={tgt_q:+.4f}"
        )
        make_order('open_short', coin, is_buy=False, sz_raw=abs(tgt_q), reason='new_short')

    # ── 実行順ソート ───────────────────────────────────────────────────────────
    ORDER_PRIORITY = {
        'close_short': 0,
        'close_long':  1,
        'resize':      2,
        'open_long':   3,
        'open_short':  4,
    }
    orders.sort(key=lambda o: ORDER_PRIORITY.get(o['type'], 99))

    return orders



# ──────────────────────────────────────────────────────────────────────────────
# 注文実行ヘルパー
# ──────────────────────────────────────────────────────────────────────────────

def _round_limit_px(mid: float, offset: float, is_buy: bool) -> tuple:
    """mid 価格から limit 価格を計算し、HL が受け付ける小数桁数に丸めて返す。"""
    mid_str  = f"{mid:.5g}"
    n_dp     = len(mid_str.split('.')[1]) if '.' in mid_str else 0
    limit_px = mid * (1.0 - offset) if is_buy else mid * (1.0 + offset)
    return round(limit_px, n_dp), n_dp


def _place_ioc(order: dict, exchange, logger: logging.Logger) -> bool:
    """
    IOC 指値注文を発注する (Strategy C のフォールバック / CLOSE 注文用)。
    Returns: True = 約定 or 部分約定, False = 未約定 or エラー
    """
    coin        = order['coin']
    is_buy      = order['is_buy']
    sz          = order['sz']
    mid         = order['price']
    reduce_only = order['reduce_only']

    # IOC は約定しやすい方向に LIMIT_SLIPPAGE 分だけ広めに設定
    # BUY:  mid*(1+SLIP) = mid*1.02  → 最大これまで出すので即時約定しやすい
    # SELL: mid*(1-SLIP) = mid*0.98  → 最低これで売るので即時約定しやすい
    if is_buy:
        limit_px, _ = _round_limit_px(mid, -LIMIT_SLIPPAGE, True)   # mid*(1+SLIP)
    else:
        limit_px, _ = _round_limit_px(mid,  LIMIT_SLIPPAGE, True)   # mid*(1-SLIP)

    try:
        result = exchange.order(coin, is_buy, sz, limit_px,
                                {'limit': {'tif': 'Ioc'}},
                                reduce_only=reduce_only)
        logger.info(f"    [IOC] {coin} limit={limit_px} (mid={mid}, slippage={LIMIT_SLIPPAGE:.1%})")
        if isinstance(result, dict) and result.get('status') == 'ok':
            try:
                status_info = result['response']['data']['statuses'][0]
                if 'filled' in status_info:
                    filled_sz = float(status_info['filled'].get('totalSz', 0.0))
                    if filled_sz < sz * 0.5:
                        logger.warning(f"    → 部分約定: filled={filled_sz:.6f} / order={sz:.6f}")
                    else:
                        logger.info(f"    → 約定: {filled_sz:.6f}")
                    return True
                else:
                    logger.warning(f"    → 未約定 (IOC キャンセル): {status_info}")
                    return False
            except Exception:
                logger.info(f"    → ok (詳細不明)")
                return True
        else:
            logger.error(f"    → IOC エラー: {result}")
            return False
    except Exception as e:
        logger.error(f"    → IOC 例外 ({coin}): {e}")
        logger.debug(traceback.format_exc())
        return False


@dataclass
class PlaceResult:
    """
    _place_alo() の返り値。status フィールドで呼び出し側が分岐する。

    status:
        'resting'  — oid が板上に残存。pending に積んで待機する。
        'filled'   — ALO が即時全量約定。IOC 追加発注は不要（二重注文になる）。
        'rejected' — would-cross-book 等でリジェクト。IOC フォールバックへ。
        'error'    — API 例外・不明応答。failed 扱いにするか保守停止する。
    """
    status:    Literal['resting', 'filled', 'rejected', 'error']
    oid:       int   = None
    filled_sz: float = 0.0
    message:   str   = field(default='')


def _place_alo(coin: str, is_buy: bool, sz: float, limit_px: float,
               exchange, logger: logging.Logger,
               reduce_only: bool = False) -> PlaceResult:
    """
    ALO (Add Liquidity Only / Post-Only) 指値注文を発注する。

    Returns:
        PlaceResult — status は 'resting' / 'filled' / 'rejected' / 'error' の 4 種類。
        'filled' は即時約定済みなので呼び出し側で IOC を追加してはいけない（二重発注になる）。
    """
    try:
        result = exchange.order(coin, is_buy, sz, limit_px,
                                {'limit': {'tif': 'Alo'}},
                                reduce_only=reduce_only)
        if isinstance(result, dict) and result.get('status') == 'ok':
            status_info = result['response']['data']['statuses'][0]
            if 'resting' in status_info:
                oid = int(status_info['resting']['oid'])
                logger.info(f"    [ALO] {coin} oid={oid} limit={limit_px} → resting")
                return PlaceResult(status='resting', oid=oid)
            elif 'filled' in status_info:
                filled_sz = float(status_info['filled'].get('totalSz', 0.0))
                logger.info(f"    [ALO] {coin} 即時全量約定: {filled_sz:.6f} (IOC 追加発注不要)")
                return PlaceResult(status='filled', filled_sz=filled_sz)
            else:
                msg = str(status_info)
                logger.info(f"    [ALO] {coin} リジェクト: {msg}")
                return PlaceResult(status='rejected', message=msg)
        else:
            msg = str(result)
            logger.warning(f"    [ALO] {coin} エラー応答: {msg}")
            return PlaceResult(status='error', message=msg)
    except Exception as e:
        logger.warning(f"    [ALO] {coin} 例外: {e}")
        return PlaceResult(status='error', message=str(e))


def _get_open_order_ids(address: str):
    """
    現在の未約定注文の order_id セットを返す。
    API 失敗時は None を返す（空セットと区別して「状態不明」として扱う）。
    """
    try:
        data = _hl_post({'type': 'openOrders', 'user': address})
        return {int(o['oid']) for o in (data or [])}
    except Exception:
        return None  # 空セット (全約定) ではなく「不明」として返す


def _cancel_order(coin: str, oid: int, exchange, logger: logging.Logger) -> bool:
    """注文をキャンセルする。Returns: True = 成功 or 既約定, False = エラー"""
    try:
        exchange.cancel(coin, oid)
        return True
    except Exception as e:
        logger.debug(f"    cancel {coin} oid={oid}: {e}")
        return False


def _cancel_and_confirm(
    coin:        str,
    oid:         int,
    exchange,
    address:     str,
    logger:      logging.Logger,
    timeout_sec: int = 10,
) -> Literal['canceled_or_filled', 'unknown']:
    """
    キャンセルを送り、openOrders から oid が消えるまで待機する。

    Returns:
        'canceled_or_filled' — oid が板から消えた（約定済み or キャンセル済み）
        'unknown'            — timeout 内に消えなかった、または API が応答しなかった
                               この場合は再発注せず保留にする（二重注文防止）
    """
    _cancel_order(coin, oid, exchange, logger)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        open_ids = _get_open_order_ids(address)
        if open_ids is None:
            time.sleep(1)
            continue
        if oid not in open_ids:
            return 'canceled_or_filled'
        time.sleep(1)
    logger.warning(f'    [cancel_confirm] {coin} oid={oid}: timeout {timeout_sec}s 内に消えず → unknown')
    return 'unknown'


# ──────────────────────────────────────────────────────────────────────────────
# 注文実行 (Strategy C)
# ──────────────────────────────────────────────────────────────────────────────
def _get_exec_policy(order: dict, ohlcv: dict) -> dict:
    """
    注文ごとの執行ポリシーを返す。CLOSE と OPEN で異なるパラメータを使い分ける。

    CLOSE (reduce_only=True):
      - タイムアウト短め (15min) でポジション決済の遅延を抑制
      - ATR オフセット小さめ (0.10) でベストビッドに近い指値
    OPEN:
      - X1 動的タイムアウト (出来高ティア × 方向)
      - ATR オフセット大きめ (0.20) で板に深めに刺さるよう配置

    Returns: {atr_ratio, min_offset, max_offset, timeout_min, check_every}
    """
    if order.get('reduce_only', False):
        return {
            'atr_ratio':   CLOSE_ATR_RATIO,
            'min_offset':  CLOSE_MIN_OFFSET,
            'max_offset':  CLOSE_MAX_OFFSET,
            'timeout_min': CLOSE_TIMEOUT_MIN,
            'check_every': CLOSE_CHECK_EVERY,
        }
    return {
        'atr_ratio':   C_ATR_RATIO,
        'min_offset':  C_MIN_OFFSET,
        'max_offset':  C_MAX_OFFSET,
        'timeout_min': _compute_timeout_min(order['coin'], order['is_buy'], ohlcv),
        'check_every': C_CHECK_EVERY,
    }


def execute_orders(
    orders: list,
    exchange,        # hyperliquid.exchange.Exchange
    live: bool,
    logger: logging.Logger,
    address: str = None,
    ohlcv: dict = None,
    eval_date: str = None,
    tranche_id: str = None,
) -> tuple:
    """
    注文リストを Strategy C + X1 アルゴリズムで実行する。

    【執行ルール】
      CLOSE と OPEN でポリシーを分けて ALO maker 優先で発注する:

      CLOSE (reduce_only=True):
        → ALO 指値発注 (ATR × 0.10 オフセット) → 5分ごとに確認・リプライス
        → 15分でタイムアウト → IOC (taker) に切り替え (exit の遅延を抑制)

      OPEN:
        → ALO 指値発注 (ATR × 0.20 オフセット) → 30分ごとに確認・リプライス
        → X1 動的タイムアウト → IOC (taker) に切り替え

    X1 動的タイムアウト (OPEN のみ):
      各銘柄の過去7日平均時間出来高 USD に基づくタイムアウト:
        > $10M/h → 1.0h base  |  > $1M/h → 1.5h base  |  その他 → 2.0h base
      BUY = base − 0.5h (短め)  SELL = base (長め)

    Args:
        orders:   build_orders() の返り値
        exchange: HL Exchange インスタンス (live=False のとき None でよい)
        live:     True = 実際に発注, False = ドライランのみ
        address:  HL ウォレットアドレス (未約定注文確認用)
        ohlcv:    {coin: DataFrame} — X1 動的タイムアウト計算用

    Returns:
        (executed_count, failed_count)
    """
    executed  = 0
    failed    = 0
    mode_str  = '' if live else ' [DRY RUN]'
    ohlcv     = ohlcv or {}

    close_cnt = sum(1 for o in orders if o.get('reduce_only', False))
    open_cnt  = len(orders) - close_cnt

    # ── ログ: 全注文一覧 ──────────────────────────────────────────────────────
    for order in orders:
        coin      = order['coin']
        is_buy    = order['is_buy']
        sz        = order['sz']
        notional  = order['notional_usd']
        reason    = order['reason']
        direction = 'BUY ' if is_buy else 'SELL'
        tag       = '[CLOSE/StratC+X1]' if order.get('reduce_only') else '[OPEN/StratC+X1]'
        logger.info(
            f"  {direction} {coin:12s} sz={sz:.6f}"
            f" (${notional:.2f}) {tag} [{order['type']}] reason={reason}{mode_str}"
        )

    # ── DRY RUN ─────────────────────────────────────────────────────────────
    if not live:
        return len(orders), 0

    if not orders:
        return executed, failed

    # ── フェーズ分割: close 先行 → open 後続 ─────────────────────────────────
    # 証拠金逼迫防止: クローズで余力を確保してからオープンを発注する
    close_orders = [o for o in orders if o.get('reduce_only', False)]
    open_orders  = [o for o in orders if not o.get('reduce_only', False)]

    logger.info(
        f'[執行] フェーズ分割: Phase1=CLOSE {len(close_orders)}件 → Phase2=OPEN {len(open_orders)}件 '
        f'(StratC+X1: CLOSE=ALO 15min/5min-check, OPEN=ALO 動的タイムアウト/30min-check)'
    )

    # 実行ログ収集用リスト
    log_rows: list = []

    if close_orders:
        logger.info(f'[執行 Phase1] CLOSE {len(close_orders)} 件を先行発注')
        e, f = _run_strat_c_phase(
            close_orders, exchange, logger, address, ohlcv,
            phase_tag='CLOSE', log_rows=log_rows,
            eval_date=eval_date, tranche_id=tranche_id,
        )
        executed += e
        failed   += f

    if open_orders:
        logger.info(f'[執行 Phase2] OPEN {len(open_orders)} 件を発注')
        e, f = _run_strat_c_phase(
            open_orders, exchange, logger, address, ohlcv,
            phase_tag='OPEN', log_rows=log_rows,
            eval_date=eval_date, tranche_id=tranche_id,
        )
        executed += e
        failed   += f

    # 実行ログを CSV に書き込む
    _write_exec_rows(log_rows)

    return executed, failed


def _run_strat_c_phase(
    orders: list,
    exchange,
    logger: logging.Logger,
    address: str,
    ohlcv: dict,
    phase_tag: str = '',
    log_rows: list = None,
    eval_date: str = None,
    tranche_id: str = None,
) -> tuple:
    """
    Strategy C + X1 の ALO → リプライス → IOC フォールバックループを実行する。
    CLOSE フェーズ / OPEN フェーズを同一ロジックで処理するために分離。

    CLOSE (reduce_only=True) と OPEN で執行ポリシーが異なる (_get_exec_policy 参照):
      CLOSE: タイムアウト短め (15min), チェック間隔短め (5min), ATR オフセット小さめ
      OPEN:  X1 動的タイムアウト (出来高ティア), チェック 30min, ATR オフセット大きめ

    pending dict は {oid: info} で管理する (コイン同名での衝突防止、リプライス時に
    旧 oid を削除して新 oid を挿入することで参照の一意性を保つ)。

    Returns: (executed_count, failed_count)
    """
    executed = 0
    failed   = 0
    ohlcv    = ohlcv or {}
    tag      = f'[StratC+X1/{phase_tag}]' if phase_tag else '[StratC+X1]'
    log_rows = log_rows if log_rows is not None else []

    from datetime import datetime as _dt
    def _log_row(order: dict, method: str, final_status: str,
                 filled_qty: float = None, slippage_pct: float = None):
        """実行ログに1行追記する"""
        coin  = order['coin']
        is_buy = order['is_buy']
        side  = 'long' if order['type'] in ('open_long', 'close_long') else 'short'
        log_rows.append({
            'exec_datetime': _dt.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'eval_date':     eval_date or '',
            'tranche_id':    tranche_id or '',
            'coin':          coin,
            'side':          side,
            'order_type':    order.get('type', ''),
            'reason':        order.get('reason', ''),
            'target_qty':    f"{order.get('sz', 0):.6f}",
            'target_px':     f"{order.get('price', 0):.6f}",
            'notional_usd':  f"{order.get('notional_usd', 0):.2f}",
            'method':        method,
            'final_status':  final_status,
            'filled_qty':    f"{filled_qty:.6f}" if filled_qty is not None else '',
            'slippage_pct':  f"{slippage_pct:.4%}" if slippage_pct is not None else '',
        })

    # ── ALO 初期発注フェーズ ─────────────────────────────────────────────────
    # pending = {oid (int): {'order', 'coin', 'init_off', 'n_dp', 'policy'}}
    pending: dict[int, dict] = {}
    # reduce_only 注文で price=0/NaN の場合に備えて現在値を一括取得しておく
    _live_mids: dict | None = None
    def _ensure_live_mids():
        nonlocal _live_mids
        if _live_mids is None:
            try:
                _live_mids = get_all_mids()
            except Exception:
                _live_mids = {}
        return _live_mids

    for order in orders:
        coin   = order['coin']
        is_buy = order['is_buy']
        mid    = order['price']
        atr    = order.get('atr', float('nan'))
        sz     = order['sz']

        # price が欠落している reduce_only 注文はライブ mid で補完する
        if (not mid or np.isnan(float(mid) if mid else float('nan'))) and order.get('reduce_only', False):
            mid = _ensure_live_mids().get(coin, 0.0)
            if not mid:
                logger.warning(f'  {tag} {coin}: CLOSE 注文の価格が取得できず → スキップ')
                failed += 1
                continue

        policy = _get_exec_policy(order, ohlcv)

        if not np.isnan(atr) and mid > 0 and atr > 0:
            init_off = float(np.clip(
                atr / mid * policy['atr_ratio'],
                policy['min_offset'],
                policy['max_offset'],
            ))
        else:
            init_off = policy['min_offset']

        limit_px, n_dp = _round_limit_px(mid, init_off, is_buy)

        atr_str = f"{atr:.4f}" if not np.isnan(atr) else 'N/A'
        logger.info(
            f"  {tag} {coin} {'BUY' if is_buy else 'SELL'}"
            f" ATR={atr_str} offset={init_off:.3%} limit={limit_px}"
            f" timeout={policy['timeout_min']}min (mid={mid})"
        )

        alo_res = _place_alo(coin, is_buy, sz, limit_px, exchange, logger,
                             reduce_only=order.get('reduce_only', False))
        if alo_res.status == 'resting':
            pending[alo_res.oid] = {
                'order':    order,
                'coin':     coin,
                'init_off': init_off,
                'n_dp':     n_dp,
                'policy':   policy,
            }
            # resting は待機ループで約定確認後に記録
        elif alo_res.status == 'filled':
            # 即時全量約定 → IOC 追加発注不要
            executed += 1
            _log_row(order, 'ALO_filled', 'filled', filled_qty=alo_res.filled_sz)
        else:
            # 'rejected' / 'error' → IOC フォールバック
            logger.info(f"  {tag} {coin}: ALO {alo_res.status} → IOC フォールバック")
            ok = _place_ioc(order, exchange, logger)
            if ok:
                executed += 1
                slip = (limit_px * (1 if is_buy else -1) - float(order.get('price', limit_px))) / max(float(order.get('price', limit_px)), 1e-12)
                _log_row(order, 'ALO_rejected_IOC', 'filled', filled_qty=sz, slippage_pct=slip)
            else:
                failed += 1
                _log_row(order, 'ALO_rejected_IOC', 'unfilled')
        time.sleep(REQUEST_DELAY)

    if not pending:
        return executed, failed

    # ── 待機ループ ──────────────────────────────────────────────────────────
    start_time  = time.time()
    max_timeout = max(info['policy']['timeout_min'] for info in pending.values())
    min_check   = min(info['policy']['check_every'] for info in pending.values())
    logger.info(
        f'{tag} {len(pending)} 件待機中 '
        f'(timeout {min(info["policy"]["timeout_min"] for info in pending.values())}'
        f'~{max_timeout}min, check every {min_check}min)'
    )

    while True:
        # min_check 分スリープ (1分刻みで shutdown チェック)
        for _sleep_i in range(min_check):
            if _shutdown_requested:
                break
            time.sleep(60)
        if _shutdown_requested:
            logger.warning(f'{tag} shutdown シグナル受信 → 残存 ALO 注文をキャンセルして終了')
            for oid, info in list(pending.items()):
                _cancel_order(info['coin'], oid, exchange, logger)
            break

        elapsed_min = (time.time() - start_time) / 60.0

        try:
            current_mids = get_all_mids()
        except Exception:
            current_mids = {}

        open_ids = _get_open_order_ids(address) if address else set()
        if open_ids is None:
            logger.warning(f'{tag} openOrders API 失敗 → 約定確認をスキップ (次回チェック時に再試行)')
        else:
            # oid が板から消えていれば約定済み
            for oid in list(pending.keys()):
                if oid not in open_ids:
                    coin = pending[oid]['coin']
                    _ord = pending[oid]['order']
                    logger.info(f'  {tag} {coin}: 約定確認 oid={oid} (elapsed={elapsed_min:.0f}min)')
                    _log_row(_ord, 'ALO_resting', 'filled', filled_qty=_ord.get('sz'))
                    del pending[oid]
                    executed += 1

        if not pending:
            logger.info(f'{tag} 全注文約定完了')
            break

        # ── リプライス / タイムアウト処理 ──────────────────────────────────
        for oid in list(pending.keys()):
            if oid not in pending:
                continue  # 上の約定確認ループで削除済みの場合
            info     = pending[oid]
            order    = info['order']
            coin     = info['coin']
            is_buy   = order['is_buy']
            sz       = order['sz']
            policy   = info['policy']
            mid      = current_mids.get(coin, order['price'])
            do_ioc   = False
            new_off  = 0.0

            if elapsed_min >= policy['timeout_min']:
                logger.info(f'  {tag} {coin}: タイムアウト ({policy["timeout_min"]}min) → IOC')
                do_ioc = True
            else:
                progress = elapsed_min / policy['timeout_min']
                new_off  = max(0.0, info['init_off'] * (1.0 - progress))
                if new_off < policy['min_offset'] / 2:
                    logger.info(f'  {tag} {coin}: offset≈0 → IOC に切り替え')
                    do_ioc = True

            if do_ioc:
                result = _cancel_and_confirm(coin, oid, exchange, address, logger)
                if result == 'unknown':
                    logger.warning(
                        f'  {tag} {coin}: キャンセル未確認 (oid={oid}) → IOC 中止, 次回確認'
                    )
                else:
                    ok = _place_ioc(order, exchange, logger)
                    if ok:
                        executed += 1
                        _log_row(order, 'ALO_timeout_IOC', 'filled', filled_qty=sz)
                    else:
                        failed += 1
                        _log_row(order, 'ALO_timeout_IOC', 'unfilled')
                    del pending[oid]
            else:
                # リプライス: _cancel_and_confirm → 新 ALO
                result = _cancel_and_confirm(coin, oid, exchange, address, logger)
                if result == 'unknown':
                    logger.warning(
                        f'  {tag} {coin}: リプライスのキャンセル未確認 (oid={oid}) → スキップ'
                    )
                else:
                    limit_px, n_dp2 = _round_limit_px(mid, new_off, is_buy)
                    alo_res = _place_alo(coin, is_buy, sz, limit_px, exchange, logger,
                                        reduce_only=order.get('reduce_only', False))
                    if alo_res.status == 'resting':
                        # 旧 oid を削除し、新 oid で登録
                        del pending[oid]
                        pending[alo_res.oid] = {
                            'order':    order,
                            'coin':     coin,
                            'init_off': info['init_off'],
                            'n_dp':     n_dp2,
                            'policy':   policy,
                        }
                        logger.info(
                            f'  {tag} {coin}: リプライス oid={oid}→{alo_res.oid}'
                            f' limit={limit_px}'
                            f' (offset={new_off:.3%}, elapsed={elapsed_min:.0f}min'
                            f'/{policy["timeout_min"]}min)'
                        )
                    elif alo_res.status == 'filled':
                        executed += 1
                        del pending[oid]
                    else:
                        # 'rejected' / 'error' → IOC フォールバック
                        ok = _place_ioc(order, exchange, logger)
                        if ok:
                            executed += 1
                        else:
                            failed += 1
                        del pending[oid]
            time.sleep(REQUEST_DELAY)

        if not pending:
            logger.info(f'{tag} 全注文完了')
            break

        # 全銘柄タイムアウト超過の安全弁
        if elapsed_min >= max_timeout + min_check:
            logger.warning(f'{tag} 最大タイムアウト超過 → 残り {len(pending)} 件を強制 IOC')
            for oid, info in list(pending.items()):
                _cancel_order(info['coin'], oid, exchange, logger)
                time.sleep(REQUEST_DELAY)
                ok = _place_ioc(info['order'], exchange, logger)
                if ok:
                    executed += 1
                else:
                    failed += 1
            pending.clear()
            break

        # 次ループのチェック間隔を再計算 (pending が変化した可能性があるため)
        if pending:
            min_check = min(info['policy']['check_every'] for info in pending.values())

    return executed, failed


# ──────────────────────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── 引数解析 ───────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='X1 Donchian(LS) Hyperliquid perp 本番スクリプト (n=4 トランシェ + X1 動的タイムアウト)'
    )
    parser.add_argument(
        '--live', action='store_true',
        help='実際に注文を送る（指定なし = ドライラン）',
    )
    parser.add_argument(
        '--force-reb', action='store_true',
        help='実行曜日に関わらず強制フルリバランス（トランシェ ID は曜日から自動決定）',
    )
    parser.add_argument(
        '--reset-state', action='store_true',
        help='全トランシェの state をリセットして初期状態から再開する',
    )
    parser.add_argument(
        '--tranche', type=int, default=None, metavar='ID',
        help=f'トランシェ ID を手動指定 (0~{TRANCHE_N-1})。省略時は実行曜日から自動決定。',
    )
    args = parser.parse_args()

    LIVE = args.live

    # ── ロガー初期化 ───────────────────────────────────────────────────────────
    logger = setup_logging(_LOG_PATH)

    # ── 並行実行ロック ─────────────────────────────────────────────────────────
    _lock_fh = open(_LOCK_PATH, 'w')
    try:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error('別のインスタンスが実行中です (donchian.lock)。終了します。')
        sys.exit(1)

    logger.info('=' * 70)
    logger.info('=== Donchian HL Bot 起動 ===')
    logger.info(f'MODE: {"LIVE 本番実行" if LIVE else "DRY RUN（--live を渡すと実際に注文）"}')
    logger.info(f'実行時刻 (UTC): {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('=' * 70)

    # ── .env 読み込み ──────────────────────────────────────────────────────────
    if _ENV_PATH.exists():
        load_dotenv(str(_ENV_PATH))
        logger.info(f'.env 読み込み: {_ENV_PATH}')
    else:
        load_dotenv()  # デフォルト位置を試みる
        logger.warning(f'.env が見つかりません: {_ENV_PATH}')

    PRIVATE_KEY    = os.getenv('PRIVATE_KEY')
    PUBLIC_ADDRESS = os.getenv('PUBLIC_ADDRESS')

    # LIVE モードでは PRIVATE_KEY 必須
    if LIVE and not PRIVATE_KEY:
        logger.error('LIVE モードには PRIVATE_KEY が必要です。.env を確認してください。')
        sys.exit(1)

    # DRY RUN の場合、PRIVATE_KEY がなくても PUBLIC_ADDRESS で読み取りのみ可能
    if not LIVE and not PRIVATE_KEY and not PUBLIC_ADDRESS:
        logger.warning(
            'PRIVATE_KEY / PUBLIC_ADDRESS ともに未設定。'
            'アカウント残高取得をスキップし、仮の equity ($10,000) でドライランを継続します。'
        )

    # ── SDK 初期化 ─────────────────────────────────────────────────────────────
    account  = None
    address  = PUBLIC_ADDRESS  # デフォルト: PUBLIC_ADDRESS のみ (DRY RUN 向け)
    exchange = None

    if PRIVATE_KEY:
        try:
            from eth_account import Account as EthAccount
        except ImportError as e:
            logger.error(f'eth-account インポート失敗: {e}')
            logger.error('pip install eth-account を実行してください。')
            sys.exit(1)
        account = EthAccount.from_key(PRIVATE_KEY)
        address = PUBLIC_ADDRESS if PUBLIC_ADDRESS else account.address

    if LIVE:
        try:
            from hyperliquid.utils import constants
            from hyperliquid.exchange import Exchange
        except ImportError as e:
            logger.error(f'SDK インポート失敗: {e}')
            logger.error('pip install hyperliquid-python-sdk を実行してください。')
            sys.exit(1)
        exchange = Exchange(account, base_url=constants.MAINNET_API_URL)
        logger.info('Exchange 初期化: MAINNET')

    logger.info(f'アドレス: {address if address else "(未設定 → DRY RUN 仮残高使用)"}')

    # ── トランシェ ID 決定 ─────────────────────────────────────────────────────
    _DOW_NAMES = ['月', '火', '水', '木', '金', '土', '日']
    today_dow  = datetime.now(timezone.utc).weekday()  # 0=月,...,6=日

    if args.tranche is not None:
        tranche_id = args.tranche
        if not (0 <= tranche_id < TRANCHE_N):
            logger.error(f'--tranche {tranche_id} は範囲外 (0~{TRANCHE_N-1})。終了します。')
            sys.exit(1)
        logger.info(
            f'トランシェ ID: {tranche_id} (--tranche で手動指定, 本日曜日={_DOW_NAMES[today_dow]})'
        )
    elif today_dow in TRANCHE_EVAL_DAYS:
        tranche_id = TRANCHE_EVAL_DAYS.index(today_dow)
        logger.info(
            f'トランシェ ID: {tranche_id} '
            f'(曜日={_DOW_NAMES[today_dow]}, TRANCHE_EVAL_DAYS={[_DOW_NAMES[d] for d in TRANCHE_EVAL_DAYS]})'
        )
    else:
        logger.error(
            f'本日 ({_DOW_NAMES[today_dow]}曜日) はトランシェ実行日ではありません。'
            f'実行日: {[_DOW_NAMES[d] for d in TRANCHE_EVAL_DAYS]} → 終了します。'
        )
        logger.error('手動実行する場合は --tranche INT を指定してください。')
        sys.exit(1)

    # n=4 トランシェでは毎回フルリバランス (do_reb=True)
    do_reb = True
    if args.force_reb:
        logger.info('  --force-reb が指定されています (n=4 トランシェでは常に do_reb=True)')

    # ── State 読み込み ─────────────────────────────────────────────────────────
    if args.reset_state:
        state = _default_state()
        save_state(_STATE_PATH, state)
        logger.info('--reset-state: 全トランシェの state をリセットしました')

    state = load_state(_STATE_PATH)

    # tranche_targets の整合性チェック (欠損トランシェを補完)
    tranche_targets = state.get('tranche_targets', {})
    for i in range(TRANCHE_N):
        if str(i) not in tranche_targets:
            tranche_targets[str(i)] = {}
    state['tranche_targets'] = tranche_targets

    logger.info(
        f'State 読み込み: last_run={state.get("last_run")}, '
        f'last_tranche={state.get("last_tranche")}'
    )
    for i in range(TRANCHE_N):
        n_pos = len(tranche_targets.get(str(i), {}))
        logger.info(f'  tranche {i}: {n_pos} ポジション (state)')

    # ────────────────────────────────────────────────────────────────────────
    # Step 1: HL ユニバース取得
    # ────────────────────────────────────────────────────────────────────────
    logger.info('[Step 1] HL ユニバース取得...')
    try:
        universe_meta = get_hl_universe()
    except Exception as e:
        logger.error(f'HL ユニバース取得失敗: {e}')
        sys.exit(1)
    logger.info(f'HL perp 銘柄数: {len(universe_meta)}')

    # ────────────────────────────────────────────────────────────────────────
    # Step 2: 対象銘柄決定（上場期間フィルタなし）
    # ────────────────────────────────────────────────────────────────────────
    eligible_coins = list(universe_meta.keys())
    logger.info(f'[Step 2] 全 {len(eligible_coins)} 銘柄を対象（上場期間フィルタなし）')
    logger.info(f'  ※ データ不足銘柄は OHLCV 取得後に自動スキップ（N_HIGH={N_HIGH}日未満）')

    if not eligible_coins:
        logger.error('対象銘柄が0件。終了します。')
        sys.exit(1)

    # ────────────────────────────────────────────────────────────────────────
    # Step 3: 日足 OHLCV 取得
    # ────────────────────────────────────────────────────────────────────────
    logger.info(f'[Step 3] 日足 OHLCV 取得 (過去 {DATA_DAYS} 日)...')
    try:
        ohlcv = fetch_ohlcv_all(eligible_coins, DATA_DAYS, logger)
    except Exception as e:
        logger.error(f'OHLCV 取得失敗: {e}')
        sys.exit(1)

    if not ohlcv:
        logger.error('有効な OHLCV データがありません。終了します。')
        sys.exit(1)

    # ────────────────────────────────────────────────────────────────────────
    # Step 3.5: ファンディングレート取得 & FR Z-score 計算 (Candidate A)
    # ────────────────────────────────────────────────────────────────────────
    logger.info(f'[Step 3.5] ファンディングレート取得 (過去 {FR_FETCH_DAYS} 日)...')
    try:
        fr_daily = fetch_funding_rates_recent(list(ohlcv.keys()), FR_FETCH_DAYS, logger)
        fr_zscore_latest = compute_fr_zscore_latest(fr_daily, CARRY_WINDOW, list(ohlcv.keys()))
        n_fr_valid = sum(1 for v in fr_zscore_latest.values() if v != 0.0)
        logger.info(
            f'[FR Z-score] CARRY_WINDOW={CARRY_WINDOW}d, '
            f'有効銘柄={n_fr_valid}/{len(fr_zscore_latest)}'
        )
    except Exception as e:
        logger.warning(f'FR 取得失敗: {e} → FR Z-score を 0 (FR ブレンドなし) にフォールバック')
        fr_zscore_latest = {}

    # ────────────────────────────────────────────────────────────────────────
    # Step 4: シグナル計算
    # ────────────────────────────────────────────────────────────────────────
    logger.info('[Step 4] シグナル計算...')
    sig = compute_signals(ohlcv, logger, fr_zscore_latest=fr_zscore_latest)

    if sig.empty:
        logger.error('シグナル計算結果が空。終了します。')
        sys.exit(1)

    # [B] 出来高フィルタ通過率ログ
    n_vol_pass = int(sig['vol_ratio'].apply(
        lambda r: (not np.isnan(r)) and r >= VOL_THRESH
    ).sum()) if 'vol_ratio' in sig.columns else len(sig)
    logger.info(
        f'[Volume] 出来高フィルタ通過: {n_vol_pass}/{len(sig)} 銘柄 '
        f'(vol_{VOL_SHORT}d/vol_{VOL_LONG}d >= {VOL_THRESH})'
    )

    # [A] BTC Regime フィルタ適用 → is_bear フラグも取得
    sig, is_bear = apply_btc_regime_filter(sig, ohlcv, logger)

    n_long_cands  = int(sig['filter_long'].sum())
    n_short_cands = int(sig['filter_short'].sum())
    regime_label  = 'Bear (ロング停止・ショート集中)' if is_bear else 'Bull (FR ブレンドショート)'
    logger.info(
        f'シグナル計算完了: 計 {len(sig)} 銘柄 / '
        f'Long候補 {n_long_cands} / Short候補 {n_short_cands} / Regime={regime_label}'
    )

    # ── デバッグ: スコア上位5 / 下位5 を表示 ──────────────────────────────────
    debug_cols = ['score', 'score_short_bull', 'close', 'sma100', 'vol_ratio', 'filter_long']
    debug_cols = [c for c in debug_cols if c in sig.columns]
    top5 = sig.nlargest(5, 'score')[debug_cols]
    bot5 = sig.nsmallest(5, 'score')[debug_cols]
    logger.debug(f'score 上位5:\n{top5.to_string()}')
    logger.debug(f'score 下位5:\n{bot5.to_string()}')

    # ────────────────────────────────────────────────────────────────────────
    # Step 5: 現在ポジション取得
    # ────────────────────────────────────────────────────────────────────────
    logger.info('[Step 5] 現在のアカウント状態取得...')
    DUMMY_EQUITY = 10_000.0   # address 未設定時の仮残高

    if address:
        try:
            acct = get_account_state(address, logger)
        except Exception as e:
            logger.error(f'アカウント状態取得失敗: {e}')
            sys.exit(1)
        equity  = acct['equity']
        current = acct['positions']
    else:
        logger.warning(f'ADDRESS 未設定 → 仮 equity ${DUMMY_EQUITY:,.0f} / 現在ポジション空でドライラン')
        equity  = DUMMY_EQUITY
        current = {}

    logger.info(f'AccountValue: ${equity:,.2f}')
    if current:
        logger.info(f'現在保有ポジション ({len(current)} 銘柄):')
        for coin, pos in current.items():
            qty   = pos['qty']
            price = pos['entry_px']
            logger.info(f'  {"L" if qty > 0 else "S"} {coin:12s} qty={qty:+.4f} entry=${price:.4f}')
    else:
        logger.info('現在保有ポジション: なし')

    if equity < 10:
        logger.error(f'AccountValue ${equity:.2f} が小さすぎます。終了します。')
        sys.exit(1)

    # n=4 トランシェ: このトランシェに割り当てる資産 = 総資産 / TRANCHE_N
    tranche_equity = equity / TRANCHE_N
    logger.info(
        f'トランシェ資産: ${tranche_equity:,.2f} (総資産 ${equity:,.2f} の 1/{TRANCHE_N})'
    )

    # ────────────────────────────────────────────────────────────────────────
    # Step 6: Mid 価格取得
    # ────────────────────────────────────────────────────────────────────────
    logger.info('[Step 6] 現在価格取得...')
    try:
        mid_prices = get_all_mids()
    except Exception as e:
        logger.error(f'価格取得失敗: {e}')
        sys.exit(1)
    logger.info(f'価格取得完了: {len(mid_prices)} 銘柄')

    # ────────────────────────────────────────────────────────────────────────
    # Step 5.5: ペーパーポートフォリオ P&L 更新 → 動的レバレッジ計算
    # ────────────────────────────────────────────────────────────────────────
    # ペーパーポートフォリオ: DynLev スケールなし（MAX_GROSS_LEV 固定）の仮想損益を追跡。
    # バックテスト案2と同様に「素のリターン」で EWM Sharpe を計算し stuck-at-0 を防止。
    #   prev_paper_positions: 前週のベースラインターゲット {coin: qty}
    #   prev_paper_prices   : 前週の mid 価格 {coin: price}
    #   paper_equity        : 前週末時点のペーパー equity 値 (トランシェ単位)
    paper_positions_all   = state.get('paper_positions', {})
    paper_prices_all      = state.get('paper_prices', {})
    paper_equities_all    = state.get('paper_equities', {})
    paper_equity_hist_all = state.get('paper_equity_history', {})

    prev_pp = paper_positions_all.get(str(tranche_id), {})
    prev_pr = paper_prices_all.get(str(tranche_id), {})
    prev_pe = float(paper_equities_all.get(str(tranche_id), tranche_equity))

    # 前週ポジションの損益 = Σ qty_i × (mid_now_i − mid_prev_i)
    # 価格が欠損している銘柄があると PnL が過小評価されるため、欠損銘柄数を記録する
    missing_price_coins = [c for c in prev_pr if c not in mid_prices]
    if missing_price_coins:
        logger.warning(
            f'[Paper] 価格欠損銘柄 {len(missing_price_coins)} 件: {missing_price_coins} '
            f'→ paper equity 更新をスキップ (前週値 {prev_pe:,.2f} を引き継ぎ)'
        )
        new_paper_equity = prev_pe   # スキップ: 前週値をそのまま引き継ぐ
        paper_pnl        = 0.0       # ログ参照用のデフォルト値
        skip_paper_update = True
    else:
        paper_pnl = sum(
            prev_pp.get(c, 0.0) * (mid_prices[c] - prev_pr[c])
            for c in prev_pr
        )
        new_paper_equity = max(prev_pe + paper_pnl, 1.0)  # ゼロ以下を防止
        skip_paper_update = False

    paper_hist = list(paper_equity_hist_all.get(str(tranche_id), []))
    if not skip_paper_update:
        paper_hist.append(new_paper_equity)
        paper_hist = paper_hist[-52:]

    # ── paper_hist バックフィル ───────────────────────────────────────────────
    # paper_equity_history が MEQ_SMA_WINDOW 未満の場合、equity_history（dict形式）
    # から補完する。MAX_GROSS_LEV=1.0 で運用していた期間は実際の equity ≈ paper equity。
    if len(paper_hist) < MEQ_SMA_WINDOW:
        raw_eq_hist = state.get('equity_history', {})
        src_total = raw_eq_hist.get(str(tranche_id), []) if isinstance(raw_eq_hist, dict) else []

        if src_total:
            # 合計 equity → トランシェ単位に変換
            src_per_tranche = [v / TRANCHE_N for v in src_total]
            # すでに paper_hist にある週数分は末尾から除外して重複を避ける
            n_already = len(paper_hist)
            backfill_src = src_per_tranche[:-n_already] if n_already > 0 else src_per_tranche
            # MEQ_SMA_WINDOW に達するために必要な週数だけ使う
            n_needed = MEQ_SMA_WINDOW - len(paper_hist)
            backfill = backfill_src[-n_needed:] if n_needed > 0 else []
            if backfill:
                paper_hist = list(backfill) + paper_hist
                paper_hist = paper_hist[-52:]
                logger.info(
                    f'[MEQ] paper_hist バックフィル: equity_history から {len(backfill)}週補完 '
                    f'→ 合計 {len(paper_hist)}週'
                )

    logger.info(
        f'[Paper] tranche={tranche_id} | '
        f'prev_eq={prev_pe:,.2f} pnl={paper_pnl:+.2f} → new_eq={new_paper_equity:,.2f} | '
        f'履歴={len(paper_hist)}週 (prev_positions={len(prev_pp)}銘柄)'
    )

    # equity_history は現在週の equity を含まない（1週ラグ） → ルックアヘッドなし
    _eq_hist_all = state.get('equity_history', {})
    equity_history = list(_eq_hist_all.get(str(tranche_id), []))
    eff_max_gross_lev = compute_dynamic_leverage(
        equity_history, logger, paper_equity_history=paper_hist
    )

    logger.info(
        f'[DynLev] 今週の有効グロスレバレッジ上限: {eff_max_gross_lev:.3f}x '
        f'(equity_history 長さ={len(equity_history)}週, is_bear={is_bear})'
    )

    # ── M_eq レバレッジ倍率 ─────────────────────────────────────────────────
    # 変更3: M_eq を全トランシェ共通化
    # 全トランシェの paper_equity_history の中で最長のものを M_eq 計算に使用する。
    # これにより全4トランシェが同一の Bull/Bear シグナルを共有し、
    # トランシェ間でレバレッジ判定が矛盾するリスクを排除する。
    # ※ 現在のトランシェの paper_hist（最新値追加済み）も候補に含める。
    meq_paper_hist = paper_hist  # デフォルト: 現在のトランシェの履歴
    for _tid_str, _h in paper_equity_hist_all.items():
        if len(_h) > len(meq_paper_hist):
            meq_paper_hist = list(_h)
    if meq_paper_hist is not paper_hist:
        logger.info(
            f'[MEQ] 共通化: tranche {tranche_id} ({len(paper_hist)}週) → '
            f'最長トランシェ使用 ({len(meq_paper_hist)}週)'
        )
    meq_lev = compute_meq_leverage(meq_paper_hist, logger)
    # gross lev 上限も MEQ_LEV_HIGH 以上に引き上げ (eff_max_gross_lev がデフォルト 1.0x でも詰まらないよう)
    eff_max_gross_lev = max(eff_max_gross_lev, meq_lev)
    logger.info(
        f'[MEQ] rf_l_scale={meq_lev:.2f}x → eff_max_gross_lev={eff_max_gross_lev:.2f}x'
    )

    # ────────────────────────────────────────────────────────────────────────
    # Step 7: ターゲットポジション計算
    # ────────────────────────────────────────────────────────────────────────
    logger.info(f'[Step 7] ターゲットポジション計算 (tranche={tranche_id}, is_bear={is_bear})...')

    # このトランシェの新ターゲット (1/4 資産ベース, MEQ スケール + DynLev 上限適用)
    new_tranche_target = compute_target_positions(
        sig, tranche_equity, mid_prices, universe_meta, logger,
        is_bear=is_bear, max_gross_lev=eff_max_gross_lev,
        rf_l_scale=meq_lev, rf_s_scale=meq_lev,
    )
    logger.info(
        f'  tranche {tranche_id} 新ターゲット: '
        f'ロング {sum(1 for q in new_tranche_target.values() if q > 0)} / '
        f'ショート {sum(1 for q in new_tranche_target.values() if q < 0)}'
    )

    # ペーパーターゲット: MEQ/DynLev スケールなし（MAX_GROSS_LEV 固定, scale=1.0）で計算
    # → 次週の paper P&L 計算の基準ポジションとして state に保存する
    paper_target = compute_target_positions(
        sig, tranche_equity, mid_prices, universe_meta, logger,
        is_bear=is_bear, max_gross_lev=MAX_GROSS_LEV,
        rf_l_scale=1.0, rf_s_scale=1.0,
    )
    logger.info(
        f'  [Paper] tranche {tranche_id} ペーパーターゲット: '
        f'ロング {sum(1 for q in paper_target.values() if q > 0)} / '
        f'ショート {sum(1 for q in paper_target.values() if q < 0)}'
    )

    # 旧合成ターゲット = 全トランシェの state 合算
    old_combined: dict[str, float] = {}
    for k in range(TRANCHE_N):
        for coin, qty in tranche_targets.get(str(k), {}).items():
            old_combined[coin] = old_combined.get(coin, 0.0) + qty

    # 新合成ターゲット = 旧合成 − 旧 tranche_id + 新 tranche_id
    new_combined = dict(old_combined)
    for coin, qty in tranche_targets.get(str(tranche_id), {}).items():
        new_combined[coin] = new_combined.get(coin, 0.0) - qty
        if abs(new_combined.get(coin, 0.0)) < 1e-10:
            new_combined.pop(coin, None)
    for coin, qty in new_tranche_target.items():
        new_combined[coin] = new_combined.get(coin, 0.0) + qty

    target = {c: q for c, q in new_combined.items() if abs(q) > 1e-10}

    logger.info(
        f'合成ターゲット: ロング {sum(1 for q in target.values() if q > 0)} / '
        f'ショート {sum(1 for q in target.values() if q < 0)} '
        f'(旧合成 {len(old_combined)} → 新合成 {len(target)} 銘柄)'
    )

    # ────────────────────────────────────────────────────────────────────────
    # Step 8 & 9: 注文リスト構築
    # ────────────────────────────────────────────────────────────────────────
    logger.info(f'[Step 8] 注文構築 (do_reb={do_reb})...')
    orders = build_orders(
        target, current, mid_prices, sig, universe_meta, do_reb, logger
    )
    logger.info(f'注文リスト: {len(orders)} 件')

    executed = 0
    failed   = 0
    if not orders:
        logger.info('実行すべき注文がありません。')
    else:
        logger.info('--- 注文実行開始 ---')
        executed, failed = execute_orders(
            orders, exchange, LIVE, logger, address=address, ohlcv=ohlcv,
            eval_date=datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            tranche_id=str(tranche_id),
        )
        logger.info(
            f'--- 注文実行完了: 成功 {executed} / 失敗 {failed} / 合計 {len(orders)} ---'
        )

    # ────────────────────────────────────────────────────────────────────────
    # Step 10: 実行サマリー
    # ────────────────────────────────────────────────────────────────────────
    logger.info('[Step 10] 実行サマリー:')

    # ターゲットポジションのノショナル計算
    tgt_long_notional  = sum(q * mid_prices.get(c, 0) for c, q in target.items() if q > 0)
    tgt_short_notional = sum(abs(q) * mid_prices.get(c, 0) for c, q in target.items() if q < 0)
    tgt_gross          = (tgt_long_notional + tgt_short_notional) / equity if equity > 0 else 0
    tgt_net            = (tgt_long_notional - tgt_short_notional) / equity if equity > 0 else 0

    logger.info(f'  AccountValue     : ${equity:,.2f}')
    logger.info(f'  Target Long      : ${tgt_long_notional:,.2f} (Lev={tgt_long_notional/equity:.2f}x)')
    logger.info(f'  Target Short     : ${tgt_short_notional:,.2f} (Lev={tgt_short_notional/equity:.2f}x)')
    logger.info(f'  Target Gross Lev : {tgt_gross:.2f}x')
    logger.info(f'  Target Net Lev   : {tgt_net:+.2f}x')
    logger.info(f'  注文数           : {len(orders)} 件 (実行: {"DRY RUN" if not LIVE else "LIVE"})')

    # ────────────────────────────────────────────────────────────────────────
    # Step 11: 実行後ポジション検証・自動修正（LIVE のみ）
    # ────────────────────────────────────────────────────────────────────────
    if LIVE and address:
        logger.info('[Step 11] 実行後ポジション検証...')
        time.sleep(3)  # 注文決済を待つ
        try:
            acct2   = get_account_state(address, logger)
            actual  = acct2['positions']  # {coin: {qty, entry_px}}
            retry_orders = []

            # target にあるのに actual と大きく乖離（30%以上）→ 差分を再発注
            for coin, tgt_q in target.items():
                if abs(tgt_q) < 1e-8:
                    continue
                act_q = actual.get(coin, {}).get('qty', 0.0)
                if abs(act_q - tgt_q) > abs(tgt_q) * 0.30:
                    delta = tgt_q - act_q
                    price = mid_prices.get(coin, 0.0)
                    if abs(delta * price) < MIN_ORDER_USD:
                        continue
                    logger.warning(
                        f'  {coin}: target={tgt_q:+.4f} actual={act_q:+.4f}'
                        f' 乖離={abs(act_q-tgt_q)/abs(tgt_q)*100:.0f}% → 再発注'
                    )
                    meta     = universe_meta.get(coin, {})
                    decimals = meta.get('szDecimals', 2)
                    sz       = round_sz(abs(delta), decimals)
                    is_buy   = delta > 0
                    retry_orders.append({
                        'type':         'open_long' if is_buy else 'open_short',
                        'coin':         coin,
                        'is_buy':       is_buy,
                        'sz':           sz,
                        'price':        price,
                        'notional_usd': sz * price,
                        'reduce_only':  False,
                        'reason':       'verify_retry',
                    })

            # actual にあるのに target にない → クローズ
            for coin, pos in actual.items():
                tgt_q = target.get(coin, 0.0)
                if abs(tgt_q) < 1e-8:
                    act_q = pos['qty']
                    price = mid_prices.get(coin, 0.0)
                    if abs(act_q * price) < MIN_ORDER_USD:
                        continue
                    logger.warning(
                        f'  {coin}: 不要ポジション qty={act_q:+.4f} → クローズ'
                    )
                    meta     = universe_meta.get(coin, {})
                    decimals = meta.get('szDecimals', 2)
                    sz       = round_sz(abs(act_q), decimals)
                    is_buy   = act_q < 0
                    retry_orders.append({
                        'type':         'close_long' if act_q > 0 else 'close_short',
                        'coin':         coin,
                        'is_buy':       is_buy,
                        'sz':           sz,
                        'price':        price,
                        'notional_usd': sz * price,
                        'reduce_only':  True,
                        'reason':       'verify_close',
                    })

            if retry_orders:
                logger.info(f'  乖離検出: {len(retry_orders)} 件を再発注 [ペーパー外調整]')
                execute_orders(retry_orders, exchange, LIVE, logger, address=address, ohlcv=ohlcv)
            else:
                logger.info('  全ポジション正常 ✓')

        except Exception as e:
            logger.error(f'ポジション検証失敗: {e}')
            logger.debug(traceback.format_exc())

    # ── State 更新 (ライブモードのみ) ─────────────────────────────────────────
    if LIVE:
        # 注文の一部でも失敗した場合は state を更新しない（次回に実口座と整合した差分計算をさせる）
        # executed==0 かつ orders>0: 全件失敗
        # failed>0: 部分失敗（約定した注文があっても state を前進させると乖離が累積する）
        any_failed = failed > 0 or (len(orders) > 0 and executed == 0)
        if any_failed:
            logger.warning(
                f'注文失敗あり (executed={executed}, failed={failed}) → '
                f'tranche {tranche_id} の state を更新しません。'
                f'次回実行時に実口座ポジションとの差分で再試行します。'
            )
        else:
            tranche_targets[str(tranche_id)] = new_tranche_target
            state['tranche_targets'] = tranche_targets
            state['last_run']        = datetime.now(timezone.utc).isoformat()
            state['last_tranche']    = tranche_id
            # A_R: equity_history を更新（今週実行開始時の tranche_equity を追記、最大52週保持）
            # tranche 単位の equity を保存することで backfill 時のスケールと一致させる
            eq_hist_all = state.get('equity_history', {})
            if isinstance(eq_hist_all, list):
                eq_hist_all = {}  # 旧フォーマット（flat list）からの移行
            eq_hist = list(eq_hist_all.get(str(tranche_id), []))
            eq_hist.append(float(tranche_equity))   # 総口座ではなくトランシェ単位で保存
            eq_hist_all[str(tranche_id)] = eq_hist[-52:]
            state['equity_history'] = eq_hist_all
            logger.info(
                f'[DynLev] equity_history 更新: tranche={tranche_id} {len(eq_hist)}週分 '
                f'(最新=${tranche_equity:,.2f} = 総資産${equity:,.2f}/4)'
            )
            # ── ペーパーポートフォリオ state 更新 ──────────────────────────
            # 今週のペーパーターゲット・価格・equity を保存 → 次週の P&L 計算に使用
            paper_positions_all[str(tranche_id)] = paper_target
            paper_prices_all[str(tranche_id)]    = {
                c: float(mid_prices[c]) for c in paper_target if c in mid_prices
            }
            paper_equities_all[str(tranche_id)]  = float(new_paper_equity)
            paper_equity_hist_all[str(tranche_id)] = paper_hist
            state['paper_positions']      = paper_positions_all
            state['paper_prices']         = paper_prices_all
            state['paper_equities']       = paper_equities_all
            state['paper_equity_history'] = paper_equity_hist_all
            logger.info(
                f'[Paper] state 更新: tranche {tranche_id} | '
                f'paper_equity={new_paper_equity:,.2f} | '
                f'paper_positions={len(paper_target)}銘柄 | '
                f'履歴={len(paper_hist)}週'
            )
            save_state(_STATE_PATH, state)
            logger.info(
                f'State 更新: tranche {tranche_id} ターゲット {len(new_tranche_target)} 銘柄 → '
                f'{_STATE_PATH}'
            )
    else:
        logger.info(
            f'DRY RUN: State 更新スキップ '
            f'(tranche {tranche_id} targets={len(new_tranche_target)} 銘柄)'
        )

    logger.info('=== Donchian HL Bot 完了 ===')
    logger.info('=' * 70)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
