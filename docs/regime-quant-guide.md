# MNQ/MES 震荡区间 & 趋势识别器 — 使用指南

适用于：CME Micro E-mini Nasdaq-100 (MNQ) / Micro E-mini S&P 500 (MES)（同样适用于 NQ/ES）。
两个版本：**TradingView 指标版**（优先交付）与 **Freqtrade 量化自动交易版**。
设计契约：`docs/regime-indicator-spec.md`（§2 参数、§3 计算、§4 信号、§6 平价规则）。

## 交付物地图
| 文件 | 说明 |
|---|---|
| `tradingview/mnq_mes_range_trend.pine` | Pine Script v6 指标（叠加图 + 状态着色 + 信号标记 + 表格 + 9 个 alertcondition） |
| `mnq_mes/src/mrind.py` | 规范参考实现（TA-Lib），Pine 与策略共同契约；P2 策略直接复用 |
| `mnq_mes/strategies/regime_range_trend.py` | Freqtrade 策略 `RegimeRangeTrend`（多空）+ `RegimeRangeTrendLong`（多头变体） |
| `mnq_mes/scripts/` | download_data（Yahoo MES=F/MNQ=F）→ convert_to_freqtrade（feather）→ make_futures_data（futures 三件套）→ validate_regime（统计+图） |
| `mnq_mes/config/` | backtest_mnq.json / backtest_mes.json / backtest_okx_crypto.json / download_okx.json |
| `mnq_mes/data/` | 下载的数据与回测产物（gitignored，脚本可重建） |

## 一、TradingView 指标版（P1）
将 `tradingview/mnq_mes_range_trend.pine` 粘贴到 TradingView Pine 编辑器即可加载（Pine v6）。

- 主图：Donchian 通道（20）与 EMA9/21；可选 Bollinger(20,2)；背景按状态着色
  （TREND_UP 绿 / TREND_DOWN 红 / RANGE 蓝灰）；入场信号 ▲/▼ 并标注标签。
- 右上表格：当前状态、ADX、span/ATR、RSI。
- 信号标签：`RANGE_LONG/SHORT`（区间均值回归）、`TREND_LONG/SHORT`（趋势回踩）、
  `BRKOUT_LONG/SHORT`（区间→趋势突破）。同根 K 优先级 BRKOUT > TREND > RANGE。
- 9 个 alertcondition：三个状态切换 + 六类信号。
- 参数在"Regime"/"Display"两组中，默认值见 spec §2。

## 二、Freqtrade 量化版（P2）
策略复用 `mrind.py`（`compute_regime` + `signal_events`），入场/出场按 spec §4。
注意（如实说明）：
- 成交假设：freqtrade 将信号列**后移一根**（防未来函数），信号在第 t 根触发、**第 t+1 根开盘**成交。
- 止损：`custom_stoploss` 实现 spec §4 的 ATR 跟踪止损（2×ATR，自运行高点/低点测算，
  freqtrade 只能在 K 线高低点触发，故为文档化的近似）；ATR 取已收盘的前一根。
- 合成市场注入（`bot_start`）：MNQ/USD:USD 不在加密交易所市场表中，回测时注入最小 swap
  市场条目。**仅用于回测**；实盘 MNQ/MES 需真实券商连接器（CME 微型期货不在 gate/okx 上市）。

### 数据管线（重建数据）
```powershell
& ".venv\Scripts\python.exe" mnq_mes\scripts\download_data.py      # Yahoo MES=F / MNQ=F 日线→CSV
& ".venv\Scripts\python.exe" mnq_mes\scripts\convert_to_freqtrade.py # CSV→feather（1d+1h；含 futures/mark/funding 三件套，funding 仅 1d）
& ".venv\Scripts\python.exe" mnq_mes\scripts\validate_regime.py      # 状态统计 + 图表
```

### 回测命令（离线；仅启动时加载交易所市场表）
```powershell
& ".venv\Scripts\freqtrade.exe" backtesting -c mnq_mes\config\backtest_mnq.json `
  -s RegimeRangeTrend --strategy-path mnq_mes\strategies `
  --datadir mnq_mes\data\freqtrade\binance --timeframe 1d --timerange 20200101- --userdir .\user_data
# 多头变体：把 -s RegimeRangeTrendLong 换入；MES 用 backtest_mes.json。
# 真实加密数据验证（BTC/USDT、ETH/USDT，okx）：
& ".venv\Scripts\freqtrade.exe" backtesting -c mnq_mes\config\backtest_okx_crypto.json `
  -s RegimeRangeTrend --strategy-path mnq_mes\strategies `
  --datadir mnq_mes\data\freqtrade\okx --timeframe 1d --timerange 20210101- --userdir .\user_data
```
网络说明：本机直连多数加密交易所被墙；gate.io 可直连（回测 stub 默认用它），okx 需本地代理
`httpProxy=http://127.0.0.1:8888`（见 `download_okx.json`/`backtest_okx_crypto.json`）。

## 三、回测结果（诚实报告）
区间 2020-01-01→2026-08-25，日线，期货模式（隔离保证金），手续费 0.01%/边，起始 5 万美元，unlimited stake。
（止损均已修复并验证：ATR 跟踪止损正确生效、无异常回退。）

| 配置 | MNQ | MES |
|---|---|---|
| RegimeRangeTrend（多空，ATR 止损） | 420 笔，**-86.2%**，胜率 32.6% | 402 笔，**-73.5%**，胜率 34.6% |
| RegimeRangeTrendLong（多头，ATR 止损） | 220 笔，**-43.1%**，胜率 32.7% | 203 笔，**-25.4%**，胜率 35.0% |
| RegimeRangeTrend + 无硬止损（仅 ATR+信号） | 410 笔，**-87.7%** | 400 笔，**-73.8%** |
| okx 真实数据 BTC/ETH 日线（止损修复前）※ | BTC -50.5%、ETH -49.5%（合计 522 笔 -100%） | — |

※ 该行使用 fee 0.05%/边、钱包 10 万美元、且为 custom_stoploss 修复前结果——仅作"信号在加密日线同样亏损"的跨资产佐证，不可与上表 v2 行直接比较。
无硬止损变体：将 config 的 `stoploss` 设为 `-0.99` 即可（custom_stoploss 的"只收紧"clamp 会自动放宽；见上表第三行）。

分桶观察（MNQ 多空）：ATR 跟踪止损桶 **+5.7%**（工作正常、净正）；-5% 硬止损 34 次 ≈ -65.8%
（unlimited stake 复利放大）；exit_signal 出场 ≈ -26%。移除硬止损后 exit_signal 变为 -80%——
**亏损根源是入场/出场信号本身在日线上缺乏边际（胜率约 33-35%），不是风控参数**。

**结论（如实）**：默认参数下，该状态机策略在 MNQ/MES 日线上**不存在正期望**——多空、多头变体、
不同止损形态均亏损。主因：(1) 2019-2026 强趋势市（MNQ +235%）下对称做空信号被碾压；
(2) 日线入场信号无边际优势。**指标版（P1）是本次主要交付物**；量化版交付的是**经过验证、
可复现的完整管线 + 诚实的基线结果**。

### 数据与回测注意点
- Yahoo `=F` 为前月连续拼接，换月有价格跳空（未回填修正）；日线覆盖 2019-05-06 起，1h 仅最近约 60 天。
- freqtrade 回测跳过非交易日（`validate_row`），不会产生周末平盘虚假 K 线。
- CME 微型期货无资金费率（funding=0），mark≈期货价——合成文件如实反映。
- 手续费 0.01%/边对真实 CME 成本（约半跳至 1 跳）偏乐观，会小幅美化结果。
- 若要完整日内历史：Databento（GLBX.MDP3，`stype_in="continuous"` 回填连续合约，免费额度 $125）。

### 后续方向（优先级参考）
1. 日内周期（1h/15m）+ Databento 数据：日线被长期趋势主导，日内区间/趋势周期更短。
2. 做空过滤（ADX 阈值/长期均线方向确认）——注意对 2019-2026 单段回测有拟合风险。
3. 参数搜索（hyperopt）——同样有拟合风险；建议仅在换用真实连续数据与更长样本后谨慎进行。
4. 实盘 MNQ/MES 需券商连接（CME 经纪商 API），freqtrade 交易所适配器仅覆盖加密市场。

## 四、复现
- P1 验证：`validate_regime.py` 输出真实 MES/MNQ 的状态分布/信号统计（见 deepwork 进度文件）。
- P2 验证：上述回测命令可复现全部数字（结果确定性已双跑校验）。
