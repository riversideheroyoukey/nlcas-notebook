# magnetic_resonance — 実例（回転座標系の有効ハミルトニアン）

このフォルダは、磁気共鳴シナリオを実際に実行して得た**教育用の実例**です。
standard（**Maxima**）セッション・`magnetic_resonance` ドメインで再現できます（スピン代数は CAS 非経由）。

| シナリオ | ファイル | 内容 |
|---|---|---|
| [scenario_rotating_frame.md](../../../docs/scenarios/scenario_rotating_frame_jp.md) | `scenario_rotating_frame_{jp,en}.{json,tex}` | NMR/ESR の回転座標系変換を**一般のスピン操作の連続**で：カスタムスピン H $H_{\mathrm{lab}}=-\omega_0 S^z-\omega_1 S^x$ を登録 → `rotating_frame` 変換で $H_{\mathrm{rot}}=(\omega-\omega_0) S^z-\omega_1 S^x$ → `effective_field` $\omega_{\mathrm{eff}}=\sqrt{(\omega-\omega_0)^2+\omega_1^2}$ → 一般 `commutator` で歳差 EOM $[H_{\mathrm{rot}},S^z]=i\omega_1 S^y$ |

各ファイル：`.json` は実行後のセッション（`sessions/` に置けばロード可能）、`.tex` は export 文書
（uplatex/dvipdfmx でコンパイル可）。日本語版＝`_jp`／英語版＝`_en`。

**入力のポイント:**
- 導出を **4 つの一般操作**で順に解く：`H_{lab} = -\omega_0 S^z_1 - \omega_1 S^x_1`（スピン H 登録）→
  `E1 を \omega で回転座標系に移して`（`rotating_frame` 変換）→ `E2 の有効磁場を求めて`（`effective_field`）
  → `E2 と S^z_1 の交換子を求めて`（一般 `commutator`＝歳差 EOM）。
- **シナリオ専用の焼き込みは無し**。各操作は他のスピン式にも使える一般操作で、スピン式は
  `parse_spin_expression` で構造を保って連鎖する。離調 $\Omega=\omega_0-\omega$ は表示上の略記で、
  必要なら $(\omega-\omega_0)\to\Omega$ の一般代入を 1 段挟む。
