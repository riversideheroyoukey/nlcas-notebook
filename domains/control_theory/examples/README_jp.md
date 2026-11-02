# control_theory — 実例（RLC 回路の伝達関数）

このフォルダは、制御理論シナリオを実際に実行して得た**教育用の実例**です。
standard（**Maxima**）セッション・`control_theory` ドメインで再現できます。

| シナリオ | ファイル | 内容 |
|---|---|---|
| [scenario_rlc_transfer_function.md](../../../docs/scenarios/scenario_rlc_transfer_function_jp.md) | `scenario_rlc_transfer_function_jp.{json,tex}` | 直列 RLC の運動方程式 $L\ddot q+R\dot q+q/C=v$ を Laplace 変換し、伝達関数 $H(s)=\dfrac{1}{Ls^2+Rs+1/C}$ を導く。時間微分は `全微分` で本物の $d/dt$ を作り、伝達関数は定義どおりゼロ初期条件（リークなし）で抽出 |

各ファイル：`.json` は実行後のセッション（`sessions/` に置けばロード可能）、`.tex` は export 文書
（uplatex/dvipdfmx でコンパイル可）。

**入力のポイント:**
- `q を時間依存関数として定義する` で $q(t)$ を宣言し、`全微分` で本物の時間微分 $\dot q,\ddot q$ を作る
  （`\ddot{q}` 記法は解析力学では独立アトムになり Laplace 用の微分にならない）。
- ODE は数式で `L*E3 + R*E2 + E1/C = v` のように登録（左辺を保つため）。
- `E5 の伝達関数 q/v を求めて` は s 領域の式から $Q,V$ の係数を取り出して $H(s)$ を得る
  （初期条件項は自動で落ちる。一般の Laplace は影響を受けない）。
