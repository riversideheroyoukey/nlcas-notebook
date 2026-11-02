# electromagnetism — 実例（Maxwell 方程式の整合確認）

このフォルダは、電磁気シナリオを実際に実行して得た**教育用の実例**です。
standard（**Maxima**）セッション・`electromagnetism` ドメインで再現できます。

| シナリオ | ファイル | 内容 |
|---|---|---|
| [scenario_maxwell_consistency.md](../../../docs/scenarios/scenario_maxwell_consistency_jp.md) | `scenario_maxwell_consistency_jp.{json,tex}` | 平面電磁波 $\boldsymbol{E}=(0,\cos(kx-\omega t),0)$, $\boldsymbol{B}=(0,0,\cos(kx-\omega t)/c)$ で $\nabla\!\cdot\!E=\nabla\!\cdot\!B=0$、$\nabla\!\times\!E$/$\nabla\!\times\!B$、$\partial\boldsymbol{B}/\partial t$ を成分ごとに作って再組立て、Faraday の $z$ 成分等式から分散関係 $\omega=ck$ を導く |
| [scenario_coordinate_jacobian.md](../../../docs/scenarios/scenario_coordinate_jacobian_jp.md) | `scenario_coordinate_jacobian_jp.{json,tex}` | **定積分の変数変換**。範囲つき定積分を**表示専用の未評価オブジェクト**として与え（領域述語 `x^2+y^2<=1` や LaTeX `\int\int`）、変数変換すると**新しい範囲つき定積分**（被積分関数·\|J\|＋新範囲）が組み立て・表示され、評価で数値に。極座標·単位円板 $\to\pi/2$、球座標·単位球の体積 $\to 4\pi/3$、一般ヤコビアン（利用者が新範囲を指定）。変換も評価も **LLM 非依存** |

各ファイル：`.json` は実行後のセッション（`sessions/` に置けばロード可能）、`.tex` は export 文書
（uplatex/dvipdfmx でコンパイル可）。

**入力のポイント:**
- `E をベクトル場として定義する` でベクトル場を宣言（成分 `E_x,E_y,E_z` を抽象登録）、各成分は `E_y = …`
  の等式で束縛する。
- 発散・回転は **ナブラ記法を式として入力**（`\nabla \cdot E`・`\nabla \times E`）するとその場で計算される
  （`div E`・`curl E` でも可）。
- $\partial\boldsymbol{B}/\partial t$ は**成分ごとに偏微分**（`B_z を t で偏微分して` — $B_z(x,t)$ は多変数なので
  偏微分を明示）→ 新しいベクトル場 `dBdt` に組み直して利用。
- ベクトル等式の比較は**成分ごと**（$z$ 成分の等式を登録 → `w について解いて` → $\omega=ck$）。
