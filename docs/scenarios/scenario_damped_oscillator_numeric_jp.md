# 実践ループ サンプルシナリオ：減衰振動子 — ODE 一般解 → 明示的 assume → 初期条件で決定 → 数値層フォーク（S-x）

**物理的テーマ:** 減衰振動子 $\ddot x+2\gamma\dot x+\omega_0^2x=0$ の一般解を求め、初期条件で任意定数を
決定し、理論層の式をそのまま数値層（N-1）へフォークして減衰波形を描く。理論→数値の**還流一周**を初めて
実データで確認するシナリオ。
**バックエンド:** standard（Maxima）
**所要時間:** 約 10 分（理論層）＋ 数値層フォーク（API/スクリプト操作）
**検証目的:**
1. 判別式の符号のような**問題依存の物理選択はコードにハードコードせず**、判別式を明示的に定義して
   `assume()` するステップとしてセッションに記録すること（CLAUDE.md (reference omitted from the public snapshot) 参照）。
2. 一般解 → 初期条件で任意定数を決定 → 右辺抽出 → 数値層フォーク → sandbox 実行 → 来歴付き還流、の
   一気通貫。

---

## 物理的背景

減衰振動子の運動方程式 $\ddot x+2\gamma\dot x+\omega_0^2 x=0$（$\gamma$: 減衰率、$\omega_0$: 固有角振動数、
ともに正の実数）に $x=e^{rt}$（標準的な指数関数の ansatz）を代入すると特性方程式
$r^2+2\gamma r+\omega_0^2=0$ を得る。この判別式 $(2\gamma)^2-4\omega_0^2$ の符号で解の性質が変わる
（不足減衰＝振動しながら減衰／過減衰／臨界減衰）。ここでは判別式の**符号を反転させた量**
$D=4\omega_0^2-(2\gamma)^2$ を明示的に定義し、$D>0$（不足減衰、特性方程式が複素根を持つ）を仮定する。
**この符号は「扱っている問題」に依存する物理的な選択であり、CAS コマンド生成コードが勝手に決めてはならない**
（[feedback_no_hardcoded_modeling_choices] 参照）。

一般解は $x(t)=e^{-\gamma t}\left(k_1\sin(\omega t)+k_2\cos(\omega t)\right)$（$\omega=\sqrt{\omega_0^2-\gamma^2}=\sqrt{D}/2$）
の形の任意定数 $k_1,k_2$ を持つ（FREE-PARAM）。初期条件 $x(0)=1,\ \dot x(0)=0$ で $k_1,k_2$ を決定して
特定の解を得たのち、右辺を数値層へフォークして $t$ を掃引し、減衰波形の図を生成する。

---

## セッション設定

- モード: **standard（Maxima）**
- ドメイン: **differential_equations**（力学・微分方程式）
- 記号入力: **TeX 記法**（`\gamma`, `\omega_0`）を使用。手書き入力層を経由した場合の自然な入力形にそろえる
  （ASCII 綴り `gamma`/`omega_0` を直接打つのは、この記法を経由しない不自然な入力）
- 検証ハーネス: `./scripts/debug_repl.sh --no-llm --lang ja`（理論層はルールベース dispatch で再現可能）
- 数値層（フォーク以降）: 2026-07-06 時点で **NL トリガも UI パネルも未実装**（CLAUDE.md 「Numeric Layer」節の
  想定どおり API/エンジンメソッド経由の操作。下記「数値層ステップ」参照）

---

## 入力ステップ（理論層・NL・UI からそのまま操作可能）

### ステップ 1〜3：変数の準備

まず、時間に依存する変位 $x$ と、減衰率 $\gamma$・固有角振動数 $\omega_0$（ともに正の実数）を準備する。

```
x を時間依存関数として定義する
\gamma を正の実数とする
\omega_0 を正の実数とする
```

**期待する出力:** `\gamma`・`\omega_0` はそれぞれ「を正の実数として登録しました」で確定。
このとき **`Quantity.sign="positive"` として保持され、`assume(gamma > 0)` / `assume(omega_0 > 0)` が
自動的に Maxima へ送信される**（ASM1・ASM2 としてタイムラインに記録。domain/sign 分離の実装 —
[project_domain_sign_split] 参照）。内部識別子は ASCII 化された `gamma`/`omega_0` になるが、表示・以後の
数式入力は `\gamma`/`\omega_0`（TeX）のままでよい。

### ステップ 4：運動方程式の登録

以下の減衰振動子の運動方程式を考える（$\gamma$ は減衰率、$\omega_0$ は固有角振動数）。

```math
\ddot{x} + 2 \gamma \dot{x} + \omega_0^2 x = 0
```

**期待する出力:** `E1`: `\ddot{x} + 2 \gamma \dot{x} + \omega_0^2 x = 0`

### ステップ 5：特性方程式を登録（判別式の由来）

この微分方程式に指数関数の ansatz $x=e^{rt}$ を代入すると（$\dot x=re^{rt},\ \ddot x=r^2e^{rt}$、
両辺を $e^{rt}\neq0$ で割る）、この微分方程式の特性方程式は以下となる。

```math
r^2 + 2 \gamma r + \omega_0^2 = 0
```

**期待する出力:** `E2`: `r^2 + 2 \gamma r + \omega_0^2 = 0`

**検証ポイント:** この式を単なる説明文でなく実際に登録された式（`E2`）として残すことで、以降の
判別式条件がどこから来るかを、シナリオ自身の実行ログの中でたどれるようにしている。

### ステップ 6：判別式 D を定義

特性方程式は $r$ の２次式（$ar^2+br+c=0$ で $a=1,\ b=2\gamma,\ c=\omega_0^2$）なので、以下の判別式
$b^2-4ac=(2\gamma)^2-4\omega_0^2$ を得る。ここでは以降の議論を「正なら不足減衰」という自然な形にするため、
この符号を反転させた量 $D=4\omega_0^2-(2\gamma)^2=-(b^2-4ac)$ を定義する。

```math
D = 4 \omega_0^2 - (2 \gamma)^2
```

**期待する出力:** `E3`: `D = 4 \omega_0^2 - (2 \gamma)^2`

**検証ポイント:** H-1（equation-form 入力）により `D` はシンボルとして登録され、Maxima 側にも
`D : 4*omega_0^2-(2*gamma)^2` として束縛される。$D>0 \Leftrightarrow$ 判別式 $b^2-4ac<0$
（複素根＝振動解＝不足減衰）。

### ステップ 7：不足減衰を想定し、D の符号を明示的に assume（研究者の選択）

ここでは減衰振動（不足減衰、$D>0$）を想定する。この選択は扱っている問題に依存する物理的な判断であり、
ツール側が代わりに決めるものではないので、以下のように明示的に assume する。

```
D > 0 と仮定する
```

**期待する出力:** 青い仮定タイル `ASM3`: `D > 0` が追加される。

**検証ポイント（重要）:**
- この assume がないと、`ode2` の内部で判別式の符号が確定できず、Maxima が対話的に符号を尋ねる
  質問を発行して永続アダプタが**ハングする**（`_build_solve_ode_general` は各記号の正値性は
  `assume(sym>0)` で自動処理するが、判別式のような**複合式の符号**まではカバーしない）。
- 「不足減衰を扱う（$D>0$）」という選択はこのシナリオの前提（**研究者が選ぶもの**）であり、
  `maxima_backend_handler.py` 側にハードコードしてはならない。
- **`D` は `E3` で束縛された量なので、`assume(D > 0)` は Maxima 内部で `D` の値
  （`4*omega_0^2-(2*gamma)^2`）に展開されてから評価される。** これは以前確認した「Maxima の
  `assume()`/`is()` は代数的に同値な式を自動では正規化しない」という制約と矛盾しない —
  `D` を **束縛済みの量として** assume することで、`ode2` が内部でチェックする形と一致する結果になる
  （実機で確認済み。逆に、`D` を経由せず `assume(omega_0^2 > gamma^2)` のような**別形**を直接書くと
  一致せずハングする）。

### ステップ 8：一般解（FREE-PARAM）

$D>0$ の仮定のもとで、運動方程式（`E1`）の一般解を求める。

```
E1 の一般解を求めて
```

**期待する出力:** `E4`:
`x=e^{-t\gamma}\left(k_{1}\sin\left(\dfrac{t\sqrt{4\omega_{0}^2-4\gamma^2}}{2}\right)+k_{2}\cos\left(\dfrac{t\sqrt{4\omega_{0}^2-4\gamma^2}}{2}\right)\right)`

**検証ポイント:**
- 任意定数 `k_1, k_2` を `free_parameters` として保持（後続のステップで消費される）。
- `\sqrt{4\omega_0^2-4\gamma^2}/2` は $\omega=\sqrt{D}/2$（`E3` の `D`）と同値
  （Maxima は根号内の共通因数 4 を自動では約分しない — 実際の出力をそのまま示している）。
- **「一般解を求めて」「一般に解いて」「の微分方程式を解いて」「の微分方程式の解を求めて」はすべて
  同じ `solve_ode_general` にルーティングされる。** 一方「E1 を解いて」（修飾語なし）は未認識、
  「E1 を x について解いて」は**認識はされるが**微分を通常の代数変数と誤認する汎用の代数ソルバ
  （`solve_equation`）にルーティングされ、`x = -(2\gamma\dot x+\ddot x)/\omega_0^2` という
  **もっともらしいが無意味な代数的書き換え**が「成功」として返ってしまう。「一般（解）」は、
  ODE 専用の一般解ソルバを選ばせるための実質的なトリガ語である。
- **実際の解法:** Maxima の `ode2(方程式, 従属変数, 独立変数)`（1階・2階線形 ODE の組込みソルバ）を
  呼び出している。呼び出し前に各スカラー係数（`gamma`, `omega_0`）へ `assume(sym>0)` を自動付与し、
  加えてステップ7の assume がないと内部で符号確認の対話質問が発行されハングする。得られた
  `%k1,%k2`（Maxima の任意定数プレースホルダ）は `k_1,k_2` に改名される。

### ステップ 9：初期条件で任意定数を決定

一般解（`E4`）には任意定数 $k_1,k_2$ が含まれている。初期条件 $x(0)=1,\ \dot x(0)=0$ を与えて、これらを
決定する。

```
E4 の初期条件を x(0)=1, x'(0)=0 とする
```

（同義の言い回し「…で決めて」「…で定めて」「…で決定して」、境界値問題なら「…の境界条件を…とする」
でも同じ結果になる。旧称「特殊化して」も引き続き使える。）

**期待する出力:** `E5`:
`x = e^{-t\gamma}\left(\cos\left(\dfrac{t\sqrt{4\omega_{0}^2-4\gamma^2}}{2}\right)-\dfrac{\gamma\sqrt{4\omega_{0}^2-4\gamma^2}\sin\left(\frac{t\sqrt{4\omega_{0}^2-4\gamma^2}}{2}\right)}{2\gamma^2-2\omega_{0}^2}\right)`

**検証ポイント:**
- 何をしているか：各初期条件を一般解（および必要な階数の導関数）に代入して `k_1, k_2` に関する連立方程式を
  作り、Maxima の `solve` で解いて、その値を一般解の右辺に代入し直している。物理でいう
  「初期条件を適用して積分定数を決定する」操作と同じ。
- `k_1, k_2` は消費され `free_parameters == []`（完全に決定済み）。

### ステップ 10：数値層フォーク用に右辺だけを取り出す

次のステップで数値層へフォークするために、決定した解（`E5`）の右辺だけを取り出して整理する。

```
E5 の右辺を整理して
```

**期待する出力:** `E6`（右辺の生の抽出、`evaluate_rhs`）を経て `E7`（`整理`＝ratsimp 後）が登録される。
`E7` の canonical は `x = …` の等式形ではなく **裸の RHS 式**（フォーク対象として必要な形）。

**検証ポイント:**
- ODE 一般解／初期条件による決定の結果は（H-1 の「canonical は RHS のみ」規約と異なり）
  `x = …` の等式形で保存される。数値層 `fork_numeric_task` は等式をパースできない
  （`canonical_to_numpy` は `=` を式として扱えない）ため、フォーク前に「右辺を整理して」などの
  H-7 演算で**裸の RHS 式**へ変換する一手間が要る。これはシナリオ側の手順であり、コード側の欠陥ではない。

---

## 数値層ステップ（2026-07-06 時点：NL/UI トリガ未実装・API/エンジン操作）

数値層（N-1）の fork / コード生成 / sandbox 実行は、現状 REST エンドポイント
（`/numeric/fork`, `/numeric/generate/{task_id}`, `/numeric/run/{task_id}`）またはエンジンメソッド
（`fork_numeric_task` / `generate_numeric_code` / `run_numeric_task`）からのみ操作できる
（自然言語トリガもクリック可能な UI パネルもまだ存在しない — CLAUDE.md の N-1 節が定める設計どおり、
「生成コードはコーディングエージェントに渡す種」という PoC スコープのため）。以下は実機で検証済みの
Python 操作列（`E7` を対象、`gamma=0.3, omega_0=2.0` を固定し `t` を掃引）：

```python
task = engine.fork_numeric_task(
    "E7", problem_type="parameter_sweep", target_quantity="x",
    sweep_parameters={"t": {"min": 0, "max": 10, "points": 200}},
    fixed_values={"gamma": 0.3, "omega_0": 2.0},
)
engine.generate_numeric_code(task.task_id)
engine.save("<セッション保存先>.json")   # 数値層の出力は保存先ファイル名に紐づく
execution = engine.run_numeric_task(task.task_id)
```

**期待する出力（実機検証済み・2026-07-07、E7 に対して直接確認済み）:**
- `execution.return_status == 0`（sandbox 実行成功）
- `NB_RESULT {"target": "x", "sweep_var": "t", "n_points": 200, "min": {"x": 1.608…, "y": -0.620…}, "max": {"x": 0.0, "y": 1.0}}`
- タイムラインに `numeric_result` ステップが還流：
  `数値計算 N001: x(t) — 最小 -0.620…（t=1.608…）／最大 1.0（t=0.0）`
- `max` が `t=0` で `y=1.0` となっているのは初期条件 `x(0)=1` と整合。`min` は減衰振動の最初の谷。
- `figure.png`・`table.csv` が `<セッション名>_numeric/N001/<execution_id>/` に生成される。

**検証ポイント:**
- `NumericTask.source_expression_ids == ["E7"]`・`source_assumption_ids` に `ASM1, ASM2, ASM3` が
  スナップショットされ、**理論層のどの仮定のもとで数値化したか**の来歴が残る。

---

## UI での対話的フォーク実行確認（2026-07-18）

上記は API/エンジン直叩きでの検証（2026-07-07）だったが、その後「🔢 To numeric」ボタン＋
フォークモーダルが実装済みとなったため、実践ループの一環として **ブラウザ UI から対話的に**
同じシナリオを最初から最後まで実行し、再確認した。

**実行差分（この回で使った手順・上記の generator ブロックと同値だが英語 NL・別数値を使用）:**
```
x を時間依存関数として定義する
\ddot{x} + 2*gamma*\dot{x} + omega_0^2*x = 0
assume omega_0 > gamma
solve E1 generally
specialize E2 with x(0)=1, x'(0)=0
evaluate the right-hand side of E3
```
その式（`E6`、bare RHS）を選択 → 「🔢 To numeric」→ Sweep variable=`t`, Min=0, Max=20,
Fixed values: `gamma=0.3, omega_0=2` → Generate code → ▶ Run。

**結果（実機・ブラウザ、`sandbox-exec` 経由）:**
- `min -0.6204120728921954 at t=1.6080402010050252` / `max 1.0 at t=0.0` —
  2026-07-07 の API 検証と**同一の数値**（`gamma=0.3, omega_0=2.0` を同じく使用）で再現。
- 減衰波形の図（`x(0)=1` から振動しつつ 0 へ収束）がタイムラインに還流し、視認確認済み。
- 「Equation は phase-1 数値層でサポートされない（scalar only）」というモーダル側のガードも
  実機で確認 — 方程式形（`x = …`）のままフォークしようとすると `Generate code` がグレーアウトし、
  上記の赤字メッセージが出る。これは意図された制約であり、`evaluate the right-hand side of …`
  で bare RHS へ変換してから選び直す、という上記の手順どおりの回避が必要。

**追加で見つかった注意点（コードのバグではなく、シナリオ手順上の既知の必要条件）:**
この回では `\gamma`・`\omega_0` を「正の実数として定義する」ステップ（本シナリオ本来の手順1〜3）を
省略し、代わりに `omega_0 > gamma と仮定する` のみを行った。その状態で `evaluate the right-hand
side of E3`（内部的に `simplify` を呼ぶ）を実行すると、未宣言の桁下付き文字 `omega_0` が
**汎用オペレーション経由の解決規則**（「未宣言の下付きはオペレーションの入力では index 扱い」—
CLAUDE.md「Undefined Symbol Detection」節）に従って `omega(0)`（関数呼び出し）に化けてしまう
再現性のあるケースを確認した（`solve_ode_general`/`specialize_by_conditions` など FREE-PARAM の
特別扱いハンドラでは発生しない。汎用ディスパッチ経由の `simplify`/`expand` 等でのみ発生）。
`\omega_0`（正の実数として定義する）で先に宣言しておけば発生しない — これは本シナリオの手順
1〜3 が最初から回避している既知の要求であり、今回はあえて省略して踏んだことで再現確認した形。
コード側の修正は不要（宣言すれば正しく動く）だが、「汎用オペレーションへ渡す前に、桁下付きの
物理パラメータは明示的に宣言しておく」という要求は他のシナリオでも再現しうるため、ここに記録する。

---

## 発見した不具合と修正（コード修正、シナリオ回避策ではない）

このシナリオを実機で流す過程で、フォーク対象の canonical（Maxima がプリティプリントした長い式は
継続行にインデント付きで改行される）をそのまま数値層のテンプレートへ渡すと、生成される Python コードが
**`IndentationError` で構文エラーになる**不具合を発見した。

`notebook/numeric/templates.py` の `# --- theory expression: {task.source_canonical} ---` という
1行コメントに、改行を含む canonical をそのまま埋め込んでいたため、継続行が `#` を失ってそのまま
Python コードとして解釈されてしまうのが原因（例：`sin(...)\n +(...)*cos(...)` の2行目 `  +(...)` が
インデント付きの生の文（constructing statement）に見えてしまう）。

**修正:** `_single_line()` ヘルパーを追加し、コメントに埋め込む直前に改行・連続空白を1つのスペースへ
畳み込むようにした（`templates.py`）。docstring 内のプロヴェナンス表示（複数行のまま見せたい箇所）は
変更していない。回帰テスト `tests/test_numeric_layer.py::TestTemplates::test_multiline_canonical_stays_valid_python`
を追加。既存の `_task()` フィクスチャの canonical は改行を含まない単純な式だったため、これまで
気付かれていなかった。

## シナリオ手動確認で見つかった NL 面の改善（コード修正）

実際にこのシナリオを手動で追った際、以下の点が指摘され、コード側を修正した：

- **TeX 記法の対応:** `\gamma`/`\omega_0` のような TeX 記法は元からテキスト入力として正しく解釈される
  （手書き入力層を経由する場合の自然な形）。当初の文書が ASCII 綴り `gamma`/`omega_0` を直接使っていたのは
  誤り（コードの不具合ではなく文書の不備）。
- **判別式の由来が不明確だった:** 当初の文書は `(2*\gamma)^2 < 4*\omega_0^2` という不等式をいきなり
  assume させており、由来が式として残っていなかった。特性方程式（`E2`）と判別式 `D`（`E3`、H-1
  registration）を実際に登録するステップを追加し、`D > 0 と仮定する` という形に改めた
  （[project_domain_sign_split] とは別に、この D の値そのものは Maxima 側の
  `assume()` へそのまま渡る一般的な H-1 束縛シンボルの挙動であり、新規のコード実装は不要だった）。
- **「一般に」の必要性・言い回し追加:** 「E1 を解いて」「E1 の微分方程式を解いて」は当初未認識だった点は、
  `_ODE_GENERAL_RE`（および英語版 `_ODE_GENERAL_RE_EN`）に「の微分方程式を解いて」
  「の微分方程式の解を求めて」を追加して解消（`ui/interpret_free_params.py`）。「の一般解を求めて」は
  もともと対応していた言い回し。
- **「特殊化」という用語:** 物理研究者になじみのない CS 由来の用語との指摘を受け、NL トリガに
  「決めて」「定めて」「決定して」、および条件を宣言文として述べる言い回し
  「E5 の初期条件を x(0)=1, x'(0)=0 とする」「…の境界条件を…とする」を追加
  （内部の操作名 `specialize_by_conditions` 自体は変更していない。英語側にも
  `specialize E5 with …` / `determine the constants in E5 from …` /
  `set the initial/boundary conditions of E5 to …` を追加）。エラーメッセージ
  （`engine.specialize_no_params` 等）の文言も「特殊化」を避けた表現に更新。境界条件版は t=0 に
  限らない真の2点境界値問題（例: q(0)=q0, q(10)=q1）でも動作を確認済み。
- **式が唐突に現れ、説明が式に結びついていなかった:** シナリオ文書には導入文を書いていたが、それは
  `.md` だけの記述で、実行される `scenario` ブロックにも生成される TeX 出力にも反映されていなかった。
  `scripts/generate_scenario_examples.py` に `@note: <text>` 行の対応を追加し（既存の UI の
  「📝 メモ」＝`engine.record_manual_note` をそのまま呼ぶ）、各式の直前に説明メモを挿入した。メモは
  タイムラインの正式なステップとして記録され、`export_latex` は既存の `manual_note` 処理
  （`_note_to_latex`）でそのまま TeX に反映する（新規のレンダリング実装は不要だった）。このメモ機構は
  本シナリオ限定ではなく、他のシナリオ文書でも同じ `@note:` 記法が使える。

---

## カバレッジ（このシナリオで検証される機能）

| 機能 | 入力例 | 実装 |
|---|---|---|
| 判別式を明示的に定義して assume（問題依存の modeling choice） | `D = 4 \omega_0^2 - (2 \gamma)^2` → `D > 0 と仮定する` | H-1 登録 → `_try_add_assumption` → `engine.add_assumption` |
| 符号宣言が実際に assume() へ配線される | `\gamma を正の実数とする` | `Quantity.sign` + `BackendHandler.assume_sign` |
| ODE 一般解（FREE-PARAM） | `E1 の一般解を求めて` | `solve_ode_general` |
| 初期条件による任意定数の決定（宣言文） | `E4 の初期条件を x(0)=1, x'(0)=0 とする` | `specialize_by_conditions` |
| 等式結果からの裸の RHS 抽出 | `E5 の右辺を整理して` | `evaluate_rhs` (H-5/H-7) |
| 理論式 → 数値層フォーク | `fork_numeric_task("E7", …)` | N-1 phase 1（決定的テンプレート） |
| sandbox 実行 → 来歴付き還流 | `run_numeric_task(task_id)` | `numeric_result` ステップ |

**関連設計文書:** [12_symbol_domain_design.md](../developer_manuals/12_symbol_domain_design.md)、
[22_numeric_layer_design.md](../developer_manuals/22_numeric_layer_design.md)

## generator コマンド（理論層のみ・この `scenario` ブロックが生成の単一ソース）

数値層のフォーク以降は NL 化されていないため、この自動生成ブロックには含まれない
（上記「数値層ステップ」を参照し、別途スクリプトで実行・確認する）。

```scenario domain=differential_equations
@note: まず、時間に依存する変位 $x$ と、減衰率 $\gamma$・固有角振動数 $\omega_0$（ともに正の実数）を準備する。
x を時間依存関数として定義する
\gamma を正の実数とする
\omega_0 を正の実数とする
@note: 以下の減衰振動子の運動方程式を考える（$\gamma$ は減衰率、$\omega_0$ は固有角振動数）。
$\ddot{x} + 2 \gamma \dot{x} + \omega_0^2 x = 0$
@note: この微分方程式に指数関数の ansatz $x=e^{rt}$ を代入すると（$\dot x=re^{rt},\ \ddot x=r^2e^{rt}$、両辺を $e^{rt}\neq0$ で割る）、この微分方程式の特性方程式は以下となる。
$r^2 + 2 \gamma r + \omega_0^2 = 0$
@note: 特性方程式は $r$ の２次式（$ar^2+br+c=0$ で $a=1,\ b=2\gamma,\ c=\omega_0^2$）なので、以下の判別式 $b^2-4ac=(2\gamma)^2-4\omega_0^2$ を得る。ここでは以降の議論を「正なら不足減衰」という自然な形にするため、この符号を反転させた量 $D=4\omega_0^2-(2\gamma)^2=-(b^2-4ac)$ を定義する。
$D = 4 \omega_0^2 - (2 \gamma)^2$
@note: ここでは減衰振動（不足減衰、$D>0$）を想定する。この選択は扱っている問題に依存する物理的な判断であり、ツール側が代わりに決めるものではないので、以下のように明示的に assume する。
D > 0 と仮定する
@note: $D>0$ の仮定のもとで、運動方程式（E1）の一般解を求める。
E1 の一般解を求めて
@note: 一般解（E4）には任意定数 $k_1,k_2$ が含まれている。初期条件 $x(0)=1,\ \dot x(0)=0$ を与えて、これらを決定する。
E4 の初期条件を x(0)=1, x'(0)=0 とする
@note: 次のステップで数値層へフォークするために、決定した解（E5）の右辺だけを取り出して整理する。
E5 の右辺を整理して
```
