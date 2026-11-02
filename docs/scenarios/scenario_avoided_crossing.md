# 実践ループ サンプルシナリオ：2準位ハミルトニアンの固有系→避交差の数値図（S-y）

実践ループ・シナリオメニュー **S-y（量子力学、C-Ext-5×N-1 の層横断）** の詳細化。
2準位ハミルトニアン $H=\begin{pmatrix}\epsilon&\Delta\\\Delta&-\epsilon\end{pmatrix}$
を題材に、(1) C-Ext-5 の固定ワークフロー「固有状態を求めて」1指示で特性多項式→
固有値→固有ベクトルまで一気通貫に実行できるか、(2) その**プランが生成した式**を
そのまま数値層へフォークできるか（プラン出力→数値入力の**層横断接続**）、(3) 避交差
（avoided crossing）の数値図が実際に生成されるか、を実機検証した。

**バックエンド:** standard（Maxima）／**ドメイン:** quantum_mechanics。
**実行言語:** English セッション。数値層フォークは「🔢 To numeric」UI モーダルから
対話的に実行（S-x と同じく、生成コードの review・実行までブラウザ操作で完結）。

> **他のシナリオとの違い:** 本シナリオは C-Ext-5 のプラン実行 UI と N-1 の数値フォーク
> UI という**2つの独立した機能を1本のデータフローで繋ぐ**ことが主眼であり、
> `scripts/generate_scenario_examples.py` の自動生成 DSL（プレーンテキスト列のみ）
> では表現できない UI 操作（プラン実行結果の式選択・数値フォークモーダル）を含む。
> そのため本ワークシートは damped_oscillator_numeric と同様、**実ブラウザ操作による
> 手動実行の記録**である。

## モデルの背景

2準位系のハミルトニアン

$$H=\begin{pmatrix}\epsilon&\Delta\\\Delta&-\epsilon\end{pmatrix}$$

（$\epsilon$: 準位間のデチューニング、$\Delta$: 結合強度／ギャップ）の固有値は

$$E_\pm=\pm\sqrt{\epsilon^2+\Delta^2}$$

で与えられる。$\Delta=0$ なら $\epsilon=0$ で2準位が交差する（$E_+=E_-=0$）が、
$\Delta\neq0$ の結合があると交差せず、$\epsilon=0$ で最小間隔 $2\Delta$ を保って
反発する——これが「避交差（avoided crossing）」である。$\Delta$ を固定して
$\epsilon$ を掃引し、$E_+(\epsilon)$（あるいは両分岐）をプロットすると、
$\epsilon=0$ で最小値 $\Delta$ を取る双曲線が現れる。

## セッション設定

- 新規セッション、Language: English、domain: **quantum_mechanics**、
  backend: Maxima（Cartesian）。

## 入力ステップ

### ステップ 1：ハミルトニアン行列の登録（H-1）

```
H = [epsilon, Delta; Delta, -epsilon]
```

**期待する出力:** `E1`（実際は行列の**内部エイリアス**として `H` に束縛。
表示は `H = matrix(...)` 相当）。

**確認ポイント（重要・本シナリオで踏んだ入力ミス）:** 行列の入力規約は
**セミコロン区切りの行**（`[a, b; c, d]`、S-i の規約と同一）であり、Python/JSON
スタイルの入れ子リスト `[[a, b],[c, d]]` は**別の構文**（`[A, B]` はカンマ区切り
2要素の意味を持ち、コンマ区切り＋カンマなしの `[A, B]` は交換子と解釈される設計
——`scenario_matrix_diagonalization` の「`;` で行区切り／`[A, B]`（`;` なし）は
交換子」という注記のとおり）として解決される。誤って `[[epsilon, Delta],[Delta,
-epsilon]]` と入力すると、式自体は登録できてしまう（`[[...],[...]]` という
ネストしたリストとして）が、後段の `characteristic_polynomial`（Maxima の
`charpoly()`）が**真の `matrix(...)` オブジェクトを要求する**ため
`charpoly: matrix must be square; found 2 rows, 3 columns.` という Maxima 側の
エラーで失敗する（プランは s1 で安全に停止、0/3）。**コード側のバグではなく
入力規約違反**——正しい `[epsilon, Delta; Delta, -epsilon]` で再登録すると
`canonical` が `matrix([epsilon, Delta], [Delta, -epsilon])` として正しく
構築され、後続がすべて成功する。

### ステップ 2：固定ワークフロー「固有状態を求めて」（C-Ext-5）

```
find the eigenstates of E1
```

（`E1` は上記の再登録後の行列式の ID——プラン実行前に必ず「行列が正しく
`matrix(...)` として登録されているか」を確認すること。）

**期待する出力:** プラン `P2`「Find the eigenstates (characteristic polynomial →
eigenvalues → eigenvectors)」が **3/3 ステップ成功**で完了：

| step | operation | 結果 |
|---|---|---|
| s1 | `characteristic_polynomial` | `E3`: $x^2-\epsilon^2-\Delta^2$ |
| s2 | `eigenvalues` | `E4`: $[[-\sqrt{\epsilon^2+\Delta^2},\sqrt{\epsilon^2+\Delta^2}],[1,1]]$（値のリスト形） |
| s3 | `eigenvectors` | `E5`（固有ベクトルのネストしたリスト形） |

### ステップ 3：個別の固有値式を取り出す（H-9、数値フォークの前段）

**確認ポイント（既知の制約、実機再確認）:** `E4`（`eigenvalues` の生の出力）は
**リスト形**（`[[固有値のリスト],[重複度のリスト]]`）であり、数値層の
`fork_numeric_task` はこれを直接フォークできない（`numeric.list_not_supported`
の明示エラーに改善済み）。正しい経路は、s1 の特性多項式 `E3`（$x^2-\epsilon^2-
\Delta^2$、暗黙に $=0$）を**改めて解いて個別の式として登録し直す**こと：

```
solve E3 for x
```

**期待する出力:** `E6`: $x=-\sqrt{\epsilon^2+\Delta^2}$、`E8`:
$x=\sqrt{\epsilon^2+\Delta^2}$（いずれも `canonical` は bare スカラー、
"x = " を含まない——H-1 由来ではなく `solve_equation` の個別解登録の性質）。

### ステップ 4：正の分岐 `E8` を数値層へフォーク

UI で `E8` を選択 → 「🔢 To numeric」→ Sweep variable=`epsilon`（Min=-10,
Max=10, Points=200）、Fixed values: `Delta=1` → Generate code → ▶ Run。

**期待する出力（実機確認、2026-07-18）:**
- `min 1.0012617983114402 at epsilon=-0.05025125628140614`（$\epsilon\approx0$
  近傍で最小値 $\approx\Delta=1$ — 理論値と整合）
- `max 10.04987562112089 at epsilon=-10.0`（$\sqrt{10^2+1^2}\approx10.05$ と
  整合）
- 生成された図は $\epsilon\in[-10,10]$ にわたる**双曲線（避交差の上側分岐）**：
  $\epsilon=0$ 近傍で最小値 $\Delta=1$ を保ったまま、両側で $|\epsilon|$ に
  漸近する直線状に開いていく——avoided crossing の教科書的な形状を実データで
  確認した。

## 確認ポイント（本シナリオの検証目的）

- C-Ext-5 の固定ワークフロー「固有状態」が2×2対称でない一般2準位ハミルトニアンにも
  正しく適用できる（3/3 ステップ成功）。
- **プランが生成した式（`E3`）を起点に、既存の H-9（`solve`）→ 数値フォークという
  別々の機能を繋いで、避交差の図まで到達できる**——「プラン出力→数値入力の接続」
  という当初の検証目的を実データで確認。
- `eigenvalues` の生リスト出力は数値フォーク不可という既知の制約（2026-07-04時点で
  確認済み）を、実際のプラン実行結果に対しても再確認。回避経路（特性多項式→
  `solve`→個別式）は H-9 の既存機能のみで完結し、新規コードは不要だった。
- 行列入力規約（`;` 区切り行 vs ネストしたリスト）の誤りが `characteristic_polynomial`
  でのみ表面化する非対称性を発見・記録（`eigenvalues`/`eigenvectors` 単体では
  この行列オブジェクトの違いが問題にならないケースがある可能性があり、他のシナリオ
  でも同じ入力ミスをしないよう、ここに明記する）。

## コード変更

なし。すべて既存機能（C-Ext-5 固定ワークフロー・H-9 `solve`・N-1 数値フォーク UI）の
組み合わせで完走した。行列入力の失敗は入力規約の誤りであり、シナリオ手順の記録として
本ワークシートに残す（`scenario_matrix_diagonalization` が既に正しい規約を明記して
いたにもかかわらず本シナリオで再度踏んだ、という事実も含めて記録）。

**検証日:** 2026-07-18（実機・Maxima バックエンド、`.venv-numeric` サンドボックス）。
