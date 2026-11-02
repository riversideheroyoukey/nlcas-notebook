# 定義と代入の規則

**作成日:** 2026-06-05  
**対象:** ユーザー向け（バックエンドの違いを知らなくてよい）

---

## 1. 定義とは何か

「定義」とは `H = H_0 + H_1` のように **LHS = RHS** の形で式を登録する操作。

```
H = H_0 + H_1      ← H をシンボルとして登録
H_0 = p^2 / (2*m)  ← H_0 を定義
H_1 = -k * x^2     ← H_1 を定義
```

定義された式は `lhs_alias_map` に記録される（`H → E1`, `H_0 → E2`, `H_1 → E3`）。

---

## 2. バックエンドによる評価タイミングの違い

定義の「評価タイミング」はバックエンドによって異なる：

| バックエンド | H = H_0 + H_1 を登録後 | H_1 を後から更新したとき |
|---|---|---|
| **Maxima** | `H : H_0 + H_1` を即座に評価・確定 | **H は更新されない**（定義時に固定） |
| **Julia** | `H = H_0 + H_1` を構造として保持 | **H は自動更新されない**（明示的な再評価が必要） |

**どちらのバックエンドでも：** 後から H_1 を更新しても H は自動で変わらない。  
→「再評価」操作で明示的に反映させる必要がある。

---

## 3. 再評価（re_evaluate_expr）

### 3-1. 自動検出モード（推奨）

`lhs_alias_map` に登録されたシンボルを自動検出して一括代入する。

```
"H を再評価して"   または   "H を H_0, H_1 の値で展開して"
→ engine.re_evaluate_expr("E1")  （substitutions 省略）
→ H_0 → E2 の canonical、H_1 → E3 の canonical を自動検出
→ 両バックエンドで同じ結果
```

**Maxima の実行内容:**
```
subst([H_0 = p^2/(2*m), H_1 = -k*x^2], H_0 + H_1)
→ p^2/(2*m) - k*x^2
```

**Julia の実行内容:**
```
substitute(H_0 + H_1, Dict(H_0 => p^2/(2*m), H_1 => -k*x^2))
→ p^2/(2*m) - k*x^2
```

### 3-2. 明示的代入モード

値を直接指定する。数値・式・E-ref が使える。

```python
# 数値で指定
engine.re_evaluate_expr("E3", substitutions={"n_k": "0.42"})

# 別の式（E-ref）で指定
engine.re_evaluate_expr("E3", substitutions={"H_0": "E5"})

# 複数同時
engine.re_evaluate_expr("E3", substitutions={"rho": "0.42", "kappa": "0.1"})
```

自然言語での指示例：
```
"E3 の n_k を 0.42 で更新して"
"E3 の H_0 を E5 で置き換えて展開して"
```

---

## 4. シンボルのマッチング規則

### 4-1. 単語境界マッチング

`H` を検索するとき `H_0` や `H_1` の中の `H` には**マッチしない**。

```
式: "H_0 + H_1"
  → lhs_alias_map に "H" があっても検出しない（境界不一致）
  → lhs_alias_map に "H_0", "H_1" があれば両方を検出する

式: "H + V"
  → lhs_alias_map に "H" があれば検出する
```

実装: `(?<![A-Za-z0-9_])H(?![A-Za-z0-9_])` による正規表現マッチ。

### 4-2. 長い名前を優先

`H` と `H_0` が両方 `lhs_alias_map` にある場合、長い名前（`H_0`）を先に処理する。これにより誤代入を防ぐ。

### 4-3. 代入数の上限

**上限なし。** Maxima の `subst([...], expr)` も Julia の `substitute(expr, Dict(...))` もリストを一括処理する。20個以上のシンボルでも動作確認済み。

---

## 5. E-ref の解決

代入値として E-ref（`"E3"` 等）を指定した場合、実行時に canonical に展開される：

```
{"H_0": "E2"}  →  {"H_0": E2 の canonical 文字列}
```

これにより「最新の計算結果を別の式に代入する」という操作が書ける：

```
"E5 の H_0 を E3 で更新して"
→ substitutions = {"H_0": "E3"}
→ H_0 を E3 の canonical（最新値）に置き換えて E5 を再評価
```

---

## 6. 定義と代入の組み合わせパターン

### パターン A：段階的定義→一括展開

```
H_0 = p^2/(2*m)        (E1)
H_1 = -k*(x-a)^2      (E2)
H = H_0 + H_1          (E3)

"H を展開して"
→ re_evaluate_expr(E3) → H_0, H_1 を自動検出
→ p^2/(2*m) - k*(x-a)^2  (E4)
```

### パターン B：平均場の自己無撞着更新

```
H_MF = xi*n_k          (E5)   ← n_k がパラメータ

# 1回目
"E5 の n_k を 0.3 で更新して"  → n_k = 0.3 → 0.3*xi  (E6)

# 2回目（収束するまで繰り返す）
"E5 の n_k を 0.42 で更新して" → n_k = 0.42 → 0.42*xi  (E7)
```

### パターン C：定義チェーン

```
A = 1        (E1)
B = A + 1    (E2)
C = B + 1    (E3)

"C を展開して"
→ re_evaluate_expr(E3) → B を検出 → B の canonical = "A + 1"
→ subst([B = A+1], B+1) = A + 2    (E4)

# さらに A を展開したければ:
"E4 を展開して"
→ subst([A = 1], A+2) = 3   (E5)
```

チェーン参照は**1段階ずつ** 展開される（自動再帰はしない）。

---

## 7. 制限事項

| 制限 | 詳細 |
|---|---|
| 自動展開は1段階 | H = H_0 + H_1、H_0 = p^2/2m → "H を展開" では H_0 が p^2/2m に置き換わるが、さらに数値評価はされない |
| 元の式は変わらない | re_evaluate_expr は**新しい式 En を追加**する。元の式は保持される |
| Julia の構造保持 | Julia セッションでは H = H_0 + H_1 を登録しても H_0 は代入されない（明示的な re_evaluate が必要） |
| Maxima の即時確定 | Maxima セッションでは H = H_0 + H_1 を登録した時点で H_0 が即座に評価される。後から H_0 を更新しても H には反映されない |
| セッションロード後の LaTeX キー | `lhs_alias_map` に登録されているキーは、セッション保存時の LaTeX 形式（例: `\ddot{q}`）のみ残り canonical 形式（`'diff(q(t),t,2)`）は消える。E→E 代入ハンドラー（`_try_rule_dainyu`、NL 文法は `nl_patterns.json["dainyu"]`）は LaTeX キーを検出すると `BackendHandler.lhs_latex_to_canonical()` で変換してから代入 LHS として渡す（Maxima のみ変換、Julia は no-op）。 |

---

## 7.5 `lhs_alias_map` の鍵は「書かれた綴り」と「正準形」の両方（2026-08-09）

この地図の**利用者は LaTeX 側と正準側の両方にいる**。だから鍵は片方では足りない。
3つの答えを順に試して3つとも別の利用者を壊した（すべて実測）:

| 鍵にしたもの | 壊れたもの |
|---|---|
| 正規化した**表示** | `alpha_k = …` が `\alpha_{k}` で登録され、canonical を走査する `_auto_detect_substitutions` が何も見つけられない。逆変換が **`c_k = 0` という縮退解を「成功」として返した** |
| **書かれた綴り**のみ | 同じ式を plain text で打った場合と math chip で入れた場合が別の鍵になる（`sigma` / `\sigma`） |
| **正準形**のみ | LaTeX を走査する `_expand_symbol_refs` が止まる（`2 * \bar{H} + V` が `H_bar` を見つけられない） |

現在は `NotebookEngine._alias_keys()` が**両方の綴りを登録する**（同じ式を指す）。
利用者は例外なく鍵を `_extract_simple_symbol_name()` で記号名に落としてから使うので、
綴りが2つあっても二重定義にはならない（P2 の CAS 束縛は同じ命令を2回送るだけ）。

**表示の正規化を鍵に通してはいけない。** 「書かれたもの」と「照合に使う鍵」は
別の問いで、`register_expression` の表示の扉（1つに集約されている）は前者にしか
当てはまらない。壊れ方が**例外ではなく「もっともらしい成功」**なので、この境界の
テストは値まで確認すること。

- 実装: `notebook/engine.py` `_alias_keys` / `_update_lhs_alias`
- テスト: `tests/test_debug_repl.py::TestInverseTransformMaxima`,
  `tests/test_plain_text_equation_registration.py`,
  `tests/test_engine.py::TestEquationFormInput::test_symbol_expansion_in_formula`

---

## 8. UI 統合の実装箇所

再評価（`re_evaluate_expr`）・代入の UI 統合は以下に実装されている（C-Ext Phase 4 で導入）。

| ファイル | 内容 |
|---|---|
| `ui/notebook_bridge.py` | `/re_evaluate`・`/wick_contract` エンドポイント（バックエンド差はここで隠蔽。Maxima/Julia で同一挙動）|
| `notebook/engine.py` | `_auto_detect_substitutions()`（`lhs_alias_map` から自動解決）、`re_evaluate_expr` の substitutions 省略対応 |
| `ui/interpret_pipeline.py` | `re_evaluate_expr` の特別ルーティング（「再評価して」等を LLM 前にルールベース処理）|
| `ui/notebook_main.js` / `i18n/ui_messages.json` | 再評価・Wick 収縮ボタンとダイアログ、i18n ラベル |

**`/re_evaluate` の入力:** `expr_id`（必須）と `substitutions`（省略可。省略時は `lhs_alias_map` から自動解決）。

**既知の制約:** `_auto_detect_substitutions` は部分文字列マッチのため `H` が `H_0` を誤検出しうる（より堅牢には `canonical_variables(canonical)` を使う）。詳細は §4-1 の単語境界マッチを参照。

---

## 9. 参照

- `docs/developer_manuals/03_backend_paradigm_unification.md` — バックエンドパラダイムの設計決定
- `docs/user_guide.md` § 平均場の自己無撞着更新 — UI での使い方
- `notebook/engine.py` `_auto_detect_substitutions()` — 自動検出の実装
