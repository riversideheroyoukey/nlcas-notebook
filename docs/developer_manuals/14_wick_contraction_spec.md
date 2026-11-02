# Wick 収縮（wick_contract）操作 設計仕様

**作成:** 2026-06-04  
**更新:** 2026-06-06（SymPy ルーティング実装完了）  
**状態:** 実装済み（QA パス・SymPy パス両対応）  
**関連:** `docs/developer_manuals/04_second_quantization_engine_design.md`, `docs/open_items.md` A-2

> **バックエンドルーティング原則**  
> `wick_contract` は QA または SymPy のいずれかで処理される。  
> 整数サイト・単純記号インデックス → QA。代数インデックス（`-k`, `k+q`）または  
> `delta()` テンプレート → SymPy。**一度 SymPy に移行したセッションは QA に戻らない。**  
> 詳細: `docs/developer_manuals/05_algebraic_index_architecture.md` §3

---

## 1. 背景と動機

`pair_substitute`（A-2）は **2演算子積全体** を c-number で置換する：

```
c†(k) c(k)  →  n_k     （bares 全体を置換）
```

これに対して **Wick 収縮** は、**n 次演算子積の中から特定のペアを取り出し**、
そのペアを c-number に置換しつつ残りの演算子積を保持する：

```
c†(k) c†(k') c(q) c(q')
  →  V_HF(k,q) · c†(k') c(q')   （c†(k) と c(q) を収縮、残り 2 次積が残る）
  +  ...
```

物理的用途：
- Hartree-Fock 近似（通常平均 ⟨c†c⟩ = V^{HF}）
- BCS 平均場理論（異常平均 ⟨cc⟩ = Δ、⟨c†c†⟩ = Δ*）
- Hubbard モデルのデカップリング
- 一般的な n 次項の (n-2) 次への簡約

---

## 2. 収縮の数学的定義

### 2.1 単一収縮

n 次演算子積 $A_1 A_2 \cdots A_n$ において、
位置 $i$ と $j$（$i < j$）のペアを収縮する：

$$A_1 \cdots A_n
\xrightarrow{\text{収縮}(i,j)}
(-1)^{j-i-1} \langle A_i A_j \rangle
\cdot A_1 \cdots \hat{A}_i \cdots \hat{A}_j \cdots A_n$$

- **符号** $(-1)^{j-i-1}$：$A_j$ を $A_i$ の隣まで移動するのに要する
  フェルミオン反交換の回数（ボソンは常に +1）
- **収縮値** $\langle A_i A_j \rangle$：ユーザーが指定する c-number
- **残余** $A_1 \cdots \hat{A}_i \cdots \hat{A}_j \cdots A_n$：$(n-2)$ 次演算子積

### 2.2 収縮タイプと標準記法

個々の収縮値は HF ポテンシャル $V^{HF}$ そのものではなく、その構成要素である。

| op1 | op2 | 収縮値 | 記号 | 種別 |
|---|---|---|---|---|
| `c'` (c†) | `c` | $\rho(k,\sigma;\,p,\tau)$ | **ρ**（4添字） | 通常平均 — 一粒子密度行列（1-RDM）|
| `c` | `c` | $\kappa(k,\sigma;\,p,\tau)$ | **κ**（4添字） | 異常平均 — BCS 対消滅 |
| `c'` (c†) | `c'` (c†) | $\kappa^*(k,\sigma;\,p,\tau)$ | **κStar**（4添字） | 異常平均共役 — BCS 対生成 |

**記法の根拠：**

- **ρ**（rho）：$\rho(k,\sigma;\,p,\tau) \equiv \langle a^\dagger(k,\sigma)\, a(p,\tau)\rangle$  
  凝縮系物理・量子化学で標準的な「一粒子密度行列」の記号。  
  HF ポテンシャルはこれを使って $V^{HF}(k,\sigma;\,p,\tau) = \sum_{q,\nu} v(\cdots)\,\rho(q,\nu;\,q,\nu)$ と定義される。

- **κ**（kappa）：$\kappa(k,\sigma;\,p,\tau) \equiv \langle a(k,\sigma)\, a(p,\tau)\rangle$  
  BCS 理論での「異常密度行列」（anomalous density matrix）の標準記号。  
  BCS ギャップパラメータは $\Delta_k = g\,\kappa(k,\uparrow;\,-k,\downarrow)$ と表される。

- **κStar**（κ の複素共役）：$\kappa^*(k,\sigma;\,p,\tau) \equiv \langle a^\dagger(k,\sigma)\, a^\dagger(p,\tau)\rangle$  
  系が時間反転対称のとき $\kappa^*(k,\sigma;\,p,\tau) = \kappa(-p,-\tau;\,-k,-\sigma)$。

op1 のインデックス（INDEX1_0, INDEX1_1）が $k,\sigma$、op2 のインデックス（INDEX2_0, INDEX2_1）が $p,\tau$ に対応する慣例。

**value テンプレートの使い方（スピン添字あり）：**

```python
contraction_rules = [
    {"op1": "c'", "op2": "c",
     "value": "rho(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)"},     # ⟨c†(k,σ)c(p,τ)⟩ = ρ(k,σ,p,τ)
    {"op1": "c",  "op2": "c",
     "value": "kappa(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)"},   # ⟨c(k,σ)c(p,τ)⟩  = κ(k,σ,p,τ)
    {"op1": "c'", "op2": "c'",
     "value": "kappaStar(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)"},# ⟨c†(k,σ)c†(p,τ)⟩= κ*(k,σ,p,τ)
]
```

スピン添字を持たない場合（運動量のみ）：

```python
{"op1": "c'", "op2": "c",  "value": "rho(INDEX1,INDEX2)"}       # ρ(k,p) のみ
```

### 2.3 ダブル収縮（定数項）

残余演算子積にさらに収縮を適用すると定数（c-number のみ）が得られる：

$$
\langle A_i A_j \rangle \langle A_k A_l \rangle \cdot (\text{定数})
$$

`include_constant: True` のとき出力に含める。

### 2.4 異常平均の反対称性

フェルミオン統計より $\Delta_{ij} = \langle c_i c_j \rangle = -\langle c_j c_i \rangle = -\Delta_{ji}$。

Δ を反対称行列として宣言することで op1/op2 の順序による符号の曖昧さを解消：
- `Delta(i,j)` が出たとき、SymbolContext の対称性情報を参照して正規化
- `Delta(j,i) → -Delta(i,j)` の変換をシステムが自動適用

---

## 3. 操作インターフェース

### 3.1 Python API

```python
engine.run_operation("wick_contract", {
    "expr": "E3",
    "contraction_rules": [
        # 通常平均: ⟨c†(k,σ) c(p,τ)⟩ = ρ(k,σ,p,τ)
        {
            "op1":   "c'",
            "op2":   "c",
            "value": "rho(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)",
        },
        # 異常平均: ⟨c(k,σ) c(p,τ)⟩ = κ(k,σ,p,τ)
        {
            "op1":   "c",
            "op2":   "c",
            "value": "kappa(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)",
        },
        # 異常平均共役: ⟨c†(k,σ) c†(p,τ)⟩ = κ*(k,σ,p,τ)
        {
            "op1":   "c'",
            "op2":   "c'",
            "value": "kappaStar(INDEX1_0,INDEX1_1,INDEX2_0,INDEX2_1)",
        },
    ],
    "include_constant": False,
})
```

スピン添字なし（運動量のみ）の場合は `INDEX1`/`INDEX2`（= INDEX1_0/INDEX2_0 の省略形）を使う：

```python
{"op1": "c'", "op2": "c", "value": "rho(INDEX1,INDEX2)"}   # ρ(k,p)
```

### 3.2 自然言語入力

```
"E3 を Wick 収縮して：c†(k,σ)c(p,τ) の期待値は ρ(k,σ,p,τ)、c(k,σ)c(p,τ) の期待値は κ(k,σ,p,τ)"
"E3 の c† と c のペアを ρ で、c と c のペアを κ で収縮して"
"E3 を Hartree-Fock-Bogoliubov 近似して"  ← Adviser 連携（将来）
```

### 3.3 収縮値の指定方法

**関数名を指定する場合**（推奨）：
```python
"value": "V_HF"
# → 収縮した演算子のインデックスを引数として V_HF(op1_index, op2_index) を生成
```

**式として直接指定する場合**：
```python
"value": "n_k * delta(k, q)"
# → k = op1 のインデックス、q = op2 のインデックスとして評価
# ※ 式中のインデックス名は expression 内の演算子のインデックス名と一致させること
```

---

## 4. 出力フォーマット

### 4.1 個別式として列挙

全ての可能な収縮を**個別の式（各 E-ID）**として出力する。
例：4次項 `c†(k,↑) c†(-k,↓) c(-k,↓) c(k,↑)` に ρ + κ + κ* の3規則を適用：

```
E4:  -ρ(k,↑,-k,↓) · c†(-k,↓) c(k,↑)   [ρ / Fock,    (pos.1, pos.3)]
E5:  +ρ(k,↑,k,↑)  · c†(-k,↓) c(-k,↓)  [ρ / Hartree, (pos.1, pos.4)]
E6:  +ρ(-k,↓,-k,↓)· c†(k,↑) c(k,↑)   [ρ / Hartree, (pos.2, pos.3)]
E7:  -ρ(-k,↓,k,↑) · c†(k,↑) c(-k,↓)  [ρ / Fock,    (pos.2, pos.4)]
E8:  -κ(-k,↓,k,↑) · c†(-k,↓) c†(k,↑) [κ / 異常平均, (pos.3, pos.4)]
E9:  +κ*(k,↑,-k,↓) · c(-k,↓) c(k,↑)  [κ* / 異常平均共役, (pos.1, pos.2)]
```

HF ポテンシャルとの関係：

$$V^{HF}(k,\sigma;\,p,\tau) = \sum_{q,\nu} v(k,\sigma,q,\nu;\,p,\tau,q,\nu)\,\rho(q,\nu;\,q,\nu)$$

すなわち `wick_contract` が出力する個々の収縮値 $\rho(k,\sigma;\,p,\tau)$ は 1-RDM の行列要素であり、
これを相互作用ポテンシャルで畳み込んだものが $V^{HF}$ になる。

### 4.2 ラベル情報

各出力式には以下のメタデータを付与：
- 収縮したペアの位置（pos.i, pos.j）
- 適用した規則（HF / 異常平均 / 異常平均共役）
- フェルミオン符号

---

## 5. ユーザー選択フロー

```
1. wick_contract → 全収縮項を個別式 E4, E5, ... として列挙

2a. ユーザーが直接選択：
    "E4, E5, E8, E9 を残して H_MF として定義して"

2b. LLM Adviser に委ねる（将来対応）：
    "どの項が BCS 平均場ハミルトニアンになる？"
    → Adviser が物理的考察を説明し、残すべき項を提案
    → ユーザーが確認

3. 選択した式の和を新しい式として登録：
    H_MF = E4 + E5 + E8 + E9
    engine.update_lhs_alias(H_MF_id, "H")  ← 必要なら H を上書き
```

---

## 6. 反対称行列の宣言

Δ の反対称性をシステムに教えるための操作：

```python
engine.run_operation("define_scalar_parameter", {
    "name": "Delta",
    "symmetry": "antisymmetric",   # Delta(i,j) = -Delta(j,i)
})
```

または専用操作：

```python
engine.run_operation("declare_symmetry", {
    "name":     "Delta",
    "symmetry": "antisymmetric",
})
```

**サポートする対称性：**

| 値 | 意味 | 例 |
|---|---|---|
| `"symmetric"` | $f(i,j) = f(j,i)$ | 距離行列、分散共分散行列 |
| `"antisymmetric"` | $f(i,j) = -f(j,i)$ | BCS の Δ、磁場テンソル |
| `"hermitian"` | $f(i,j) = f(j,i)^*$ | ハミルトニアン行列 |

**Wick 収縮時の使用：**  
wick_contract が収縮値 `Delta(j,i)` を生成したとき、
SymbolContext の対称性情報を参照して `−Delta(i,j)` に正規化する。

---

## 7. 高次項への適用

一回の `wick_contract` で次数を 2 下げる。
高次項を複数段階で簡約するには繰り返し呼ぶ：

```
6次項 → wick_contract → 4次項（複数）
4次項 → wick_contract → 2次項（複数）
```

各ステップで全収縮項が列挙され、ユーザーが残す項を選択する。

**例：8次項 → 4次項**

```
a†_1 a†_2 a†_3 a†_4 a_5 a_6 a_7 a_8

wick_contract（2回適用）→ C(4,1) × C(4,1) = 16 通りの 4次項
→ 物理的選択（保存則、対称性）で残す項を絞り込む
```

---

## 8. 実装フェーズ

### Phase 1（優先）：固定インデックスの n 次項

対象：`c†(1) c†(2) c(3) c(4)` のような具体インデックス付き項

- `_nb_wick_contract` Julia ドライバー関数
- QuantumAlgebra の `BaseOpProduct` 内部構造へのアクセス
- フェルミオン符号の自動計算
- Python 側 `build_command` の実装

### Phase 2：代数インデックス付き項

対象：`c†(k) c†(k') c(q) c(q')` のような添字変数付き項

- 収縮値への実インデックスの代入
- Δ の反対称性による正規化

### Phase 3：IndexSum 内の項 ✅ 追加実装不要

対象：`IndexSum(c†(k) c†(-k) c(-k) c(k), k)` のような束縛インデックス項

QuantumAlgebra は `IndexSum(expr, k)` を評価する際に自動的に k → `#₁`、-k → `#₋₁` に
内部変換する。結果は通常の QuExpr として格納されるため、`_nb_wick_contract` は
そのまま適用可能。

```
IndexSum(U a†(k) a†(-k) a(-k) a(k), k)
  → QuantumAlgebra が -U a†(#₋₁) a†(#₁) a(#₋₁) a(#₁) に展開
  → _nb_wick_contract が #₁、#₋₁ インデックスで正常に収縮
  → 収縮値: V_HF(#₁,#₋₁) 等（QuantumAlgebra の内部表記）
```

Phase 2 の実装（2026-06-04）で自動的に対応済み。

### Phase 4：declare_symmetry と Adviser 連携

- `Quantity.symmetry` フィールドの追加
- `declare_symmetry` 操作の実装
- LLM Adviser への収縮結果の解釈委託インターフェース

---

## 9. pair_substitute との関係

`pair_substitute`（A-2）と `wick_contract` の使い分け：

| 操作 | 対象 | 動作 |
|---|---|---|
| `pair_substitute` | bares 全体が収縮対象 | 2演算子積を丸ごと c-number に |
| `wick_contract` | bares の一部ペアを収縮 | n 次積から 2 演算子を取り出し、(n-2) 次積が残る |

**実装上の関係：**
- `pair_substitute` の Julia 側コード（QuExpr イテレーション、新 QuTerm 構築）は `wick_contract` の参考になる
- bares 部分分解とフェルミオン符号計算は `wick_contract` で新規実装が必要
- Python 側のスキーマ・build_command 構造は再利用可能

---

## 10. 未解決事項

- [ ] `declare_symmetry` 操作の設計（`open_items.md` A-7 として登録予定）
- [ ] wick_contract の結果選択 UI（E-ID 指定 vs 自然言語）
- [ ] LLM Adviser との連携インターフェース設計
- [ ] Phase 3（IndexSum 内）の束縛インデックス対応
- [ ] ボソン系への対応（符号は常に +1、Bose-Einstein 凝縮の異常平均）
