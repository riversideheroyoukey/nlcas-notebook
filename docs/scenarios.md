# シナリオ集（実践ループ／ワークシート索引）

このプロジェクトの **実践ループ・シナリオ** の表紙。各シナリオは
**潜在的な不具合・粗さ・バグを炙り出すための材料**であると同時に、機能の使い方を
期待値つきで示す **ワークシート（worked example）** でもある。シナリオ完走そのものより、
現れた不都合な振る舞いを直すことを優先する。

- **🟢 実現済み** … `scenario_*.md` として詳細化・厳密な期待値つきで確認完了（下表）。
- **✅ 確認可（未実施）／⏳ 機能待ち** … まだ開いている候補。
  ステータスと候補一覧は [open_items.md の実践ループ・メニュー](../KNOWN_ISSUES.md) を single source とする。

---

## 実例（.json / .tex）はコードの現状を映していなければならない

`domains/<domain>/examples/*.{json,tex}` は **生成物**で、単一ソースは各 `scenario_*.md` の
` ```scenario ` ブロック。パーサ・LaTeX 変換器・表示ブリッジ・エクスポートなど
**出力が変わる修正をしたら、その修正と同じコミットで再生成する**:

```bash
PATH="$HOME/.juliaup/bin:$PATH" .venv/bin/python scripts/generate_scenario_examples.py
```

検査は `tests/test_examples_up_to_date.py`。ライブ CAS で全シナリオを回すため
**opt-in**（約3分）:

```bash
PATH="$HOME/.juliaup/bin:$PATH" .venv/bin/python -m pytest tests/test_examples_up_to_date.py --run-scenarios
```

**なぜ要るのか（EXAMPLES-DRIFT、2026-08-06 実測）。** 実例は silent-wrong の根治を5件ぶん
取り残しており、公開している `.tex` に `\dot{theta}`（組版すると θ ではなく**斜体の
"theta" という語**）と、式中の literal `*` が残っていた。**指摘も起票もされず**、
無関係な作業の副産物として偶然発覚した。しかもその場では 124 ファイル中 90 が無関係な
陳腐化で、**本物の破壊（シナリオ7本が落ちる）が危うく紛れるところだった**。

**失敗したら、再生成する前に差分を読むこと。** この検査は「良くなった」と「壊れた」を
区別できない。言えるのは「実例がもうコードを説明していない」ことだけである。

---

## ドメイン／バックエンドの決め方（着手前に決め切る）

実例の保存先**ドメインのバックエンドと、実行セッションの backend を一致させる**こと（手戻り防止）。
判定基準は「**第２量子化の演算子（QA）が要るか**」：

- 要る（Bogoliubov の k 空間演算子・ボソン/フェルミ演算子）→ **Julia**＝second_quantization 系ドメイン
  （condensed_matter / quantum_optics 等）。
- 要らない（古典・記号・spin-algebra/F-1 は CAS 非経由）→ **Maxima**＝physics/数学系ドメイン。
  スピン/格子でも spin-algebra・F-1 は Maxima。

力学系は**意味論で2ドメインに分離**（[doc 19](developer_manuals/19_analytical_mechanics_independent_variables.md)）：
ラグランジアン／オイラー・ラグランジュ等で $q$ と $\dot q$ を**独立変数**として扱うものは
`analytical_mechanics`（独立アトム・モデル）、Newton 力学で $\dot q=dq/dt$（従属）として直接積分・
軌道・衝突等を扱うものは `classical_mechanics`。CAS 操作（微分・代入・solve 等）は共通で、
$\dot q$ の意味論だけが異なる。

---

## 🟢 実現済みシナリオ（実践ループ）

| # | テーマ | 分野 | ドメイン | backend | 文書 |
|---|---|---|---|---|---|
| S-a | ハイゼンベルク模型 → 運動方程式 → 動的スペクトル（モーメント→連分数→高温展開） | 量子磁性 | statistical_mechanics | Maxima | [scenario_spin_chain_spectrum.md](scenarios/scenario_spin_chain_spectrum_jp.md)（日英: `_jp`/`_en`） |
| S-b | 連成振動：基準座標変換の逆変換と基準振動（EOM→基準モード→L-1 逆変換） | 古典力学 | analytical_mechanics | Maxima | [scenario_coupled_oscillations.md](scenarios/scenario_coupled_oscillations_jp.md)（日英: `_jp`/`_en`） |
| S-d | 二重振り子のラグランジアン → オイラー・ラグランジュ方程式（速度結合あり） | 解析力学 | analytical_mechanics | Maxima | [scenario_double_pendulum_lagrangian.md](scenarios/scenario_double_pendulum_lagrangian_jp.md)（日英: `_jp`/`_en`） |
| S-e | 直列 RLC の伝達関数：$L\ddot q+R\dot q+q/C=v$ → **定義ラプラス変換**（$\int_0^\infty f e^{-st}dt$、組み込み不使用）→ $H(s)=1/(Ls^2+Rs+1/C)$（部分積分を明示し $q(0),\dot q(0)$ を自由パラメータ化・ゼロIC で特殊化。FREE-PARAM 上で再建） | 制御・回路 | control_theory | Maxima | [scenario_rlc_transfer_function.md](scenarios/scenario_rlc_transfer_function_jp.md)（日英: `_jp`/`_en`） |
| S-f | 与えられた $\boldsymbol{E},\boldsymbol{B}$ で Maxwell 方程式の整合確認 → 分散関係 $\omega=ck$（div/curl、成分は H-1 で付与） | 電磁気 | electromagnetism | Maxima | [scenario_maxwell_consistency.md](scenarios/scenario_maxwell_consistency_jp.md)（日英: `_jp`/`_en`） |
| S-g | 2 次元イジングの分配関数（小格子・高温展開、正方 vs 三角で $\beta^3$ ↔ 三角形数） | 統計力学 | statistical_mechanics | Maxima | [scenario_ising_partition_function.md](scenarios/scenario_ising_partition_function_jp.md)（日英: `_jp`/`_en`） |
| S-h | 中心力ポテンシャルの保存量（$dL_z/dt=0$・$dE/dt=0$）。Poisson 括弧による別解あり | 古典力学 | classical_mechanics | Maxima | [scenario_central_force_conservation.md](scenarios/scenario_central_force_conservation_jp.md)（日英: `_jp`/`_en`）・[scenario_poisson_bracket.md](scenarios/scenario_poisson_bracket_jp.md)（日英: `_jp`/`_en`） |
| S-i | 行列の対角化（固有値・固有ベクトル → 直交 $P$ / 対角 $D$ → $P^{-1}AP=D$） | 線形代数 | linear_algebra | Maxima+Julia | [scenario_matrix_diagonalization.md](scenarios/scenario_matrix_diagonalization_jp.md)（日英: `_jp`/`_en`） |
| S-c | **BCS ギャップ方程式（Julia → Maxima のスカラー handoff）**：Bogoliubov 対角化 → 異常平均 $\langle c_{-k\downarrow}c_{k\uparrow}\rangle=uv/(u^2+v^2)^2$ → **セッション境界を越えて** Maxima で $\int_0^{\omega_D}d\xi/\sqrt{\xi^2+\Delta^2}$ → BCS 閉形式 $\Delta=\omega_D/\sinh\frac{1}{N_0V}$ | 超伝導 | condensed_matter | **Julia→Maxima** | [scenario_bcs_gap_equation.md](scenarios/scenario_bcs_gap_equation_jp.md)（日英: `_jp`/`_en`） |
| S-ag | k 空間の運動項 $\xi_k c^\dagger_k c_k$（添字つきスカラー）：正規順序で反交換関係 → 定数項、交換子 $[H,c_k]=\xi_k c_k-\Delta c^\dagger_{-k}$ | 凝縮系 | condensed_matter | Julia | [scenario_kspace_kinetic_term.md](scenarios/scenario_kspace_kinetic_term_jp.md)（日英: `_jp`/`_en`） |
| S-ai | スピン依存の基底で行列要素を定義する：$\int d^3x$ の成分展開、$h_{m\sigma,n\sigma'}$・$V$（6次元）が**定義のまま**残る | 量子力学 | quantum_mechanics | Maxima | [scenario_spinful_matrix_elements.md](scenarios/scenario_spinful_matrix_elements_jp.md)（日英: `_jp`/`_en`） |
| S-aj | スピンありフェルミオンの1次摂動項：スピン依存・任意基底の $\hat H_0$・$\hat V$ → $[\hat H_0,\hat V]$ が4項。新しい束縛添字が $m_1,s_1$ | 凝縮系 | condensed_matter | Julia | [scenario_spinful_perturbation.md](scenarios/scenario_spinful_perturbation_jp.md)（日英: `_jp`/`_en`） |
| S-ak | **場の演算子で書いたハミルトニアンから、グリーン関数まで**：スピン依存の $\hat\psi_\sigma(\mathbf r)$ → $\hat H_0$・$\hat V$ → 正規直交系（成分展開）→ 基底展開 → 相互作用描像 → $G_\sigma=-i\langle\Phi_0|T[\cdots]|\Phi_0\rangle$ → $G^0$・$\Sigma^{(1)}$ → Dyson 方程式 → 運動方程式 → Hartree–Fock 分解 → $G$ について解く。**2026-08-21 に「計算する版」へ書き直した**（利用者の指摘: 初版は定義を置いただけで、CAS の利用可能性を測っていなかった）。いま**計算しているのは**: 対角化による固有値の導入（δ 縮約）・グリーン関数の基底への移行（2回の代入）・**Dyson の形式解**（両辺に左から逆行列を掛ける）・フーリエ変換（逆変換の積分を微分）・$G^{(HF)}_{n\sigma}(\omega)=\hbar/(\hbar\omega-\varepsilon-\Sigma^{(HF)})$ を解く・$\Sigma^{(HF)}=0$ で自由伝播関数に戻ることの確認。**交換子だけは Maxima に演算子代数が無いので姉妹シナリオ S-al へ**。記法の実測（2026-08-20、open_items GF-NOTATION）の根治を実物で確認する回帰シナリオでもあり、**書き直しの過程でさらに 8 件の silent-wrong が出て根治した** | 量子力学 | quantum_mechanics | Maxima | [scenario_greens_function_from_hamiltonian.md](scenarios/scenario_greens_function_from_hamiltonian_jp.md)（日英: `_jp`/`_en`） |
| S-al | **運動方程式の交換子を、演算子代数で実際に計算する**：QuantumAlgebra で $H=\sum_n\varepsilon_n c^\dagger_n c_n+\sum_{pqsn}V_{pqsn}c^\dagger_p c^\dagger_q c_s c_n$ と $c_m$ の交換子を取り、$-\varepsilon_m c_m+\sum V(\ldots)$ を得る。S-ak（Maxima）が**計算できない1段だけ**を担当する姉妹シナリオで、バックエンドはセッション開始時に固定される（原則#6）という制約がそのまま形になっている | 物性 | condensed_matter | Julia full | [scenario_greens_function_eom_commutator.md](scenarios/scenario_greens_function_eom_commutator_jp.md)（日英: `_jp`/`_en`） |
| S-am | **2サイト Hubbard 模型を、非可換演算子で最後まで計算する**：$H$・$\hat N$・$\hat S^z$ とも**教科書どおりの和の記法**（$\sum_{i=1}^{2}$ は書いた範囲に展開、$\sum_\sigma$ は**宣言した粒子**の射影 $\pm1/2$ に展開。$\hat S^z=\sum_{i\sigma}\sigma\,\hat n_{i\sigma}$ では係数の位置に**値**が、添字の位置に**綴り**が入る）、交換関係は括弧記法（`[A, B]`・`\{A, B\}`・入れ子も）。登録するのは $H$ と比べたい演算子の定義だけで、$[\hat H,\hat N]=0$・$[\hat H,\hat S^z]=0$・$[\hat H,\hat S^+]=0$、SU(2) 代数 $[\hat S^+,\hat S^-]=2\hat S^z$（**仮定せず導出**し、差が 0 になることまで確認）、保存しない二重占有（$U$ が消えて跳び移りの4項だけ残る）、Pauli の $\hat n^2=\hat n$、原子極限の厳密な時間発展 $\hat c_{1\uparrow}(t)=\hat c_{1\uparrow}+(1-e^{-iUt})\hat c^\dagger_{1\downarrow}\hat c_{1\uparrow}\hat c_{1\downarrow}$（PAIR-REP）——**すべて計算結果**。$\uparrow$ は QA の添字にそのまま置ける値で、束縛できないので $\delta(\uparrow,\downarrow)=0$ が落ちる。**初版は和を長く展開して書いていた**——不自然だという指摘を受けて測ったら、和の記法は使わなかったのではなく**使うと黙って誤答していた**（範囲が消える／生の CAS エラー／反交換子と読まれない） | 物性 | condensed_matter | Julia full | [scenario_hubbard_dimer_operator_algebra.md](scenarios/scenario_hubbard_dimer_operator_algebra_jp.md)（日英: `_jp`/`_en`） |
| S-ah | 積分変数を自分の記法で書く（`θ`/`\theta`/`theta` の3綴りが一致）：極座標変換 → $\pi/2$、単変数 $\int_0^\pi\sin^2\theta\,d\theta$ | 数学・電磁気 | electromagnetism | Maxima | [scenario_integration_variable_notation.md](scenarios/scenario_integration_variable_notation_jp.md)（日英: `_jp`/`_en`） |
| S-l | 単一モードのスクイズド状態（Bogoliubov 変換）：スクイズ変換 → $[b,b^\dagger]=1$（正準性）→ 逆変換 → 数演算子の真空期待値 $\langle a^\dagger a\rangle=\sinh^2 r$（真空なのに光子がいる） | 量子光学 | quantum_optics | **Julia** | [scenario_squeezed_state.md](scenarios/scenario_squeezed_state_jp.md)（日英: `_jp`/`_en`） |
| S-p | 超微細構造 $H_{\rm hf}=A\,\mathbf I\cdot\mathbf S$（$I=3/2$・$S=1/2$）を**2経路**で：角運動量合成（F-9）→ 許される $F$ → 準位 $\frac34A,-\frac54A$ → 間隔 $2A$（ランデの間隔則）／スピン代数（F-13）→ $\langle\mathbf I\cdot\mathbf S\rangle=0$・$\langle(\mathbf I\cdot\mathbf S)^2\rangle=\frac{15}{16}$（どちらも CAS 非経由） | 磁気共鳴 | magnetic_resonance | Maxima | [scenario_hyperfine_structure.md](scenarios/scenario_hyperfine_structure_jp.md)（日英: `_jp`/`_en`） |
| S-j | 2準位系の摂動エネルギー補正：$2\times2$ の厳密固有値 → 結合定数 $\lambda$ で Taylor 展開＝Rayleigh–Schrödinger 級数（1次は消え、2次が $\pm V^2\lambda^2/(E_b-E_a)$＝準位反発）→ 和がトレース $E_a+E_b$ に戻る独立検算。**専用の摂動操作を使わない** | 量子力学 | quantum_mechanics | Maxima | [scenario_two_level_perturbation.md](scenarios/scenario_two_level_perturbation_jp.md)（日英: `_jp`/`_en`） |
| S-k | 格子図（ハニカム/正方・周期）の描画 → TeX エクスポート（CAS 非経由） | 可視化 | statistical_mechanics | Maxima | [scenario_lattice_figures.md](scenarios/scenario_lattice_figures_jp.md)（日英: `_jp`/`_en`） |
| S-m | 回転座標系の有効ハミルトニアン（NMR/ESR）：カスタムスピン H 登録 → `rotating_frame` 変換 $H_{\rm rot}=(\omega-\omega_0) S^z-\omega_1 S^x$・`effective_field` $\sqrt{(\omega-\omega_0)^2+\omega_1^2}$・一般 `commutator` で歳差 EOM $[H_{\rm rot},S^z]=i\omega_1 S^y$（一般スピン操作の連続・CAS 非経由） | 磁気共鳴 | magnetic_resonance | Maxima | [scenario_rotating_frame.md](scenarios/scenario_rotating_frame_jp.md)（日英: `_jp`/`_en`） |
| S-n | 交流 RLC のインピーダンス $Z=R+j\omega L+1/(j\omega C)$ → 実部/虚部分離 → 共振 $\omega_0=1/\sqrt{LC}$（EE の `j`＝虚数単位・`i`＝電流。実部/虚部抽出はルールベース、`%i→j` 往復表示。添字/指数の `i` は非破壊） | 電気工学 | electrical_engineering | Maxima | [scenario_ac_impedance.md](scenarios/scenario_ac_impedance_jp.md)（日英: `_jp`/`_en`） |
| S-o | 定積分の変数変換：範囲つき定積分を表示専用で与え、変数変換で被積分関数·\|J\|＋新範囲を表示→評価（極座標·円板→π/2、球座標·球→4π/3） | 解析（体積積分） | electromagnetism | Maxima | [scenario_coordinate_jacobian.md](scenarios/scenario_coordinate_jacobian_jp.md)（日英: `_jp`/`_en`） |
| S-q | Lotka–Volterra の安定性：固定点（共存平衡 $(c/d,a/b)$）→ ヤコビアン → 固有値 $\pm i\sqrt{ac}$（中立中心） | 数理生物学 | mathematical_biology | Maxima | [scenario_lotka_volterra_stability.md](scenarios/scenario_lotka_volterra_stability_jp.md)（日英: `_jp`/`_en`） |
| S-r | M/M/1 待ち行列：利用率 $\rho=\lambda/\mu$ → 平均系内人数 $L=\rho/(1-\rho)$ → $\rho$ 代入 → 両辺を $\lambda$ で割って Little 則 $W=1/(\mu-\lambda)$ | 経営工学・OR | operations_research | Maxima | [scenario_mm1_queue.md](scenarios/scenario_mm1_queue_jp.md)（日英: `_jp`/`_en`） |
| S-s | 2変数線形計画：有効制約の交点を solve → 最適頂点 → 目的 $z=c_1x+c_2y$ 最大化（SUBST-DIGIT-SUB／SUBST-P2-EAGER の2件のバグを根治） | 経営工学・OR | operations_research | Maxima | [scenario_lp_vertex_optimization.md](scenarios/scenario_lp_vertex_optimization_jp.md)（日英: `_jp`/`_en`） |
| S-t | EOQ：総費用 $C(Q)=KD/Q+hQ/2$ を $Q$ で全微分→0 → $Q^*=\sqrt{2KD/h}$ → 代入で最小総費用 | 生産最適化 | production_optimization | Maxima | [scenario_eoq_optimal_order_quantity.md](scenarios/scenario_eoq_optimal_order_quantity_jp.md)（日英: `_jp`/`_en`） |
| S-u | ラインバランシング：作業時間和 $T$・サイクル $c$ → 最小工程数 $n=\lceil T/c\rceil$・効率 $\eta=T/(nc)$（天井関数・H-1 P2自動代入の伝播を確認） | 生産最適化 | production_optimization | Maxima | [scenario_line_balancing.md](scenarios/scenario_line_balancing_jp.md)（日英: `_jp`/`_en`） |
| S-v | ニュースベンダー：過小/過大在庫費用の期待費用を微分→solve → 最適発注量 $Q^*=(bc_u+ac_o)/(c_u+c_o)$（一様需要の臨界比と一致・単一アルファベット添字の一括宣言バグ J-4／J-5 を発見・根治） | SCM・物流 | supply_chain_logistics | Maxima | [scenario_newsvendor_critical_ratio.md](scenarios/scenario_newsvendor_critical_ratio_jp.md)（日英: `_jp`/`_en`） |
| S-w | 在庫配置（リスクプーリング）：安全在庫 $SS=z\sigma\sqrt{L}$ の $L\to nL$ 代入と比較 → 平方根則 $1/\sqrt{n}$（循環代入ガードの誤検出 SUBST-LINEAR-SELFREF を根治） | SCM・物流 | supply_chain_logistics | Maxima | [scenario_risk_pooling_safety_stock.md](scenarios/scenario_risk_pooling_safety_stock_jp.md)（日英: `_jp`/`_en`） |
| S-x | 減衰振動：ODE 一般解 → 判別式の明示 assume（$D=4\omega_0^2-4\gamma^2>0$）→ 初期条件で特殊化 → 右辺抽出 → **数値層で当たり**（N-1 初の実戦、理論→数値の還流一周）。API 検証（2026-07-07）に続き、「🔢 To numeric」UI モーダルからの対話的フォーク実行を実機確認（2026-07-18、同一数値で再現） | 力学・微分方程式 | differential_equations | Maxima＋N-1 | [scenario_damped_oscillator_numeric.md](scenarios/scenario_damped_oscillator_numeric_jp.md)（日英: `_jp`/`_en`） |
| S-y | 2準位ハミルトニアン $H=\begin{pmatrix}\epsilon&\Delta\\\Delta&-\epsilon\end{pmatrix}$ の固有系**プラン**（C-Ext-5「固有状態を求めて」1指示・3/3成功）→ 特性多項式を `solve` で個別固有値式へ → 「🔢 To numeric」で $\epsilon$ 掃引・$\Delta$ 固定 → 避交差（avoided crossing）の数値図を実機確認（C-Ext-5×N-1 の層横断・プラン出力→数値入力の接続） | 量子力学 | quantum_mechanics | Maxima＋N-1 | [scenario_avoided_crossing.md](scenarios/scenario_avoided_crossing.md) |

---

## 🟢 実現済みシナリオ（横断・評価）

物理ワークフローの worked example ではなく、LLM 提案品質など**システム横断の指標**を
測る評価系シナリオ。個別の `scenario_*.md` ワークシートではなく
docs/evaluations.md (reference omitted from the public snapshot) の台帳（1回の実施＝1行、追記型）に記録される。

| # | テーマ | 分野 | ドメイン | backend | 台帳 |
|---|---|---|---|---|---|
| S-aa | Adviser 解法候補の実地評価：実 LLM で3種の問題（固有系・ODE・ガウス積分）に「解法の候補を出して」→ JSON契約遵守率・棄却率・完走率を測定 | 横断（評価） | （複数） | Maxima | evaluations.md (reference omitted from the public snapshot) §S-aa |
| S-ab | notation 候補の実使用＋R-9 需要計測：`\|x-1\|`・未宣言 $\Delta$・`E[X]` | 横断（記法） | （domain なし） | Maxima | evaluations.md (reference omitted from the public snapshot) §S-ab |
| S-ac | バイブコーディングの実地評価：実 coder LLM で3段階の式からコード生成（S-aa のバイブ版） | 横断（評価） | （複数） | Maxima＋Julia＋N-1 | evaluations.md (reference omitted from the public snapshot) §S-ac |
| S-ae | 自由計画（LLM プランナー）の実地評価：実 LLM で「計画を立てて」→提案→選択→実行 | 横断（評価） | （複数） | Maxima | evaluations.md (reference omitted from the public snapshot) §S-ae |
| S-af | OLS 推定量 $\hat\beta=(X'X)^{-1}X'y$：計量経済セッションで設計行列 $X$・応答 $y$ を宣言→ `X'X`・`X'y`（**プライム＝転置**, AUDIT-XTX）が行列積へ→逆行列・積で $\hat\beta=[2/3,1/2]$ を実計算。記法ブリッジ（X'／E[X]／A[i,j]）の実戦検証 | 計量経済 | econometrics | Maxima | [tests/test_ols_scenario.py](../tests/test_ols_scenario.py) |

---

## その他の実例・検証シナリオ

メニュー（S-a〜）には属さないが、機能検証・初期の実践ループ・新テーマとして詳細化済みの文書。

| テーマ | 区分 | backend | 文書 |
|---|---|---|---|
| 汎用作図：曲線のパラメータ比較・非線形の穴のある領域・等高線・注目点 | N-1 共通作図の再現手順（JSON実行確認・回帰テスト、自然言語は依頼例） | Maxima＋N-1 | [scenario_cartesian_plots.md](scenarios/scenario_cartesian_plots.md) |
| 単振り子のラグランジアン → EL（解析力学の基本形） | 実践ループ シナリオ 2 | Maxima | [scenario_pendulum_lagrangian.md](scenarios/scenario_pendulum_lagrangian_jp.md)（日英: `_jp`/`_en`） |
| ラーモア歳差運動（スピン・磁場） | 実践ループ第1回；2026-07-17 workflow版リプレイ（S-z）追記 | Julia | [scenario_larmor_precession.md](scenarios/scenario_larmor_precession_jp.md)（日英: `_jp`/`_en`）— カタログのバックエンド欄が誤って「Maxima」だったのを訂正（本文冒頭は元々 Julia と明記） |
| tight-binding バンド構造（グラフェン・カゴメ・Lieb・SSH） | F-1 3c-fourier 新テーマ | Maxima | [scenario_tight_binding_bands.md](scenarios/scenario_tight_binding_bands_jp.md)（日英: `_jp`/`_en`） |
| BCS ハミルトニアン（Julia 実用検証＝塊A） | バックエンド検証 | Julia | [scenario_bcs_hamiltonian.md](scenarios/scenario_bcs_hamiltonian_jp.md)（日英: `_jp`/`_en`） |
| 横磁場イジングモデル（Julia 実用検証＝塊A） | バックエンド検証 | Julia | [scenario_transverse_field_ising.md](scenarios/scenario_transverse_field_ising_jp.md)（日英: `_jp`/`_en`） |
| フリー電子モデルのフェルミ面（Julia バックエンド動作検証） | バックエンド検証 | Julia | [scenario_free_electron_fermi_surface.md](scenarios/scenario_free_electron_fermi_surface_jp.md)（日英: `_jp`/`_en`） |
| Jaynes-Cummings モデル（光子・2準位分子）| 量子光学（旧 E-3 分割）| Julia | [scenario_jaynes_cummings.md](scenarios/scenario_jaynes_cummings_jp.md)（日英: `_jp`/`_en`） |
| ハイゼンベルク 2サイトスピン鎖 | スピン系（旧 E-3 分割）| Julia | [scenario_heisenberg_two_site.md](scenarios/scenario_heisenberg_two_site_jp.md)（日英: `_jp`/`_en`） |
| クーロンゲージの確認 | 電磁気（旧 E-3 分割）| Maxima | [scenario_coulomb_gauge.md](scenarios/scenario_coulomb_gauge_jp.md)（日英: `_jp`/`_en`） |
| 理想気体の統計力学 | 統計力学（旧 E-3 分割）| Maxima | [scenario_ideal_gas.md](scenarios/scenario_ideal_gas_jp.md)（日英: `_jp`/`_en`） |
| 情報量測度（エントロピー・KLダイバージェンス・相互情報量） | F-16 ドメイン活性化 | Maxima | [scenario_information_theory.md](scenarios/scenario_information_theory_jp.md)（日英: `_jp`/`_en`） |
| 効用最大化とラグランジュ未定乗数法（コブ・ダグラス需要関数・比較静学） | F-14 ドメイン活性化 | Maxima | [scenario_utility_maximization.md](scenarios/scenario_utility_maximization_jp.md)（日英: `_jp`/`_en`） |
| ゲーム理論の均衡（Battle of the Sexes、純戦略・混合戦略ナッシュ均衡） | F-17 ドメイン活性化 | Maxima | [scenario_battle_of_the_sexes.md](scenarios/scenario_battle_of_the_sexes_jp.md)（日英: `_jp`/`_en`） |
| 定義に基づくフーリエ変換・Z変換・畳み込み（ガウス関数の往復・幾何級数） | F-8 ドメイン活性化 | Maxima | [scenario_fourier_convolution.md](scenarios/scenario_fourier_convolution_jp.md)（日英: `_jp`/`_en`） |
| 伊藤の補題とBlack–Scholes方程式（$f=\ln S$・$f=S^2$の教科書的検証） | F-15 ドメイン活性化 | Maxima | [scenario_ito_black_scholes.md](scenarios/scenario_ito_black_scholes_jp.md)（日英: `_jp`/`_en`） |
| ノートブック UI のブラウザ操作確認 | UI 検証 | — | archive/scenario_browser_test.md (reference omitted from the public snapshot)（アーカイブ） |

---

## 未確認・待ちの候補（✅/⏳）

✅（追加機能なしに確認可だが未実施）・⏳（機能待ち）の候補テーマと最新ステータスは、
課題トラッカー側に一元管理している：

→ **[open_items.md ＞ 実践ループ シナリオ・メニュー](../KNOWN_ISSUES.md)**

実現（🟢）したら本ファイルのカタログへ移し、open_items 側からは外す。
