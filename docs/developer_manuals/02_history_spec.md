# 理論研究ノートの履歴仕様まとめ

## 1. 履歴の目的

このシステムの履歴は、単なるチャットログではなく、**理論計算の再現・再開・分岐・検証のための計算記録**として扱う。

保存すべき対象は、利用者の自然文だけでは不十分である。自然文は後から解釈が変わる可能性があるため、履歴には、自然文に加えて、解釈済みの操作、対象式、利用した仮定、使用した CAS、生成された CAS コマンド、計算結果、採用・不採用の状態、branch 情報を保存する。

つまり、履歴の正本は「会話ログ」ではなく、**計算ステップ列と分岐構造**である。

---

## 2. 履歴は線形ログではなく DAG として扱う

理論計算では、途中から別の方針を試すことが多い。

たとえば、以下のような作業が頻繁に起こる。

```text
ここまでは同じ
ここから別の仮定を試す
ここから別の変形を試す
この結果を採用せず別ルートを試す
```

そのため、履歴は単純な時系列ログではなく、**branch を持つ DAG、有向非巡回グラフ**として設計する。

各計算ステップは、どの branch に属するか、どの step を親としているかを持つ。
フォークが発生した場合、新しい branch を作成し、どの step から分岐したかを記録する。

---

## 3. セッション、branch、step、expression の関係

### 3.1 session

session は、1つの研究ノート全体を表す。

session は、タイトル、計算モード、primary backend、現在アクティブな branch などを持つ。

例:

```text
session:
  title: Hamiltonian derivation
  mode: standard_symbolic
  primary_backend: maxima
  active_branch: main
```

### 3.2 branch

branch は、同じ session 内の計算分岐を表す。

main branch は通常の導出本線である。
別方針を試すときには、新しい branch を作成する。

branch は、以下の情報を持つ。

```text
branch:
  branch id
  label
  forked from step
  head step
```

### 3.3 step

step は、1回の意味ある操作を表す。

step には、CAS 操作だけでなく、以下も含める。

```text
式登録
仮定追加
仮定削除
CAS 操作
LLM supervisor の判断
LLM adviser の考察
曖昧性解消
fallback 試行
branch fork
他 branch からの式採用
手入力メモ
```

step は、自分が属する branch、親 step、入力式、出力式、操作の種類、成功・失敗、状態に影響するかどうかを持つ。

### 3.4 expression

expression は、ノート上の数式を表す。

利用者には `E1`, `E2`, `E3` のような式番号として表示する。
内部的には、branch をまたいでも衝突しない一意な ID を持つ。

expression は、表示用 LaTeX、CAS 用 canonical expression、生成元 step、所属 branch などを持つ。

---

## 4. 表示 ID と内部 ID は分ける

UI では、利用者に分かりやすいように `E1`, `E2`, `E3` のような式番号を使う。

しかし、branch があると、別 branch に同じ `E13` が存在し得る。
そのため、内部では globally unique な ID を使う。

方針は以下とする。

```text
利用者に見せる ID:
  E1, E2, E3 ...

内部 ID:
  expr_uid のような一意 ID

branch:
  どの branch に属する式かを記録
```

利用者は式番号で操作する。
内部処理では一意 ID を使う。

---

## 5. 履歴に保存する step の種類

履歴には、少なくとも以下の step 種別を持たせる。

```text
user_input
expression_register
assumption_add
assumption_remove
cas_operation
llm_supervision
llm_advisory
ambiguity_resolution
fallback_attempt
cross_backend_check
branch_fork
adopt_expression_from_branch
manual_note
```

### 5.1 user_input

利用者が入力した自然文や操作指示を記録する。

### 5.2 expression_register

手書き入力またはテキスト入力により、式がノートに登録されたことを記録する。

### 5.3 assumption_add / assumption_remove

仮定を追加・削除したことを記録する。

例:

```text
m > 0 を仮定する
x は実数とする
a, a† をボソン演算子とする
```

### 5.4 cas_operation

CAS に計算を依頼した操作を記録する。

例:

```text
E1 を展開する
E2 を有理式として簡約する
E3 を x について解く
```

### 5.5 llm_supervision

LLM supervisor が、利用者の操作を解釈したり、曖昧性を解消したり、実行前後のチェックを行った記録。

### 5.6 llm_advisory

LLM adviser が、計算結果に対して物理的考察や次手提案を行った記録。

### 5.7 ambiguity_resolution

曖昧な指示に対して、利用者がどの操作を選んだかを記録する。

### 5.8 fallback_attempt

primary backend で失敗した操作を、別 backend で試したことを記録する。

### 5.9 cross_backend_check

同じ式や操作を複数 backend で比較した記録。

### 5.10 branch_fork

既存履歴の途中から新しい branch を作った記録。

### 5.11 adopt_expression_from_branch

別 branch の結果を現在の branch に採用した記録。

### 5.12 manual_note

利用者が明示的に書いた研究メモ、観察、コメント。

---

## 6. replay 可能性を最優先する

履歴は、セッション再開時に CAS workspace を復元できる必要がある。

再開時の基本手順は以下である。

```text
1. session metadata を読む
2. primary backend を起動する
3. 起動用 preamble / startup を実行する
4. active branch の lineage を求める
5. state_affecting な step だけを順に replay する
6. 式番号・仮定・名前付き式を復元する
7. 作業を再開する
```

重要なのは、**再開時に自然文を再解釈しない**ことである。

避けるべき方式:

```text
保存された user_text を、再開時にもう一度 LLM に解釈させる
```

採用すべき方式:

```text
保存済みの semantic operation、operation arguments、rendered command を replay する
```

これにより、LLM の揺らぎや backend spec の変更による再現性低下を避ける。

---

## 7. state_affecting フラグ

すべての step を replay する必要はない。

各 step には、ノート状態や CAS workspace に影響するかどうかを示す `state_affecting` 相当の情報を持たせる。

### replay 対象

以下は replay 対象とする。

```text
式登録
仮定追加・削除
名前付き式定義
採用済み CAS 結果
ユーザ定義ルール
branch で採用された式
```

### replay 不要

以下は通常 replay 不要とする。

```text
表示だけの操作
LLM advisory の未採用提案
失敗した CAS 実行
不採用の fallback 結果
単なる説明文
比較検証だけの副次結果
```

---

## 8. CAS command は保存するが、通常 UI には見せない

履歴には、CAS に実際に送った command を保存する。

理由は以下である。

```text
再現性
デバッグ
監査
将来の backend spec 変更への備え
```

ただし、通常 UI では CAS 文法を見せない。

利用者に見せるのは、以下のような数学的操作である。

```text
E1 を有理式として簡約
E2 を x で微分
E3 に E1 を代入
```

CAS command は、開発者モードまたは詳細表示でのみ確認可能にする。

---

## 9. primary backend はセッション単位で固定する

操作ごとに Maxima / Julia を自動で切り替えると、履歴再開が複雑になる。

そのため、通常はセッション開始時に primary backend を固定する。

例:

```text
標準記号計算モード:
  primary backend = Maxima

物性・量子多体系モード:
  primary backend = Julia full
```

fallback や cross-backend check は可能だが、明示的な履歴イベントとして扱う。

重要な方針:

```text
通常操作は primary backend に閉じる
別 backend を使う場合は利用者に確認する
fallback の結果を本線に採用するかどうかも明示的に記録する
```

---

## 10. branch の replay

branch を再開するには、その branch の lineage をたどる。

例:

```text
main:
  S1 → S2 → S3 → S4 → S5

branch_002:
  S1 → S2 → S3 → S8b → S9b
```

`branch_002` を再開する場合、replay するのは以下である。

```text
S1, S2, S3, S8b, S9b
```

main の `S4`, `S5` は replay しない。

これにより、branch ごとの作業状態を独立に復元できる。

---

## 11. フォーク操作

ロード済み履歴の途中からフォークできる。

フォーク手順は以下とする。

```text
1. 任意 step S_k を選ぶ
2. 新しい branch を作る
3. forked_from_step = S_k として保存する
4. S_k までの状態を再構成する
5. 以後の操作は新 branch に追加する
6. 元 branch は変更しない
```

フォークは、理論計算における以下の作業に対応する。

```text
別の仮定を試す
別の変形順序を試す
別の近似を入れる
別の基底で計算する
別 CAS で検証する
```

---

## 12. branch 間比較

branch 間の式を比較できるようにする。

例:

```text
main の E18 と branch_002 の E18 を比較
```

比較方法は式の種類に依存する。

### 可換な式

差を取り、簡約して 0 になるか確認する。

```text
difference = simplify(left - right)
```

### 方程式

両辺の差、または同値変形で比較する。

### 行列

要素ごとの差を比較する。

### 非可換式

非可換代数用の正規形や専用比較 operation を用いる。

---

## 13. branch merge は初期実装では自動化しない

自動 merge は難しいため、初期実装では不要とする。

代わりに、利用者が明示的に結果を採用する。

例:

```text
branch_002 の E18 を main に採用
```

この場合、採用操作を step として保存する。

採用操作は、新しい式を現在 branch に作成する。
元 branch は変更しない。

---

## 14. LLM supervisor / adviser も履歴に残す

LLM の判断も研究過程の一部として保存する。

### 14.1 LLM supervisor

LLM supervisor は以下を行う。

```text
利用者指示の解釈
操作候補の提示
曖昧性解消
実行前チェック
実行後チェック
履歴 step の要約
```

たとえば、利用者が「E3 を整理して」と言った場合、supervisor は以下のような候補を出す。

```text
x についてまとめる
展開する
因数分解する
有理式として簡約する
```

利用者が選んだ操作も履歴に保存する。

### 14.2 LLM adviser

LLM adviser は、CAS 結果に対して以下を行う。

```text
物理的考察
数理的解釈
注意点の指摘
次に試すべき操作の提案
検証方法の提案
```

LLM adviser の出力は、CAS 結果とは別扱いにする。
adviser の出力は仮説・考察・提案であり、自動でノート本線に採用しない。

利用者が提案を採用した場合のみ、対応する CAS 操作または branch fork が作成される。

---

## 15. LLM step と CAS step は分ける

LLM が提案したことと、CAS が実行したことは別の step として保存する。

例:

```text
L23:
  LLM supervisor が「整理して」を collect / expand / factor に分解

S24:
  利用者が collect を選び、CAS が実行

A25:
  LLM adviser が結果の物理的意味を考察
```

このように分けることで、後から以下を追跡できる。

```text
なぜこの操作を選んだか
どの候補があったか
利用者がどれを選んだか
CAS が何を返したか
LLM がどのように考察したか
その考察を採用したか
```

---

## 16. 履歴表示 UI

通常 UI では、CAS 文法ではなく、数学的操作と結果を表示する。

表示例:

```text
E1: (x^2 - 1)/(x - 1)

S2: E1 を有理式として簡約
E2: x + 1

S3: E2 を x で微分
E3: 1
```

branch 表示例:

```text
main
 ├─ S1 E1 を登録
 ├─ S2 E1 を簡約 → E2
 ├─ S3 E2 を微分 → E3
 │
 └─ branch_002 from S2
     ├─ S4 E2 を展開 → E4
     └─ S5 E4 を x について解く → E5
```

詳細表示では、operation 名や backend を表示してよい。
開発者表示では、実際の CAS command も表示してよい。

---

## 17. 履歴保存ファイルの最小構成

MVP では、以下の情報を保存できればよい。

```text
session metadata
branches
expressions
assumptions
steps
```

### session metadata

セッション全体の情報。

```text
session id
title
mode
primary backend
active branch
created at
updated at
```

### branches

各 branch の情報。

```text
branch id
label
forked from step
head step
primary backend override の有無
```

### expressions

式の情報。

```text
内部 ID
表示 ID
branch
LaTeX
canonical expression
生成元 step
状態
```

### assumptions

仮定の情報。

```text
仮定 ID
内容
適用 branch
追加 step
削除 step
```

### steps

計算・判断・分岐の履歴。

```text
step id
kind
branch
parent steps
input refs
output refs
operation
operation args
backend
rendered command
status
state_affecting
user confirmation
```

---

## 18. MVP で実装すべき履歴機能

最初に実装する履歴機能は以下で十分である。

```text
1. セッション作成
2. main branch 作成
3. 式登録 step
4. CAS 操作 step
5. 結果式の登録
6. session 保存
7. session 読み込み
8. active branch の replay
9. 任意 step から branch fork
10. branch 選択
```

後回しでよいもの:

```text
自動 branch merge
高度な branch diff
複数 backend branch の完全同期
LLM advisory の高度な要約
大規模履歴の圧縮
```

---

## 19. 開発上の注意

履歴設計で避けるべきこと:

```text
自然文だけを保存する
線形ログ前提で作る
式番号だけを内部 ID にする
CAS 結果を無条件に本線採用する
fallback を黙って実行する
LLM の考察を CAS 結果と同じ扱いにする
replay 時に自然言語を再解釈する
```

守るべきこと:

```text
semantic operation を保存する
branch_id と parent_step_ids を保存する
expr_uid と display_id を分ける
state_affecting を明示する
primary_backend を session metadata に保存する
rendered_command を監査用に保存する
LLM step と CAS step を分ける
```

---

## 20. 一文での要約

このノート履歴は、**会話履歴ではなく、再現可能な理論計算の分岐付き実験記録**として設計する。

利用者は、黒板や紙のノートのように計算を進めながら、必要に応じて過去の任意地点に戻り、別 branch を作り、別の仮定や変形を試せる。
CAS と LLM は、その過程を支援するが、履歴の正本は、利用者が採用した計算 step とその分岐構造である。

---

## 20b. E 番号に穴を空けない（INTERNAL-INTERMEDIATE-EID-GAP、2026-08-06）

**不変条件：タイムラインに出る式番号は `E1, E2, E3, …` と連続し、欠番が無い。**

E 番号は研究者が**書き留めて後から参照する識別子**である。だから規則は2つあり、
向きが逆であることが要点：

1. **一度表示した番号は動かさない。** 後から詰め直す（振り直す）ことはしない。
   保存済みセッションには研究者自身の `user_text`（「E7 に E6 を代入して」）が残っており、
   番号を動かせばその履歴が嘘になる。
2. **表示しないものに番号を与えない。** 足場（scaffolding）は誕生時から別名前空間
   `_T1`, `_T2`, … を使う（`_alloc_internal_expr_id`、`Expression.internal`）。

この2つを合わせると「穴が空いたら塞ぐ」ではなく「そもそも穴を空けない」になる。

### 穴が空いていた2つの機構（別物）

| 機構 | 何が起きていたか | 直し方 |
|---|---|---|
| 足場 | 一時式を登録→操作→**後から**隠す。隠すのは操作の後（でないと `_active_expr_ids()` が見つけられず当の操作が動かない）なので番号は消費済み | 誕生時に `internal=True`（`register_expression` / `run_operation` の両方にある） |
| solve / build_inverse_transform | 結果枠を CAS 実行前に確保 → 第1解を**別の新番号で登録**して枠へコピー → 元の鍵を pop。カウンタは進んだままで2解が E3・E5 に。さらに pop した鍵を `output_refs` に持つ**登録ステップが残っていた** | 第1解を確保済みの枠へ直接登録（昇格と lhs_alias_map 付け替えが不要に） |

### 意図的に残した2箇所

- **plan の中間物**（`engine_plan_mixin`）：隠すか否かが*後続*ステップの失敗（`failed_at`）に
  依存し、操作実行時には決められない。
- **読込時マイグレーション**（`engine_persistence_mixin`）：対象は保存済みセッションで、
  上の番号は後続ステップと `user_text` から参照済み。規則1に反するので振り直さない。

### 代償：番号を書き留めた台本は書き換えが要る

欠番を埋めれば後続の番号は必ず動く。動く先は次の2つで、影響が違う：

| | 影響 |
|---|---|
| **保存済みセッション** | **無し。** 自分の番号を持って読み込まれ、既存の式は動かない（既存の保存ファイル全件で実測） |
| **E 番号を書いた台本** | **書き換えが要る。** 再走すると番号が別の式を指す |

後者の実例が `docs/scenarios` の台本で、変更直後にシナリオ7本（jp/en 計14走）が
`Unknown expression references` で落ちた。**手作業の確認では通っており、generator を
通して初めて見えた。** 新番号は各シナリオの committed 例（旧番号で生成済み）から
位置対応で機械的に求められる——可視の式を順に E1..En と振り直したものが、そのまま
新しい番号になる。

落ちるのは幸運な側で、危険なのは**落ちずに別の式を指す**場合である。だから
「失敗したものだけ直す」ではなく、番号を含む台本を全て機械的に振り直すこと。

---

### 「式参照か」の判定は解決側と NL 側で別の問い

`is_expr_ref`（`notebook/runtime/expr_ref_fields.py`、`IDENTIFIER_KEYS` の隣）は
**「expr_lookup に引いて意味があるか」**を答える。`_T3` を含む。
NL パーサと supervisor の `E\d+` は**「研究者が E 番号を打ったか」**を答える。`_T3` を含まない
——研究者は打たないし、含めれば足場が参照可能になって目的に反する。

この区別を付けずに内部鍵を導入したところ、`^E\d+$` が**11箇所に手書きで散っていた**ため
鍵が CAS へ裸の記号として渡り、`E1 の両辺に 2 をかけて` が `2 E = _{T}_{1}` を**エラー無しで**
返した。9箇所は `and ... in expr_lookup` と対で書かれ個々には正しく、個々に新しい鍵が
見えていなかった。

---

## 21. Maxima スナップショット方式（notebook JSON との関係）

保存時: `MaximaAdapter.save_snapshot()` が `functions` / `values` / `facts()` / `dependencies` を問い合わせ、再実行可能な Maxima コマンド列として `.mac` ファイルに書き出す。

読み込み時: `.mac` ファイルが存在すればコマンドを順次実行して状態を復元。ファイルが存在しない、または読み込みに失敗した場合は従来の `replay_current_branch()` にフォールバックする。

**設計原則:**
- **notebook JSON が正本**。Maxima snapshot は notebook JSON の代替ではなく、計算継続用の補助キャッシュである。
- Maxima snapshot の読み込みに失敗しても、notebook JSON からノートは開ける（replay にフォールバック）。
- Maxima snapshot は保存時点の Maxima 状態をそのまま保存するため、保存時点で状態が汚染されていれば、その汚染も保存される。

**実装上の制限:**
- `save(file, all)` は Lisp 形式で全履歴を含むため使用しない。代わりにユーザー状態のみを抽出した `.mac` を生成する。
- 内部アダプター変数（`_res_` 等）は保存対象外。
- Maxima の `assumptions` データベースは `facts()` 経由で取得するため、`forget()` で削除した仮定は正しく除外される。
- `assume_positive` 等の特殊フラグは保存されない（通常のセッションでは使用しない）。
- `declare_time_dependent` が Maxima に送る `depends()` は現行コードでは rendered_command に記録されないが、スナップショット取得時に `dependencies` を直接問い合わせるため正しく保存される。

---

## 実装状況（2026-05-16 時点）

### 実装済み ✅

| 要素 | 備考 |
|------|------|
| Session / Branch / Step / Expression / Assumption モデル | §3 |
| expr_uid（内部 UUID）/ display_id（E1, E2...）の分離 | §4 — モデルに存在、エンジンは display_id を主キーとして使用中 |
| StepKind: cas_operation / expression_registration / branch_adoption / llm_advisory | §5 |
| StepKind: user_input / assumption_add / assumption_remove | §5.1–5.3 |
| StepKind: llm_supervision / ambiguity_resolution / branch_fork / manual_note | §5.5, 5.7, 5.10, 5.12 |
| Step.input_refs（DAG 辺） | §17 |
| Step.user_confirmation（曖昧性解消記録） | §17 |
| state_affecting フラグ | §7 |
| rendered_command 保存（show_command_to_user=False） | §8 |
| primary_backend セッション固定 | §9 |
| active branch replay（lineage 計算 + state_affecting step のみ再実行） | §6, §10 |
| 任意 step から live branch fork（branch_fork step 記録付き、`fork_branch`/`switch_branch`） | §11 |
| branch 操作 UI（ファイルスナップショット fork モーダル + live branch fork/switch UI・branch セレクタ、2026-07-16） | §11 |
| Assumption 管理（add / remove / remove_by_canonical、engine.assumptions dict） | §5.3 |
| llm_supervision step の記録（bridge で record_llm_supervision） | §5.5 |
| ambiguity_resolution step（候補リスト + user_confirmation 記録） | §5.7 |
| manual_note の削除と lineage 再配線（delete_note_step） | §5.12 |
| LLM adviser（CAS 結果への考察・次手提案、多 LLM 選択、メモ保存） | §5.6, §14.2 |
| session 保存 / 読み込み（JSON、assumptions 含む） | §17 |
| Maxima スナップショット保存・読み込み（.mac、replay フォールバック） | — |
| ブランチサマリ自動生成（lineage から deterministic テキスト生成） | §16 代替 |
| 保存ダイアログ（上書き / 名前付き）+ 自動保存ローテーション（5 スロット） | — |
| LaTeX エクスポート（タイムライン → 自己完結 .tex ドキュメント） | — |
| タイムライン UI — 仮定タイル（青）・メモタイル（黄）、個別削除 | §16 |

### 未実装 ❌

| 要素 | 仕様参照 |
|------|---------|
| StepKind: fallback_attempt / cross_backend_check | §5.8, 5.9 |
| branch 間比較（式の差分・同値確認） | §12 |
| UI ブランチ可視化（DAG ツリー表示）— サマリテキストで代替中 | §16 |
