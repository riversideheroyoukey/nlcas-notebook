# インタープリタのレジストリ化とルータ分割の設計

**背景:** 2026-07-11の第3次コード監査（保守性主眼——数値計算層 N-1 G1〜G3 で系が
一段大きくなった機会の棚卸し）を発端に、2026-07-19に Stage R0〜R6 のリファクタが
完了した。本文書はその**設計判断・根本診断**（恒久的に有効）と、各 Stage が適用した
**パターン**をまとめたもの。逐次の実行ログ（各スライスの日付・行数増減・テスト件数）
は git 履歴に譲り、ここには残さない。

**前史:** [17_code_audit_2026-06-13.md](17_code_audit_2026-06-13.md)（第1次 AUD-1〜4:
god-object の mixin 分割）、docs/archive/audit/（第2次・2026-07-03 独立監査: 正しさ
R-1〜R-8＋リファクタ Stage 1〜5）。本監査は**正しさではなく保守性**が主眼。

---

## 1. 維持できている規律（変えないこと）

監査時点で、以下は過去2回の監査の成果が**定着している**ことを確認済み：

- **Backend Encapsulation**: shared 層での `primary_backend ==`／`isinstance(ハンドラ)` 分岐は
  **正当な1箇所**（backend_factory）のみ。CLAUDE.md 規約が守られている。
- **TODO/FIXME/XXX/HACK: 0 件** — 課題は open_items に一元化する規律が機能。
- **engine の mixin 構成は安定**: AUD-4（6/13）で 3,367→2,263 行に分割後、3週間でほぼ
  成長ゼロ。新機能（numeric/plan/wick…）が新 mixin ファイルに載る構造が効いている。
- テストは挙動命名（`test_h9_solve` 等）——モジュール鏡映でないのは意図的で正しい。
- モデル層（`notebook/models/`）は小さく明示的なまま。numeric パッケージも平均関数長は健全。

## 2. 監査所見（保守性リスクの序列）

### 所見A（最重要）: インタープリタ層の「巨大逐次関数」— 分割が定着しなかった唯一の箇所

`run_interpret_pipeline`（AUD-5、6/19 に 2,595→827 行へ分割済みだったが再肥大化）の
実態は、約52個の `_try_*`／`_parse_*` インターセプタが巨大関数内に逐次 if ブロックとして
並ぶ形だった：

- **成長経路が「関数内部への追記」**: 新しい NL 機能のたびにブロックが増える。
- **順序が暗黙**: インターセプタの発火順は if ブロックの物理位置。順序は意味を持つのに
  データとして見えない。
- **個別テスト不能**: 1 インターセプタの検証にパイプライン全体の実行が要る。

### 所見B: bridge の多責務化

`ui/notebook_bridge.py` に FastAPI エンドポイント49本＋タイムラインシリアライズ＋
セッションナラティブ生成＋操作の副作用ハンドラが同居していた。

### 所見C: `_resolve_node` のif/elif ディスパッチ（semantic_resolver）

両バックエンドのリゾルバは共通基底化済みだが、中心メソッドはノード型ごとの巨大 if/elif
のまま。**注意**: 未宣言 subscript→IndexedSymbol の設計（CLAUDE.md）はハード制約——
挙動保存必須。

### 所見D: `extra_pre_phase1`（Julia）— 宣言 NL 解析の一枚岩

スピン演算子・フェルミ/ボソン演算子・一括宣言・確認ダイアログ経路などの宣言族解析が
1 関数。所見Aと同種（逐次正規表現ブロック）。

### 所見E: engine の中核関数（`run_operation` 等）

検証→ソース解決→コマンド構築→実行→結果正規化→step 記録→SymbolContext 更新の
7 相を 1 関数で持つ。mixin 分割の対象外だった「背骨」。

### 所見F: フロントエンド `notebook_main.js` — 単一ファイル・ユニットテスト無し

numeric 系モーダルで直近 +900 行。検証手段が opt-in のブラウザ E2E のみで、
**リファクタのリスクが Python 側より高い**（先送りするほど悪化）。

### 所見G（方針レベル）: NL パターンの日本語直書き正規表現

`nl_patterns.json` 外部化ルール制定**以前**のインターセプタに残っていた。詳細・
悉皆調査の経緯は 28_nl_pattern_ratchet_handoff_2026-07.md (reference omitted from the public snapshot)。

### 所見H（低）: broad except／両ハンドラの同名メソッド

except は AUD-3（6月）でルーティング経路は精査済み。同名メソッドは Backend
Encapsulation 上**正当な差分**が大半——構造が本当に同一の部分のみ基底ヘルパー化の
余地（→ R5 で3件のみ対象化、他は正当な差分と判定し据え置き）。

## 3. 根本診断 — 「分割」より「追加の形」

第1次・第2次リファクタの追跡が示す教訓:

> **分割の成果が定着するのは、その後の機能追加が「新ファイル追加」で済む形にした箇所だけ。**
> 「編集による追加」が残る箇所（パイプラインの if ブロック）は必ず再肥大化する。

したがって本計画の主眼は行数削減ではなく、**成長ホットスポットの追加様式を additive に変える**
こと（レジストリ・ルータ分割・テーブルディスパッチ）と、**後退を検知するラチェット**の導入。

## 4. 各 Stage が適用したパターン（現在のアーキテクチャの由来）

**不変条件（全 Stage 共通）:** 挙動保存（純粋な移動・機械的変換のみ）／Stage ごとに
フルスイート green を確認してからマージ／機能追加と混ぜない。

| Stage | 適用したパターン | 対応する所見 |
|---|---|---|
| **R0** | **衛生ラチェットテスト** `tests/test_code_hygiene.py` を新設——ファイル行数・関数行数・broad except 総数・フロント JS 単一ファイルの上限表を持ち、超過をテスト失敗として検知。新規肥大化は「意識的な上限引き上げコミット」を強制する。CLAUDE.md コーディング規約にも明記 | 再肥大化の構造的防止（AUD-5 の教訓の制度化） |
| **R1** | **インターセプタ・レジストリ化**: 各 `_try_*` をモジュールレベル関数へ切り出し、発火順を明示的な順序付きリスト（データ）で宣言、ディスパッチャはループ1本に（`ui/interpret_registry.py`）。5つの巨大逐次関数（`run_interpret_pipeline`／`_try_early_intercepts`／`_try_equation_intercepts`／`_try_rule_short_circuits`）を全てレジストリ化——新 NL 機能は「`_try_*` を書いてリストに1行」で追加される形に | 所見A |
| **R2** | **bridge のルータ分割**: 全49エンドポイントを `ui/routes/{numeric,history,persistence,export,operations,session,advisory}.py` へ`APIRouter`で分割（パス不変）、シリアライザを別モジュール（`ui/timeline_serializer.py`）へ抽出。bridge 2376→350 行——エンジン singleton ライフサイクル＋アクセサ＋ルータ組立専用に | 所見B |
| **R3** | **engine 背骨のフェーズ抽出**: `run_operation` を検証/解決/構築/実行/正規化/記録の私的メソッド列に分解（mixin 化はしない、ファイル内分解） | 所見E |
| **R4** | **リゾルバのテーブル化＋Julia 宣言族分割**: `_resolve_node` をノード型→メソッドのディスパッチテーブルに（`_RESOLVE_DISPATCH`）、`extra_pre_phase1` を宣言族ごとの関数群に分割 | 所見C／所見D（一部） |
| **R5** | **両ハンドラ同名メソッドの構造同一部の基底化**: 同名ペア約18〜30組を全数調査し、真に同型の3件のみ `BackendHandler` の具象実装へ昇格（残りは正当な差分と判定し据え置き） | 所見H |
| **R6** | **フロントエンド分割**: `notebook_main.js` を ES modules（ビルド不要、`type="module"`）で機能族ごとに分割——`notebook_utils.js`／`notebook_numeric.js`／`notebook_timeline.js`／`notebook_input.js`／`notebook_advise.js`／`notebook_persistence.js`。DOM/状態ハブ（`appendEntry`等）は依存注入で疎結合化。2685→1145 行 | 所見F |

**継続方針（Stage にしない）:**
- NL パターン外部化（所見G）: 機能変更が入るときに個別に行う（「触った時に移す」）。
- broad except: 新規追加は理由コメント必須（R0 のラチェットが総数増を検知）。

**見送った項目:**
- mypy 等の型検査導入——pydantic 中心で恩恵はあるが 38k 行への後付けは費用対効果が
  合わない。新パッケージ限定のパイロットなら将来検討。
- パッケージ再編（`notebook/runtime` の細分化等）——import 互換の破壊に見合う痛みが現状ない。
- `debug_repl.py`——開発ツールで本体非依存。テストもあり現状維持。

**関連:** `docs/open_items.md`（C-Ref-3 アーカイブ項目）、
28_nl_pattern_ratchet_handoff_2026-07.md (reference omitted from the public snapshot)（所見G の詳細）、
`tests/test_code_hygiene.py`／`tests/test_nl_pattern_ratchet.py`（継続運用のラチェット）。
