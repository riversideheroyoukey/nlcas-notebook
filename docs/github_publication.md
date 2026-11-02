# GitHub公開準備 — NLCAS Notebook

初回公開: 2026-11-02。公開名は `riversideheroyoukey/nlcas-notebook`。
[公開リポジトリ](https://github.com/riversideheroyoukey/nlcas-notebook)。
APIEMS2026の論文・発表に対応するResearch Notebookの実装を保存する。
公開対象はソース・公開用説明・一般的な再現例に限定する。
本体ライセンスは検討中のまま公開する。公開自体にライセンス選択は必須ではない
（[GitHub公式説明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)）。
旧候補ZIPは取り下げ、`nlcas-notebook-public.zip` を公開用のファイル一式とする。

## 公開物の構成

- トップの [README](../README.md) は英語の概要・導入・発表の再現例・後継への導線。
  [日本語README](../README_jp.md) と [英語詳細README](../README_en.md) は従来の詳しい説明を保つ。
- [ソフトウェア引用情報](../CITATION.cff) は著者・プロジェクト名・公開予定URLを記録する。
  論文のDOIや確定していない巻号・公開日を補って記入しない。
- [汎用作図の再現例](scenarios/scenario_cartesian_plots.md) に、検証済みJSON仕様と実行画像を掲載。
- [OOCAS](https://github.com/riversideheroyoukey/oocas) は後継。正式名はOut-of-Order CAS。
  調査時のGitHubメタデータはprivateなので、発表参加者のアクセス可否は別途確認する。
  NLCAS Notebook側の準備作業はOOCAS側の公開設定を変更しない。
- 本体コードのライセンスは未選定。[第三者コンポーネント](../THIRD_PARTY_NOTICES.md)の
  ライセンスと分けて扱う。MIT等を選択したことにして公開しない。
- 紙面・スライドの最終版や未公開研究ノートは、このソース公開には含めない。

## 公開候補を作る

プロジェクトルートで以下を実行する。Python標準ライブラリとGitを使う。

```bash
python3 scripts/prepare_public_release.py
```

作成物:

- `public_release/nlcas-notebook-public/` — 公開用のファイル一式。
- `public_release/nlcas-notebook-public.zip` — 同じ内容の配布候補。
- `public_release/nlcas-notebook-public-report.json` — ファイルごとのSHA-256、除外・置換一覧、元のcommit。

公開対象はルートの `public_release_files.txt` に列挙したファイルだけ。
最新の作図機能のような未コミットの変更も、列挙されたファイルは現在の作業ツリーから作る。
新しいファイルが増えても自動では含めず、公開してよいことを確認した上で一覧に追加する。
個人設定 `.claude/`、環境ファイル、作業用セッション、論文作業フォルダ、巨大バイナリ、
仮想環境、ダウンロード済みKaTeXキャッシュ、既存の `.git` 履歴は含めない。
`Documents/apiems2026` はプロジェクト外なのでコピー元に含めない。
プロジェクト内でも、名前にAPIEMSを含むフォルダ・ファイル、生産計画の発表用保存ノート・TeX・
ワークシート、内部引継ぎ・測定・評価ログ、CLAUDE/PROGRESS/未解決事項の内部台帳、
文書・スライド・表計算添付や実行ログを除外する。一般的な曲線比較・非線形領域の再現例を公開する。
公開用の既知の制限は [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) に要約する。
除外した文書・データへのリンクと作図シナリオの発表固有の節も、公開コピーでは取り除く。
公開候補は新規の初回commitとして扱い、未監査の過去履歴をそのままpushしない。

公開用コピーでは、既定Adviserを同梱のlocalhostプロファイルにし、モデル設定のプライベートLAN
アドレスをloopbackへ置換する。記録中の個人ホームディレクトリも汎用パスに置換する。
元の開発環境の設定・記録は変更しない。
既定のローカルLLMを使う場合も、モデルの別途入手・起動が必要。

公開前のスキャンは、既知形式のトークン、秘密鍵、認証情報の文字列リテラルを対象にし、
疑いがあれば値を表示せず停止する。パターン検査だけで機密情報全般を判定できるわけではないため、
reportの対象ファイル・除外内容も確認する。初回公開後の更新でも、公開対象の差分を確認する。

既存の候補は上書きしない。更新時は別の出力先を指定する。

```bash
python3 scripts/prepare_public_release.py --output public_release/nlcas-notebook-next
```

## GitHub公開と今後の更新

1. ライセンスが未選定の間はREADMEの「検討中」を維持する。選定後にLICENSEと記載を揃える。
2. 更新時にもreport・README・導入手順・実行例を確認し、公開範囲を確定する。
3. 初回公開では空のPublicリポジトリを作り、公開一覧に含むファイルだけを登録する。
   開発用リポジトリの履歴・内部データは登録しない。以後も公開用コピーの差分を登録する。
4. 公開URLを未ログインで開き、READMEの画像・文書リンク・ファイル取得を確認する。
5. 発表から参照する版を確定してタグを付ける。論文・スライドにはその公開URLを記載する。

GitHub接続の認証アカウントは `riversideheroyoukey` と確認済み。
この準備ではパスワードやアクセストークンをソースや手順書に記入しない。
