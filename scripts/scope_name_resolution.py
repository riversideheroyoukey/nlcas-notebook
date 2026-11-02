"""名前の解決を**スコープごとに**追う静的走査。

`tests/test_static_name_resolution.py` が読む測定器。「この名前は実行時に
`NameError` になるか」を、モジュール全体ではなく**そのコードが立っている
スコープの鎖**で答える。

**なぜスコープを追うのか（起票 FUNC-LOCAL-IMPORT-SCOPE、2026-09-06）。**
元の走査はモジュール全体の束縛を1つの集合にしていた——意図的に保守的で、
「当たったら必ず本物」を保つ代わりに取りこぼす、と docstring に書いてあった。
その取りこぼしが実際に出た: `backend_handler.py` から24行を新しいメソッドへ
切り出したとき、`InterpretResult` の import は**元の関数の中**にあったので
新メソッドは `NameError` を出したが、**検査は170件通過したまま**だった。
この検査は「リファクタで落とした import」のために書かれており、関数内 import が
主流のこのコードベースでは、その目的の半分しか果たしていなかった。

**採用前に測ったこと（2026-09-10）。** 起票は「今の0件が穴なのか本当に無いのか、
スコープを追ってみないと分からない」と書いていた。実測: 製品170本に対して
**0件**——本当に無い。穴だったのは**コードではなく検査のほう**で、直したのは
「この族を将来捕まえられるようにする」ことである。

**校正**（極端な数字はまず測定器を疑う、という規則に従って先に実施）:

- 報告する4形 —— 関数内 import を別のメソッドが使う（起票の形そのもの）／
  関数内の代入を別の関数が読む／クラス属性をメソッドが素の名前で読む
  （メソッドからクラスのスコープは見えない）／内包表記の変数を外で読む。
- 沈黙する12形 —— `global`・`nonlocal`・`try/except ImportError` の別名・
  デコレータと既定値（外側で評価される）・walrus・内包表記が外側を読む・
  クラス本体が外側を読む・`TYPE_CHECKING` と文字列注釈・0引数 `super()`・
  f-string・`match` の捕捉名・`except*`・PEP 695 の型引数。
  起票が「踏むだろう」と名指した3つ（`global`/`nonlocal`・内包表記・
  `try/except ImportError`）は全部ここに入っている。

**当たったら必ず本物**という性質は元の走査から引き継ぐ（だから ratchet ではなく
0本である）。スコープ版は元の版より**厳密に強い**——どこにも束縛が無い名前は、
どのスコープ鎖でも解決しないので、元が報告するものは全部報告する。

単体で走らせると製品全体の報告を出す::

    .venv/bin/python scripts/scope_name_resolution.py
"""
import ast
import builtins
import collections
import os

SAFE = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__debug__", "__builtins__", "__path__", "__class__",
}


class Scope:
    __slots__ = ("kind", "parent", "names")
    def __init__(self, kind, parent):
        self.kind, self.parent = kind, parent
        self.names = set()

    def resolve(self, name):
        s, first = self, True
        while s is not None:
            # クラスのスコープは入れ子の関数からは見えない（Python の規則）
            if s.kind != "class" or first:
                if name in s.names:
                    return True
            s, first = s.parent, False
        return False


def _bind_target(node, scope):
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            scope.names.add(n.id)


class Collector(ast.NodeVisitor):
    """1周目: スコープの木と、その束縛を集める。"""
    def __init__(self):
        self.module = Scope("module", None)
        self.by_node = {}          # AST ノード -> そのノードが作るスコープ
        self.stack = [self.module]

    @property
    def cur(self):
        return self.stack[-1]

    def _push(self, node, kind):
        s = Scope(kind, self.cur)
        self.by_node[id(node)] = s
        self.stack.append(s)
        return s

    def _pop(self):
        self.stack.pop()

    # --- 束縛を作る文 ---
    def visit_Import(self, node):
        for a in node.names:
            self.cur.names.add(a.asname or a.name.split(".")[0])
    visit_ImportFrom = visit_Import

    def visit_Global(self, node):
        # **`global` は本当に効く。** 名前がその関数の中でしか代入されていなくても
        # モジュールの束縛になるので、他の関数から読める（実測、これを外すと
        # `global _c; _c = 1` を別関数が読む形が偽陽性になる）。
        self.module.names.update(node.names)
        self.cur.names.update(node.names)

    # `nonlocal` は**扱わない**。変異試験で生き残ったので削った状態を実測した
    # （2026-09-10）: `nonlocal x` は外側の関数に束縛が無ければ Python 自身が
    # コンパイル時に撥ねるので、**合法なコードでは必ず親の鎖に在る**——つまり
    # ここで足しても解決は変わらない。効くのは `ast.parse` は通すが `compile`
    # は通さない形だけで、それは検査でなく import が捕まえる。
    def visit_Assign(self, node):
        for t in node.targets: _bind_target(t, self.cur)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.target is not None: _bind_target(node.target, self.cur)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        _bind_target(node.target, self.cur)
        self.generic_visit(node)

    def visit_NamedExpr(self, node):
        _bind_target(node.target, self.cur)
        self.generic_visit(node)

    def visit_For(self, node):
        _bind_target(node.target, self.cur)
        self.generic_visit(node)
    visit_AsyncFor = visit_For

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars is not None: _bind_target(item.optional_vars, self.cur)
        self.generic_visit(node)
    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node):
        if node.name: self.cur.names.add(node.name)
        self.generic_visit(node)

    def visit_Match(self, node):
        for n in ast.walk(node):
            if isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
                self.cur.names.add(n.name)
            elif isinstance(n, ast.MatchMapping) and n.rest:
                self.cur.names.add(n.rest)
        self.generic_visit(node)

    # --- 新しいスコープを作る文 ---
    def _function(self, node):
        self.cur.names.add(node.name)
        self.cur.names.update(tp.name for tp in getattr(node, "type_params", []))
        for d in node.decorator_list: self.visit(d)
        args = node.args
        for d in (args.defaults + [x for x in args.kw_defaults if x]): self.visit(d)
        for a in (args.posonlyargs + args.args + args.kwonlyargs
                  + ([args.vararg] if args.vararg else [])
                  + ([args.kwarg] if args.kwarg else [])):
            if a.annotation is not None: self.visit(a.annotation)
        if node.returns is not None: self.visit(node.returns)
        s = self._push(node, "function")
        for a in (args.posonlyargs + args.args + args.kwonlyargs
                  + ([args.vararg] if args.vararg else [])
                  + ([args.kwarg] if args.kwarg else [])):
            s.names.add(a.arg)
        s.names.update(tp.name for tp in getattr(node, "type_params", []))
        for st in node.body: self.visit(st)
        self._pop()

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_Lambda(self, node):
        args = node.args
        for d in (args.defaults + [x for x in args.kw_defaults if x]): self.visit(d)
        s = self._push(node, "function")
        for a in (args.posonlyargs + args.args + args.kwonlyargs
                  + ([args.vararg] if args.vararg else [])
                  + ([args.kwarg] if args.kwarg else [])):
            s.names.add(a.arg)
        self.visit(node.body)
        self._pop()

    def visit_ClassDef(self, node):
        self.cur.names.add(node.name)
        self.cur.names.update(tp.name for tp in getattr(node, "type_params", []))
        for d in node.decorator_list: self.visit(d)
        for b in node.bases: self.visit(b)
        for k in node.keywords: self.visit(k.value)
        self._push(node, "class")
        for st in node.body: self.visit(st)
        self._pop()

    def _comp(self, node):
        # 最初の iterable は外側のスコープで評価される
        self.visit(node.generators[0].iter)
        self._push(node, "function")
        for i, gen in enumerate(node.generators):
            _bind_target(gen.target, self.cur)
            if i: self.visit(gen.iter)
            for c in gen.ifs: self.visit(c)
        for part in ([node.elt] if hasattr(node, "elt") else [node.key, node.value]):
            self.visit(part)
        self._pop()

    visit_ListComp = _comp
    visit_SetComp = _comp
    visit_GeneratorExp = _comp
    visit_DictComp = _comp


class Checker(ast.NodeVisitor):
    """2周目: Load をそのスコープ鎖で解決する。"""
    def __init__(self, collector, skip_ids):
        self.c, self.skip = collector, skip_ids
        self.stack = [collector.module]
        self.hits = []

    @property
    def cur(self): return self.stack[-1]

    def _enter(self, node, body_visit):
        s = self.c.by_node.get(id(node))
        if s is None:
            self.generic_visit(node); return
        self.stack.append(s); body_visit(); self.stack.pop()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in SAFE \
           and id(node) not in self.skip and not self.cur.resolve(node.id):
            self.hits.append((node.lineno, node.id))

    def _function(self, node):
        for d in node.decorator_list: self.visit(d)
        args = node.args
        for d in (args.defaults + [x for x in args.kw_defaults if x]): self.visit(d)
        # **注釈は def 実行時に、外側のスコープで評価される。**
        # `from __future__ import annotations` があれば文字列なので `skip` が
        # 外すが、無ければ解決しない注釈は本物の def 時 NameError である。
        # 直す前はここを訪ねておらず、`def f(x: Missing) -> Other` を
        # 黙って通していた（校正テストが捕まえた、2026-09-10）。
        for a in (args.posonlyargs + args.args + args.kwonlyargs
                  + ([args.vararg] if args.vararg else [])
                  + ([args.kwarg] if args.kwarg else [])):
            if a.annotation is not None: self.visit(a.annotation)
        if node.returns is not None: self.visit(node.returns)
        self._enter(node, lambda: [self.visit(st) for st in node.body])
    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_Lambda(self, node):
        for d in (node.args.defaults + [x for x in node.args.kw_defaults if x]): self.visit(d)
        self._enter(node, lambda: self.visit(node.body))

    def visit_ClassDef(self, node):
        for d in node.decorator_list: self.visit(d)
        for b in node.bases: self.visit(b)
        for k in node.keywords: self.visit(k.value)
        self._enter(node, lambda: [self.visit(st) for st in node.body])

    def _comp(self, node):
        self.visit(node.generators[0].iter)
        def body():
            for i, gen in enumerate(node.generators):
                if i: self.visit(gen.iter)
                for c in gen.ifs: self.visit(c)
            for part in ([node.elt] if hasattr(node, "elt") else [node.key, node.value]):
                self.visit(part)
        self._enter(node, body)
    visit_ListComp = _comp
    visit_SetComp = _comp
    visit_GeneratorExp = _comp
    visit_DictComp = _comp


def annotation_ids(tree):
    out = set()
    for node in ast.walk(tree):
        parts = []
        if isinstance(node, ast.AnnAssign): parts.append(node.annotation)
        elif isinstance(node, ast.arg) and node.annotation is not None:
            parts.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
            parts.append(node.returns)
        for p in parts:
            out.update(id(n) for n in ast.walk(p))
    return out


def scan(source):
    tree = ast.parse(source)
    if any(isinstance(n, ast.ImportFrom) and n.module is None and
           any(a.name == "*" for a in n.names) for n in ast.walk(tree)):
        return []
    if any(isinstance(n, (ast.Import, ast.ImportFrom)) and
           any(a.name == "*" for a in n.names) for n in ast.walk(tree)):
        return []
    lazy = any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(a.name == "annotations" for a in n.names) for n in ast.walk(tree))
    skip = annotation_ids(tree) if lazy else set()
    c = Collector(); c.visit(tree)
    ch = Checker(c, skip); ch.visit(tree)
    return ch.hits


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = [os.path.join(root, "cli.py")]
    for d in ("notebook", "ui"):
        for r, _dd, ff in os.walk(os.path.join(root, d)):
            if "__pycache__" in r: continue
            paths += [os.path.join(r, f) for f in ff if f.endswith(".py")]
    total, byfile = 0, collections.Counter()
    rows = []
    for p in sorted(paths):
        hits = scan(open(p, encoding="utf-8").read())
        if hits:
            rel = os.path.relpath(p, root)
            byfile[rel] = len(hits); total += len(hits)
            for ln, nm in hits: rows.append((rel, ln, nm))
    print(f"走査ファイル {len(paths)} 本 / 解決しない名前 {total} 件 / {len(byfile)} ファイル\n")
    for rel, n in byfile.most_common():
        print(f"  {n:4}  {rel}")
    print()
    names = collections.Counter(nm for _r, _l, nm in rows)
    print("名前の多い順（上位25）:")
    for nm, n in names.most_common(25):
        print(f"  {n:4}  {nm}")
