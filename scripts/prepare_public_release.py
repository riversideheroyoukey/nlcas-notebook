#!/usr/bin/env python3
"""Export current source files for first publication without local Git history.

Uses the working tree (including uncommitted changes), not HEAD. Never edits
runtime settings in the source tree. No upload, commit, or license selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import posixpath
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PREFIXES = ('.git/', '.claude/', '.codex/', '.agents/', '.venv/',
                     '.venv-numeric/', 'sessions/', 'docs/temp/', 'docs/misc/',
                     'public_release/', 'ui/static/lib/katex/', 'docs/archive/',
                     'docs/handoffs/', 'docs/measurements/', 'docs/evaluations/')
EXCLUDED_NAMES = {'.DS_Store', '.numeric_account_pw.dat', 'debug_preprocessed.png',
                  '.coverage', 'CLAUDE.md', 'PROGRESS.md', 'VS_start.sh',
                  'open_items.md', 'evaluations.md', '28_nl_pattern_ratchet_handoff_2026-07.md'}
FILE_LIST = ROOT / 'public_release_files.txt'
SECRET_PATTERNS = {
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'GitHub token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'OpenAI key': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b'),
    'AWS access key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'credential literal': re.compile(r'''(?im)["']?(?:api_key|access_token|auth_token|password|client_secret)["']?\s*[:=]\s*["']([^"'\s]{12,})["']'''),
}


def candidate_files():
    raw = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT)
    return sorted(set(raw.decode().split('\0')) - {''})


def exclusion(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Invalid publication path: {name}')
    if any('apiems' in part.casefold() for part in p.parts):
        return 'conference data directory/file'
    if 'scenario_or_production_planning' in p.name:
        return 'conference production-planning notebook/export or worksheet'
    if p.suffix.lower() in {'.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.pdf', '.zip', '.log', '.env', '.key', '.db', '.sqlite', '.sqlite3', '.bak'}:
        return 'manuscript, slide, attachment, archive or execution log'
    if name.startswith(EXCLUDED_PREFIXES) or '__pycache__' in p.parts:
        return 'local configuration, runtime workspace, history or downloaded assets'
    if (p.name in EXCLUDED_NAMES or p.name == '.env' or p.name.startswith('.env.')
            and p.name != '.env.example' or p.suffix.lower() in {'.iso', '.pem', '.p12', '.pfx', '.pyc', '.dylib', '.so'}):
        return 'local/generated file or credential container'
    return None


def prepare_bytes(name, approved):
    source = ROOT / name
    if not source.resolve().is_relative_to(ROOT) or source.is_symlink():
        raise ValueError(f'Symlink requires explicit review: {name}')
    data = source.read_bytes()
    changes = []
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return data, changes
    # Home directories in archived logs/session examples are not portable.
    text, count = re.subn(r'/path/to/user/\s"\x27]+/', '/path/to/user/', text)
    if count:
        changes.append('replaced personal macOS home-directory prefix')
    if name == 'config/default_settings.json':
        config = json.loads(text)
        config['adviser_target'] = 'ollama_gpt_oss_20b'
        text = json.dumps(config, ensure_ascii=False, indent=2) + '\n'
        changes.append('selected bundled localhost Adviser profile')
    if name.startswith(('models/', 'models.windows/')) and name.endswith('/endpoint.json'):
        endpoint = json.loads(text)
        url = endpoint.get('base_url', '')
        match = re.search(r'://(\d+\.\d+\.\d+\.\d+)', url)
        if match and ipaddress.ip_address(match[1]).is_private and not ipaddress.ip_address(match[1]).is_loopback:
            endpoint['base_url'] = url.replace(match[1], '127.0.0.1', 1)
            text = json.dumps(endpoint, ensure_ascii=False, indent=2) + '\n'
            changes.append('replaced private LAN endpoint with loopback')
    if name == 'docs/developer_manual.md':
        public_start = text.index('## 全体像・仕様')
        text = ('# Developer manual — public index\n\n'
                'Source architecture and implementation references. Start with '
                '[the user guide](user_guide.md) and [the setup guide](setup_guide.md). '
                'Internal development policies, measurements and handoff logs are not distributed.\n\n'
                + text[public_start:])
        start = text.find('**引き継ぎメモ')
        end = text.find('**その他**', start)
        if start >= 0 and end >= 0:
            text = text[:start] + text[end:]
        changes.append('prepared public developer index without internal reading order/logs')
    if name == 'docs/scenarios/scenario_cartesian_plots.md':
        text = text.split('## 生産計画の図へ適用する場合')[0].rstrip() + '\n'
        text += '\nこの公開用手順は一般的な曲線比較と非線形領域の例に限定する。発表用データは同梱しない。\n'
        changes.append('omitted conference-specific case-study section')
    if name == 'docs/scenarios.md':
        text = '\n'.join(line for line in text.split('\n') if 'scenario_or_production_planning' not in line)
        changes.append('omitted conference worksheet entry')
    if name.endswith('.md'):
        # Keep published links navigable without including internal working files.
        def public_link(match):
            image, label, target = match.groups()
            if re.match(r'^[A-Za-z][\w+.-]*:', target) or target.startswith('#'):
                return match.group(0)
            destination = unquote(target.split('#', 1)[0])
            relative = posixpath.normpath(posixpath.join(posixpath.dirname(name), destination))
            if relative == 'docs/open_items.md':
                replacement = posixpath.relpath('KNOWN_ISSUES.md', posixpath.dirname(name) or '.')
                return '[' + label + '](' + replacement + ')'
            if relative in approved or any(v.startswith(relative.rstrip('/') + '/') for v in approved):
                return match.group(0)
            return label + ' (reference omitted from the public snapshot)'
        cleaned = re.sub(r'(!?)\[([^\]]*)\]\(([^\s)]+)\)', public_link, text)
        if cleaned != text:
            text = cleaned
            changes.append('removed links to files outside the publication list')
    # Values are never printed in scan errors or written to the report.
    for label, pattern in SECRET_PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group(1) if label == 'credential literal' else match.group(0)
            if any(s in value.lower() for s in ('example', 'your_', 'your-', 'dummy', 'placeholder', 'getenv', 'environ')):
                continue
            raise ValueError(f'Possible {label} requires review: {name}')
    return text.encode('utf-8'), changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'public_release'/'nlcas-notebook-public')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; choose a new directory to keep the previous candidate.')
    if output == ROOT or ROOT.is_relative_to(output):
        parser.error('Output cannot replace or contain the source tree.')
    archive = output.with_suffix('.zip')
    report_file = output.with_name(output.name + '-report.json')
    if archive.exists() or report_file.exists():
        parser.error('Archive/report already exists; choose a new directory.')
    if not FILE_LIST.is_file():
        parser.error('Missing reviewed public_release_files.txt; no automatic broad export is allowed.')
    approved = {line.strip() for line in FILE_LIST.read_text().splitlines()
                if line.strip() and not line.startswith('#')}
    for name in approved:
        reason = exclusion(name)
        if reason:
            parser.error(f'Publication list contains a prohibited file: {name} ({reason})')
        if not (ROOT/name).is_file():
            parser.error(f'Approved publication file is missing: {name}')
    # Prepare and inspect everything before creating a partial distribution.
    prepared, excluded, transformed = [], [], []
    for name in candidate_files():
        reason = exclusion(name) or (None if name in approved else 'not in the reviewed publication list')
        if reason:
            excluded.append({'file':name, 'reason':reason})
            continue
        source = ROOT/name
        if not source.exists():
            continue
        if source.is_file():
            data, changes = prepare_bytes(name, approved)
            prepared.append((name, data, source.stat().st_mode & 0o777))
            if changes:
                transformed.append({'file':name, 'changes':changes})
    output.mkdir(parents=True)
    inventory = []
    for name, data, mode in prepared:
        dest = output/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        dest.chmod(mode)
        inventory.append({'file':name, 'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()})
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, _, _ in prepared:
            z.write(output/name, arcname='nlcas-notebook/'+name)
    report = {
        'project':'nlcas-notebook', 'owner':'riversideheroyoukey',
        'created_at':datetime.now(timezone.utc).isoformat(),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'source_includes_working_tree_changes':True, 'includes_git_history':False,
        'license':'not selected', 'uploaded':False,
        'secret_scan':'selected token/private-key/literal patterns; no matches in included text files',
        'scope':'reviewed public_release_files.txt allowlist; other files excluded',
        'external_conference_folder_included':False,
        'excluded':excluded, 'transformed':transformed, 'files':inventory,
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f'Prepared {len(prepared)} files; excluded {len(excluded)}; normalized {len(transformed)}.')
    print(f'Folder: {output}\nArchive: {archive}\nReport: {report_file}')


if __name__ == '__main__':
    main()
