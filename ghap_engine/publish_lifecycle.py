"""Additive metadata updates, verified completions and tracked analytics jobs."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


@contextmanager
def publication_lock(folder):
    """Prevent overlapping local publishers; the OS releases locks after a crash."""
    path = Path(folder) / '.dhis2_publication.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    if path.stat().st_size == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    import os
    acquired = False
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        yield
    except OSError as exc:
        if not acquired:
            raise RuntimeError('Another publisher is using this output folder; wait for it to finish.') from exc
        raise
    finally:
        if acquired:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    # Windows indexers/antivirus can briefly hold JSON files open without sharing delete access.
    for attempt in range(6):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * 2 ** attempt)


def check_import(body, expected=None):
    summary = body.get('response', body)
    status = summary.get('status', body.get('status'))
    counts = summary.get('importCount', summary.get('stats', {}))
    def errors(obj):
        if isinstance(obj, dict):
            return bool(obj.get('errorReports')) or any(errors(v) for v in obj.values())
        return isinstance(obj, list) and any(errors(v) for v in obj)
    if status not in {'OK', 'SUCCESS'} or summary.get('conflicts') or errors(body):
        raise RuntimeError('DHIS2 import was not successful: ' + json.dumps(body, default=str)[:2000])
    if expected is not None:
        accepted = sum(int(counts.get(k, 0)) for k in ('imported', 'updated', 'ignored'))
        if accepted != expected:
            raise RuntimeError(f'DHIS2 accounted for {accepted} of {expected} requested values: {counts}')
    return counts


def prepare_metadata(client, metadata, ids):
    """Reuse objects by code and union relationships, preserving older members/sharing."""
    metadata = json.loads(json.dumps(metadata))
    existing, remap = {}, {}
    for kind, objects in metadata.items():
        existing[kind] = {}
        for start in range(0, len(objects), 80):
            codes = [item['code'] for item in objects[start:start + 80]]
            found = client.get(kind, {'filter': 'code:in:[' + ','.join(codes) + ']',
                                      'fields': ':owner', 'paging': 'false'}).get(kind, [])
            for item in found:
                if item['code'] in existing[kind]:
                    raise RuntimeError('Ambiguous DHIS2 metadata code: ' + item['code'])
                existing[kind][item['code']] = item
        for item in objects:
            old = existing[kind].get(item['code'])
            if old:
                remap[item['id']] = old['id']

    def remapped(obj):
        if isinstance(obj, dict):
            return {key: remap.get(value, value) if key == 'id' and isinstance(value, str)
                    else remapped(value) for key, value in obj.items()}
        return [remapped(v) for v in obj] if isinstance(obj, list) else obj

    metadata = remapped(metadata)
    ids = {**ids, 'combo': remap.get(ids['combo'], ids['combo']),
           'lookup': {key: remap.get(value, value) for key, value in ids['lookup'].items()},
           'options': {key: remap.get(value, value) for key, value in ids['options'].items()},
           'datasets': {key: remap.get(value, value) for key, value in ids['datasets'].items()}}
    lists = {'categories': ['categoryOptions'], 'categoryCombos': ['categories'],
             'dataSets': ['dataSetElements', 'organisationUnits', 'sections'], 'sections': ['dataElements']}
    climate_added = any(
        {v['id'] for v in item.get('categoryOptions', [])}
        - {v['id'] for v in existing['categories'].get(item['code'], {}).get('categoryOptions', [])}
        for item in metadata.get('categories', []))
    for kind, objects in metadata.items():
        for index, item in enumerate(objects):
            old = existing[kind].get(item['code'], {})
            merged = {**old, **item}
            for field in lists.get(kind, []):
                def key(value):
                    return value['dataElement']['id'] if field == 'dataSetElements' else value['id']
                members = {key(value): value for value in old.get(field, [])}
                for value in item.get(field, []):
                    members[key(value)] = {**members.get(key(value), {}), **value}
                merged[field] = list(members.values())
            if kind == 'dataSets' and old:
                def membership(record, field):
                    return {value['dataElement']['id'] if field == 'dataSetElements' else value['id']
                            for value in record.get(field, [])}
                changed = any(membership(merged, field) != membership(old, field) for field in lists[kind])
                if changed or (climate_added and item['code'] == 'GHAP2_DS_ATTRIBUTION'):
                    merged['version'] = int(old.get('version') or 1) + 1
            objects[index] = merged
    all_ids = [item['id'] for objects in metadata.values() for item in objects]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Metadata UID collision; nothing was imported.')
    return metadata, ids


def reference(value):
    return value.get('id') if isinstance(value, dict) else value


def completion_rows(client, dataset, targets):
    found = {}
    pages = []
    for period in sorted({period for _, period in targets}):
        units = sorted(unit for unit, month in targets if month == period)
        for start in range(0, len(units), 80):
            pages.append((period, units[start:start + 80]))
    for page, (period, units) in enumerate(pages, 1):
        print(f'Registrations {dataset}: page {page}/{len(pages)} ({period}, {len(units)} units)', flush=True)
        body = client.get('completeDataSetRegistrations', {'dataSet': dataset, 'period': period,
                          'orgUnit': units, 'children': 'false'})
        for item in body.get('completeDataSetRegistrations', []):
            if item.get('completed') is False:
                continue
            unit, month = reference(item.get('organisationUnit')), reference(item.get('period'))
            if (unit, month) in targets and reference(item.get('dataSet')) == dataset:
                found[(unit, month)] = item
    return found


def targets_by_dataset(values):
    return {dataset: set(zip(group['orgUnit'], group['period'])) for dataset, group in values.groupby('dataSet')}


def reopen_forms(client, values, path):
    """Persist original completions before reopening only the affected output forms."""
    originals = []
    for dataset, targets in targets_by_dataset(values).items():
        for item in completion_rows(client, dataset, targets).values():
            originals.append({key: reference(value) for key, value in item.items()
                              if key in {'dataSet', 'organisationUnit', 'period', 'attributeOptionCombo', 'date', 'storedBy'}})
    save_json(path, {'original_completions': originals, 'status': 'reopening'})
    attributes = {}
    print(f'Reopening {len(originals)} completed forms', flush=True)
    for index, item in enumerate(originals, 1):
        params = {'ds': item['dataSet'], 'ou': item['organisationUnit'], 'pe': item['period'],
                  'multiOu': 'false'}
        if item.get('attributeOptionCombo'):
            aoc = item['attributeOptionCombo']
            if aoc not in attributes:
                combo = client.get('categoryOptionCombos/' + aoc,
                                   {'fields': 'id,categoryCombo[id],categoryOptions[id]'})
                cc = reference(combo.get('categoryCombo'))
                options = [reference(option) for option in combo.get('categoryOptions', [])]
                if not cc or not options or not all(options):
                    raise RuntimeError('Cannot resolve completion attribute category: ' + aoc)
                attributes[aoc] = {'cc': cc, 'cp': ';'.join(sorted(options))}
            params.update(attributes[aoc])
        response = client.delete('completeDataSetRegistrations', params)
        if response.get('status') == 'ERROR' or int(response.get('httpStatusCode', 0)) >= 400:
            raise RuntimeError('DHIS2 refused to reopen form '
                               + f"{item['dataSet']}/{item['organisationUnit']}/{item['period']}: "
                               + json.dumps(response, default=str)[:2000])
        if index == 1 or index % 25 == 0 or index == len(originals):
            print(f'Reopened {index}/{len(originals)} forms ({index / len(originals):.0%})', flush=True)
    # A successful DELETE is insufficient: confirm the selected registrations are gone.
    for dataset, targets in targets_by_dataset(values).items():
        if completion_rows(client, dataset, targets):
            raise RuntimeError('Some output forms remain complete; data upload was not started.')
    return originals


def complete_forms(client, values, batch=500):
    rows = [{'dataSet': dataset, 'organisationUnit': unit, 'period': period,
             'date': pd.Timestamp.now(tz='UTC').date().isoformat()}
            for dataset, targets in targets_by_dataset(values).items() for unit, period in sorted(targets)]
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        check_import(client.post('completeDataSetRegistrations', {'completeDataSetRegistrations': chunk},
                                 {'importStrategy': 'CREATE_AND_UPDATE'}), len(chunk))
        print(f'Completed forms: page {start // batch + 1}/{math.ceil(len(rows) / batch)} '
              f'({min(start + batch, len(rows))}/{len(rows)} forms)', flush=True)
    verified = 0
    for dataset, targets in targets_by_dataset(values).items():
        actual = completion_rows(client, dataset, targets)
        missing = targets - set(actual)
        if missing:
            raise RuntimeError(f'Completion readback failed for {dataset}: {sorted(missing)[:5]}')
        verified += len(targets)
    return {'status': 'verified', 'forms': verified}


def verify_values(client, values):
    """Compare every uploaded key with raw DHIS2 storage, including retained zeros."""
    verified = 0
    groups = values.groupby(['dataSet', 'orgUnit'], sort=False)
    for page, ((dataset, unit), group) in enumerate(groups, 1):
        print(f'Raw readback: page {page}/{groups.ngroups} ({verified}/{len(values)} values verified)', flush=True)
        months = pd.PeriodIndex(group['period'], freq='M')
        body = client.get('dataValueSets', {'dataSet': dataset, 'orgUnit': unit,
                          'startDate': months.min().start_time.strftime('%Y-%m-%d'),
                          'endDate': months.max().end_time.strftime('%Y-%m-%d')})
        stored = {}
        for item in body.get('dataValues', []):
            key = (item['dataElement'], item['period'], item.get('categoryOptionCombo'))
            stored.setdefault(key, []).append(item['value'])
        for item in group.to_dict('records'):
            coc = item.get('categoryOptionCombo')
            if not isinstance(coc, str):
                # Forecasts use the default category combo; reject ambiguous categories.
                matches = [v for (de, period, _), vals in stored.items()
                           if de == item['dataElement'] and period == item['period'] for v in vals]
            else:
                matches = stored.get((item['dataElement'], item['period'], coc), [])
            numeric = pd.to_numeric(item['value'], errors='coerce')
            if pd.isna(numeric):
                equal = len(matches) == 1 and str(matches[0]) == str(item['value'])
            else:
                actual = pd.to_numeric(matches[0], errors='coerce') if len(matches) == 1 else float('nan')
                equal = math.isclose(actual, float(numeric), rel_tol=1e-6, abs_tol=0.0)
            if not equal:
                raise RuntimeError(f"Value readback mismatch: {unit}/{item['period']}/{item['dataElement']}/{coc}: {matches}")
            verified += 1
    return verified


def analytics_completed(messages):
    """DHIS2 LOOP notifications complete individual steps, not the entire job."""
    return any(m.get('level') == 'INFO' and m.get('completed') is True
               and str(m.get('message', '')).lower().startswith('analytics tables updated')
               for m in messages)


def refresh_analytics(client, values, folder, timeout=3600, poll_seconds=5, skip_outliers=True):
    """Submit once, track this exact task and resume interrupted polling."""
    path = Path(folder) / 'analytics_report.json'
    digest = hashlib.sha256(values.to_json(orient='records').encode()).hexdigest()
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous.get('status') == 'completed' and not analytics_completed(previous.get('notifications', [])):
        # Older publishers incorrectly accepted completed LOOP progress messages.
        # Preserve the evidence, then submit a replacement only after the server is idle.
        save_json(path.with_name('analytics_report_invalid_completion.json'), previous)
        previous = {}
    if previous.get('status') in {'requesting', 'submission_unconfirmed'}:
        raise RuntimeError('Analytics submission is uncertain; inspect analytics_report.json and server job status before resubmitting.')
    deadline = time.monotonic() + timeout
    if previous.get('payload_sha256') == digest and previous.get('status') in {'running', 'completed'}:
        report = previous
    else:
        while 'ANALYTICS_TABLE' in client.get('scheduling/running/types'):
            if time.monotonic() >= deadline:
                raise TimeoutError('Another analytics job is running; retry to monitor after it finishes.')
            time.sleep(poll_seconds)
        report = {'status': 'requesting', 'payload_sha256': digest}
        save_json(path, report)
        try:
            earliest = int(values['period'].astype(str).str[:4].min())
            params = {'skipResourceTables': 'false', 'skipAggregate': 'false',
                      # Outlier SQL is unsupported on separate analytics databases.
                      # GHAP uses aggregate values, not DHIS2 outlier statistics.
                      'skipOutliers': 'true' if skip_outliers else 'false',
                      'skipEvents': 'true', 'skipEnrollment': 'true', 'skipTrackedEntities': 'true',
                      'lastYears': str(max(1, pd.Timestamp.now(tz='UTC').year - earliest + 1))}
            report['parameters'] = params
            body = client.post_once('resourceTables/analytics', {}, params)
            if body.get('status') not in {None, 'OK', 'SUCCESS'}:
                raise RuntimeError('Analytics request rejected: ' + str(body))
            task = body.get('response', body)
            task_id = task.get('id') or task.get('uid')
            match = re.search(r'/ANALYTICS_TABLE/([A-Za-z0-9]+)', body.get('location', ''))
            task_id = task_id or (match.group(1) if match else None)
            if not task_id:
                raise RuntimeError('Analytics accepted without a trackable task ID: ' + str(body))
        except Exception as exc:
            report.update(status='submission_unconfirmed', error=str(exc))
            save_json(path, report)
            raise
        report.update(status='running', task_id=task_id)
        save_json(path, report)
    while report['status'] != 'completed':
        if time.monotonic() >= deadline:
            raise TimeoutError('Analytics still running. Rerun to resume task ' + report['task_id'])
        messages = client.get('system/tasks/ANALYTICS_TABLE/' + report['task_id'])
        if not isinstance(messages, list):
            raise RuntimeError('Unexpected analytics task notifications: ' + str(messages))
        report['notifications'] = messages
        latest = str(messages[0].get('message', 'Waiting for notifications')) if messages else 'Waiting for notifications'
        if latest != report.get('last_progress_message'):
            print(f"Analytics {report['task_id']}: {latest}", flush=True)
            report['last_progress_message'] = latest
        if any(m.get('level') == 'ERROR' for m in messages):
            report['status'] = 'failed'
            save_json(path, report)
            raise RuntimeError('Analytics task failed: ' + str(messages))
        if analytics_completed(messages):
            report['status'] = 'completed'
        save_json(path, report)
        if report['status'] != 'completed':
            time.sleep(poll_seconds)
    # Check one nonzero value per output dataset through analytics, including its COC.
    checks = []
    for dataset, group in values.groupby('dataSet'):
        numeric = pd.to_numeric(group['value'], errors='coerce')
        nonzero = group.loc[numeric.notna() & numeric.ne(0)]
        if nonzero.empty:
            checks.append({'dataset': dataset, 'status': 'all_values_zero; raw storage verified'})
            continue
        elements = sorted(nonzero['dataElement'].unique())
        aggregatable = set()
        for start in range(0, len(elements), 80):
            metadata = client.get('dataElements', {'filter': 'id:in:[' + ','.join(elements[start:start + 80]) + ']',
                                  'fields': 'id,aggregationType', 'paging': 'false'})
            aggregatable.update(de['id'] for de in metadata.get('dataElements', [])
                                if de.get('aggregationType') and de['aggregationType'] != 'NONE')
        candidates = nonzero.loc[nonzero['dataElement'].isin(aggregatable)]
        if candidates.empty:
            raise RuntimeError('No aggregatable nonzero value available for analytics visibility check: ' + dataset)
        item = candidates.iloc[0]
        dx = item['dataElement']
        if isinstance(item.get('categoryOptionCombo'), str):
            dx += '.' + item['categoryOptionCombo']
        body = client.get('analytics.json', {'dimension': ['dx:' + dx, 'ou:' + item['orgUnit'], 'pe:' + item['period']],
                                            'skipMeta': 'true', 'skipRounding': 'true'})
        headers = [h['name'] for h in body.get('headers', [])]
        rows = [dict(zip(headers, row)) for row in body.get('rows', [])]
        matches = [r for r in rows if r.get('dx') == dx and r.get('ou') == item['orgUnit'] and r.get('pe') == item['period']]
        if len(matches) != 1 or not math.isclose(float(matches[0]['value']), float(item['value']), rel_tol=1e-6, abs_tol=0.0):
            report['visibility'] = checks + [{'dataset': dataset, 'status': 'failed',
                'dataElement': dx, 'orgUnit': str(item['orgUnit']), 'period': str(item['period']),
                'expected': str(item['value']), 'rows': rows[:3]}]
            save_json(path, report)
            raise RuntimeError('Analytics visibility check failed for ' + dataset + ': ' + str(rows[:3])
                               + '. Uploaded values remain stored; use --analytics-only to retry without uploading again.')
        checks.append({'dataset': dataset, 'status': 'verified', 'dataElement': dx})
    report['visibility'] = checks
    save_json(path, report)
    return report
