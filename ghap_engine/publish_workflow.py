"""Notebook publishing workflow for the two scalable monthly GHAP datasets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .publish_lifecycle import (check_import, prepare_metadata, save_json, reopen_forms,
                                complete_forms, verify_values, refresh_analytics, publication_lock)


def create_or_update_datasets(client, data, settings, folder, mode='dry_run'):
    with publication_lock(settings['output_folder']):
        return _create_or_update_datasets(client, data, settings, folder, mode)


def _create_or_update_datasets(client, data, settings, folder, mode='dry_run'):
    """Create/update both datasets and outcome sections; preserve older members.

    dry_run validates metadata on the server without committing or running maintenance.
    live validates first, commits and confirms dataset membership and category combos.
    """
    from .publish import build_metadata, resolve_cocs
    if mode not in {'dry_run', 'live'}:
        raise ValueError('Metadata mode must be dry_run or live.')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta, ids = build_metadata(data, settings)
    meta, ids = prepare_metadata(client, meta, ids)
    save_json(folder / 'dhis2_metadata.json', meta)
    params = {'importStrategy': 'CREATE_AND_UPDATE', 'atomicMode': 'ALL', 'mergeMode': 'MERGE'}
    validated = client.post('metadata', meta, {**params, 'importMode': 'VALIDATE'})
    check_import(validated)
    report = {'status': 'metadata_validated', 'datasets': ids['datasets'],
              'data_elements': len(meta['dataElements']), 'validation': validated}
    save_json(folder / 'metadata_report.json', report)
    if mode == 'live':
        committed = client.post('metadata', meta, {**params, 'importMode': 'COMMIT'})
        check_import(committed)
        report['commit'] = committed
        report['status'] = 'metadata_committed'
        save_json(folder / 'metadata_report.json', report)
        try:
            cocs = resolve_cocs(client, ids['combo'], set(ids['options'].values()))
        except Exception as exc:
            report.update(status='category_setup_failed', error=str(exc))
            save_json(folder / 'metadata_report.json', report)
            raise
        for kind, uid in ids['datasets'].items():
            actual = client.get('dataSets/' + uid, {'fields': 'id,periodType,dataSetElements[dataElement[id]],organisationUnits[id]'})
            if actual.get('periodType') != 'Monthly':
                raise RuntimeError('Output dataset does not use Monthly periods: ' + uid)
            expected = {value for (table, _, _), value in ids['lookup'].items()
                        if ('forecast' if table == 'forecast_pair' else table) == kind}
            found = {entry['dataElement']['id'] for entry in actual.get('dataSetElements', [])}
            expected_units = {item['id'] for ds in meta['dataSets'] if ds['id'] == uid for item in ds['organisationUnits']}
            found_units = {item['id'] for item in actual.get('organisationUnits', [])}
            if expected - found or expected_units - found_units:
                raise RuntimeError('Dataset membership readback failed: ' + uid)
        report['status'] = 'metadata_ready'
    else:
        cocs = None
    save_json(folder / 'metadata_report.json', report)
    return report, ids, cocs


def prepare_values(client, values, ids, settings, folder, check_dates=True):
    """Validate import keys and exclude periods outside facility operating dates."""
    values = values.copy()
    by_element = {uid: ids['datasets']['forecast' if table == 'forecast_pair' else table]
                  for (table, _, _), uid in ids['lookup'].items()}
    values['dataSet'] = values['dataElement'].map(by_element)
    if values.empty:
        return values
    if values[['dataSet', 'dataElement', 'orgUnit', 'period', 'value']].isna().any().any():
        raise ValueError('Publish payload contains a missing required key/value.')
    if not values['period'].astype(str).str.fullmatch(r'\d{4}(0[1-9]|1[0-2])').all():
        raise ValueError('Publication periods must be monthly YYYYMM strings.')
    text_ids = {uid for (_, _, param), uid in ids['lookup'].items() if param == 'selected_climate_name'}
    text = values['dataElement'].isin(text_ids)
    if values.loc[text, 'value'].map(lambda v: not isinstance(v, str) or not v.strip() or len(v) > 500).any():
        raise ValueError('Selected climate labels must be nonempty text of at most 500 characters.')
    if not np.isfinite(pd.to_numeric(values.loc[~text, 'value'], errors='coerce')).all():
        raise ValueError('Publish payload contains nonnumeric/nonfinite values.')
    att_elements = {uid for (table, _, _), uid in ids['lookup'].items() if table in {'attribution', 'forecast_pair'}}
    if values.loc[values['dataElement'].isin(att_elements), 'categoryOptionCombo'].isna().any():
        raise ValueError('An attribution climate category option combination is unresolved.')
    keys = ['dataElement', 'period', 'orgUnit', 'categoryOptionCombo']
    if values.duplicated(keys).any():
        raise ValueError('Duplicate DHIS2 data value keys; refusing an ambiguous upload.')
    if len(values) > settings['publish_max_values']:
        raise ValueError(f"{len(values)} values exceed publish_max_values={settings['publish_max_values']}.")
    if check_dates:
        dates = {}
        units = sorted(values['orgUnit'].unique())
        for start in range(0, len(units), 80):
            body = client.get('organisationUnits', {'filter': 'id:in:[' + ','.join(units[start:start + 80]) + ']',
                              'fields': 'id,openingDate,closedDate', 'paging': 'false'})
            dates.update({item['id']: item for item in body.get('organisationUnits', [])})
        if set(units) - set(dates):
            raise RuntimeError('Operating dates were not returned for every output organisation unit.')
        pairs = values[['orgUnit', 'period']].drop_duplicates().copy()
        def reason(row):
            unit = dates[row['orgUnit']]
            month = pd.Period(row['period'], freq='M')
            if unit.get('openingDate') and month.end_time < pd.Timestamp(unit['openingDate']):
                return 'before_opening_date'
            if unit.get('closedDate') and month.start_time > pd.Timestamp(unit['closedDate']):
                return 'after_closure_date'
            return ''
        pairs['exclusion_reason'] = pairs.apply(reason, axis=1)
        excluded = pairs.loc[pairs['exclusion_reason'].ne('')]
        excluded.to_csv(Path(folder) / 'dhis2_excluded_periods.csv', index=False)
        values = values.merge(pairs, on=['orgUnit', 'period'], validate='many_to_one')
        values = values.loc[values['exclusion_reason'].eq('')].drop(columns='exclusion_reason')
    return values.sort_values(keys, na_position='last').reset_index(drop=True)


def iter_value_batches(values, batch):
    """Materialise only one request's row dictionaries at a time."""
    for dataset, group in values.groupby('dataSet', sort=False):
        for start in range(0, len(group), batch):
            rows = group.iloc[start:start + batch].drop(columns='dataSet').to_dict('records')
            rows = [{key: value for key, value in item.items() if value is not None and not
                     (isinstance(value, float) and np.isnan(value))} for item in rows]
            yield {'dataSet': dataset, 'dataValues': rows}


def publish(client, results, forecasts, data, settings, run, mode='off', reviewed_by='', batch=5000, verify_sample=25):
    """Prepare, validate or publish; live mode verifies ALL values before completion.

    verify_sample is retained for API compatibility; live verification is exhaustive.
    Metadata-only setup is available separately through create_or_update_datasets.
    """
    from .publish import build_metadata, build_data_values, resolve_cocs
    if mode not in {'off', 'dry_run', 'live'} or batch < 1:
        raise ValueError('Use mode off/dry_run/live and a positive batch size.')
    if mode == 'live' and not reviewed_by.strip():
        raise PermissionError('Live model-result publishing requires reviewed_by; metadata setup does not.')
    report = {'mode': mode, 'run_id': run.run_id, 'status': 'preparing'}
    report_path = run.dir / 'publish_report.json'
    save_json(report_path, report)
    originals = []
    try:
        if mode == 'off':
            meta, ids = build_metadata(data, settings)
            save_json(run.dir / 'dhis2_metadata.json', meta)
            cocs = {option: '<coc:' + option + '>' for option in ids['options'].values()}
        else:
            meta_report, ids, cocs = _create_or_update_datasets(client, data, settings, run.dir, mode)
            report['metadata'] = meta_report
            if mode == 'dry_run':
                try:
                    cocs = resolve_cocs(client, ids['combo'], set(ids['options'].values()), generate=False)
                except RuntimeError:
                    report.update(status='metadata_validated', note='Run metadata setup first to enable data-value validation. No writes were performed.')
                    save_json(report_path, report)
                    return report
        values = build_data_values(results, forecasts, data, settings, ids, cocs, run.run_id)
        values = prepare_values(client, values, ids, settings, run.dir, check_dates=mode != 'off')
        values.to_csv(run.dir / 'dhis2_datavalues_preview.csv', index=False)
        report.update(data_values=len(values), datasets=ids['datasets'])
        save_json(report_path, report)
        if mode == 'off' or values.empty:
            report['status'] = 'payload_written' if mode == 'off' else 'no_eligible_values'
            save_json(report_path, report)
            return report
        # Validate every batch before reopening forms or committing any data values.
        validated_batches = 0
        total_batches = sum((len(group) + batch - 1) // batch for _, group in values.groupby('dataSet'))
        for payload in iter_value_batches(values, batch):
            check_import(client.post('dataValueSets', payload, {'importStrategy': 'CREATE_AND_UPDATE', 'dryRun': 'true'}), len(payload['dataValues']))
            validated_batches += 1
            print(f"Validated batch {validated_batches}/{total_batches} ({len(payload['dataValues'])} values)", flush=True)
            report.update(status='validating', validated_batches=validated_batches)
            save_json(report_path, report)
        if mode == 'dry_run':
            report['status'] = 'dry_run_completed'
            save_json(report_path, report)
            return report
        if settings.get('publish_complete', True):
            report['status'] = 'reopening'
            save_json(report_path, report)
            originals = reopen_forms(client, values, run.dir / 'completion_snapshot.json')
        counts = dict(imported=0, updated=0, ignored=0, deleted=0, conflicts=[])
        for index, payload in enumerate(iter_value_batches(values, batch)):
            stats = check_import(client.post('dataValueSets', payload, {'importStrategy': 'CREATE_AND_UPDATE'}), len(payload['dataValues']))
            for key in ('imported', 'updated', 'ignored', 'deleted'):
                counts[key] += int(stats.get(key, 0))
            report.update(status='uploading', accepted_batches=index + 1, **{'import': counts})
            save_json(report_path, report)
            print(f"Published batch {index + 1}/{validated_batches}", flush=True)
        verified = verify_values(client, values)
        report['readback_verified'] = f'{verified} / {len(values)}'
        if settings.get('publish_complete', True):
            report['completion'] = complete_forms(client, values)
            originals = []
        if settings.get('publish_analytics', True):
            report['analytics'] = refresh_analytics(client, values, run.dir,
                timeout=settings.get('analytics_timeout_seconds', 3600), poll_seconds=settings.get('analytics_poll_seconds', 5),
                skip_outliers=settings.get('analytics_skip_outliers', True))
        report['status'] = 'published'
        registry = run.root / 'model_registry.csv'
        if registry.exists():
            reg = pd.read_csv(registry, dtype=str, keep_default_na=False)
            reg.loc[reg['run_id'].eq(run.run_id), ['published', 'reviewed_by']] = ['True', reviewed_by]
            reg.to_csv(registry, index=False)
        save_json(report_path, report)
        run.log('publish', status='published', data_values=len(values), verified=verified)
        return report
    except (Exception, KeyboardInterrupt) as exc:
        if report.get('status') == 'reopening' and not originals:
            snapshot = run.dir / 'completion_snapshot.json'
            if snapshot.exists():
                originals = json.loads(snapshot.read_text()).get('original_completions', [])
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}')
        if originals:
            try:
                for start in range(0, len(originals), 500):
                    check_import(client.post('completeDataSetRegistrations', {'completeDataSetRegistrations': originals[start:start + 500]},
                                             {'importStrategy': 'CREATE_AND_UPDATE'}))
                report['original_completions_restored'] = len(originals)
            except Exception as restore_error:
                report['completion_restore_error'] = str(restore_error)
        save_json(report_path, report)
        run.log('publish_failed', error=report['error'])
        raise
