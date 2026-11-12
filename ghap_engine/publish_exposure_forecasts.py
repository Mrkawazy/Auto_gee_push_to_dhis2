"""Publish saved per-exposure forecasts into climate-column forecast sections.

Run: python -m ghap_engine.publish_exposure_forecasts --apply
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .publish import build_data_values
from .publish_saved import saved_context
from .publish_lifecycle import publication_lock, check_import, save_json, verify_values
from .publish_workflow import _create_or_update_datasets, prepare_values, iter_value_batches


def changed_values(client, values):
    """Read every key and return only missing or changed numeric forecasts."""
    pending = []
    groups = values.groupby(['dataSet', 'orgUnit'], sort=False)
    for page, ((dataset, unit), group) in enumerate(groups, 1):
        print(f'Compare exposure forecasts: page {page}/{groups.ngroups}', flush=True)
        months = pd.PeriodIndex(group['period'], freq='M')
        body = client.get('dataValueSets', {'dataSet': dataset, 'orgUnit': unit,
            'startDate': str(months.min().start_time.date()), 'endDate': str(months.max().end_time.date())})
        stored = {}
        for row in body.get('dataValues', []):
            key = (row['dataElement'], row['period'], row.get('categoryOptionCombo'))
            stored.setdefault(key, []).append(row['value'])
        for index, row in group.iterrows():
            matches = stored.get((row['dataElement'], row['period'], row['categoryOptionCombo']), [])
            if len(matches) > 1:
                raise RuntimeError('Ambiguous stored forecast key; refusing to overwrite: ' + str(index))
            actual = pd.to_numeric(matches[0], errors='coerce') if matches else float('nan')
            if not math.isclose(float(actual), float(row['value']), rel_tol=1e-6, abs_tol=0.0):
                pending.append(index)
    return values.loc[pending].copy()


def publish_exposures(client, root='ghap_outputs', apply=False):
    data, results, forecasts, settings, run = saved_context(root)
    pair = forecasts.get('forecast_pair', pd.DataFrame())
    if pair.empty:
        raise ValueError('No saved exposure forecasts are available; run forecasting first.')
    folder = Path(root) / 'publications' / run.run_id / 'exposure_forecasts'
    folder.mkdir(parents=True, exist_ok=True)
    # Do not rebuild or re-upload attribution or the selected-model forecast.
    empty = folder / 'empty_facility_month'
    empty.mkdir(exist_ok=True)
    results = {'facility_month_dir': empty, 'tables': {
        'model_summary': results['tables']['model_summary'], 'aggregates': pd.DataFrame()}}
    with publication_lock(root):
        _, ids, cocs = _create_or_update_datasets(client, data, settings, folder,
                                                'live' if apply else 'dry_run')
        if not apply:
            cocs = {option: '<coc:' + option + '>' for option in ids['options'].values()}
        values = build_data_values(results, {'forecast_pair': pair}, data, settings, ids, cocs, run.run_id)
        values = prepare_values(client, values, ids, settings, folder, check_dates=apply)
        values.to_csv(folder / 'exposure_forecast_values.csv', index=False)
        report = {'status': 'preview', 'run_id': run.run_id, 'values': len(values),
                  'model_pairs': len(pair[['outcome_id', 'climate_uid']].drop_duplicates())}
        save_json(folder / 'publish_report.json', report)
        if not apply or values.empty:
            return report
        pending = changed_values(client, values)
        report.update(unchanged_values=len(values) - len(pending), uploaded_values=0)
        if pending.empty:
            report.update(status='verified', verified=len(values), accepted_pages=0)
            save_json(folder / 'publish_report.json', report)
            return report
        total = (len(pending) + 4999) // 5000
        for page, payload in enumerate(iter_value_batches(pending, 5000), 1):
            check_import(client.post('dataValueSets', payload, {
                'importStrategy': 'CREATE_AND_UPDATE', 'dryRun': 'true'}), len(payload['dataValues']))
            print(f'Exposure forecasts validated: page {page}/{total}', flush=True)
        for page, payload in enumerate(iter_value_batches(pending, 5000), 1):
            check_import(client.post('dataValueSets', payload, {'importStrategy': 'CREATE_AND_UPDATE'}),
                         len(payload['dataValues']))
            report.update(status='uploading', accepted_pages=page,
                          uploaded_values=report['uploaded_values'] + len(payload['dataValues']))
            save_json(folder / 'publish_report.json', report)
            print(f'Exposure forecasts uploaded: page {page}/{total}', flush=True)
        report.update(status='verified', verified=verify_values(client, values))
        save_json(folder / 'publish_report.json', report)
        return report


if __name__ == '__main__':
    from .dhis2 import make_client
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='ghap_outputs')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(publish_exposures(make_client(), args.root, args.apply), indent=2))
