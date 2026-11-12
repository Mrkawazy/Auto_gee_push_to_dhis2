"""Backfill selected exposure names without rewriting existing numeric forecasts.

Run: python -m ghap_engine.repair_forecast_labels --apply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .publish import forecast_climate_names
from .publish_lifecycle import check_import, publication_lock, save_json, verify_values
from .publish_saved import saved_context
from .publish_workflow import _create_or_update_datasets, prepare_values, iter_value_batches


def repair(client, root='ghap_outputs', apply=False):
    data, _, forecasts, settings, run = saved_context(root)
    folder = Path(root) / 'repairs' / run.run_id / 'forecast_labels'
    folder.mkdir(parents=True, exist_ok=True)
    fc = forecasts['forecast_outcome'].copy()
    fc['selected_climate_name'] = forecast_climate_names(fc, data['climate_vars'])
    report = {'status': 'preparing', 'run_id': run.run_id, 'input_rows': len(fc), 'rows': len(fc)}
    with publication_lock(root):
        metadata, ids, _ = _create_or_update_datasets(client, data, settings, folder,
                                                     'live' if apply else 'dry_run')
        values = pd.DataFrame({
            'dataElement': fc['outcome_id'].map(lambda oid: ids['lookup'][('forecast', oid, 'selected_climate_name')]),
            'period': fc['period'].astype(str), 'orgUnit': fc['org_unit_id'],
            'categoryOptionCombo': None, 'value': fc['selected_climate_name'],
            'comment': run.run_id})
        values = prepare_values(client, values, ids, settings, folder, check_dates=apply)
        values.to_csv(folder / 'forecast_label_values.csv', index=False)
        report.update(status='preview', rows=len(values), excluded_rows=len(fc) - len(values),
                      metadata_status=metadata['status'])
        save_json(folder / 'repair_report.json', report)
        if not apply:
            return report
        batches = list(iter_value_batches(values, 2000))
        for page, payload in enumerate(batches, 1):
            check_import(client.post('dataValueSets', payload, {
                'importStrategy': 'CREATE_AND_UPDATE', 'dryRun': 'true'}), len(payload['dataValues']))
            print(f'Forecast labels validated: page {page}/{len(batches)}', flush=True)
        # Existing completion registrations are preserved. Respect any server lock
        # rejection rather than reopening thousands of already published forms.
        report['status'] = 'uploading'
        save_json(folder / 'repair_report.json', report)
        for page, payload in enumerate(batches, 1):
            check_import(client.post('dataValueSets', payload, {'importStrategy': 'CREATE_AND_UPDATE'}),
                         len(payload['dataValues']))
            report['accepted_pages'] = page
            save_json(folder / 'repair_report.json', report)
            print(f'Forecast labels uploaded: page {page}/{len(batches)}', flush=True)
        report['verified'] = verify_values(client, values)
        report['status'] = 'verified'
        save_json(folder / 'repair_report.json', report)
        return report


def main():
    from .dhis2 import make_client
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='ghap_outputs')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(repair(make_client(), args.root, args.apply), indent=2))


if __name__ == '__main__':
    main()
