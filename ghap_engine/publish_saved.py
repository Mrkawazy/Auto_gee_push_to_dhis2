"""Publish saved GHAP results without rerunning modelling or input retrieval.

Examples:
  python -m ghap_engine.publish_saved --metadata-only --mode live
  python -m ghap_engine.publish_saved --mode dry_run
  python -m ghap_engine.publish_saved --mode live --reviewed-by "Reviewer name"
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .config import default_settings
from .dhis2 import make_client
from .publish import create_or_update_datasets, publish
from .publish_lifecycle import save_json, refresh_analytics, publication_lock


def saved_context(root='ghap_outputs'):
    root = Path(root)
    current = root / 'current'
    settings = {**default_settings(), **json.loads((current / 'settings.json').read_text())}
    # Use the expanded output configuration even if this saved model predates it.
    settings['publish_facility_params'] = default_settings()['publish_facility_params']
    settings['publish_complete'] = True
    settings['publish_analytics'] = True
    snapshot = json.loads((root / 'inputs' / 'snapshot.json').read_text())
    data = {'outcomes': {o['outcome_id']: o for o in snapshot['outcomes']},
            'facilities': pd.read_parquet(root / 'inputs' / 'facilities.parquet'),
            'climate_vars': pd.read_parquet(root / 'inputs' / 'climate_vars.parquet')}
    results = {'tables': {'model_summary': pd.read_parquet(current / 'model_summary.parquet'),
                          'aggregates': pd.read_parquet(current / 'aggregates.parquet')},
               'facility_month_dir': current / 'facility_month'}
    forecasts = {'forecast_outcome': pd.read_parquet(current / 'forecast_outcome.parquet')}
    if (current / 'forecast_pair.parquet').exists():
        forecasts['forecast_pair'] = pd.read_parquet(current / 'forecast_pair.parquet')
    run_id = (current / 'RUN_ID.txt').read_text().strip()
    folder = root / 'publications' / run_id
    folder.mkdir(parents=True, exist_ok=True)
    def log(event, **detail):
        with (folder / 'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'event': event, **detail}, default=str) + '\n')
    run = SimpleNamespace(dir=folder, root=root, run_id=run_id, log=log)
    return data, results, forecasts, settings, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='ghap_outputs')
    parser.add_argument('--mode', choices=['off', 'dry_run', 'live'], default='dry_run')
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--analytics-only', type=Path, metavar='RUN_FOLDER',
                        help='Refresh/verify analytics using a saved payload, without uploading data or metadata.')
    parser.add_argument('--reviewed-by', default='')
    parser.add_argument('--batch-size', type=int, default=5000)
    args = parser.parse_args()
    if args.analytics_only:
        if args.metadata_only:
            parser.error('--analytics-only and --metadata-only cannot be combined.')
        folder = args.analytics_only
        values = pd.read_csv(folder / 'dhis2_datavalues_preview.csv', dtype=str)
        with publication_lock(args.root):
            report = refresh_analytics(make_client(), values, folder)
        print(json.dumps(report, indent=2))
        return
    data, results, forecasts, settings, run = saved_context(args.root)
    client = make_client()
    if args.metadata_only:
        if args.mode == 'off':
            parser.error('Metadata setup requires dry_run or live.')
        report, _, _ = create_or_update_datasets(client, data, settings, run.dir, args.mode)
    else:
        report = publish(client, results, forecasts, data, settings, run, mode=args.mode,
                         reviewed_by=args.reviewed_by, batch=args.batch_size)
    save_json(run.dir / 'command_report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in {'validation', 'commit', 'metadata'}}, indent=2))


if __name__ == '__main__':
    main()
