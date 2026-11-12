"""Readable historical attribution and selected forecast dataframes."""
from pathlib import Path

import pandas as pd


def monthly_output_views(results=None, forecasts=None, folder='ghap_outputs/current'):
    """Use the current in-memory run, or reload the latest saved run offline.

    Attribution retains each outcome/climate pair separately and exposes qa_pass.
    These are inspection tables; calling this function does not publish data.
    """
    from .pipeline import load_facility_month

    if (results is None) != (forecasts is None):
        raise ValueError('Run both attribution and forecast cells first, or pass neither to reload both saved tables.')
    root = Path(folder)
    if results is not None:
        summary = results['tables']['model_summary']
        frames = [load_facility_month(results, oid) for oid in summary['outcome_id'].unique()]
    else:
        summary = pd.read_parquet(root / 'model_summary.parquet')
        frames = [pd.read_parquet(path) for path in sorted((root / 'facility_month').glob('*.parquet'))]
    frames = [frame for frame in frames if not frame.empty]
    attribution = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not attribution.empty:
        labels = summary[['outcome_id', 'climate_uid', 'indicator_name', 'qa_pass', 'family']].drop_duplicates(['outcome_id', 'climate_uid'])
        attribution = attribution.merge(labels, on=['outcome_id', 'climate_uid'], how='left', validate='many_to_one')
        cols = ['indicator_name', 'outcome_id', 'facility_name', 'facility_id', 'period',
                'climate_name', 'climate_uid', 'family', 'qa_pass', 'observed', 'expected',
                'expected_without_exposure', 'an', 'an_low', 'an_high',
                'af', 'af_low', 'af_high', 'rr', 'rr_low', 'rr_high']
        attribution = attribution[cols].sort_values(['indicator_name', 'climate_name', 'facility_name', 'period']).reset_index(drop=True)
    forecast = (forecasts['forecast_outcome'].copy() if forecasts is not None
                else pd.read_parquet(root / 'forecast_outcome.parquet'))
    return attribution, forecast
