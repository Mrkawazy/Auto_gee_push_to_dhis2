"""Make the forecast entry form mirror attribution without deleting data values."""
import json
import re
import uuid
from pathlib import Path

from .publish_lifecycle import publication_lock, save_json
from .publish_workflow import _create_or_update_datasets


def redesign(client, data, settings, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with publication_lock(settings['output_folder']):
        # sections is an inverse association and is excluded by :owner alone.
        dataset = client.get('dataSets/u4LeniuSnf2', {'fields': ':owner,sections[id]'})
        old_sections = [client.get('sections/' + s['id'], {'fields': ':owner'})
                        for s in dataset.get('sections', [])]
        backup = folder / ('forecast_structure_before_' + uuid.uuid4().hex + '.json')
        save_json(backup, {'dataSets': [dataset], 'sections': old_sections})
        _, ids, _ = _create_or_update_datasets(client, data, settings, folder, 'live')
        selected_ids = {de for (table, _, _), de in ids['lookup'].items() if table == 'forecast'}
        removed = []
        for section in old_sections:
            if not re.fullmatch(r'GHAP2_SECTION_FORECAST_[A-F0-9]{6}', section.get('code', '')):
                continue
            ds = section.get('dataSet', {})
            if (ds.get('id') if isinstance(ds, dict) else ds) != ids['datasets']['forecast']:
                raise RuntimeError('Legacy section belongs to another dataset: ' + section['id'])
            if any(e['id'] not in selected_ids for e in section.get('dataElements', [])):
                raise RuntimeError('Legacy section has unexpected elements; inspect backup: ' + str(backup))
            # A section contains presentation metadata, not data values. Retain
            # all original elements and dataset membership for stored records.
            body = client.delete('sections/' + section['id'])
            if body.get('status') == 'ERROR' or int(body.get('httpStatusCode', 0)) >= 400:
                raise RuntimeError('Cannot remove legacy presentation section: ' + json.dumps(body))
            removed.append(section['id'])
            print(f'Removed duplicate forecast section {len(removed)}', flush=True)
        actual = client.get('dataSets/' + ids['datasets']['forecast'], {
            'fields': 'id,name,formType,sections[id,name,dataElements[id,categoryCombo[id]]],dataSetElements[dataElement[id]]'})
        expected = {s['id'] for s in json.loads((folder / 'dhis2_metadata.json').read_text())['sections']
                    if s['dataSet']['id'] == ids['datasets']['forecast']}
        sections = actual.get('sections', [])
        if {s['id'] for s in sections} != expected:
            raise RuntimeError('Forecast section readback differs from the redesigned layout.')
        members = {e['dataElement']['id'] for e in actual.get('dataSetElements', [])}
        if selected_ids - members:
            raise RuntimeError('Legacy selected-model dataset membership was not preserved.')
        report = {'status': 'verified', 'sections': len(sections), 'removed_duplicate_sections': len(removed),
                  'legacy_elements_preserved': len(selected_ids), 'backup': str(backup)}
        save_json(folder / 'redesign_report.json', report)
        return report


if __name__ == '__main__':
    from .dhis2 import make_client
    from .publish_saved import saved_context
    data, _, _, settings, run = saved_context()
    print(json.dumps(redesign(make_client(), data, settings,
                             Path(settings['output_folder']) / 'repairs' / run.run_id / 'forecast_structure'), indent=2))
