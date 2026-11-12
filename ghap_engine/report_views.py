"""Climate-separated alternatives to the incompatible built-in dataset report."""
from __future__ import annotations

import json
from pathlib import Path

from .publish import make_uid
from .publish_lifecycle import check_import, save_json


def create_attribution_reports(client, folder):
    """Create/update only GHAP-owned report views; never change model data or skipTotal."""
    dataset = client.get('dataSets/JCo5G20MVHs', {
        'fields': 'id,name,sections[id,name,dataElements[id,aggregationType]],sharing'})
    category = client.get('categories/IsNLje1v92K', {
        'fields': 'id,name,categoryOptions[id]'})
    roots = client.get('organisationUnits', {
        'filter': 'level:eq:1', 'fields': 'id', 'paging': 'false'})['organisationUnits']
    if len(roots) != 1 or not dataset.get('sections') or not category.get('categoryOptions'):
        raise RuntimeError('Expected one national unit, outcome sections and climate options.')
    views = []
    for section in dataset['sections']:
        code = 'GHAP2_REPORT_CLIMATE_' + section['id']
        found = client.get('visualizations', {
            'filter': 'code:eq:' + code, 'fields': ':owner', 'paging': 'false'}).get('visualizations', [])
        if len(found) > 1:
            raise RuntimeError('Ambiguous report code: ' + code)
        old = found[0] if found else {}
        view = {**old, 'id': old.get('id', make_uid(code)), 'code': code,
                'name': ('GHAP Attribution by climate | ' + section['name'])[:230],
                'description': 'Monthly additive case counts only. Each climate is a separate model; do not sum across climates. '
                               'RR, fractions and interval bounds use NONE and cannot be queried in aggregate reports.',
                'type': 'PIVOT_TABLE',
                'columnDimensions': ['pe'],
                'rowDimensions': ['dx', category['id']],
                'filterDimensions': ['ou'],
                'dataDimensionItems': [{'dataDimensionItemType': 'DATA_ELEMENT', 'dataElement': {'id': de['id']}}
                                       for de in section['dataElements'] if de.get('aggregationType') == 'SUM'],
                'categoryDimensions': [{'category': {'id': category['id']},
                                        'categoryOptions': category['categoryOptions']}],
                'organisationUnits': roots,
                'periods': [], 'relativePeriods': {'last12Months': True},
                'rowTotals': False, 'colTotals': False,
                'rowSubTotals': False, 'colSubTotals': False,
                'hideEmptyRows': True, 'hideEmptyColumns': True,
                'hideEmptyRowItems': 'ALL'}
        # New reports are private; preserve sharing on an existing managed report.
        if not old:
            view['sharing'] = {'public': '--------', 'users': {}, 'userGroups': {}}
        views.append(view)
    folder = Path(folder)
    payload = {'visualizations': views}
    save_json(folder / 'attribution_report_views.json', payload)
    # Replace this report's selections so obsolete nonaggregatable items are removed.
    # Existing report owner fields/sharing were included above.
    params = {'importStrategy': 'CREATE_AND_UPDATE', 'atomicMode': 'ALL', 'mergeMode': 'REPLACE'}
    check_import(client.post('metadata', payload, {**params, 'importMode': 'VALIDATE'}))
    check_import(client.post('metadata', payload, {**params, 'importMode': 'COMMIT'}))
    reports = []
    for view in views:
        actual = client.get('visualizations/' + view['id'], {
            'fields': 'id,name,rows,rowTotals,colTotals,categoryDimensions[category[id]]'})
        if (category['id'] not in {row['id'] for row in actual.get('rows', [])}
                or actual.get('rowTotals') or actual.get('colTotals')):
            raise RuntimeError('Climate-separated report readback failed: ' + view['id'])
        reports.append({'name': actual['name'], 'id': actual['id'],
                        'url': client.base + '/dhis-web-data-visualizer/#/' + actual['id']})
    save_json(folder / 'attribution_report_links.json', reports)
    return reports


if __name__ == '__main__':
    from .dhis2 import make_client
    print(json.dumps(create_attribution_reports(make_client(), 'ghap_outputs/report_views'), indent=2))
