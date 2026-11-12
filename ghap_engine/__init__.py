"""GHAP climate-health modelling engine (self-contained, DHIS2-integrated).

Layers follow the GHAP architecture diagram:
  data source  -> dhis2.py (retrieval), inputs.py (mapping workbook)
  processing   -> panel.py (validation, alignment, lag windows, analysis panel)
  analytics    -> bases.py, model.py (exposure-response, lag-response, attribution),
                  forecast.py (forecast and early warning), registry.py (versioning)
  service      -> tables.py (pre-computed metric tables), publish.py (DHIS2 Web API)
"""

ENGINE_VERSION = "2.0.0"
