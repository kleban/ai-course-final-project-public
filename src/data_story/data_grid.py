"""Read-only interactive table of a dataset: AG Grid Community bundled with NiceGUI (MIT).

Pages are loaded on demand (infinite row model + pagination): the browser asks the local API
for the visible block only, and the server filters, sorts and slices the parsed file. Numbers
sort as numbers, recognised dates as dates (ISO text shown as ДД.ММ.РРРР). Nothing is edited.
"""

import json
from collections.abc import Callable
from typing import Any

from nicegui import ui

PAGE_SIZE = 50
BLOCK_SIZE = 100
MISSING = "—"
TEXT = f"p => p.value == null || p.value === '' ? '{MISSING}' : p.value"
# Decimal comma as in the rest of the interface; no thousands separator (years, postcodes).
NUMBER = f"p => p.value == null ? '{MISSING}' : String(p.value).replace('.', ',')"
DATE = (
    f"p => p.value == null ? '{MISSING}' : p.value.slice(8, 10) + '.' + p.value.slice(5, 7)"
    " + '.' + p.value.slice(0, 4) + (p.value.length > 10 ? ' ' + p.value.slice(11, 16) : '')"
)
FILTERS = {"number": "agNumberColumnFilter", "date": "agDateColumnFilter"}

LOCALE = {
    "noRowsToShow": "Немає рядків",
    "loadingOoo": "Завантаження…",
    "filterOoo": "Значення…",
    "dateFormatOoo": "дд.мм.рррр",
    "equals": "Дорівнює",
    "notEqual": "Не дорівнює",
    "contains": "Містить",
    "notContains": "Не містить",
    "startsWith": "Починається з",
    "endsWith": "Закінчується на",
    "blank": "Порожнє",
    "notBlank": "Не порожнє",
    "lessThan": "Менше",
    "greaterThan": "Більше",
    "lessThanOrEqual": "Менше або дорівнює",
    "greaterThanOrEqual": "Більше або дорівнює",
    "inRange": "Між",
    "inRangeStart": "Від",
    "inRangeEnd": "До",
    "before": "До",
    "after": "Після",
    "andCondition": "І",
    "orCondition": "АБО",
    "applyFilter": "Застосувати",
    "resetFilter": "Скинути",
    "clearFilter": "Очистити",
    "cancelFilter": "Скасувати",
    "columnFilter": "Фільтр стовпця",
    "sortAscending": "За зростанням",
    "sortDescending": "За спаданням",
    "sortUnSort": "Без сортування",
    "pinColumn": "Закріпити стовпець",
    "pinLeft": "Ліворуч",
    "pinRight": "Праворуч",
    "noPin": "Не закріплювати",
    "autosizeThisColumn": "Підігнати ширину",
    "autosizeAllColumns": "Підігнати всі стовпці",
    "resetColumns": "Скинути стовпці",
    "page": "Сторінка",
    "of": "з",
    "to": "–",
    "more": "більше",
    "firstPage": "Перша сторінка",
    "previousPage": "Попередня сторінка",
    "nextPage": "Наступна сторінка",
    "lastPage": "Остання сторінка",
    "pageSizeSelectorLabel": "Рядків на сторінці:",
    "ariaPageSizeSelectorLabel": "Рядків на сторінці",
    "thousandSeparator": "\u202f",
    "decimalSeparator": ",",
    "ariaFilterInput": "Значення фільтра",
    "ariaSearch": "Пошук",
}


def column_def(index: int, name: str, kind: str) -> dict:
    column: dict = {
        "headerName": name,
        "field": f"c{index}",
        "filter": FILTERS.get(kind, "agTextColumnFilter"),
        ":valueFormatter": {"date": DATE, "number": NUMBER}.get(kind, TEXT),
    }
    if kind == "number":
        column["type"] = "rightAligned"
        column["cellClass"] = "num"
    if kind == "date":
        column["filterParams"] = {"inRangeInclusive": True, "maxNumConditions": 2}
    return column


def datasource(dataset_id: str) -> str:
    """JS datasource: POST the visible block request to the local API (no external hosts)."""
    url = json.dumps(f"/api/datasets/{dataset_id}/rows")
    return (
        "{ getRows: (p) => fetch(" + url + ", { method: 'POST',"
        " headers: { 'Content-Type': 'application/json' },"
        " body: JSON.stringify({ start: p.startRow, limit: p.endRow - p.startRow,"
        " sort: p.sortModel, filter: p.filterModel }) })"
        ".then((r) => r.json())"
        ".then((j) => j.data ? p.successCallback(j.data.rows, j.data.total) : p.failCallback())"
        ".catch(() => p.failCallback()) }"
    )


def grid_options(dataset_id: str, columns: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "columnDefs": [column_def(i, name, kind) for i, (name, kind) in enumerate(columns)],
        "defaultColDef": {
            "sortable": True,
            "resizable": True,
            "floatingFilter": True,
            "minWidth": 90,
            "filterParams": {"inRangeInclusive": True, "maxNumConditions": 2},
        },
        "autoSizeStrategy": {"type": "fitCellContents"},
        "rowModelType": "infinite",
        ":datasource": datasource(dataset_id),
        "cacheBlockSize": BLOCK_SIZE,
        "maxBlocksInCache": 10,
        "pagination": True,
        "paginationPageSize": PAGE_SIZE,
        "paginationPageSizeSelector": [25, 50, 100],
        "localeText": LOCALE,
        "enableCellTextSelection": True,
        "ensureDomOrder": True,
        "animateRows": False,
        "suppressMovableColumns": True,
    }


def data_grid(
    dataset_id: str,
    columns: list[tuple[str, str]],
    on_count: Callable[[int], None] | None = None,
) -> ui.aggrid:
    grid = ui.aggrid(grid_options(dataset_id, columns), auto_size_columns=False).classes(
        "data-grid"
    )

    async def counted() -> None:
        if on_count is not None:
            on_count(int(await grid.run_grid_method("paginationGetRowCount")))

    grid.on("paginationChanged", counted)
    return grid
