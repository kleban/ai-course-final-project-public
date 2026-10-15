"""Line icons as CSS masks (`.ico.ico-<name>`), no network and no icon fonts.

Style and most drawings are taken from the author's dashboard «Keeping the Lights On»
(E:\\lun_ua_project\\...\\dashboard, u06/viz/icons.py): 24x24 SVG, stroke 1.9, round caps.
Icons take the text colour (currentColor), so they work in any theme.
`assets/icons.css` is generated from this module; `tests/test_icons.py` keeps them in sync.
"""

from pathlib import Path
from urllib.parse import quote

_C = '<circle cx="{}" cy="{}" r="{}"/>'
_F = ' fill="black"'

ICONS = {
    # from the dashboard
    "overview": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>'
    '<rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    "story": _C.format(12, 12, 9) + f'<path d="M10 8.5l5 3.5-5 3.5z"{_F}/>',
    "data": '<ellipse cx="12" cy="5.5" rx="8" ry="2.8"/><path d="M4 5.5v13c0 1.5 3.6 2.8 8 2.8s8-1.3 8-2.8v-13"/>'
    '<path d="M4 12c0 1.5 3.6 2.8 8 2.8s8-1.3 8-2.8"/>',
    "when": _C.format(12, 12, 9) + '<path d="M12 7v5l3.5 2.5"/>',
    "lab": '<path d="M9 3h6M10 3v6.5L4.6 18.6A1.6 1.6 0 0 0 6 21h12a1.6 1.6 0 0 0 1.4-2.4L14 9.5V3M7.2 15h9.6"/>',
    "sources": '<path d="M4 4.5h5.5A2.5 2.5 0 0 1 12 7v13a2 2 0 0 0-2-2H4zM20 4.5h-5.5A2.5 2.5 0 0 0 12 7v13a2 2 0 0 1 2-2h6z"/>',
    "methods": '<path d="M18 4.5H6.5l6 7.5-6 7.5H18"/>',
    "info": _C.format(12, 12, 9) + '<path d="M12 11v6M12 7.6v.2"/>',
    "formula": '<path d="M14.5 4.5c-2 0-2.6 1.4-3 3.5l-2 9c-.4 2-1 3.5-3 3.5M8 10h7M15 13l5 6M20 13l-5 6"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8M15 7h6v6"/>',
    "compare": '<rect x="3" y="10" width="4" height="10" rx="1"/><rect x="10" y="4" width="4" height="16" rx="1"/>'
    '<rect x="17" y="13" width="4" height="7" rx="1"/>',
    "shield": '<path d="M12 3l8 3v6c0 4.5-3.4 8-8 9-4.6-1-8-4.5-8-9V6z"/><path d="M8.5 12l2.5 2.5 4.5-4.5"/>',
    "idea": '<path d="M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0 0 12 3z"/>'
    '<path d="M9.5 18.5h5M10.5 21h3M12 7.5v3"/>',
    "warning": '<path d="M12 3.5L2.5 20h19z"/><path d="M12 10v4.5M12 17.2v.2"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "search": _C.format(10.5, 10.5, 6.5) + '<path d="M15.5 15.5L21 21"/>',
    "next": '<path d="M9 5l7 7-7 7"/>',
    "prev": '<path d="M15 5l-7 7 7 7"/>',
    "upload": '<path d="M12 16V5M7 9.5l5-5 5 5M4 20h16"/>',
    # types of analysis (Data Story)
    "regression": '<path d="M4 3.5V20h16.5"/><path d="M6.5 17.5L19.5 6"/>'
    + "".join(
        f'<circle cx="{x}" cy="{y}" r=".55"{_F}/>'
        for x, y in ((8.5, 12.5), (11.5, 16), (14.5, 9), (17.5, 12.5))
    ),
    "classification": '<path d="M3.5 20.5L20.5 3.5" stroke-dasharray="2.2 2.6"/>'
    + "".join(
        f'<circle cx="{x}" cy="{y}" r=".6"{_F}/>' for x, y in ((5, 5.5), (10, 5), (5.5, 10.5))
    )
    + "".join(
        f'<rect x="{x}" y="{y}" width="1.8" height="1.8"/>'
        for x, y in ((12.5, 17.5), (17.5, 12), (17.5, 17.5))
    ),
    "clustering": "".join(
        f'<circle cx="{x}" cy="{y}" r=".7"{_F}/>'
        for x, y in (
            (4.5, 6),
            (8.5, 4.5),
            (7, 9),
            (16, 5),
            (19.5, 7.5),
            (16.5, 9.5),
            (9.5, 16.5),
            (13.5, 15.5),
            (12, 20),
        )
    ),
    # actions and states
    "settings": '<path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1"/>'
    + _C.format(15, 6, 2)
    + _C.format(9, 12, 2)
    + _C.format(17, 18, 2),
    "edit": '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.5 6.5l4 4"/>',
    "reset": '<path d="M4 4v5h5"/><path d="M4.6 9A8 8 0 1 1 6 17.5"/>',
    "chevron": '<path d="M6 9l6 6 6-6"/>',
    "doc": '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 16h6"/>',
    "gallery": '<rect x="3" y="4" width="18" height="16" rx="2"/>'
    + _C.format(8.5, 9.5, 1.6)
    + '<path d="M21 15.5l-5-5L6 20"/>',
    # Data Story additions in the same style
    "folder": '<path d="M3 6.5A1.5 1.5 0 0 1 4.5 5H9l2 2.5h8.5A1.5 1.5 0 0 1 21 9v9.5a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 18.5z"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "copy": '<rect x="8" y="8" width="12" height="12" rx="1"/>'
    + '<path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
    "journal": '<path d="M8 6h12M8 12h12M8 18h12"/>'
    + _C.format(4, 6, 0.6)
    + _C.format(4, 12, 0.6)
    + _C.format(4, 18, 0.6),
    "spark": f'<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"{_F}/>'
    '<path d="M19 15.5v4M17 17.5h4"/>',
    "chat": '<path d="M4 5.5h16v10H10l-4.5 3.5v-3.5H4z"/><path d="M8 9.5h8M8 12.5h5"/>',
    "menu": '<path d="M4 7h16M4 12h16M4 17h16"/>',
    "lock": '<rect x="5" y="10.5" width="14" height="10" rx="1"/><path d="M8 10.5V7.5a4 4 0 0 1 8 0v3"/>',
    "heading": '<path d="M5 5v14M15 5v14M5 12h10M19 15v4M18 15h1"/>',
    "bold": '<path d="M7 4h6a4 4 0 0 1 0 8H7V4zM7 12h7a4 4 0 0 1 0 8H7v-8z"/>',
    "italic": '<path d="M10 4h9M5 20h9M15 4L9 20"/>',
    "list": '<path d="M9 6h11M9 12h11M9 18h11M4 6h.1M4 12h.1M4 18h.1"/>',
    "table": '<rect x="3" y="4" width="18" height="16"/><path d="M3 9h18M3 14h18M9 4v16M15 4v16"/>',
    "code": '<path d="M7 7l-5 5 5 5M17 7l5 5-5 5M14 4l-4 16"/>',
    "undo": '<path d="M8 4L3 9l5 5M3 9h10a7 7 0 0 1 0 14"/>',
    "redo": '<path d="M16 4l5 5-5 5M21 9H11a7 7 0 0 0 0 14"/>',
    "preview": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/>'
    + _C.format(12, 12, 3),
    # chart themes menu item (history 096): an artist's palette with three paint wells
    "palette": '<path d="M12 3a9 9 0 0 0 0 18c1.3 0 2-.8 2-1.8 0-1.4-1.2-1.6-1.2-2.8 '
    '0-1 .8-1.6 1.8-1.6H17a4 4 0 0 0 4-4c0-4.3-4-7.8-9-7.8z"/>'
    + "".join(
        f'<circle cx="{x}" cy="{y}" r="1.1"{_F}/>'
        for x, y in ((7.5, 11.5), (9.5, 7.5), (14.5, 7.5))
    ),
}


# Review candidates only: not registered in ICONS or used by the application yet.
REGRESSION_PROPOSALS = {
    "1": '<path d="M4 4v16h16M7 17L19 6"/>'
    '<circle cx="8" cy="10" r="1"/><circle cx="13" cy="16" r="1"/><circle cx="18" cy="3" r="1"/>',
    "2": '<path d="M4 19L20 5"/>'
    '<circle cx="5" cy="14" r=".8"/><circle cx="9" cy="17" r=".8"/>'
    '<circle cx="11" cy="10" r=".8"/><circle cx="15" cy="12" r=".8"/>'
    '<circle cx="18" cy="5" r=".8"/><circle cx="20" cy="9" r=".8"/>',
    "3": '<path d="M4 4v16h16M7 17c3-8 6-9 12-10"/>'
    '<circle cx="7" cy="13" r=".8"/><circle cx="12" cy="9" r=".8"/><circle cx="18" cy="5" r=".8"/>',
    "4": '<path d="M4 4v16h16M7 17L19 5"/>'
    '<path d="M9 10v5M15 14V9" stroke-dasharray="1 2"/>'
    '<circle cx="9" cy="9" r="1"/><circle cx="15" cy="15" r="1"/>',
}


# Groups for the gallery page (/icons); every icon belongs to exactly one group.
GROUPS = {
    "Редактор Markdown": [
        "heading",
        "bold",
        "italic",
        "list",
        "table",
        "code",
        "undo",
        "redo",
        "preview",
    ],
    "Типи аналізу": ["regression", "classification", "clustering"],
    "Розділи й навігація": [
        "overview",
        "folder",
        "data",
        "story",
        "settings",
        "gallery",
        "sources",
        "journal",
        "doc",
        "menu",
        "search",
        "prev",
        "next",
        "chevron",
    ],
    "Аналіз і AI": [
        "compare",
        "trend",
        "formula",
        "palette",
        "methods",
        "lab",
        "spark",
        "chat",
        "idea",
    ],
    "Дії": ["plus", "upload", "copy", "edit", "reset", "check"],
    "Стани й підказки": ["info", "warning", "shield", "lock", "when"],
}


def svg(name: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
        'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
        f"{ICONS[name]}</svg>"
    )


def css() -> str:
    head = (
        "/* Generated from src/data_story/icons.py - do not edit by hand. */\n"
        ".ico{display:inline-block;width:1.15em;height:1.15em;flex:0 0 auto;"
        "background-color:currentColor;-webkit-mask:var(--ico) center/contain no-repeat;"
        "mask:var(--ico) center/contain no-repeat;vertical-align:-0.2em}\n"
    )
    rules = [f'.ico.ico-{k}{{--ico:url("data:image/svg+xml,{quote(svg(k))}")}}\n' for k in ICONS]
    return head + "".join(rules)


CSS_PATH = Path(__file__).parent / "assets" / "icons.css"


if __name__ == "__main__":
    CSS_PATH.write_text(css(), encoding="utf-8")
