"""Local TOAST UI component; Markdown remains the persisted instruction format."""

from nicegui import ui


class MarkdownEditor(  # type: ignore[call-arg]  # NiceGUI runtime component registration
    ui.element,
    component="assets/markdown_editor.js",
    dependencies=[
        "assets/vendor/toastui/toastui-editor-all.min.js",
        "assets/vendor/toastui/uk-ua.js",
    ],
):
    def __init__(self, value: str, max_length: int = 20000) -> None:
        super().__init__()
        self._props["value"] = value
        self._props["maxLength"] = max_length
        self.classes("markdown-editor w-full")
        ui.add_head_html('<link rel="stylesheet" href="/assets/vendor/toastui/toastui-editor.css">')

    async def markdown(self) -> str:
        """Read from the browser at save time, including the last keystroke."""
        return await self.run_method("getMarkdown", timeout=10)
