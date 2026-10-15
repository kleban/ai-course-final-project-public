"""Settings → knowledge base: rule documents and a local search check for RAG (stage 07)."""

from functools import partial

from nicegui import run, ui

from .contracts import Failure
from .markdown_editor import MarkdownEditor
from .model_screen import panel
from .screen import Shell, button, error_line, error_text, page_header, progress_line
from .service import ProjectService


def hit_card(hit: dict) -> None:
    with ui.element("div").classes("knowledge-hit"):
        ui.label(f"{hit['title']} · {hit['section']}").classes("knowledge-section")
        ui.label(hit["text"]).classes("model-explain")
        detail = f"Джерело: {hit['source']} · оцінка {hit['score']:.2f}"
        # Book fragments have no lexical score (history 138: None broke the search page).
        parts = [f"{label} {hit[key]:.2f}" for key, label in (("semantic", "зміст"), ("lexical", "слова"))
                 if hit.get(key) is not None]  # fmt: skip
        if parts and hit.get("semantic") is not None:
            detail += f" ({', '.join(parts)})"
        ui.label(detail).classes("model-note mono-wrap")


def knowledge_screen(service: ProjectService) -> None:
    ui.page_title("База знань · Data Story")
    Shell(service, "knowledge", trail=[("Налаштування", None), ("База знань", None)])
    with ui.column().classes("page"):
        page_header(
            "Налаштування",
            "doc",
            "База знань",
            "Правила, на які спирається AI-помічник. Пошук локальний, без інтернету: "
            "за змістом (нейронні embeddings) і за словами (TF-IDF).",
        )
        documents = service.knowledge_documents()
        if isinstance(documents, Failure):
            error_line(documents)
            return
        with ui.element("div").classes("model-grid model-grid-main"):
            search_box = panel("Перевірити пошук")
            docs_box = panel("Документи")
        with search_box:
            # Field and button in one row, the same height (author feedback 4, history 097).
            search_row = ui.element("div").classes("search-row")
            with search_row:
                query = ui.input(
                    "Запитання", placeholder="Наприклад: що означають зірочки біля коефіцієнта"
                ).props("outlined square")

            searching = {"on": False}

            async def search() -> None:
                if searching["on"]:
                    return  # one search at a time: repeated clicks do not queue more
                results.clear()
                if not (query.value or "").strip():
                    with results:
                        ui.label("Введіть запитання.").classes("error-message")
                    return
                searching["on"] = True
                find.props("disabled")
                with results:
                    progress_line(
                        "Шукаємо в правилах і книжках…",
                        "Перший пошук після запуску сервісу завантажує модель bge-m3 "
                        "(до ~30 с); далі пошук займає кілька секунд.",
                    )
                try:
                    response = await run.io_bound(
                        service.knowledge_search, {"query": query.value, "limit": 3}
                    )
                finally:
                    searching["on"] = False
                    find.props(remove="disabled")
                results.clear()
                with results:
                    if isinstance(response, Failure):
                        error_line(response)
                        return
                    data = response.data
                    if not data["hits"]:
                        ui.label(data["reason"]).classes("section-lead")
                    for hit in data["hits"]:
                        hit_card(hit)
                    if data.get("book_hits"):
                        ui.label("З книжок (переказ — у відповіді AI)").classes("knowledge-section")
                        for hit in data["book_hits"]:
                            hit_card(hit)
                    elif data.get("books_reason"):
                        ui.label(data["books_reason"]).classes("model-note")
                    ui.label(
                        f"Метод: {data['method']} · фрагментів: {data['fragment_count']}"
                    ).classes("model-note")
                    if data.get("semantic_reason"):
                        ui.label(data["semantic_reason"]).classes("model-warning")

            query.on("keydown.enter", search)
            with search_row:
                find = button("Знайти", search, "search", "primary")
            results = ui.column().classes("w-full")
        with docs_box:
            documents_list(service)
        library = service.knowledge.library.status()
        with panel("Книжки"):
            if library["reason"]:
                ui.label(library["reason"]).classes("model-note")
            ui.label(
                f"Обрані розділи: {len(library['books'])} книжок, {library['fragments']} фрагментів. "
                "Тексти лежать лише локально (runtime), не в Git."
            ).classes("model-note")
            for book in library["books"]:
                with ui.element("div").classes("knowledge-hit"):
                    ui.label(f"{book['title']} · {book['authors']}, {book['year']}").classes(
                        "knowledge-section"
                    )
                    ui.label(f"{book['why']} Фрагментів: {book['chunks']}.").classes("model-note")


TEMPLATE = """# Назва документа
Джерело: звідки правило (лекція, книга, власний досвід)
Пріоритет: 2

## Перший розділ
Текст правила. Кожен розділ «##» — окремий фрагмент для пошуку.
"""
FORMAT_HINT = (
    "Формат: перший рядок «# Назва», далі «Джерело: …» і «Пріоритет: 1–9» "
    "(1 — найважливіше), потім розділи «## Назва розділу». Кожен розділ — окремий фрагмент."
)


def documents_list(service: ProjectService) -> None:
    @ui.refreshable
    def listing() -> None:
        documents = service.knowledge_documents()
        if isinstance(documents, Failure):
            error_line(documents)
            return
        total = sum(d["fragments"] for d in documents.data)
        ui.label(f"Документів: {len(documents.data)} · фрагментів (розділів): {total}").classes(
            "model-note"
        )
        button("Новий документ", lambda: edit(None, TEMPLATE), "plus", "small")
        for document in documents.data:
            state = (
                "типовий" if document["is_default"]
                else "змінений" if document["builtin"]
                else "ваш"
            )  # fmt: skip
            with ui.expansion(
                f"{document['title']} · {document['fragments']} розд. · {state}"
            ).classes("w-full knowledge-doc"):
                ui.label(f"Джерело: {document['source']}").classes("model-note mono-wrap")
                with ui.row().classes("gap-1"):
                    button(
                        "Редагувати",
                        partial(edit, document["id"], document["text"]),
                        "edit",
                        "small",
                    )
                    if document["builtin"] and not document["is_default"]:
                        button(
                            "Повернути типовий",
                            partial(reset, document["id"]),
                            "undo",
                            "small",
                        )
                    if not document["builtin"]:
                        button("Видалити", partial(remove, document), "warning", "small")
                ui.markdown(document["text"].split("\n", 3)[-1]).classes("md-text")

    def notify(result, text: str) -> bool:
        if isinstance(result, Failure):
            ui.notify(error_text(result), type="negative")
            return False
        ui.notify(text, type="positive")
        listing.refresh()
        return True

    def reset(document_id: str) -> None:
        notify(service.knowledge_reset(document_id), "Повернуто типовий текст.")

    def remove(document: dict) -> None:
        with ui.dialog() as dialog, ui.card().classes("knowledge-dialog"):
            ui.label(f"Видалити документ «{document['title']}»?").classes("panel-title")
            ui.label("Попередній текст лишиться в журналі налаштувань.").classes("model-note")
            with ui.row().classes("gap-2"):

                def confirm() -> None:
                    if notify(service.knowledge_delete(document["id"]), "Документ видалено."):
                        dialog.close()

                button("Видалити", confirm, "warning", "primary")
                button("Скасувати", dialog.close, "undo")
        dialog.open()

    def edit(document_id: str | None, text: str) -> None:
        with (
            ui.dialog().props("maximized=false") as dialog,
            ui.card().classes("knowledge-dialog knowledge-editor"),
        ):
            ui.label("Новий документ" if document_id is None else "Редагування документа").classes(
                "panel-title"
            )
            ui.label(FORMAT_HINT).classes("model-note")
            editor = MarkdownEditor(text)
            message = ui.label("").classes("error-message").props('role="alert"')

            async def save() -> None:
                try:
                    value = await editor.markdown()
                except TimeoutError:
                    message.set_text("Редактор не відповів. Спробуйте ще раз.")
                    return
                result = (
                    service.knowledge_create({"text": value})
                    if document_id is None
                    else service.knowledge_save(document_id, {"text": value})
                )
                if isinstance(result, Failure):
                    message.set_text(error_text(result))
                    return
                notify(result, "Документ збережено; пошук уже використовує нову версію.")
                dialog.close()

            with ui.row().classes("gap-2"):
                button("Зберегти", save, "check", "primary")
                button("Скасувати", dialog.close, "undo")
        dialog.open()

    listing()
