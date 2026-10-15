import "toastui-editor-all";
import "uk-ua";

// The library is local. Neither usage tracking nor automatic remote assets are enabled.
export default {
  template: `
    <section :class="{'show-preview': preview}" aria-label="Редактор Markdown">
      <div class="markdown-tools" role="toolbar" aria-label="Форматування Markdown">
        <button v-for="tool in tools" :key="tool.command" type="button"
          class="btn ghost" :title="tool.hint" :aria-label="tool.label"
          @mousedown.prevent @click="format(tool)">
          <span :class="'ico ico-' + tool.icon" aria-hidden="true"></span>
          <span class="markdown-tooltip" role="tooltip">{{ tool.hint }}</span>
        </button>
        <button type="button" class="btn ghost" :aria-pressed="preview"
          aria-label="Перегляд" title="Показати або приховати оформлений текст" @click="preview = !preview">
          <span class="ico ico-preview" aria-hidden="true"></span>
          <span class="markdown-tooltip" role="tooltip">Показати або приховати оформлений текст</span>
        </button>
      </div>
      <details class="markdown-help">
        <summary>Підказки Markdown</summary>
        <p>Виділіть текст і натисніть кнопку форматування. Ctrl+B — жирний,
          Ctrl+I — курсив, Ctrl+Z — скасувати зміну.</p>
        <dl>
          <dt>Заголовок</dt><dd><code>## Назва розділу</code></dd>
          <dt>Жирний / курсив</dt><dd><code>**текст**</code> / <code>*текст*</code></dd>
          <dt>Список</dt><dd><code>- Пункт списку</code></dd>
          <dt>Посилання</dt><dd><code>[Назва](адреса)</code></dd>
          <dt>Таблиця</dt><dd>Кнопка «Таблиця» вставляє готовий шаблон.</dd>
          <dt>Код</dt><dd>Кнопка «Код» додає блок для прикладу коду.</dd>
        </dl>
      </details>
      <div ref="host" class="markdown-host"></div>
      <div class="markdown-status" :class="{'error-message': length > maxLength}" aria-live="polite">
        {{ length.toLocaleString('uk-UA') }} / {{ maxLength.toLocaleString('uk-UA') }} символів
        <span v-if="length > maxLength"> — скоротіть текст перед збереженням</span>
      </div>
    </section>`,
  props: { value: String, maxLength: {type: Number, default: 20000} },
  data() {
    return {
      preview: false,
      length: Array.from(this.value || "").length,
      tools: [
        {label: "Заголовок", icon: "heading", command: "heading", args: {level: 2}, hint: "Заголовок розділу: ## Назва"},
        {label: "Жирний", icon: "bold", command: "bold", hint: "Жирний текст: **текст** · Ctrl+B"},
        {label: "Курсив", icon: "italic", command: "italic", hint: "Курсив: *текст* · Ctrl+I"},
        {label: "Список", icon: "list", command: "bulletList", hint: "Список: - Пункт"},
        {label: "Таблиця", icon: "table", command: "table", hint: "Вставити шаблон Markdown-таблиці"},
        {label: "Код", icon: "code", command: "codeBlock", hint: "Вставити блок коду"},
        {label: "Скасувати", icon: "undo", command: "undo", hint: "Скасувати останню зміну · Ctrl+Z"},
        {label: "Повторити", icon: "redo", command: "redo", hint: "Повторити скасовану зміну"},
      ],
    };
  },
  mounted() {
    this.editor = new toastui.Editor({
      el: this.$refs.host,
      initialValue: this.value || "",
      initialEditType: "markdown",
      previewStyle: "vertical",
      hideModeSwitch: true,
      toolbarItems: [],
      language: "uk-UA",
      height: "480px",
      usageStatistics: false,
      autofocus: false,
      events: {change: () => { this.length = Array.from(this.editor.getMarkdown()).length; }},
    });
    this.$refs.host.querySelector('.toastui-editor-md-container .ProseMirror')
      .setAttribute('aria-label', 'Текст інструкції Markdown');
    this.$refs.host.querySelector('.toastui-editor-md-container .ProseMirror')
      .setAttribute('role', 'textbox');
    this.$refs.host.querySelector('.toastui-editor-md-container .ProseMirror')
      .setAttribute('aria-multiline', 'true');
  },
  beforeUnmount() {
    this.editor?.destroy();
  },
  methods: {
    getMarkdown() { return this.editor.getMarkdown(); },
    format(tool) {
      this.editor.focus();
      if (tool.command === "table") {
        this.editor.insertText('\n| Поле | Опис |\n| --- | --- |\n| Назва | Пояснення |\n');
      } else {
        this.editor.exec(tool.command, tool.args);
      }
    },
  },
};
