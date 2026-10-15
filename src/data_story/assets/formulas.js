/* Local MathJax SVG, fixed application formulas only; no autoload/CDN. */
(() => {
  async function render() {
    for (const el of document.querySelectorAll('[data-formula]:not([data-rendered])')) {
      el.dataset.rendered = 'pending';
      try {
        if (!window.MathJax?.startup?.promise) throw new Error('MathJax unavailable');
        await MathJax.startup.promise;
        const node = await MathJax.tex2svgPromise(el.dataset.formula, {display: true});
        // Conversion alone does not perform the document's stylesheet update.
        MathJax.startup.document.addStyleSheet();
        el.replaceChildren(node);
        el.dataset.rendered = 'ok';
      } catch (_) {
        el.textContent = 'Не вдалося показати формулу. Пояснення методу доступне нижче.';
        el.dataset.rendered = 'error';
      }
    }
  }
  document.addEventListener('focusin', detail);
  document.addEventListener('click', detail);
  function detail(event) {
    const cell = event.target.closest?.('.heatmap-cell, .chart-item');
    const out = cell?.closest('.heatmap-wrap')?.querySelector('.heatmap-detail');
    if (out) out.textContent = cell.getAttribute('aria-label');
  }
  new MutationObserver(render).observe(document.body, {childList: true, subtree: true});
  render();
})();
