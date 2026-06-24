// Global MathJax configuration for the tipopac docs.
//
// `pymdownx.arithmatex` (generic = true) wraps math in `\(...\)` (inline)
// and `\[...\]` (display) spans with class `arithmatex`; MathJax is told to
// typeset only those. The `document$` hook re-typesets after Zensical's
// instant navigation swaps page content without a full reload.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
