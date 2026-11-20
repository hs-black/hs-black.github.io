/* Local resource adapter for SwiftLaTeX (AGPL-3.0; see NOTICE.md). */
'use strict';
importScripts('swiftlatexpdftex.js');

const files = new Set();
// kpathsea format numbers from the upstream engine's kpse_file_format_type.
const suffixes = { 3: '.tfm', 6: '.bib', 7: '.bst', 10: '.fmt', 11: '.map', 26: '.tex', 32: '.pfb', 44: '.enc' };
kpse_find_file_impl = function (namePointer, format) {
  const name = UTF8ToString(namePointer);
  const candidate = files.has(name) ? name : name + (suffixes[format] || '');
  if (!files.has(candidate)) return 0;
  return allocate(intArrayFromString('/tex/' + candidate), 'i8', ALLOC_NORMAL);
};
kpse_find_pk_impl = function () { return 0; };

// Run BibTeX only when the document requests a bibliography. The PDF for this
// pass is already complete; subsequent LaTeX passes read the generated bbl.
function compileDocument() {
  prepareExecutionContext();
  cwrap('setMainEntry', 'number', ['string'])('main.tex');
  const status = _compileLaTeX();
  if (status !== 0) throw new Error('pdfTeX status ' + status);
  const pdf = FS.readFile('/work/main.pdf');
  const aux = FS.analyzePath('/work/main.aux').exists
    ? FS.readFile('/work/main.aux', {encoding:'utf8'}) : '';
  if (aux.includes('\\bibdata{') && aux.includes('\\bibstyle{') && _compileBibtex() !== 0) {
    throw new Error('BibTeX 无法生成参考文献。');
  }
  self.postMessage({cmd:'compile', result:'ok', pdf:pdf.buffer, log:self.memlog}, [pdf.buffer]);
}

const upstreamMessage = self.onmessage;
self.onmessage = async function (event) {
  if (event.data.cmd === 'compilelatex') {
    try {
      compileDocument();
    } catch (error) {
      self.postMessage({cmd:'compile', result:'failed', log:self.memlog + '\n' + error});
    }
    return;
  }
  if (event.data.cmd !== 'load-local-library') {
    upstreamMessage(event);
    return;
  }
  try {
    const response = await fetch('texlive.bin');
    if (!response.ok) throw new Error('TeX 宏包下载失败（HTTP ' + response.status + '）');
    const stream = response.body.pipeThrough(new DecompressionStream('gzip'));
    const data = new Uint8Array(await new Response(stream).arrayBuffer());
    const headerLength = new DataView(data.buffer).getUint32(0, true);
    let offset = 4 + headerLength;
    const index = JSON.parse(new TextDecoder().decode(data.subarray(4, offset)));
    for (const [name, length] of index) {
      FS.writeFile('/tex/' + name, data.subarray(offset, offset + length));
      files.add(name);
      offset += length;
    }
    self.postMessage({cmd:'library', result:'ok'});
  } catch (error) {
    self.postMessage({cmd:'library', result:'failed', log:String(error)});
  }
};
