// Compile one isolated project in a fresh pdfTeX worker. Resources stay in the
// browser's normal static HTTP cache, while all document files are transient.
export async function compilePDF(source, {signal, onProgress, files = []} = {}) {
  signal?.throwIfAborted();
  const inputs = await Promise.all(Array.from(files, async file => {
    const name = file.name;
    if (!name || /[/\\\u0000-\u001f]/.test(name) || name === '.' || name === '..') {
      throw new Error('附件名称需要是单个文件名。');
    }
    if (name === 'main.tex') throw new Error('main.tex 已用于当前编辑的主文档。');
    return {name, bytes: new Uint8Array(await file.arrayBuffer())};
  }));
  signal?.throwIfAborted();
  const worker = new Worker(new URL('./vendor/pdftex/worker.js', import.meta.url));
  let onAbort, timer;
  try {
    return await new Promise((resolve, reject) => {
      const fail = (message, log = '') => reject(Object.assign(new Error(message), {log}));
      onAbort = () => reject(new DOMException('编译已取消', 'AbortError'));
      signal?.addEventListener('abort', onAbort, {once:true});
      timer = setTimeout(() => fail('编译超时，请检查宏命令并重试。'), 90000);
      worker.onerror = event => fail('排版引擎运行失败。', event.message);
      let pass = 0;
      worker.onmessage = ({data}) => {
        if (data.result === 'failed') {
          fail(data.cmd === 'library' ? '无法加载本地排版资源。' : 'LaTeX 编译失败。', data.log);
          return;
        }
        if (data.cmd === undefined && data.result === 'ok') {
          onProgress?.('正在加载排版资源…');
          worker.postMessage({cmd:'load-local-library'});
        } else if (data.cmd === 'library') {
          inputs.forEach(({name,bytes}) => worker.postMessage({cmd:'writefile', url:name, src:bytes}, [bytes.buffer]));
          worker.postMessage({cmd:'writefile', url:'main.tex', src:source});
          worker.postMessage({cmd:'setmainfile', url:'main.tex'});
          pass = 1;
          onProgress?.('正在排版 PDF…');
          worker.postMessage({cmd:'compilelatex'});
        } else if (data.cmd === 'compile') {
          if (!(data.pdf instanceof ArrayBuffer) || data.pdf.byteLength < 8 ||
              new TextDecoder().decode(new Uint8Array(data.pdf, 0, 5)) !== '%PDF-') {
            fail('排版引擎未生成有效的 PDF。', data.log);
            return;
          }
          // Cross references and the table of contents depend on the previous
          // pass's auxiliary files. BibTeX may require a third pass for labels.
          if (pass < 3 && (pass === 1 || /Rerun to get|rerun LaTeX|undefined references/i.test(data.log))) {
            pass++;
            onProgress?.('正在更新目录与交叉引用…');
            worker.postMessage({cmd:'compilelatex'});
          } else {
            resolve({blob:new Blob([data.pdf], {type:'application/pdf'}), log:data.log || ''});
          }
        }
      };
    });
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
    worker.terminate();
  }
}
