#!/usr/bin/env node
/** Publish an encrypted HTML fragment without placing its plaintext in the repo.
 * Requires Node.js >= 20; no installed npm packages.
 * Passwords are read from a hidden terminal prompt, never argv/environment.
 * --password-stdin accepts one UTF-8 line for automation; protect its producer.
 */
import { createCipheriv, createHash, pbkdf2Sync, randomBytes, timingSafeEqual } from 'node:crypto';
import { existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, realpathSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { StringDecoder } from 'node:string_decoder';

export const ROOT = realpathSync(resolve(dirname(fileURLToPath(import.meta.url)), '..'));
export const AAD = 'hs-black.github.io/encrypted-post/v1';
export const ITERATIONS = 600000;
const MAX_SOURCE_BYTES = 8 * 1024 * 1024;
const MAX_PAYLOAD_BYTES = 16 * 1024 * 1024;

const HELP = `Create an encrypted article (Node.js >= 20).

  node scripts/encrypt-post.mjs --source /private/article.html \\
    --slug private-note --title "公开标题" [--date YYYY-MM-DD] [--math] [--steps]

Required:
  --source FILE       Trusted HTML body fragment, stored outside this repository.
  --slug SLUG         Lowercase letters/digits separated by hyphens.
  --title TITLE       Public title; also appears on the locked page.

Options:
  --date YYYY-MM-DD   Public date; defaults to today's date in Asia/Shanghai.
  --math             Bundle local KaTeX, data-URL fonts, and scoped symbol hints.
  --steps            Bundle the shared ArticleSteps controls before your HTML.
  --replace          Replace an existing encrypted page; refuses ordinary posts.
  --output DIR       Preview/test destination outside this repository.
  --password-stdin   Read one password line from stdin instead of hidden prompts.
  --help             Show this help.

The password must contain at least 12 Unicode characters. Keep the source and
password outside Git. Use inline SVG or data-URL images; encrypted documents do
not load external images, scripts, styles, fonts, or attachments. Public title,
slug and date remain visible. Register the published page in the archive/RSS
using only public metadata and the neutral description from the template.
`;

function inside(path, parent) {
  const rel = relative(parent, path);
  return rel === '' || (!rel.startsWith('..' + sep) && rel !== '..' && !isAbsolute(rel));
}

// Resolve existing ancestors as well, so nonexistent children of symlinks
// cannot evade the source/output boundary checks.
function realFuturePath(path) {
  const absolute = resolve(path);
  if (existsSync(absolute)) return realpathSync(absolute);
  const parent = dirname(absolute);
  if (parent === absolute) throw new Error('无法解析输出路径。');
  return join(realFuturePath(parent), relative(parent, absolute));
}

function parseArgs(args) {
  const options = {};
  const flags = new Set(['math', 'steps', 'replace', 'password-stdin', 'help']);
  const values = new Set(['source', 'slug', 'title', 'date', 'output']);
  for (let i = 0; i < args.length; i++) {
    const key = args[i].startsWith('--') ? args[i].slice(2) : '';
    if ((!flags.has(key) && !values.has(key)) || key in options) throw new Error('未知或重复的参数。请使用 --help 查看用法；密码不能作为命令参数传入。');
    if (flags.has(key)) options[key] = true;
    else {
      if (i + 1 >= args.length || args[i + 1].startsWith('--')) throw new Error(`参数 --${key} 缺少值。`);
      options[key] = args[++i];
    }
  }
  return options;
}

function publicDate(value) {
  const date = value || new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error('日期需要使用 YYYY-MM-DD 格式。');
  const parsed = new Date(date + 'T00:00:00Z');
  if (!Number.isFinite(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== date) throw new Error('日期无效。');
  return date;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function inlineStyle(css) {
  return '<style>' + css.replace(/<\/style/gi, '<\\/style') + '</style>';
}

function inlineScript(js) {
  return '<script>' + js.replace(/<\/script/gi, '<\\/script') + '</script>';
}

function asset(name) {
  return readFileSync(join(ROOT, 'assets', name), 'utf8');
}

function versionedAsset(name) {
  const file = join(ROOT, 'assets', name);
  if (!existsSync(file)) return '/assets/' + name;
  return '/assets/' + name + '?v=' + createHash('sha256').update(readFileSync(file)).digest('hex').slice(0, 12);
}

function embeddedKatexCss() {
  const base = join(ROOT, 'assets/vendor/katex/0.16.11');
  return readFileSync(join(base, 'katex.min.css'), 'utf8').replace(/src:url\(fonts\/([^()]+\.woff2)\)[^}]+/g, (_, name) => {
    if (!/^[A-Za-z0-9_-]+\.woff2$/.test(name)) throw new Error('KaTeX 字体路径无效。');
    const data = readFileSync(join(base, 'fonts', name)).toString('base64');
    return 'src:url(data:font/woff2;base64,' + data + ') format("woff2")';
  });
}

function validateFragment(source) {
  if (!source.trim()) throw new Error('正文为空。');
  if (/<!doctype\b|<\/?(?:html|head|body)\b/i.test(source)) throw new Error('源文件应只包含正文 HTML 片段，不要包含 html、head 或 body 外壳。');
  if (/<(?:iframe|frame|object|embed|base|link|form)\b/i.test(source)) throw new Error('正文包含需要外部页面或资源的标签。请使用内联 HTML、SVG、样式与 data URL 图片。');
  if (/<script\b[^>]*\bsrc\s*=/i.test(source)) throw new Error('脚本需要内联在正文中；共享公式和图解功能可使用 --math、--steps。');
  const attributes = /\b(?:src|poster|srcset)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/gi;
  for (const match of source.matchAll(attributes)) {
    if (!/^data:/i.test(match[1] ?? match[2] ?? match[3])) throw new Error('图片等资源需要转换为 data URL 后放入正文，不能另行公开明文附件。');
  }
  if (/@import\b/i.test(source)) throw new Error('请把所需样式内联在正文中，不能使用 CSS @import。');
  for (const match of source.matchAll(/url\(\s*(['"]?)(.*?)\1\s*\)/gi)) {
    if (!/^(?:data:|#)/i.test(match[2])) throw new Error('CSS 资源需使用 data URL 或当前 SVG 的片段引用。');
  }
}

export function buildPrivateDocument(source, { math = false, steps = false } = {}) {
  validateFragment(source);
  let styles = inlineStyle(asset('site.css')) + inlineStyle(asset('article.css'));
  let before = '', after = '';
  if (math) {
    styles += inlineStyle(embeddedKatexCss()) + inlineStyle(asset('reader-math.css'));
    before += inlineScript(asset('vendor/katex/0.16.11/katex.min.js'));
    before += inlineScript(asset('vendor/katex/0.16.11/contrib/auto-render.min.js'));
    after += inlineScript(asset('reader-math.js'));
  }
  if (steps) before += inlineScript(asset('article-steps.js'));
  // Width comes from the outer public page; no second full-page margin.
  styles += inlineStyle('html,body{margin:0;padding:0}body{background:transparent}.prose{margin:0}img{height:auto}');
  return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">' + styles + before + '</head><body><article class="prose">' + source + '</article>' + after + '</body></html>';
}

export function encryptDocument(document, password) {
  if ([...password].length < 12) throw new Error('请使用至少 12 个字符的独立长密码。');
  const plaintext = Buffer.from(document, 'utf8');
  if (plaintext.length > MAX_PAYLOAD_BYTES - 16) throw new Error('加密文档超过 16 MiB，请减小内嵌资源。');
  const salt = randomBytes(16), iv = randomBytes(12);
  const key = pbkdf2Sync(password, salt, ITERATIONS, 32, 'sha256');
  try {
    const cipher = createCipheriv('aes-256-gcm', key, iv);
    cipher.setAAD(Buffer.from(AAD, 'utf8'));
    const ciphertext = Buffer.concat([cipher.update(plaintext), cipher.final(), cipher.getAuthTag()]);
    return { version: 1, cipher: 'AES-GCM', kdf: 'PBKDF2-SHA256', iterations: ITERATIONS,
      salt: salt.toString('base64'), iv: iv.toString('base64'), ciphertext: ciphertext.toString('base64') };
  } finally {
    key.fill(0);
    plaintext.fill(0);
  }
}

export function renderPublicPage({ title, date, envelope }) {
  const values = {
    TITLE: escapeHtml(title), DATE: escapeHtml(date), PAYLOAD: JSON.stringify(envelope),
    SITE_CSS: versionedAsset('site.css'), ENCRYPTED_CSS: versionedAsset('encrypted-post.css'),
    ENCRYPTED_JS: versionedAsset('encrypted-post.js'),
  };
  return readFileSync(join(ROOT, 'templates/encrypted-article.html'), 'utf8').replace(/\{\{([A-Z_]+)\}\}/g, (_, key) => {
    if (!(key in values)) throw new Error('加密文章模板含未知占位符。');
    return values[key];
  });
}

function readHidden(prompt) {
  if (!process.stdin.isTTY || !process.stdout.isTTY || !process.stdin.setRawMode) throw new Error('请在交互终端运行以隐藏输入密码；自动化可使用 --password-stdin。');
  return new Promise((resolvePromise, reject) => {
    let value = '';
    const decoder = new StringDecoder('utf8');
    const priorRaw = process.stdin.isRaw;
    const done = (error) => {
      process.stdin.off('data', onData);
      process.stdin.off('error', onError);
      process.stdin.setRawMode(priorRaw);
      process.stdin.pause();
      process.stdout.write('\n');
      if (error) reject(error); else resolvePromise(value);
    };
    const onError = () => done(new Error('密码输入失败。'));
    const onData = chunk => {
      for (const ch of decoder.write(chunk)) {
        if (ch === '\r' || ch === '\n') { done(); return; }
        if (ch === '\u0003' || ch === '\u0004') { done(new Error('已取消。')); return; }
        if (ch === '\u007f' || ch === '\b') value = [...value].slice(0,-1).join('');
        else if (ch >= ' ' && ch !== '\u001b') value += ch;
      }
    };
    process.stdout.write(prompt);
    process.stdin.setRawMode(true);
    process.stdin.on('data', onData);
    process.stdin.on('error', onError);
    process.stdin.resume();
  });
}

async function readPassword(fromStdin) {
  let password;
  if (fromStdin) {
    if (process.stdin.isTTY) throw new Error('--password-stdin 需要管道输入；直接运行可使用隐藏密码提示。');
    let data = Buffer.alloc(0);
    for await (const chunk of process.stdin) {
      data = Buffer.concat([data, chunk]);
      if (data.length > 16384) throw new Error('密码输入过长。');
    }
    password = new TextDecoder('utf-8', { fatal: true }).decode(data);
    data.fill(0);
    password = password.replace(/\r?\n$/, '');
    if (/[\r\n]/.test(password)) throw new Error('--password-stdin 只接收一行密码。');
  } else {
    password = await readHidden('设置阅读密码：');
    const confirmation = await readHidden('再次输入密码：');
    const a = Buffer.from(password, 'utf8'), b = Buffer.from(confirmation, 'utf8');
    const match = a.length === b.length && timingSafeEqual(a,b);
    a.fill(0); b.fill(0);
    if (!match) throw new Error('两次密码不一致。');
  }
  if ([...password].length < 12) throw new Error('请使用至少 12 个字符的独立长密码。');
  return password;
}

function checkDestination(directory, replace) {
  if (!existsSync(directory)) return;
  if (!statSync(directory).isDirectory()) throw new Error('输出位置必须是目录。');
  const files = readdirSync(directory);
  if (!files.length) return;
  if (files.length !== 1 || files[0] !== 'index.html') throw new Error('输出目录中已有其他文件。请使用专用的新目录，避免公开明文附件。');
  const index = join(directory, 'index.html');
  if (lstatSync(index).isSymbolicLink() || !statSync(index).isFile()) throw new Error('输出文件不能是符号链接或特殊文件。');
  const current = readFileSync(index, 'utf8');
  const match = current.match(/<script\b[^>]*\bid=["']encrypted-post-payload["'][^>]*>([\s\S]*?)<\/script>/i);
  let payload;
  try { payload = match && JSON.parse(match[1]); } catch {}
  if (!payload || payload.version !== 1 || payload.cipher !== 'AES-GCM' || payload.kdf !== 'PBKDF2-SHA256') throw new Error('不能覆盖普通博文。已公开的明文及其 Git 历史不会因重新加密而消失。');
  if (!replace) throw new Error('加密文章已存在；确认替换时使用 --replace。');
}

export async function main(args = process.argv.slice(2)) {
  if (Number(process.versions.node.split('.')[0]) < 20) throw new Error('需要 Node.js 20 或更高版本。');
  const options = parseArgs(args);
  if (options.help) { process.stdout.write(HELP); return; }
  if (!options.source || !options.slug || !options.title) throw new Error('需要 --source、--slug 和 --title；使用 --help 查看用法。');
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(options.slug) || options.slug.length > 80) throw new Error('slug 需要由小写字母、数字和单个连字符组成，最长 80 字符。');
  const title = options.title.trim();
  if (!title || [...title].length > 200 || /[\r\n\0]/.test(title)) throw new Error('公开标题不能为空，需为不超过 200 字符的单行文本。');
  const date = publicDate(options.date);
  const sourcePath = realpathSync(resolve(options.source));
  if (inside(sourcePath, ROOT)) throw new Error('明文源文件必须放在博客仓库之外；gitignore 不能撤回已经提交的内容。');
  if (!statSync(sourcePath).isFile() || statSync(sourcePath).size > MAX_SOURCE_BYTES) throw new Error('源文件需要是小于 8 MiB 的普通 UTF-8 HTML 文件。');
  const directory = realFuturePath(options.output || join(ROOT, 'posts', options.slug));
  if (options.output && inside(directory, ROOT)) throw new Error('--output 用于仓库外预览。正式发布请省略该参数。');
  if (!options.output && directory !== join(ROOT, 'posts', options.slug)) throw new Error('发布目录必须是仓库 posts 下的指定目录，不能经过符号链接。');
  if (inside(sourcePath, directory)) throw new Error('明文源文件不能放在输出目录里。');
  checkDestination(directory, options.replace);
  const sourceBytes = readFileSync(sourcePath);
  const source = new TextDecoder('utf-8', { fatal: true }).decode(sourceBytes);
  sourceBytes.fill(0);
  const document = buildPrivateDocument(source, options);
  const password = await readPassword(options['password-stdin']);
  const envelope = encryptDocument(document, password);
  const page = renderPublicPage({ title, date, envelope });
  // Recheck before writing, then atomically replace with ciphertext only.
  checkDestination(directory, options.replace);
  mkdirSync(directory, { recursive: true });
  const temporary = join(directory, '.encrypted-' + randomBytes(8).toString('hex') + '.tmp');
  try {
    writeFileSync(temporary, page, { flag: 'wx', mode: 0o644 });
    renameSync(temporary, join(directory, 'index.html'));
  } finally {
    rmSync(temporary, { force: true });
  }
  process.stdout.write('已生成加密文章：' + join(directory, 'index.html') + '\n');
  process.stdout.write('列表和 RSS 仅填写公开标题、日期及中性摘要。密码与正文没有写入发布文件。\n');
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  main().catch(error => {
    // Deliberately avoid stack traces or logging source/password values.
    process.stderr.write('加密失败：' + error.message + '\n');
    process.exitCode = 1;
  });
}
