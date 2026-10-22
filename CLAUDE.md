# hs-black.github.io

Personal site and blog of Zunzhe Zhang (Tsinghua IIIS; embodied AI, VLA
models, LLMs), served by GitHub Pages straight from this repo.
There is **no generator and no build step**: every page is hand-written HTML,
and whatever is in the publishing branch is what goes live.
The owner asks Claude to write the posts; this file is how to do that.

## Layout

```
index.html            home: profile card + "Now" (sidebar), intro, experience timeline,
                      selected papers, writing (series cards + newest 3 posts), awards
papers/index.html     full publication list (abstract + BibTeX toggles)
projects/index.html   projects and tools
blog/index.html       archive: timeline / category / series / tag views + search,
                      rendered from assets/posts.js
posts/<slug>/index.html   one folder per post
tools/<name>/index.html   standalone interactive tools (linked from projects)
feed.xml              RSS; hand-maintained, one <item> per post
404.html
.nojekyll             stops GitHub Pages from running Jekyll
assets/site.css       shared styles + color tokens (light/dark)
assets/article.css    shared inline figure and exercise styles
assets/article-steps.js manual, rewindable figure controls
assets/posts.js       list of all posts (category, tags, desc) + tags/related box on posts
assets/post-links.js  generated cross-reference index (scripts/post-links.py)
assets/series.js      series list + nav for multi-part posts
assets/curriculum.js  ten-stage LLM engineering map + per-post route links
assets/curriculum.css shared learning-map styles
assets/anim.js        timeline animation engine (smooth, scrubbable, step-able)
assets/img/           avatar and other shared images
assets/favicon.svg
files/resume.pdf      linked from the home page
templates/article.html    starting point for an article / interactive post
```

All internal links and assets use root-absolute paths (`/assets/site.css`,
`/posts/<slug>/`). Post-specific images or data go next to the post in its
folder and are referenced relatively.

## Adding a post

1. Pick a short lowercase slug (`kv-cache`, `rlhf-notes`) and copy
   `templates/article.html` to `posts/<slug>/index.html`.
2. Fill in `<title>` (`标题 · Zunzhe Zhang`), `<meta name="description">`, the
   `<h1>`, the kind label and the date.
3. Add an entry at the **top** of `POSTS` in `assets/posts.js` (slug, title,
   date, kind, cat, tags, desc). The blog page renders from it; reuse
   existing categories and tags where possible. Load
   `/assets/post-links.js` and `/assets/posts.js` before `</body>` (the
   templates already do); post pages then show their category and tags under
   the header, a 相关文章 box at the end, and preview cards on links to
   other posts.
4. Update the post list under "Writing" on `index.html`: it shows the newest
   **3** posts, so add the new one on top and drop the fourth. If the post
   belongs to a series, also add it to `assets/series.js` (and bump the count
   on the series card under "Writing").
5. Add an `<item>` at the **top** of `feed.xml` (RFC 822 `pubDate`).
6. Link related posts to each other in the text where a concept is
   explained elsewhere, then run `python3 scripts/post-links.py` to refresh
   the cross-reference index, the blog's content-versioned data URLs, and
   matching shared script/style URLs.
7. Preview locally (below), then commit and push.

The header nav (Home / Papers / Projects / Blog) is copied into every page;
mark the current section with `aria-current="page"`. If the nav changes,
change it in every page and the article template.

## Adding a paper

Add it at the top of `papers/index.html` (authors with the owner in
`<span class="me">`, venue, one-line summary, links, abstract, BibTeX), and
add the short form to "Selected Papers" on `index.html` if it should be
featured. Only use metadata the owner gives or that you have verified (arXiv
page etc.); never guess venues, author order or years.

Kinds, used in both the post header and the home list:

| kind | label | class |
| --- | --- | --- |
| plain article | 文章 | `kind` |
| article with interactive demos | 交互 | `kind interactive` |

## Writing

- Engineering lessons should read as continuous articles, with the question,
  reasoning, derivation, worked numbers and engineering consequence connected
  in prose. Use inline interactive figures for the steps that benefit from
  motion. All topics use the article format. Develop the explanation in
  continuous paragraphs, with figures at the point where they clarify a step.
- Explain variables, tensor axes and units beside each important formula.
  Repeat the needed definitions after a substantial gap or context change.
  A symbol tooltip supplements this text; it does not replace it. Avoid using
  one letter for unrelated quantities in the same derivation.
- Math uses the local pinned KaTeX runtime in
  `/assets/vendor/katex/0.16.11/` and `reader-math.js` / `reader-math.css`.
  Explicit `data-symbol` annotations and the page's `#math-definitions` JSON
  provide hover, focus and tap explanations, with `data-math-scope` for local
  meanings. Never infer a definition by matching letters across the whole page.
  Formula labels in SVG need explicitly sized XHTML `foreignObject` content
  with `data-tex`, or an adjacent HTML formula; raw TeX in SVG `text` is not math.
- Inline teaching animations start paused. Give the reader named objects,
  stable positions, a visible initial state, controls, and a nearby explanation
  of each change. Every arrow and color needs a clear meaning; motion should
  make a computation or state transition easier to follow.
- Personal / academic pages (home, papers, projects) are in English; posts
  are in Chinese unless the owner asks otherwise. Write in the owner's
  first-person voice, plainly; no filler intros or hype.
- The point of a post is that the reader understands. Write as a narrator
  explaining the idea, not as someone annotating sources: no "本文基于 X",
  "Playbook 给的公式", "书里的例子", "我核对过", "原文有误" in the body. Motivate
  each concept before defining it and don't skip steps in derivations. When
  unsure of a fact, research it and state the settled answer plainly. Source
  chapters and papers go in a final 参考与延伸阅读 section, which is the only
  place sources are narrated.
- Don't talk about the post itself: no reading plans or structure
  announcements ("下面按报告的顺序走一遍", "本文分三部分", "每遇到一个新部件先讲…再讲…",
  "本节讲…"). Open with the subject — a question, a number, the problem —
  and let headings and the TOC carry the structure.
- Use the English term when the Chinese one is uncommon in the ML community
  (micro-batch, all-reduce, kernel, warp, rollout, KV cache, prefill/decode…).

- Technical accuracy matters more than flourish. When numbers in a demo are
  made up for illustration, say so on the page. Do not state specific facts
  (token ids, benchmark scores, paper results) that you have not verified.
- Each post page is self-contained: post-specific CSS goes in its `<style>`,
  JS in its own `<script>`. Only move something into `assets/` once a second
  post needs it.
- Use the color tokens from `site.css` (`var(--accent)`, `var(--text-soft)`,
  …) instead of hard-coded colors, so pages work in light and dark mode.
- Pages must work at phone width (no horizontal scroll) and respect
  `prefers-reduced-motion` (site.css already shortens CSS animations; JS
  animation loops should check it too when they matter).
- External libraries only when they earn it, pinned to an exact version from
  a CDN (jsDelivr), or vendored with their license. Math: local KaTeX (snippet
  in `templates/article.html`). Charts:
  hand-written SVG/HTML first; D3 if really needed.

## Series and shared helpers

Multi-part posts belong to a series defined in `assets/series.js`
(`scaling`, `smol`, `pretraining`, `hyperconnections`, `architectures`,
`sparsity`, `systems`, `reports`). A series post carries
`<nav class="series" data-series="…">` under its header and
`<nav class="series-pager" data-series="…">` after the article, and loads
`/assets/series.js`; titles in that file must match the posts' `<h1>`.

The LLM engineering learning path is `/posts/llm-engineering/`, an ordinary
entry in the blog list. `assets/curriculum.js` defines the ten stages, their
prerequisites, completion tasks, main lessons (`core`) and further reading
(`extra`). The archive also has a 学习路径 view using that same data. Register
new engineering lessons there and keep the `systems` series in main-path
order. Load `curriculum.js` after the post/player scripts on participating
pages; it adds a link back to the relevant stage. Course completion is an
explicit learner checkbox stored locally in the browser. Do not add a
separate promotional block to the home or archive pages.

The `architectures` series connects state-space recurrence, shared depth,
lookup memory (OE / Engram), and concept-level computation. The `sparsity`
series separates weight pruning, sparse pretraining, and activation sparsity.
Effective depth follows the Hyper-Connections series; upcycling extends
pretraining, and model merging belongs to the adaptation path. Expand the
existing mixed-precision article when adding numerical training recipes.

`site.css` has helpers for long technical posts: `.toc`, `.table-wrap` (wide
tables scroll inside it), `.callout` / `.callout.warn`, `details`, and demo
controls `.ctrls` > `.ctrl` (label · range · output), `.seg` toggle buttons and
`.readout` stat tiles.

## Inline figures, exercises, and engineering practice

- Define each visual object positively and precisely, for example “每格表示当前层
  一个位置的一对 K/V 向量”. Keep essential assumptions beside calculations.
  Remove boilerplate about what a picture does not represent. Explain a concrete
  failure with its mechanism and correction when that distinction teaches something.
- Symbol hints are part of the reading interface. Keep their resting underline
  subtle. Do not add tooltip usage instructions or a separate symbol glossary;
  define the needed symbols in the paragraph beside each derivation.
- Use `/assets/article.css` and `/assets/article-steps.js` for figures that need
  shared controls. `ArticleSteps.mount(root, {steps, render, duration})` calls
  `render(step, index, progress)`, starts paused, and supports rewind and scrubbing.
  A step has a title, a caption, and optional data. Render the scene from its
  arguments so moving backward restores the exact state. Give the figure
  meaningful coordinates, units, a stable layout and numerical values.
- Existing continuous timelines can use `Anim.timeline()` and `Anim.controls()`.
  Keep reader controls in the normal document flow; figures must work at phone
  width and with reduced motion. Prefer custom topic-specific views over a row
  of generic boxes when the computation has a useful spatial representation.
- Keep worked exercises with full reasoning and units in closed `<details>`.
  Engineering articles link their runnable `lab.py`, setup, expected results and
  a concrete modification for the learner. Label example timings as model inputs;
  report measured GPU performance only after running the corresponding GPU code.
- When writing an engineering topic, research concrete implementation problems
  through upstream GitHub issues/discussions, Stack Overflow and official docs or
  source. Read the discussion and verify the mechanism against primary material.
  Choose broadly useful or technically interesting mechanisms: algorithm/hardware
  interactions, performance tradeoffs, numerical contracts and storage lifetimes.
  Exclude installation/dependency/environment failures and one-off implementation
  bugs. Issues are research leads, not a bug anthology. Integrate the symptom,
  mechanism, diagnostic experiment and applicable design into the explanation. Record relevant version or configuration conditions and link
  direct sources in the references. Historical issues illustrate failure modes;
  a closed issue alone is not evidence that current versions remain affected.

## Preview

```
python3 -m http.server 8000
```

then open http://localhost:8000/. Root-absolute paths need a server at the
site root; opening files directly with `file://` will not load the styles.
For a visual check, Playwright with the preinstalled Chromium can screenshot
pages at desktop and phone widths.

Article and figure checks: `python3 scripts/check-articles.py` and
`python3 scripts/check-article-steps.py` (Python Playwright and Chromium).

After changing the shared math reader, run `python3 scripts/check-reader-math.py`
(Python Playwright and Chromium required). The test serves its own local fixture
and checks scoped definitions, rendering, keyboard/touch access and retry behavior.
Also validate rendered TeX with KaTeX's `throwOnError: true`: checking only the
`.katex-error` class can miss unknown commands rendered inline in red.

## Publishing

GitHub Pages serves the `main` branch (Settings → Pages → "Deploy from a
branch", `main`, folder `/`). Changes are live a minute or so after they land
on that branch.
