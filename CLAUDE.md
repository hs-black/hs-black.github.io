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
blog/index.html       archive of all posts, grouped by year, filter by kind
posts/<slug>/index.html   one folder per post
tools/<name>/index.html   standalone interactive tools (linked from projects)
feed.xml              RSS; hand-maintained, one <item> per post
404.html
.nojekyll             stops GitHub Pages from running Jekyll
assets/site.css       shared styles + color tokens (light/dark)
assets/slides.css     slide-deck styles (load after site.css)
assets/slides.js      slide engine (navigation, fragments, slidechange event)
assets/series.js      series list + nav for multi-part posts
assets/img/           avatar and other shared images
assets/favicon.svg
files/resume.pdf      linked from the home page
templates/article.html    starting point for an article / interactive post
templates/deck.html       starting point for a slide deck
```

All internal links and assets use root-absolute paths (`/assets/site.css`,
`/posts/<slug>/`). Post-specific images or data go next to the post in its
folder and are referenced relatively.

## Adding a post

1. Pick a short lowercase slug (`kv-cache`, `rlhf-notes`) and copy the right
   template to `posts/<slug>/index.html`.
2. Fill in `<title>` (`标题 · Zunzhe Zhang`), `<meta name="description">`, the
   `<h1>`, the kind label and the date.
3. Add an entry at the **top** of the current year in `blog/index.html`
   (start a new `<section class="archive-year">` for a new year), with the
   right `data-kind`.
4. Update the post list under "Writing" on `index.html`: it shows the newest
   **3** posts, so add the new one on top and drop the fourth. If the post
   belongs to a series, also add it to `assets/series.js` (and bump the count
   on the series card under "Writing").
5. Add an `<item>` at the **top** of `feed.xml` (RFC 822 `pubDate`).
6. Preview locally (below), then commit and push.

The header nav (Home / Papers / Projects / Blog) is copied into every page;
mark the current section with `aria-current="page"`. If the nav changes,
change it in every page and both templates.

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
| slide deck | 幻灯片 | `kind deck` |

## Writing

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
- Use the English term when the Chinese one is uncommon in the ML community
  (micro-batch, all-reduce, kernel, warp, rollout, KV cache, prefill/decode…).
  Decks read like a talk: "目录", not "这套幻灯片讲什么".
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
  a CDN (jsDelivr). Math: KaTeX (snippet in `templates/article.html`). Charts:
  hand-written SVG/HTML first; D3 if really needed.

## Series and shared helpers

Multi-part posts belong to a series defined in `assets/series.js`
(`scaling`, `smol`, `reports`). A series post carries
`<nav class="series" data-series="…">` under its header and
`<nav class="series-pager" data-series="…">` after the article, and loads
`/assets/series.js`; titles in that file must match the posts' `<h1>`.

`site.css` has helpers for long technical posts: `.toc`, `.table-wrap` (wide
tables scroll inside it), `.callout` / `.callout.warn`, `details`, and demo
controls `.ctrls` > `.ctrl` (label · range · output), `.seg` toggle buttons and
`.readout` stat tiles.

## Slide decks

- Each `<section class="slide">` inside `.stage` is one slide, laid out on a
  fixed 1280×720 canvas that `slides.js` scales to the window. Sizes in deck
  CSS are canvas pixels.
- `.rise` elements animate in whenever their slide is entered; stagger them
  with `style="--d: 1"`, `--d: 2`, ….
- `.fragment` elements are revealed one per key press, in document order.
  Add `dim-after` to fade earlier fragments once a later one is shown.
- `slidechange` (and `fragmentchange`) fire on `document` with
  `detail = { index, step, slide }`. Give a slide an `id` and start its custom
  animation in a listener; clear timers when another slide is entered (see
  `posts/next-token/` for the pattern).
- Navigation: ← → / space / PageUp/Down, click (left third goes back), swipe,
  `F` for fullscreen, `#N` in the URL opens slide N.
- Live widgets go in `<div class="widget" data-interactive>`: clicks inside
  don't turn the page, and focused inputs/buttons keep their keys. See
  `posts/scaling-pipeline/` and `posts/scaling-collectives/`.

## Preview

```
python3 -m http.server 8000
```

then open http://localhost:8000/. Root-absolute paths need a server at the
site root; opening files directly with `file://` will not load the styles.
For a visual check, Playwright with the preinstalled Chromium can screenshot
pages at desktop and phone widths.

## Publishing

GitHub Pages serves the `main` branch (Settings → Pages → "Deploy from a
branch", `main`, folder `/`). Changes are live a minute or so after they land
on that branch.
