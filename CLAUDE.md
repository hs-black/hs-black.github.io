# hs-black.github.io

Personal blog about LLMs, served by GitHub Pages straight from this repo.
There is **no generator and no build step**: every page is hand-written HTML,
and whatever is in the publishing branch is what goes live.
The owner asks Claude to write the posts; this file is how to do that.

## Layout

```
index.html            home page; the post list is hand-maintained here
feed.xml              RSS; hand-maintained, one <item> per post
404.html
.nojekyll             stops GitHub Pages from running Jekyll
assets/site.css       shared styles + color tokens (light/dark)
assets/slides.css     slide-deck styles (load after site.css)
assets/slides.js      slide engine (navigation, fragments, slidechange event)
assets/favicon.svg
posts/<slug>/index.html   one folder per post
templates/article.html    starting point for an article / interactive post
templates/deck.html       starting point for a slide deck
```

All internal links and assets use root-absolute paths (`/assets/site.css`,
`/posts/<slug>/`). Post-specific images or data go next to the post in its
folder and are referenced relatively.

## Adding a post

1. Pick a short lowercase slug (`kv-cache`, `rlhf-notes`) and copy the right
   template to `posts/<slug>/index.html`.
2. Fill in `<title>` (`标题 · hs-black`), `<meta name="description">`, the
   `<h1>`, the kind label and the date.
3. Add an entry at the **top** of the list in `index.html`, matching the
   existing entries (title, kind, `<time>`, one-line description).
4. Add an `<item>` at the **top** of `feed.xml` (RFC 822 `pubDate`).
5. Preview locally (below), then commit and push.

Kinds, used in both the post header and the home list:

| kind | label | class |
| --- | --- | --- |
| plain article | 文章 | `kind` |
| article with interactive demos | 交互 | `kind interactive` |
| slide deck | 幻灯片 | `kind deck` |

## Writing

- Posts are in Chinese unless the owner asks otherwise. Write in the owner's
  first-person voice, plainly; no filler intros or hype.
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

## Preview

```
python3 -m http.server 8000
```

then open http://localhost:8000/. Root-absolute paths need a server at the
site root; opening files directly with `file://` will not load the styles.
For a visual check, Playwright with the preinstalled Chromium can screenshot
pages at desktop and phone widths.

## Publishing

GitHub Pages serves the repo's publishing branch (Settings → Pages → "Deploy
from a branch", folder `/`). Changes are live a minute or so after they land
on that branch.
