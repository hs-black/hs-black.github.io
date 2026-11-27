# hs-black.github.io

https://hs-black.github.io/ — personal site of Zunzhe Zhang: profile, papers, and a blog with in-depth articles, interactive figures and runnable engineering labs.

Plain hand-written HTML served by GitHub Pages, no build step. See [CLAUDE.md](CLAUDE.md) for the layout and how posts are added. Run `python3 scripts/sync.py` (or `python3 scripts/post-links.py`) after adding or editing posts to sync cross-links, reading time, `sitemap.xml`, Open Graph metadata, asset cache hashes, and run link checks.

The homepage presents the owner's profile, experience, selected papers, and awards. Blog articles and series are listed in [`/blog/`](https://hs-black.github.io/blog/), accessible through the Blog navigation link.

Preview locally with `python3 -m http.server 8000`.

Password-protected posts use browser-side decryption. Keep their plaintext sources outside this repository and publish only the generated ciphertext page. See [the encrypted-post workflow](docs/encrypted-posts.md).
