# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Overview

Personal Jekyll blog ("Biao's Blog") hosted at hebiao064.github.io via GitHub Pages. Forked from the `heiswayi/the-plain` theme (loaded as a `remote_theme`), with bilingual (English/Chinese) support added on top.

## Commands

```bash
# Ruby path (Apple Silicon Homebrew)
export PATH="/opt/homebrew/opt/ruby/bin:$PATH"

# Install dependencies
bundle install

# Serve locally with live reload
bundle exec jekyll serve
```

Deployment: `git push` to `main`. GitHub Pages CI builds and publishes automatically — there is no separate build step to run.

## Architecture

**Theme resolution.** The site uses `remote_theme: heiswayi/the-plain` (see `_config.yml`), so layouts/includes/sass are pulled from that gem at build time. The local `_layouts/`, `_includes/`, and `_sass/` directories **override** files of the same name from the remote theme. When styling or layout changes aren't taking effect, check whether a local override exists or needs to be created to shadow the remote theme file.

**Bilingual posts.** All posts live in the single `_posts/` directory. Chinese posts are distinguished only by `lang: zh` in front matter (see `CHINESE_SETUP.md`). Routing:
- `index.html` → lists posts *without* `lang: zh`
- `chinese.html` → lists posts *with* `lang: zh`
- The nav toggle lives in `_includes/header.html` with styles in `_sass/main.scss`.

When adding a Chinese post, set `lang: zh` in front matter; otherwise it will appear on the English home page.

**Post front matter** uses `title` + `updated` (date), e.g.:
```yaml
---
title: Post Title
updated: 2025-08-27 11:11
image: /assets/home/open-art/example-cover.jpg
summary: One sentence describing the post for homepage cards.
lang: zh   # optional, omit for English
---
```

**Assets.** Per-post image/asset folders live under `assets/<post-slug>/` (e.g. `assets/fa3-basics/`, `assets/rl-memory-management/`). Reference them via site-relative URLs.

**Homepage cover art.** Each post should have a 4:3 cover image in front matter via `image:`. The current visual direction is editorial/open-access artwork: public-domain museum images cropped into consistent 1200×900 covers. Prefer this over AI-generated abstract images unless the user explicitly asks for generated art.

Mobile cover behavior is configured in `_config.yml`: `hide_cover_art_on_mobile: true` plus `mobile_cover_art_breakpoint: 719px` means cover images are not loaded or displayed on single-column phone layouts. The templates route cover images through `_includes/responsive-cover-image.html`; keep using that include instead of raw `<img>` tags for homepage/post covers.

Workflow for adding a new blog cover:
1. Find an official open-access/public-domain artwork from a stable museum source such as the Art Institute of Chicago, The Met, or the National Gallery of Art. Do not scrape Google Arts & Culture directly; use it only for inspiration, then fetch from the original museum record.
2. Verify rights on the official source page/API. For Art Institute of Chicago API results, prefer records with `is_public_domain: true`; their API data includes CC0 metadata. For The Met API, prefer `isPublicDomain: true`.
3. Create a cropped site asset under `assets/home/open-art/` named `<artist-or-theme>-<short-title>-cover.jpg`, resized to 1200×900, `object-fit: cover` friendly, and reasonably compressed. Do not commit huge original downloads unless specifically needed.
4. Add the source to `assets/home/open-art/README.md` with filename, artwork title, artist, and official source URL.
5. Add the artwork metadata to `_data/open_art.yml` so post pages can render the museum-style caption with artwork title, artist, year, and source link.
6. Set the post front matter `image:` to the new cover path. If the post has English and Chinese versions of the same article, reuse the same cover for both.
7. Add or update `summary:` so homepage cards have concise editorial copy.
8. Run `bundle exec jekyll build`, then preview the homepage and the post page. Check that the image loads and crops well on desktop, that the post page caption appears, that phone-width layouts hide cover art, and that there is no horizontal overflow.

Useful Art Institute IIIF image URL pattern:
```text
https://www.artic.edu/iiif/2/<image_id>/full/2000,/0/default.jpg
```

**Math & diagrams.** MathJax is enabled globally (`mathjax: true` in `_config.yml`). Markdown is kramdown with GFM input and Rouge highlighting.

## Notes

- `_config.yml` changes require restarting `jekyll serve` (not picked up by live reload).
- Large binaries (`code`, `cursor`, `vscode_cli_alpine_x64_cli.tar.gz`) in the repo root are unrelated to the site build and should not be edited or committed.
