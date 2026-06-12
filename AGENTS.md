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
lang: zh   # optional, omit for English
---
```

**Assets.** Per-post image/asset folders live under `assets/<post-slug>/` (e.g. `assets/fa3-basics/`, `assets/rl-memory-management/`). Reference them via site-relative URLs.

**Math & diagrams.** MathJax is enabled globally (`mathjax: true` in `_config.yml`). Markdown is kramdown with GFM input and Rouge highlighting.

## Notes

- `_config.yml` changes require restarting `jekyll serve` (not picked up by live reload).
- Large binaries (`code`, `cursor`, `vscode_cli_alpine_x64_cli.tar.gz`) in the repo root are unrelated to the site build and should not be edited or committed.
