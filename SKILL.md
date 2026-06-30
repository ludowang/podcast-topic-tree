---
name: podcast-to-obsidian
description: "Turn Xiaoyuzhou podcast links, direct audio URLs, local audio files, or existing transcripts into Obsidian notes with two Chinese outputs: 精读整理版 and 层级全文版. Use when the user wants podcast/video/audio content transcribed with Doubao ASR or Whisper fallback, cleaned and reorganized with DeepSeek, written atomically to Obsidian, with content-aware tags, article-like prose, optional Xiaohei/structure illustrations, and no hallucinated additions."
---

# Podcast To Obsidian

## What This Skill Does

Convert long audio/video content into Obsidian notes:

1. Resolve Xiaoyuzhou audio, direct audio URL, local audio file, or existing transcript.
2. Transcribe with Doubao ASR by default; use local faster-whisper only as fallback.
3. Use DeepSeek to clean, proofread, and reorganize.
4. Produce two versions:
   - `精读整理版`: article-like, readable, moderately detailed.
   - `层级全文版`: Tab-indented logical hierarchy for Obsidian folding and close reading.
5. Write atomically to Obsidian.
6. Add only useful images: generate an opening illustration and structure diagrams only when they clarify a real relationship, save PNGs beside the note, and insert them into Markdown.

Default output target:

```text
/Users/wangluda03/Desktop/抽空学习/播客逐字稿/
```

## Required Env Files

Load these as needed with repeated `--env-file`:

```text
/Users/wangluda03/Desktop/AI时代/蚩尤/podcast.env
/Users/wangluda03/Desktop/专家访谈/doubaoyuyin.env
/Users/wangluda03/Desktop/专家访谈/豆包TOS配置.env
/Users/wangluda03/Desktop/AI时代/蚩尤/gpt_image2.env
```

Do not print keys. Confirm env loading only with boolean/prefix/length if debugging.

## Setup

Use the bundled script:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r /Users/wangluda03/.codex/skills/podcast-to-obsidian/scripts/requirements.txt
```

## Main Commands

Xiaoyuzhou link, formal run:

```bash
. .venv/bin/activate
python /Users/wangluda03/.codex/skills/podcast-to-obsidian/scripts/podcast_to_obsidian.py \
  "https://www.xiaoyuzhoufm.com/episode/..." \
  --env-file "/Users/wangluda03/Desktop/AI时代/蚩尤/podcast.env" \
  --env-file "/Users/wangluda03/Desktop/专家访谈/doubaoyuyin.env" \
  --env-file "/Users/wangluda03/Desktop/专家访谈/豆包TOS配置.env" \
  --env-file "/Users/wangluda03/Desktop/AI时代/蚩尤/gpt_image2.env" \
  --asr doubao \
  --quality final \
  --visual-mode auto \
  --output-version both \
  --article-merge merge \
  --work-base .work \
  --keep-workdir
```

Manual audio URL fallback:

```bash
python /Users/wangluda03/.codex/skills/podcast-to-obsidian/scripts/podcast_to_obsidian.py \
  --audio-url "https://..." \
  --source-url "https://www.xiaoyuzhoufm.com/episode/..." \
  --env-file "/Users/wangluda03/Desktop/AI时代/蚩尤/podcast.env" \
  --env-file "/Users/wangluda03/Desktop/专家访谈/doubaoyuyin.env" \
  --env-file "/Users/wangluda03/Desktop/专家访谈/豆包TOS配置.env" \
  --env-file "/Users/wangluda03/Desktop/AI时代/蚩尤/gpt_image2.env" \
  --asr doubao \
  --quality final \
  --visual-mode auto \
  --output-version both \
  --work-base .work \
  --keep-workdir
```

Local audio fallback:

```bash
python /Users/wangluda03/.codex/skills/podcast-to-obsidian/scripts/podcast_to_obsidian.py \
  --audio-file "/absolute/path/audio.m4a" \
  --source-url "https://source.example" \
  --env-file "/Users/wangluda03/Desktop/AI时代/蚩尤/podcast.env" \
  --env-file "/Users/wangluda03/Desktop/专家访谈/doubaoyuyin.env" \
  --env-file "/Users/wangluda03/Desktop/专家访谈/豆包TOS配置.env" \
  --env-file "/Users/wangluda03/Desktop/AI时代/蚩尤/gpt_image2.env" \
  --asr doubao \
  --quality final \
  --visual-mode auto \
  --output-version both \
  --work-base .work \
  --keep-workdir
```

Existing transcript / debug mode:

```bash
python /Users/wangluda03/.codex/skills/podcast-to-obsidian/scripts/podcast_to_obsidian.py \
  --skip-whisper "/absolute/path/transcript.txt" \
  --source-url "https://source.example" \
  --env-file "/Users/wangluda03/Desktop/AI时代/蚩尤/podcast.env" \
  --skip-cleaning \
  --work-base .work \
  --keep-workdir
```

If ASR has known mistakes, pass corrections:

```bash
--replace "追友=最右" --keep-term "push" --keep-term "ROI"
```

## YouTube Handling

The bundled script is strongest for Xiaoyuzhou/audio/local files. For YouTube:

- First try `yt-dlp` to download audio or subtitles in the workspace.
- If audio download is blocked, use available subtitles/transcript text as `--skip-whisper`.
- If there are no usable subtitles and audio cannot be downloaded, report the blocker clearly and ask for a local audio/video file.

## Quality Rules Learned From Practice

These are hard requirements, not style preferences.

- Do not let opening metadata cover only the first section. `这篇内容在回答什么问题`, `核心观点`, `原文金句`, and `对照总结` must cover the whole episode.
- Do not use generic tags such as `播客`, `逐字稿`, `层级全文`. Generate topic tags such as company, industry, method, event, and theme. Keep tags broad, not overly granular.
- Do not force every article into a fixed framework such as “目的、机制、影响、案例、边界”. Infer the source's own structure first.
- Do not hallucinate. Do not add outside facts unless the user explicitly asks for external research. If a point is inferred from the source, mark it as `嘉宾判断`, `材料中提到`, or similar.
- Do not preserve transcript-like prose in the article body. Rewrite oral fragments into readable written Chinese while preserving details.
- Do not leave long sticky paragraphs. Split dense reasoning into short paragraphs, lists, or tables when they improve readability.
- Do not duplicate sections after chunked DeepSeek calls. Run a global merge/deduplication pass and check title-content alignment.
- Preserve important detail: dates, numbers, names, product names, examples, causal claims, caveats, and uncertainty.
- Proper nouns matter. Use page context from Xiaoyuzhou, title/description, and user corrections. Ask or flag uncertainty when a term affects meaning.

## Image Rules

Images are optional. Bad images are worse than no images.

Default image mode is `--visual-mode auto`:

- Generate visual briefs from the organized article.
- Call GPT-Image through env-configured API credentials.
- Download PNGs into `播客逐字稿/assets/<note-title>/`.
- Insert Markdown image links into the note.
- If the API key is missing or image generation fails, skip images without failing the whole note.

Use `--visual-mode brief` when debugging image selection only. Use `--visual-mode none` when no image work is needed.

Use `ian-xiaohei-illustrations` only when the image can explain the whole article or a real conceptual relationship. For structure-heavy diagrams, deterministic HTML/CSS/PNG or a clean infographic can be better than AI illustration.

Opening image:

- Must explain the whole article, not just the first section.
- Prefer one clean metaphor over many objects.
- Xiaohei must perform the core action, not decorate.
- Avoid clutter: one table/one sieve/one map/one object is usually enough.
- Do not make decorative “interest” images that add no comprehension.

Structure images:

- Draw only when they reduce understanding cost in 3 seconds.
- Suitable: growth middle-platform structure, LTV/ROI ledger, buy-to-organic flywheel, ByteDance vs traditional marketing, business migration path.
- Not suitable: “this episode talks about X”, guest identity, generic importance, already-clear lists.

Before inserting an image into Markdown, inspect it. Reject if it has:

- confusing or invented relationships,
- too many objects,
- unreadable or wrong Chinese,
- cute/childish Xiaohei,
- PPT-like clutter,
- title/text that tells the reader how to read the chart,
- a scope mismatch with the surrounding section.

## Output Contract

Write notes atomically. File name format:

```text
YYYY-MM-DD 标题 精读整理版.md
YYYY-MM-DD 标题 层级全文版.md
```

The `精读整理版` must contain:

- YAML frontmatter with source, audio, date, draft, and content-specific tags.
- Title.
- Optional opening image if it passes image rules.
- `这篇内容在回答什么问题`: whole-article questions.
- `核心观点`: whole-article core points.
- `原文金句`: source-grounded quotes, not invented summaries.
- `全文整理`: article-like prose with headings that match content.
- `对照总结`: whole-article summary for comparison and review.

The `层级全文版` must use Tab-indented nested Markdown lists for Obsidian folding. It should preserve logical hierarchy:

- Child topics attach by logical dependency, not conversational order.
- B is A's child only when B expands, exemplifies, proves, or specifies A.
- B is parallel with A when both independently support a higher-level point.
- “A introduced B” is not enough for dependency.
- Merge topics that branch away and return later.

## Troubleshooting

- Xiaoyuzhou parsing is fragile. If audio extraction fails, rerun with `--audio-url` or `--audio-file`.
- Doubao ASR uses TOS temporary upload for local files and should clean objects unless `--keep-tos-object` is set.
- Whisper fallback requires `ffmpeg`; formal Whisper mode should use `large-v3`, but Doubao ASR is preferred for accuracy and speaker separation.
- If DeepSeek output becomes repetitive, switch `--article-merge merge`, inspect `organized_article.md`, and manually deduplicate before writing final.
- If Obsidian does not show the note, verify the actual vault path and subdirectory, not just the Markdown file path.
