#!/usr/bin/env python3
"""
Xiaoyuzhou/audio/transcript -> Doubao or Whisper ASR -> DeepSeek整理 -> Obsidian.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urljoin, urlparse

import requests


DEFAULT_VAULT = Path("/Users/wangluda03/Desktop/抽空学习")
DEFAULT_SUBDIR = "播客逐字稿"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DOUBAO_SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
DOUBAO_QUERY_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
DEFAULT_DOUBAO_RESOURCE_ID = "volc.seedasr.auc"
GPT_IMAGE_BASE_URL = "https://dragoncode.codes/gpt-image/v1"
GPT_IMAGE_MODEL = "gpt-image-2"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
AUDIO_EXTENSIONS = (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".mp4")
SENTENCE_RE = re.compile(r"([^。！？!?；;\n]+[。！？!?；;]?)")


@dataclass
class EpisodeInfo:
    source_url: str
    audio_url: str
    page_title: Optional[str] = None
    page_context: str = ""


@dataclass
class NoteMetadata:
    title: str
    tags: List[str]
    questions: List[str]
    core_points: List[str]
    quotes: List[str]
    conclusion: List[str]


@dataclass
class VisualSpec:
    kind: str
    title: str
    why_draw: str
    core_relationship: str
    prompt_seed: Dict[str, Any]


@dataclass
class GeneratedVisual:
    rel_path: str
    caption: str
    abs_path: Path


def log(message: str) -> None:
    print(f"[podcast] {message}", file=sys.stderr)


def require_deps(
    skip_whisper: bool = False,
    needs_parse: bool = False,
    needs_download: bool = False,
    needs_doubao: bool = False,
) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("缺少环境变量 DEEPSEEK_API_KEY。请先 export DEEPSEEK_API_KEY='...'")
    if needs_parse:
        try:
            import bs4  # noqa: F401
        except ImportError as exc:
            raise SystemExit(f"缺少 Python 依赖：{exc.name}。请先运行：pip install -r requirements.txt")
    if needs_download or not skip_whisper:
        try:
            import tqdm  # noqa: F401
        except ImportError as exc:
            raise SystemExit(f"缺少 Python 依赖：{exc.name}。请先运行：pip install -r requirements.txt")
    if not skip_whisper and not shutil.which("ffmpeg"):
        raise SystemExit("未找到 ffmpeg。请先安装：brew install ffmpeg")
    if not skip_whisper:
        try:
            import faster_whisper  # noqa: F401
        except ImportError as exc:
            raise SystemExit(f"缺少 Python 依赖：{exc.name}。请先运行：pip install -r requirements.txt")
    if needs_doubao:
        try:
            import tos  # noqa: F401
        except ImportError as exc:
            raise SystemExit(f"缺少 Python 依赖：{exc.name}。请先运行：pip install -r requirements.txt")


def load_env_file(path: Optional[Any]) -> None:
    if not path:
        return
    if isinstance(path, list):
        for item in path:
            load_env_file(item)
        return
    env_path = path.expanduser()
    if not env_path.exists():
        raise SystemExit(f"环境变量文件不存在：{env_path}")

    raw = env_path.read_text(encoding="utf-8", errors="ignore")
    text = rtf_to_plain_text(raw) if raw.lstrip().startswith("{\\rtf") else raw
    key_match = re.search(r"(DEEPSEEK_API_KEY)\s*=\s*([^\s;{}]+)", text)
    if key_match and key_match.group(1) not in os.environ:
        os.environ[key_match.group(1)] = key_match.group(2).strip().strip('"').strip("'")

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def rtf_to_plain_text(raw: str) -> str:
    text = raw.replace("\\par", "\n").replace("\\line", "\n")
    text = re.sub(r"\\'[0-9a-fA-F]{2}", "", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    text = re.sub(r"[{}]", "", text)
    return text


def fetch(url: str, timeout: int = 30) -> requests.Response:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp


def find_audio_urls_in_obj(obj: Any) -> List[str]:
    found: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_l = str(key).lower()
            if isinstance(value, str) and ("audio" in key_l or "media" in key_l or "url" in key_l):
                if looks_like_audio_url(value):
                    found.append(value)
            found.extend(find_audio_urls_in_obj(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(find_audio_urls_in_obj(item))
    elif isinstance(obj, str) and looks_like_audio_url(obj):
        found.append(obj)
    return unique(found)


def looks_like_audio_url(value: str) -> bool:
    decoded = unquote(value).replace("\\u002F", "/").replace("\\/", "/")
    if not decoded.startswith(("http://", "https://")):
        return False
    lower = decoded.lower().split("?")[0]
    return lower.endswith(AUDIO_EXTENSIONS) or any(ext + "?" in decoded.lower() for ext in AUDIO_EXTENSIONS)


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in seen:
            output.append(normalized)
            seen.add(normalized)
    return output


def extract_json_from_script(script_text: str) -> Optional[Any]:
    text = script_text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_xiaoyuzhou(source_url: str) -> EpisodeInfo:
    from bs4 import BeautifulSoup

    log(f"解析小宇宙页面：{source_url}")
    resp = fetch(source_url)
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    page_title = None

    if soup.title and soup.title.string:
        page_title = soup.title.string.strip()
    page_context = extract_page_context(soup)

    candidates: List[str] = []

    for attr_name in ("content", "src", "href"):
        for tag in soup.find_all(attrs={attr_name: True}):
            value = tag.get(attr_name)
            if isinstance(value, str) and looks_like_audio_url(value):
                candidates.append(urljoin(source_url, value))

    for script in soup.find_all("script"):
        text = script.string or script.get_text() or ""
        parsed = extract_json_from_script(text)
        if parsed is not None:
            candidates.extend(find_audio_urls_in_obj(parsed))

    candidates.extend(regex_audio_urls(html))
    candidates.extend(audio_urls_from_rss_links(soup, source_url))

    candidates = unique(clean_url(u) for u in candidates)
    if not candidates:
        raise RuntimeError(
            "没能从小宇宙页面解析到音频直链。页面可能改版了，请改用 --audio-url 或 --audio-file。"
        )

    log(f"找到音频候选 {len(candidates)} 个，使用第一个。")
    return EpisodeInfo(source_url=source_url, audio_url=candidates[0], page_title=page_title, page_context=page_context)


def parse_page_metadata(source_url: str) -> Tuple[Optional[str], str]:
    from bs4 import BeautifulSoup

    try:
        resp = fetch(source_url)
    except Exception as exc:
        log(f"读取来源页面上下文失败，继续执行：{exc}")
        return None, ""
    soup = BeautifulSoup(resp.text, "html.parser")
    page_title = soup.title.string.strip() if soup.title and soup.title.string else None
    return page_title, extract_page_context(soup)


def extract_page_context(soup: Any) -> str:
    parts: List[str] = []
    if soup.title and soup.title.string:
        parts.append(soup.title.string.strip())

    meta_keys = {
        "description",
        "keywords",
        "og:title",
        "og:description",
        "twitter:title",
        "twitter:description",
    }
    for tag in soup.find_all("meta"):
        key = (tag.get("name") or tag.get("property") or "").strip().lower()
        content = (tag.get("content") or "").strip()
        if key in meta_keys and content:
            parts.append(content)

    visible = soup.get_text(" ", strip=True)
    if visible:
        parts.append(visible[:2000])
    return re.sub(r"\s+", " ", " ".join(unique(parts))).strip()[:4000]


def regex_audio_urls(text: str) -> List[str]:
    decoded = unquote(text).replace("\\u002F", "/").replace("\\/", "/")
    pattern = r"https?://[^\"'<>\s]+?(?:mp3|m4a|aac|wav|flac|ogg|opus|mp4)(?:\?[^\"'<>\s]*)?"
    return re.findall(pattern, decoded, flags=re.IGNORECASE)


def audio_urls_from_rss_links(soup: BeautifulSoup, source_url: str) -> List[str]:
    urls: List[str] = []
    rss_links = []
    for link in soup.find_all("link", href=True):
        link_type = (link.get("type") or "").lower()
        href = link.get("href")
        if href and ("rss" in link_type or "xml" in link_type or "rss" in href):
            rss_links.append(urljoin(source_url, href))

    for rss_url in unique(rss_links):
        try:
            rss = fetch(rss_url, timeout=20).text
        except Exception:
            continue
        rss_soup = BeautifulSoup(rss, "xml")
        for enclosure in rss_soup.find_all("enclosure", url=True):
            url = enclosure.get("url")
            if url and looks_like_audio_url(url):
                urls.append(url)
    return urls


def clean_url(url: str) -> str:
    return unquote(url).replace("\\u002F", "/").replace("\\/", "/")


def download_audio(audio_url: str, workdir: Path) -> Path:
    from tqdm import tqdm

    parsed = urlparse(audio_url)
    suffix = Path(parsed.path).suffix
    if not suffix or len(suffix) > 8:
        suffix = ".mp3"
    output = workdir / f"audio{suffix}"
    log(f"下载音频：{audio_url}")
    with requests.get(audio_url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with output.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, disable=total == 0) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    return output


def transcribe(
    audio_file: Path,
    model_name: str,
    language: Optional[str],
    compute_type: str,
    initial_prompt: Optional[str] = None,
) -> str:
    from faster_whisper import WhisperModel
    from tqdm import tqdm

    log(f"加载 Whisper 模型：{model_name}（CPU，{compute_type}）")
    model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio_file),
        language=language,
        vad_filter=True,
        beam_size=5,
        word_timestamps=False,
        initial_prompt=initial_prompt,
    )
    log(f"开始转录，识别语言：{info.language}，概率：{info.language_probability:.2f}")
    lines = []
    for segment in tqdm(segments, desc="transcribing"):
        text = segment.text.strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def doubao_transcribe(
    audio_file: Path,
    workdir: Path,
    poll_interval: int,
    timeout: int,
    keep_object: bool = False,
) -> str:
    cfg = load_doubao_config(require_tos=True)
    object_key = make_tos_object_key(audio_file)
    uploaded = False
    try:
        log("上传音频到火山 TOS 私有桶")
        audio_url = tos_upload_and_presign(audio_file, object_key, cfg)
        uploaded = True
        log("提交豆包录音文件识别任务")
        task_id, submit_headers = doubao_submit_asr(audio_url, audio_file, cfg)
        log(f"轮询豆包 ASR 结果：task_id={task_id}")
        result = doubao_poll_result(
            task_id,
            cfg,
            interval=poll_interval,
            timeout=timeout,
            log_id=submit_headers.get("X-Tt-Logid"),
        )
        output = {"task_id": task_id, "submit_headers": submit_headers, "result": result}
        json_path = workdir / "doubao_asr.json"
        md_path = workdir / "doubao_transcript.md"
        json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        transcript_markdown = doubao_result_to_markdown(result)
        md_path.write_text(transcript_markdown, encoding="utf-8")
        return transcript_markdown
    finally:
        if uploaded and not keep_object:
            try:
                tos_delete_object(object_key, cfg)
                log("已删除 TOS 临时音频对象")
            except Exception as exc:
                log(f"警告：删除 TOS 临时对象失败：{exc}")


def doubao_transcribe_url(
    audio_url: str,
    audio_name: str,
    workdir: Path,
    poll_interval: int,
    timeout: int,
) -> str:
    cfg = load_doubao_config(require_tos=False)
    log("使用音频直链提交豆包录音文件识别任务")
    task_id, submit_headers = doubao_submit_asr(audio_url, Path(audio_name), cfg)
    log(f"轮询豆包 ASR 结果：task_id={task_id}")
    result = doubao_poll_result(
        task_id,
        cfg,
        interval=poll_interval,
        timeout=timeout,
        log_id=submit_headers.get("X-Tt-Logid"),
    )
    output = {"task_id": task_id, "submit_headers": submit_headers, "result": result}
    json_path = workdir / "doubao_asr.json"
    md_path = workdir / "doubao_transcript.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    transcript_markdown = doubao_result_to_markdown(result)
    md_path.write_text(transcript_markdown, encoding="utf-8")
    return transcript_markdown


def load_doubao_config(require_tos: bool = True) -> Dict[str, Any]:
    cfg = {
        "doubao_api_key": env_first("DOUBAO_API_KEY", "X_API_KEY"),
        "doubao_app_key": env_first("DOUBAO_APP_KEY", "VOLC_APP_ID", "X_API_APP_KEY"),
        "doubao_access_key": env_first("DOUBAO_ACCESS_KEY", "VOLC_ACCESS_TOKEN", "X_API_ACCESS_KEY"),
        "doubao_resource_id": os.environ.get("DOUBAO_RESOURCE_ID", DEFAULT_DOUBAO_RESOURCE_ID),
        "tos_ak": env_first("VOLC_ACCESS_KEY_ID", "TOS_ACCESS_KEY_ID"),
        "tos_sk": env_first("VOLC_SECRET_ACCESS_KEY", "TOS_SECRET_ACCESS_KEY"),
        "tos_endpoint": env_first("VOLC_TOS_ENDPOINT", "TOS_ENDPOINT"),
        "tos_region": env_first("VOLC_TOS_REGION", "TOS_REGION"),
        "tos_bucket": env_first("VOLC_TOS_BUCKET", "TOS_BUCKET"),
        "expires": int(os.environ.get("TOS_PRESIGN_EXPIRES_SECONDS", "7200")),
    }
    optional = {"doubao_api_key", "doubao_app_key", "doubao_access_key"}
    if not require_tos:
        optional.update({"tos_ak", "tos_sk", "tos_endpoint", "tos_region", "tos_bucket"})
    missing = [key for key, value in cfg.items() if value in (None, "") and key not in optional]
    if not cfg["doubao_api_key"] and not (cfg["doubao_app_key"] and cfg["doubao_access_key"]):
        missing.append("DOUBAO_API_KEY 或 DOUBAO_APP_KEY+DOUBAO_ACCESS_KEY")
    if missing:
        raise SystemExit("缺少豆包/TOS 环境变量：" + ", ".join(missing))
    return cfg


def env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def tos_upload_and_presign(audio_path: Path, object_key: str, cfg: Dict[str, Any]) -> str:
    import tos

    client = tos.TosClientV2(cfg["tos_ak"], cfg["tos_sk"], cfg["tos_endpoint"], cfg["tos_region"])
    content_type = mimetypes.guess_type(str(audio_path))[0] or "application/octet-stream"
    with audio_path.open("rb") as f:
        client.put_object(cfg["tos_bucket"], object_key, content=f, content_type=content_type)

    if hasattr(client, "pre_signed_url"):
        return client.pre_signed_url(
            tos.HttpMethodType.Http_Method_Get,
            cfg["tos_bucket"],
            object_key,
            expires=cfg["expires"],
        ).signed_url
    if hasattr(client, "preSignedURL"):
        return client.preSignedURL("GET", cfg["tos_bucket"], object_key, cfg["expires"]).signed_url
    raise SystemExit("当前 tos SDK 未提供已知的预签名 URL 方法。")


def tos_delete_object(object_key: str, cfg: Dict[str, Any]) -> None:
    import tos

    client = tos.TosClientV2(cfg["tos_ak"], cfg["tos_sk"], cfg["tos_endpoint"], cfg["tos_region"])
    client.delete_object(cfg["tos_bucket"], object_key)


def doubao_submit_asr(audio_url: str, audio_file: Path, cfg: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    request_id = str(uuid.uuid4())
    payload = {
        "user": {"uid": "podcast-to-obsidian"},
        "audio": {
            "url": audio_url,
            "format": audio_format(audio_file),
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": False,
            "enable_punc": True,
            "enable_ddc": False,
            "enable_speaker_info": True,
            "ssd_version": "200",
        },
    }
    headers, body = post_json(DOUBAO_SUBMIT_URL, payload, doubao_headers(cfg, request_id, sequence="-1"))
    status = headers.get("X-Api-Status-Code")
    if status != "20000000":
        raise SystemExit(f"豆包 ASR submit 失败：{redact_headers(headers)} {body[:1000]}")
    return request_id, redact_headers(headers)


def doubao_poll_result(
    task_id: str,
    cfg: Dict[str, Any],
    interval: int,
    timeout: int,
    log_id: Optional[str] = None,
) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        headers, body = post_json(DOUBAO_QUERY_URL, {}, doubao_headers(cfg, task_id, log_id=log_id))
        status = headers.get("X-Api-Status-Code")
        if status == "20000000":
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"raw": body}
        if status and status not in {"20000001", "20000002"}:
            raise SystemExit(f"豆包 ASR query 失败：{redact_headers(headers)} {body[:1000]}")
        time.sleep(interval)
    raise SystemExit("等待豆包 ASR 结果超时。")


def doubao_headers(
    cfg: Dict[str, Any],
    request_id: str,
    sequence: Optional[str] = None,
    log_id: Optional[str] = None,
) -> Dict[str, str]:
    headers = {
        "X-Api-Resource-Id": cfg["doubao_resource_id"],
        "X-Api-Request-Id": request_id,
        "Content-Type": "application/json",
    }
    if sequence is not None:
        headers["X-Api-Sequence"] = sequence
    if log_id:
        headers["X-Tt-Logid"] = log_id
    if cfg.get("doubao_api_key"):
        headers["X-Api-Key"] = cfg["doubao_api_key"]
    else:
        headers["X-Api-App-Key"] = cfg["doubao_app_key"]
        headers["X-Api-Access-Key"] = cfg["doubao_access_key"]
    return headers


def post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Tuple[Dict[str, str], str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return dict(resp.headers.items()), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {body[:1000]}")


def make_tos_object_key(path: Path) -> str:
    ext = path.suffix or ".mp3"
    return f"podcast-audio-temp/{time.strftime('%Y%m%d')}/{uuid.uuid4().hex}{ext}"


def audio_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"mp3", "wav", "m4a", "mp4", "aac", "ogg", "opus", "flac"}:
        return suffix
    return "mp3"


def redact_headers(headers: Dict[str, Any]) -> Dict[str, Any]:
    redacted = {}
    for key, value in headers.items():
        if key.lower() in {"x-api-access-key", "x-api-key", "authorization"}:
            redacted[key] = "[REDACTED]"
        else:
            redacted[key] = value
    return redacted


def doubao_result_to_markdown(result: Dict[str, Any]) -> str:
    inner = result.get("result", {}) if isinstance(result, dict) else {}
    text = inner.get("text", "") if isinstance(inner, dict) else ""
    utterances = inner.get("utterances", []) if isinstance(inner, dict) else []
    lines = [
        "# 播客逐字稿",
        "",
        "**转写说明**：由豆包录音文件识别模型生成；保留说话人和时间戳，后续会再做书面化校对。",
        "",
    ]
    if utterances:
        lines.append("## 对话转写")
        lines.append("")
        for item in utterances:
            segment = str(item.get("text", "")).strip()
            if not segment:
                continue
            start = ms_to_time(item.get("start_time", 0))
            end = ms_to_time(item.get("end_time", 0))
            lines.append(f"【{speaker_label(item)}】{segment}")
            lines.append("")
            lines.append(f"> 时间：{start}-{end}")
            lines.append("")
    if text:
        lines.append("## 全文")
        lines.append("")
        lines.append(text.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def ms_to_time(value: Any) -> str:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        ms = 0
    seconds = ms // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def speaker_label(item: Dict[str, Any]) -> str:
    for key in ("speaker", "speaker_id", "speaker_id_str", "spk_id", "spkid"):
        value = item.get(key)
        if value not in (None, ""):
            return f"说话人{value}"
    additions = item.get("additions")
    if isinstance(additions, dict):
        for key in ("speaker", "speaker_id", "spk_id"):
            value = additions.get(key)
            if value not in (None, ""):
                return f"说话人{value}"
    return "未知发言人"


def build_whisper_prompt(page_context: str, source_title: Optional[str]) -> str:
    terms = extract_context_terms(" ".join([source_title or "", page_context]))
    if not terms:
        return "以下是中文商业、消费、零售、行业研究类播客，请使用简体中文转写，保留英文缩写。"
    return (
        "以下是中文商业、消费、零售、行业研究类播客，请使用简体中文转写，保留英文缩写。"
        "可能出现的节目、机构、人名、品牌、行业词包括："
        + "、".join(terms[:60])
    )


def extract_context_terms(text: str) -> List[str]:
    cleaned = re.sub(r"https?://\S+", " ", text)
    raw_terms = re.findall(r"[A-Za-z]{2,12}|[\u4e00-\u9fffA-Za-z0-9]{2,18}", cleaned)
    stop = {
        "小宇宙",
        "播客",
        "听播客",
        "episode",
        "https",
        "www",
        "com",
        "分享",
        "节目",
        "标题",
    }
    terms = []
    for term in raw_terms:
        term = term.strip(" -_|，。！？、：；（）()[]【】")
        if len(term) < 2 or term.lower() in stop or term in stop:
            continue
        if re.fullmatch(r"\d+", term):
            continue
        terms.append(term)
    return unique(terms)


def split_by_sentence(text: str, max_chars: int) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return [normalized] if normalized else []

    sentences = [m.group(1).strip() for m in SENTENCE_RE.finditer(normalized) if m.group(1).strip()]
    if not sentences:
        return [normalized[i : i + max_chars] for i in range(0, len(normalized), max_chars)]

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def deepseek_chat(
    messages: List[Dict[str, str]],
    model: Optional[str],
    temperature: float = 0.2,
    max_tokens: int = 8192,
    retries: int = 4,
) -> str:
    api_key = os.environ["DEEPSEEK_API_KEY"]
    actual_model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    url = normalize_deepseek_url(os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_URL))
    payload = {
        "model": actual_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            sleep_for = 2 ** attempt
            log(f"DeepSeek 调用失败，第 {attempt} 次重试前等待 {sleep_for}s：{exc}")
            time.sleep(sleep_for)
    raise RuntimeError(f"DeepSeek 调用失败：{last_error}")


def normalize_deepseek_url(value: str) -> str:
    url = value.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/chat/completions" if url != DEEPSEEK_URL.rstrip("/") else DEEPSEEK_URL


def clean_transcript(transcript: str, model: str, chunk_chars: int, page_context: str = "") -> str:
    chunks = split_by_sentence(transcript, chunk_chars)
    cleaned: List[str] = []
    system = (
        "你是严谨的中文播客文字编辑，目标是把逐字稿整理成可阅读的行业研究文章底稿。"
        "使用简体中文。去掉口癖、寒暄式衔接、无意义语气词和重复表达；"
        "如果输入是英文或中英混合，先忠实翻译为简体中文，保留必要英文术语、人名、产品名和公司名；"
        "把碎句、倒装、口语语序整理成通顺书面句。"
        "不得压缩信息、不得总结替代原意、不得删掉观点、例子、数据、品牌、人名和细节。"
        "可以把第一人称闲聊改写成客观表述，但不能改变含义。输出连续正文。"
    )
    for index, chunk in enumerate(chunks, 1):
        log(f"DeepSeek 清理逐字稿分段 {index}/{len(chunks)}")
        context_line = (
            "以下是页面元信息，只能用于识别专名、节目背景和时间线，严禁原样输出到清理结果：\n"
            f"<page_context>{page_context[:1500]}</page_context>\n\n"
            if page_context
            else ""
        )
        cleaned.append(
            deepseek_chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{context_line}清理以下播客逐字稿片段：\n\n{chunk}"},
                ],
                model=model,
            )
        )
    return "\n\n".join(cleaned)


def proofread_cleaned_text(cleaned_text: str, model: str, chunk_chars: int, page_context: str = "") -> str:
    chunks = split_by_sentence(cleaned_text, chunk_chars)
    proofread: List[str] = []
    system = (
        "你是中文商业内容校对编辑。请在不删减信息、不改变观点的前提下校对文本。"
        "任务：统一为简体中文；修正明显错别字、同音误识别、繁简混用、标点和专有名词不一致；"
        "如果仍有英文原句，除必要术语、人名、产品名和公司名外，翻译成自然简体中文；"
        "把残留口语改成自然书面表达。不要新增原文没有的信息，不要做摘要。输出校对后的连续正文。"
    )
    for index, chunk in enumerate(chunks, 1):
        log(f"DeepSeek 校对清理稿分段 {index}/{len(chunks)}")
        context_line = (
            "以下是页面元信息，只能用于识别专名、节目背景和时间线，严禁原样输出到校对结果：\n"
            f"<page_context>{page_context[:1500]}</page_context>\n\n"
            if page_context
            else ""
        )
        proofread.append(
            deepseek_chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{context_line}请校对以下文本：\n\n{chunk}"},
                ],
                model=model,
                temperature=0.1,
            )
        )
    return "\n\n".join(proofread)


def build_topic_tree(cleaned_text: str, model: str, chunk_chars: int) -> str:
    chunks = split_by_sentence(cleaned_text, chunk_chars)
    if len(chunks) == 1:
        return topic_tree_for_chunk(chunks[0], model, final=True)

    partial_trees = []
    for index, chunk in enumerate(chunks, 1):
        log(f"DeepSeek 生成局部层级全文结构 {index}/{len(chunks)}")
        partial_trees.append(topic_tree_for_chunk(chunk, model, final=False))

    return merge_topic_trees(partial_trees, model, chunk_chars)


def topic_tree_for_chunk(text: str, model: str, final: bool) -> str:
    system = topic_tree_system_prompt()
    scope = "完整文本" if final else "文本片段"
    return deepseek_chat(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"请把以下播客{scope}重组成 Markdown 嵌套层级全文结构。\n\n"
                    "要求：只输出嵌套列表，不要 frontmatter，不要代码块。\n\n"
                    f"{text}"
                ),
            },
        ],
        model=model,
        temperature=0.1,
        max_tokens=8192,
    )


def topic_tree_system_prompt() -> str:
    return (
        "你是播客内容结构化专家。播客是线性的，但内容是树状的；你的任务是把被压平的内容逻辑重新立起来。\n"
        "输出格式必须是 Markdown 嵌套列表，使用 Tab 缩进表达层级，适配 Obsidian 原生折叠。全篇使用简体中文和书面表达。\n"
        "每个节点格式：'- **一句话标题**\\n\\t正文...'；标题加粗在前，正文另起一行。\n"
        "依附判断规则：B 是 A 的展开、举例、论证或细化，B 作 A 的子节点；"
        "B 和 A 是并列的两个点，或共同支撑上层观点，则 B 与 A 平级；"
        "'由 A 引出 B' 只是叙述衔接，不构成依附关系。\n"
        "其他原则：子话题就近挂在逻辑父话题下；岔出又绕回的话题合并回原节点；"
        "深度不限，跟逻辑自然分叉；不要为了对称硬凑层级；每个节点必须是独立自足的语义块。\n"
        "内容原则：不压缩信息，不概括省略；观点、例子、数据、细节都要保留；只去掉无意义重复。"
        "表达原则：像行业研究文章一样清晰，不保留无意义寒暄、口水话和访谈衔接。"
    )


def merge_topic_trees(trees: List[str], model: str, chunk_chars: int) -> str:
    round_no = 1
    current = trees
    while len("\n\n".join(current)) > chunk_chars or len(current) > 1:
        batches = batch_texts(current, chunk_chars)
        merged: List[str] = []
        for index, batch in enumerate(batches, 1):
            log(f"DeepSeek 合并层级全文结构 round {round_no} batch {index}/{len(batches)}")
            merged.append(merge_topic_tree_batch(batch, model))
        if len(merged) == len(current):
            return merge_topic_tree_batch(current, model)
        current = merged
        round_no += 1
    return current[0]


def batch_texts(texts: List[str], max_chars: int) -> List[List[str]]:
    batches: List[List[str]] = []
    current: List[str] = []
    current_len = 0
    for text in texts:
        text_len = len(text)
        if current and current_len + text_len > max_chars:
            batches.append(current)
            current = [text]
            current_len = text_len
        else:
            current.append(text)
            current_len += text_len
    if current:
        batches.append(current)
    return batches


def merge_topic_tree_batch(trees: List[str], model: str) -> str:
    system = topic_tree_system_prompt() + "\n你现在要合并多个局部层级全文结构，消除重复节点，把绕回的话题合并回原节点。"
    joined = "\n\n--- 局部层级全文结构分隔 ---\n\n".join(trees)
    return deepseek_chat(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "请把以下多个局部层级全文结构合并成一个全局层级全文结构。"
                    "只输出 Markdown 嵌套列表，不要 frontmatter，不要代码块。\n\n"
                    f"{joined}"
                ),
            },
        ],
        model=model,
        temperature=0.1,
        max_tokens=8192,
    )


def build_organized_article(
    cleaned_text: str,
    model: Optional[str],
    chunk_chars: int,
    glossary: str = "",
    merge_mode: str = "merge",
) -> str:
    chunks = split_by_sentence(cleaned_text, chunk_chars)
    if len(chunks) == 1:
        article = article_for_chunk(chunks[0], model, final=True, glossary=glossary)
        return repair_article_if_needed(article, cleaned_text, model)

    partials = []
    for index, chunk in enumerate(chunks, 1):
        log(f"DeepSeek 生成局部全文整理 {index}/{len(chunks)}")
        partials.append(article_for_chunk(chunk, model, final=False, glossary=glossary))
    if merge_mode == "concat":
        return "\n\n".join(part.strip() for part in partials if part.strip())
    article = merge_articles(partials, model, chunk_chars, glossary=glossary)
    return repair_article_if_needed(article, cleaned_text, model)


def article_for_chunk(text: str, model: Optional[str], final: bool, glossary: str = "") -> str:
    scope = "完整文本" if final else "文本片段"
    system = article_system_prompt()
    glossary_line = f"本次术语表：\n{glossary}\n\n" if glossary else ""
    draft = deepseek_chat(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"请把以下播客{scope}整理成可读的中文文章化 Markdown。\n\n"
                    f"{glossary_line}"
                    "只输出正文整理部分，不要 frontmatter，不要标题 H1，不要代码块。\n\n"
                    f"{text}"
                ),
            },
        ],
        model=model,
        temperature=0.15,
        max_tokens=8192,
    )
    return ground_article_against_material(draft, text, model, glossary=glossary)


def article_system_prompt() -> str:
    return (
        "你是中文行业研究编辑。任务不是输出逐字稿，也不是大幅摘要，而是把播客内容整理成可读文章。"
        "必须使用简体中文、书面表达，保留观点、案例、数字、人物、品牌、机制和关键细节。\n"
        "最重要的组织原则：忠实理解原文自己的结构和逻辑，再把它整理清楚。"
        "不要套固定模板；不要默认文章都必须有目的、机制、影响、案例、边界。"
        "原文怎么展开论证，就顺着它的内在问题、转折、递进、并列、回收关系来组织。"
        "先找全文真正的中心机制/核心命题，再安排背景材料；时代窗口、嘉宾经历、行业背景只能作为铺垫，不能抢占主轴。"
        "如果原文核心是在解释某套系统、中台、方法论或能力迁移，一级结构必须围绕这套系统如何形成、如何运转、如何迁移、在哪里失效来组织。"
        "如果原文只讲了三个问题，就只整理这三个问题；如果原文没有讲风险、边界、原因、方法或结论，就不要补。\n"
        "品牌、公司、产品和人物通常只是证据或案例，必须挂在相应逻辑段落下；"
        "但如果原文本身就是横向案例比较，也可以按案例组织；判断依据是原文逻辑，不是固定禁令。\n"
        "合并同类项：同一概念、同一案例、同一机制在不同时间点反复出现时，应合并到同一个逻辑段落中，不要在后文突然重新开一个平级段落。"
        "跨段绕回的话题要回收到原节点下，尤其是海外、本地化、风控、ROI、push、回流、红包等主题。\n"
        "排版规则：使用 Markdown 标题层级，一级整理标题用 ###，二级用 ####；"
        "一级标题必须来自原文的真实论证节点，用判断句或问题式标题表达该部分在全文中的作用。"
        "不要为了整齐、完整或像行业报告而重排成原文没有的框架。\n"
        "每段尽量 2-4 行，避免大段文字黏连；能用表格清楚对比时使用表格；"
        "机制、步骤、案例、风险只有在原文确实讨论时才可以用局部列表。不要使用巨大的多层嵌套列表。\n"
        "内容规则：可以合并重复口语和访谈衔接，但不要删掉有信息量的例子；"
        "不要把所有内容压成少数结论；严禁编造、补充或外推原文没有的事实。"
        "所有数字、比例、年份、商品例子、公司案例、政策、风险、边界、人物判断都必须来自原文；"
        "如果原文没有具体数字或例子，只能写原文已有的定性表达，不要用常识或行业知识补全。"
        "不能为了让文章更完整而加入“例如某平台曾经”“通常只有多少 SKU”“占比多少”等原文没有的信息。\n"
        "如果材料里有多个案例，先说明它们共同服务的上层观点，再分别作为例子展开细节。\n"
        "风格：像一篇经过整理的讲义/行业文章，读者不需要听过播客也能读懂。"
    )


def ground_article_against_material(article: str, material: str, model: Optional[str], glossary: str = "") -> str:
    if not article.strip():
        return article
    material_sample = material[:14000]
    article_sample = article[:14000]
    glossary_line = f"本次术语表：\n{glossary}\n\n" if glossary else ""
    return deepseek_chat(
        [
            {
                "role": "system",
                "content": (
                    "你是事实校对编辑。任务是把整理稿严格对齐材料，不做润色发挥。"
                    "只输出修订后的 Markdown 正文，不要解释，不要代码块。"
                    "必须保留整理稿中被材料支持的信息；必须删除或改写所有材料没有支持的猜测、幻觉、补充、行业常识、数字、例子、风险、边界和结论。"
                    "不要新增材料没有的标题或段落。不要为了完整性补齐框架。"
                    "如果整理稿结构偏离材料逻辑，也要按材料本身的论证关系重排。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请对照材料校正整理稿。\n\n"
                    f"{glossary_line}"
                    "材料：\n"
                    f"{material_sample}\n\n"
                    "待校正整理稿：\n"
                    f"{article_sample}"
                ),
            },
        ],
        model=model,
        temperature=0.0,
        max_tokens=8192,
    )


def merge_articles(articles: List[str], model: Optional[str], chunk_chars: int, glossary: str = "") -> str:
    round_no = 1
    current = articles
    while len("\n\n".join(current)) > chunk_chars or len(current) > 1:
        batches = batch_texts(current, chunk_chars)
        merged: List[str] = []
        for index, batch in enumerate(batches, 1):
            log(f"DeepSeek 合并全文整理 round {round_no} batch {index}/{len(batches)}")
            merged.append(merge_article_batch(batch, model, glossary=glossary))
        if len(merged) == len(current):
            return merge_article_batch(current, model, glossary=glossary)
        current = merged
        round_no += 1
    return current[0]


def merge_article_batch(articles: List[str], model: Optional[str], glossary: str = "") -> str:
    joined = "\n\n--- 局部整理分隔 ---\n\n".join(articles)
    system = (
        article_system_prompt()
        + "\n你现在要合并多个局部整理稿，去掉重复标题，合并绕回的话题，保留细节。"
        "输出仍然是文章化 Markdown，使用 ###/####、短段落、局部列表和必要表格。"
    )
    glossary_line = f"本次术语表：\n{glossary}\n\n" if glossary else ""
    merged = deepseek_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"请合并以下局部整理稿：\n\n{glossary_line}{joined}"},
        ],
        model=model,
        temperature=0.12,
        max_tokens=8192,
    )
    return ground_article_against_material(merged, joined, model, glossary=glossary)


LOGIC_HEADING_KEYWORDS = (
    "为什么",
    "目的",
    "影响",
    "机制",
    "怎么",
    "如何",
    "做法",
    "方法",
    "路径",
    "边界",
    "风险",
    "本质",
    "核心",
    "变化",
    "逻辑",
    "问题",
    "结果",
    "代价",
    "能力",
)


def extract_level3_headings(markdown: str) -> List[str]:
    headings = []
    for line in markdown.splitlines():
        match = re.match(r"^###\s+(?!#)(.+?)\s*$", line)
        if match:
            headings.append(match.group(1).strip())
    return headings


def looks_like_case_heading(title: str) -> bool:
    compact = re.sub(r"\s+", "", title)
    if not compact:
        return False
    if any(keyword in compact for keyword in LOGIC_HEADING_KEYWORDS):
        return False
    if re.match(r"^[\w\u4e00-\u9fff]{2,18}[：:]", compact):
        return True
    if "案例" in compact:
        return True
    return False


def is_case_led_article(markdown: str) -> bool:
    headings = extract_level3_headings(markdown)
    if len(headings) < 3:
        return False
    case_count = sum(1 for heading in headings if looks_like_case_heading(heading))
    return case_count >= 3 and case_count / len(headings) >= 0.5


NUMBER_FACT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|％|个|家|年|月|日|元|万|亿|倍|小时|分钟|SKU|sku|百分点)?")


def extract_number_facts(text: str) -> List[str]:
    return unique(match.group(0).strip() for match in NUMBER_FACT_RE.finditer(text) if match.group(0).strip())


def has_unsupported_number_facts(article: str, source_text: str) -> bool:
    source_numbers = set(extract_number_facts(source_text))
    for number in extract_number_facts(article):
        if number not in source_numbers:
            return True
    return False


def repair_article_if_needed(article: str, source_text: str, model: Optional[str]) -> str:
    reasons = []
    if is_case_led_article(article):
        reasons.append("偏案例/品牌分组")
    if has_unsupported_number_facts(article, source_text):
        reasons.append("出现原文没有的数字事实")
    if not reasons:
        return article
    log(f"检测到整理稿{ '、'.join(reasons) }，重新按论证逻辑和原文事实组织")
    sample = source_text[:10000]
    return deepseek_chat(
        [
            {"role": "system", "content": article_system_prompt()},
            {
                "role": "user",
                "content": (
                    "下面这版整理稿存在问题：可能按案例/品牌组织导致原文逻辑丢失，或加入了原文没有的数字/事实。"
                    "请重写为严格贴合原文结构和论证逻辑的文章化 Markdown。\n\n"
                    "硬性要求：\n"
                    "1. 先判断原文自己的中心问题、层级关系、并列关系、转折关系和回收关系，再决定标题结构。\n"
                    "2. 不套固定模板，不强行补目的、影响、机制、案例、边界；原文没讲的维度不要出现。\n"
                    "3. 品牌、公司、人物如何放置取决于原文逻辑：如果它们只是证据，就挂在上层观点下；如果原文本身就在做横向比较，才可作为主要结构。\n"
                    "4. 严禁添加原文没有的数字、比例、商品名、公司案例、风险、边界、推测和行业常识；只能使用原文事实。\n"
                    "5. 只输出正文整理部分，不要 frontmatter，不要 H1，不要代码块。\n\n"
                    f"原始材料节选：\n{sample}\n\n"
                    f"待重写整理稿：\n{article}"
                ),
            },
        ],
        model=model,
        temperature=0.12,
        max_tokens=8192,
    )


def generate_title(tree_markdown: str, fallback: Optional[str], model: str) -> str:
    sample = tree_markdown[:8000]
    title = deepseek_chat(
        [
            {"role": "system", "content": "你是中文文件标题生成器。输出一句话标题，12到28个中文字符，不能含标点解释。"},
            {
                "role": "user",
                "content": f"根据以下播客层级结构生成一个适合作为 Obsidian 文件名的标题。参考原标题：{fallback or '无'}\n\n{sample}",
            },
        ],
        model=model,
        temperature=0.2,
        max_tokens=128,
    )
    return sanitize_filename(title or fallback or "未命名播客")


def generate_note_metadata(
    cleaned_text: str,
    article_body: str,
    fallback_title: Optional[str],
    model: Optional[str],
) -> NoteMetadata:
    sample = (cleaned_text[:7000] + "\n\n" + article_body[:5000]).strip()
    system = (
        "你是中文行业研究编辑。根据播客清理稿和整理稿生成 Obsidian 笔记元信息。"
        "只输出 JSON，不要代码块。所有内容使用简体中文。"
        "严禁补充材料中没有的数字、案例、商品名、风险、边界或行业知识。"
    )
    user = (
        "请生成 JSON，字段如下：\n"
        "- title: 12到28个中文字符，适合作为文件名，不要标点解释。\n"
        "- tags: 3到6个粗粒度内容标签，不要写播客、逐字稿、层级全文、音频、笔记等流程标签；不要太细。\n"
        "- questions: 3到5个这篇内容试图回答的阅读问题，引导读者进入文章；不要提出材料没有回答的问题。\n"
        "- core_points: 4到6条核心观点，必须是判断句，帮助读者快速抓住文章立场；只能基于材料，不要加入新事实。\n"
        "- quotes: 3到6条原文金句，必须从材料中抽取或做极轻微清理，不能自己总结成漂亮话，不能补写原文没有的句子；保留原意和原文表达质感。\n"
        "- conclusion: 3到6条对照总结，逐条回应 questions 或核心讨论；不要补充材料外的风险、建议或判断。\n\n"
        f"参考原标题：{fallback_title or '无'}\n\n"
        f"材料：\n{sample}"
    )
    raw = deepseek_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        temperature=0.2,
        max_tokens=2048,
    )
    data = parse_json_object(raw)
    title = sanitize_filename(str(data.get("title") or fallback_title or "未命名播客"))
    tags = sanitize_tags(data.get("tags") or [])
    questions = sanitize_list(data.get("questions") or [], limit=5)
    core_points = sanitize_list(data.get("core_points") or [], limit=6)
    quotes = sanitize_list(data.get("quotes") or [], limit=6)
    conclusion = sanitize_list(data.get("conclusion") or [], limit=6)

    if not tags:
        tags = ["消费", "行业研究"]
    if not questions:
        questions = ["这期内容讨论的核心问题是什么？"]
    if not core_points:
        core_points = ["本文围绕核心议题整理了主要观点、案例与判断框架。"]
    if not quotes:
        quotes = []
    if not conclusion:
        conclusion = ["回到开头的问题，本文提供了可对照的观点、案例与判断框架。"]
    return NoteMetadata(
        title=title,
        tags=tags,
        questions=questions,
        core_points=core_points,
        quotes=quotes,
        conclusion=conclusion,
    )


def parse_json_object(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def sanitize_tags(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    blocked = {"播客", "逐字稿", "话题树", "层级全文", "音频", "笔记", "小宇宙"}
    tags: List[str] = []
    for value in values:
        tag = re.sub(r"[#\[\]\n\r\t,，/\\]+", " ", str(value)).strip()
        tag = re.sub(r"\s+", "", tag)
        if not tag or tag in blocked or len(tag) > 12:
            continue
        tags.append(tag)
    return unique(tags)[:6]


def sanitize_list(values: Any, limit: int) -> List[str]:
    if not isinstance(values, list):
        return []
    output = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if text:
            output.append(text)
    return output[:limit]


def parse_replacements(values: Optional[List[str]]) -> List[Tuple[str, str]]:
    replacements: List[Tuple[str, str]] = []
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"术语替换格式错误：{value}。请使用 --replace 错词=正确词")
        old, new = value.split("=", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise SystemExit(f"术语替换格式错误：{value}。请使用 --replace 错词=正确词")
        replacements.append((old, new))
    return replacements


def apply_replacements(text: str, replacements: List[Tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def strip_prompt_leaks(text: str) -> str:
    text = re.sub(r"<page_context>.*?</page_context>", "", text, flags=re.S)
    text = re.sub(r"以下是页面元信息，只能用于识别专名、节目背景和时间线，严禁原样输出到(?:清理|校对)结果：", "", text)
    text = re.sub(
        r"可参考的小宇宙页面上下文：.*?增长只是锦上添花，它不能雪中送炭\s*•\s*",
        "",
        text,
        flags=re.S,
    )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def glossary_context(replacements: List[Tuple[str, str]], keep_terms: Optional[List[str]]) -> str:
    lines = []
    if replacements:
        lines.append("强制术语替换：")
        lines.extend(f"- {old} => {new}" for old, new in replacements)
    if keep_terms:
        lines.append("必须保留的术语写法：")
        lines.extend(f"- {term}" for term in keep_terms if term.strip())
    return "\n".join(lines)


def generate_visual_briefs(article_body: str, model: Optional[str], max_visuals: int = 4) -> List[VisualSpec]:
    sample = article_body[:12000]
    system = (
        "你是中文文章配图策划，只为真正能降低理解门槛的复杂关系生成配图 brief。"
        "只输出 JSON，不要代码块。宁可少画，不要为了好看硬塞图。"
        "配图必须严格基于整理稿明说的关系，不能补充、推断或可视化整理稿没有讲的因果链。"
    )
    user = (
        f"请从整理稿中筛选最多 {max_visuals} 个值得画成结构说明图或小黑解释图的复杂结构。\n\n"
        "准入标准：能解释复杂结构；能降低理解门槛；能把文字里的关系画出来；"
        "能帮读者在 3 秒内抓住一个框架。不满足就不要输出。\n\n"
        "适合类型：organization_structure（组织结构）、ledger_formula（LTV/ROI账本）、"
        "flywheel（增长飞轮）、comparison（对比）、migration_path（能力迁移路径）。\n"
        "不适合：嘉宾介绍、单纯主题、情绪判断、已经很清楚的列表、装饰图、结尾隐喻图、"
        "只是把多个品牌/案例并排罗列的图。"
        "如果画案例，必须是为了说明一个上层机制、因果链路、分工结构或能力迁移，而不是展示案例本身。\n\n"
        "事实约束：prompt_seed 的 composition、suggested_elements、labels 只能使用整理稿里明确出现的概念。"
        "不要加入“用户增长、数据反哺、飞轮循环、降本、让利、规模效应”等整理稿没有明说的内容。\n\n"
        "JSON 字段：visuals 数组。每项包含：\n"
        "- kind: 上面五种类型之一\n"
        "- title: 8到18字\n"
        "- why_draw: 为什么这张图有助理解，必须具体\n"
        "- core_relationship: 这张图要画出的关系，1到2句话\n"
        "- prompt_seed: 对 ian-xiaohei-illustrations 生图有用的对象，包含 theme、structure_type、core_idea、composition、suggested_elements、labels。\n\n"
        f"整理稿：\n{sample}"
    )
    raw = deepseek_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        temperature=0.15,
        max_tokens=4096,
    )
    data = parse_json_object(raw)
    visuals = data.get("visuals", []) if isinstance(data, dict) else []
    specs: List[VisualSpec] = []
    allowed = {"organization_structure", "ledger_formula", "flywheel", "comparison", "migration_path"}
    if isinstance(visuals, list):
        for item in visuals[:max_visuals]:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip()
            if kind not in allowed:
                continue
            title = re.sub(r"\s+", "", str(item.get("title") or ""))[:24]
            why_draw = re.sub(r"\s+", " ", str(item.get("why_draw") or "")).strip()
            core_relationship = re.sub(r"\s+", " ", str(item.get("core_relationship") or "")).strip()
            prompt_seed = item.get("prompt_seed") if isinstance(item.get("prompt_seed"), dict) else {}
            if title and why_draw and core_relationship and prompt_seed:
                specs.append(
                    VisualSpec(
                        kind=kind,
                        title=title,
                        why_draw=why_draw,
                        core_relationship=core_relationship,
                        prompt_seed=prompt_seed,
                    )
                )
    return specs


def generate_opening_visual_brief(
    article_body: str,
    metadata: NoteMetadata,
    model: Optional[str],
) -> Optional[VisualSpec]:
    sample = article_body[:10000]
    system = (
        "你是中文长文开头配图策划。只在配图能概括全文核心问题、帮助读者进入文章时输出 brief。"
        "只输出 JSON，不要代码块。不能补充原文没有的信息。"
    )
    user = (
        "请判断这篇整理稿是否适合生成一张开头图。开头图不是装饰，必须覆盖全文核心问题，"
        "不能只表达第一部分，也不能只是“本期讲了什么”。如果不适合，输出 {\"visual\": null}。\n\n"
        "如果适合，输出 JSON：visual 对象包含 kind=opening、title、why_draw、core_relationship、prompt_seed。"
        "prompt_seed 包含 theme、core_idea、composition、suggested_elements、labels。"
        "画面应是一个清晰隐喻或解释场景，元素少、关系明确，不要加入整理稿没有的概念。\n\n"
        f"标题：{metadata.title}\n"
        f"核心观点：{'；'.join(metadata.core_points[:5])}\n\n"
        f"整理稿：\n{sample}"
    )
    raw = deepseek_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        temperature=0.15,
        max_tokens=2048,
    )
    data = parse_json_object(raw)
    item = data.get("visual") if isinstance(data, dict) else None
    if not isinstance(item, dict):
        return None
    title = re.sub(r"\s+", "", str(item.get("title") or ""))[:24]
    why_draw = re.sub(r"\s+", " ", str(item.get("why_draw") or "")).strip()
    core_relationship = re.sub(r"\s+", " ", str(item.get("core_relationship") or "")).strip()
    prompt_seed = item.get("prompt_seed") if isinstance(item.get("prompt_seed"), dict) else {}
    if not title or not why_draw or not core_relationship or not prompt_seed:
        return None
    return VisualSpec(
        kind="opening",
        title=title,
        why_draw=why_draw,
        core_relationship=core_relationship,
        prompt_seed=prompt_seed,
    )


def make_visual_prompt(spec: VisualSpec) -> str:
    seed = spec.prompt_seed
    labels = seed.get("labels", [])
    if isinstance(labels, list):
        label_text = "；".join(str(item) for item in labels[:8])
    else:
        label_text = str(labels)
    elements = seed.get("suggested_elements", [])
    if isinstance(elements, list):
        element_text = "；".join(str(item) for item in elements[:8])
    else:
        element_text = str(elements)

    if spec.kind == "opening":
        return (
            "生成一张 16:9 中文文章开头配图，白底、干净、手绘感，小黑风格但不要恐怖、不要幼稚。"
            "画面用于吸引读者进入一篇商业分析文章，同时解释全文核心问题。"
            "只保留少量中文标签，标签必须清晰可读，不要写读图说明、不要写“原文说法”。"
            "不要堆砌物品，不要装饰性元素。"
            f"主题：{seed.get('theme', spec.title)}。"
            f"核心关系：{spec.core_relationship}。"
            f"构图：{seed.get('composition', '')}。"
            f"可用元素：{element_text}。"
            f"允许出现的标签：{label_text}。"
        )

    return (
        "请生成一张专业中文商业信息图，用于深度商业分析文章正文。"
        "风格：白色背景、扁平现代、咨询报告质感、分区清晰、留白充足。"
        "只用 2-3 个克制主色系，浅色卡片背景、深色标题文字；使用极简线性图标。"
        "禁止渐变、阴影、发光、3D、玻璃拟态、装饰色块、乱码和无意义图标。"
        "不要写读图说明，不要加入未提供的数字、因果或结论。"
        "中文文字必须少而准，宁可少字，也不能错字、漏括号或挤出边界。"
        f"图表标题：{spec.title}。"
        f"要解释的关系：{spec.core_relationship}。"
        f"为什么需要画：{spec.why_draw}。"
        f"结构类型：{spec.kind}。"
        f"构图要求：{seed.get('composition', '')}。"
        f"可用元素：{element_text}。"
        f"允许出现的标签：{label_text}。"
    )


def load_image_config() -> Dict[str, Any]:
    api_key = env_first("GPT_IMAGE_API_KEY", "GPT_IMAGE2_API_KEY", "OPENAI_IMAGE_API_KEY", "IMAGE_API_KEY", "api_key")
    if not api_key:
        return {}
    return {
        "api_key": api_key,
        "base_url": os.environ.get("GPT_IMAGE_BASE_URL", GPT_IMAGE_BASE_URL).rstrip("/"),
        "model": os.environ.get("GPT_IMAGE_MODEL", GPT_IMAGE_MODEL),
        "size": os.environ.get("GPT_IMAGE_SIZE", "16:9"),
        "resolution": os.environ.get("GPT_IMAGE_RESOLUTION", "1k"),
        "poll_interval": int(os.environ.get("GPT_IMAGE_POLL_INTERVAL", "5")),
        "timeout": int(os.environ.get("GPT_IMAGE_TIMEOUT", "600")),
    }


def submit_gpt_image(prompt: str, cfg: Dict[str, Any]) -> str:
    url = f"{cfg['base_url']}/images/generations"
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    payload = {
        "model": cfg["model"],
        "prompt": prompt,
        "n": 1,
        "size": cfg["size"],
        "resolution": cfg["resolution"],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"GPT-Image submit 失败：HTTP {resp.status_code} {resp.text[:500]}")
    data = resp.json()
    try:
        return data["data"][0]["task_id"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"GPT-Image submit 返回格式异常：{redact_json(data)}") from exc


def poll_gpt_image(task_id: str, cfg: Dict[str, Any]) -> str:
    url = f"{cfg['base_url']}/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    deadline = time.time() + cfg["timeout"]
    while time.time() < deadline:
        resp = requests.get(url, headers=headers, timeout=60)
        if not resp.ok:
            raise RuntimeError(f"GPT-Image query 失败：HTTP {resp.status_code} {resp.text[:500]}")
        payload = resp.json()
        data = payload.get("data", {})
        status = data.get("status")
        if status == "completed":
            try:
                return data["result"]["images"][0]["url"][0]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"GPT-Image completed 返回格式异常：{redact_json(payload)}") from exc
        if status == "failed":
            error = data.get("error", {})
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise RuntimeError(f"GPT-Image 任务失败：{message or 'unknown error'}")
        time.sleep(cfg["poll_interval"])
    raise RuntimeError("GPT-Image 任务超时。")


def download_image(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile("wb", dir=str(output_path.parent), delete=False) as tmp:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    tmp.write(chunk)
            tmp_path = Path(tmp.name)
    os.replace(tmp_path, output_path)


def generate_visual_images(
    article_body: str,
    metadata: NoteMetadata,
    output_path: Path,
    model: Optional[str],
    workdir: Path,
    max_visuals: int = 4,
) -> List[GeneratedVisual]:
    cfg = load_image_config()
    if not cfg:
        log("未配置 GPT-Image API key，跳过 PNG 生成；仍会保留 visual_briefs.json。")
        return []

    specs: List[VisualSpec] = []
    opening = generate_opening_visual_brief(article_body, metadata, model)
    if opening:
        specs.append(opening)
    specs.extend(generate_visual_briefs(article_body, model, max_visuals=max_visuals))
    (workdir / "visual_briefs.json").write_text(
        json.dumps([spec.__dict__ for spec in specs], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not specs:
        log("没有通过准入标准的配图 brief，跳过 PNG 生成。")
        return []

    asset_dir = output_path.parent / "assets" / output_path.stem
    generated: List[GeneratedVisual] = []
    for index, spec in enumerate(specs, 1):
        filename = f"{index:02d}-{safe_asset_name(spec.title)}.png"
        image_path = asset_dir / filename
        try:
            log(f"生成配图 {index}/{len(specs)}：{spec.title}")
            prompt = make_visual_prompt(spec)
            (workdir / f"visual_prompt_{index:02d}.txt").write_text(prompt, encoding="utf-8")
            task_id = submit_gpt_image(prompt, cfg)
            image_url = poll_gpt_image(task_id, cfg)
            download_image(image_url, image_path)
            rel_path = image_path.relative_to(output_path.parent).as_posix()
            generated.append(GeneratedVisual(rel_path=rel_path, caption=spec.title, abs_path=image_path))
        except Exception as exc:
            log(f"配图生成失败，已跳过「{spec.title}」：{exc}")
    return generated


def safe_asset_name(value: str) -> str:
    name = sanitize_filename(value)
    name = re.sub(r"\s+", "-", name)
    return name[:48] or uuid.uuid4().hex[:8]


def redact_json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    text = re.sub(r"(sk-[A-Za-z0-9_-]{8})[A-Za-z0-9_-]+", r"\1[REDACTED]", text)
    return text[:1000]


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|#\[\]\n\r\t]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value[:80] or "未命名播客"


def yaml_escape(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_markdown(
    source_url: str,
    audio_url: Optional[str],
    metadata: NoteMetadata,
    article_body: str,
    visuals: Optional[List[Tuple[str, str]]] = None,
    draft: bool = False,
) -> str:
    visuals = visuals or []
    today = dt.date.today().isoformat()
    frontmatter = [
        "---",
        f"source: {yaml_escape(source_url)}",
        f"audio: {yaml_escape(audio_url or '')}",
        f"date: {today}",
        f"draft: {'true' if draft else 'false'}",
        "tags:",
        *[f"  - {tag}" for tag in metadata.tags],
        "---",
        "",
        f"# {metadata.title}",
        "",
        "## 这篇内容在回答什么问题",
        "",
        *[f"- {question}" for question in metadata.questions],
        "",
        "## 核心观点",
        "",
        *[f"- {point}" for point in metadata.core_points],
        "",
    ]
    if metadata.quotes:
        frontmatter.extend(
            [
                "## 原文金句",
                "",
                *[f"> {quote}" for quote in metadata.quotes],
                "",
            ]
        )
    if visuals:
        frontmatter.extend(["## 结构图", ""])
        for rel, caption in visuals:
            frontmatter.extend([f"![]({rel})", ""])
            if caption:
                frontmatter.extend([f"*{caption}*", ""])
    frontmatter.extend(
        [
            "## 全文整理",
            "",
        ]
    )
    conclusion = [
        "",
        "## 对照总结",
        "",
        *[f"- {item}" for item in metadata.conclusion],
        "",
    ]
    return "\n".join(frontmatter) + article_body.strip() + "\n" + "\n".join(conclusion)


def build_hierarchy_markdown(
    source_url: str,
    audio_url: Optional[str],
    metadata: NoteMetadata,
    hierarchy_body: str,
    draft: bool = False,
) -> str:
    today = dt.date.today().isoformat()
    frontmatter = [
        "---",
        f"source: {yaml_escape(source_url)}",
        f"audio: {yaml_escape(audio_url or '')}",
        f"date: {today}",
        f"draft: {'true' if draft else 'false'}",
        "version: 层级全文版",
        "tags:",
        *[f"  - {tag}" for tag in metadata.tags],
        "---",
        "",
        f"# {metadata.title}｜层级全文版",
        "",
        "## 使用说明",
        "",
        "- 以下内容按逻辑关系重组为可折叠层级，不按对话时间线逐段排列。",
        "- 子节点表示展开、举例、论证或细化；平级节点表示共同支撑上层观点。",
        "",
        "## 层级全文",
        "",
    ]
    return "\n".join(frontmatter) + hierarchy_body.strip() + "\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


class Workdir:
    def __init__(self, keep: bool, base_dir: Optional[Path] = None) -> None:
        parent = base_dir.expanduser().resolve() if base_dir else None
        if parent:
            parent.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.path = Path(tempfile.mkdtemp(prefix="podcast-to-obsidian-", dir=str(parent) if parent else None))

    def cleanup(self) -> None:
        if not self.keep and self.path.exists():
            shutil.rmtree(self.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="小宇宙播客/音频/逐字稿 -> Doubao/Whisper -> DeepSeek整理 -> Obsidian")
    parser.add_argument("link", nargs="?", help="小宇宙播客链接")
    parser.add_argument("--source-url", help="写入 frontmatter 的来源链接；使用 --audio-url/--audio-file 时可传")
    parser.add_argument("--audio-url", help="音频直链；用于绕过小宇宙页面解析")
    parser.add_argument("--audio-file", type=Path, help="本地音频文件；用于绕过下载")
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--output-subdir", default=DEFAULT_SUBDIR)
    parser.add_argument("--asr", choices=("doubao", "whisper"), default="doubao", help="音频识别引擎；正式默认 doubao，whisper 作为本地 fallback")
    parser.add_argument("--quality", choices=("smoke", "draft", "final"), default="final", help="smoke=tiny 快速验链路；draft=small；final=large-v3 正式稿")
    parser.add_argument("--model", help="faster-whisper 模型名；不传时由 --quality 决定")
    parser.add_argument("--language", default="zh", help="Whisper 语言；不确定可传空字符串")
    parser.add_argument("--compute-type", default="int8", help="CPU 推荐 int8；也可用 float32")
    parser.add_argument("--deepseek-model", help="DeepSeek 模型名；不传则读取 DEEPSEEK_MODEL 或使用 deepseek-chat")
    parser.add_argument("--chunk-chars", type=int, default=12000)
    parser.add_argument("--env-file", type=Path, action="append", help="加载 env 文件；可重复传入，也兼容误存成 RTF 的文本")
    parser.add_argument("--doubao-poll-interval", type=int, default=10)
    parser.add_argument("--doubao-timeout", type=int, default=3600)
    parser.add_argument("--keep-tos-object", action="store_true", help="调试用：不删除上传到 TOS 的临时音频")
    parser.add_argument(
        "--visual-mode",
        choices=("none", "brief", "auto"),
        default="auto",
        help="auto=生成 PNG 并插入 Markdown；brief=只生成配图 brief；none=不处理配图",
    )
    parser.add_argument(
        "--output-version",
        choices=("article", "hierarchy", "both"),
        default="both",
        help="article=只输出精读整理版；hierarchy=只输出层级全文版；both=两个版本都输出",
    )
    parser.add_argument("--article-merge", choices=("merge", "concat"), default="merge", help="merge=全局合并去重；concat=长稿模式，串接局部整理以优先保留细节")
    parser.add_argument("--work-base", type=Path, help="中间文件目录；默认使用系统临时目录")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--skip-whisper", type=Path, help="调试用：直接传已有逐字稿 txt")
    parser.add_argument("--skip-cleaning", action="store_true", help="调试用：把 --skip-whisper 输入视为已清理/校对文本，直接进入全文整理")
    parser.add_argument("--replace", action="append", help="术语强制替换，可重复传入，格式：错词=正确词")
    parser.add_argument("--keep-term", action="append", help="要求 DeepSeek 保留的术语写法，可重复传入，例如 push")
    return parser.parse_args()


def resolve_whisper_model(quality: str, model: Optional[str]) -> Tuple[str, bool]:
    if model:
        resolved = model
    elif quality == "smoke":
        resolved = "tiny"
    elif quality == "draft":
        resolved = "small"
    else:
        resolved = "large-v3"

    draft = quality != "final" or resolved in {"tiny", "base", "small"}
    if quality == "final" and resolved in {"tiny", "base"}:
        raise SystemExit("final 质量不允许使用 tiny/base。请改用 --quality smoke，或使用 --model large-v3。")
    if quality == "final" and resolved == "small":
        log("警告：final 质量使用 small 模型，转录准确率可能不足；建议 large-v3。")
    return resolved, draft


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    replacements = parse_replacements(args.replace)
    glossary = glossary_context(replacements, args.keep_term)

    if not args.link and not args.audio_url and not args.audio_file and not args.skip_whisper:
        raise SystemExit("请提供小宇宙链接，或使用 --audio-url/--audio-file/--skip-whisper。")

    source_url = args.source_url or args.link or ""
    needs_page_context = bool(source_url.startswith(("http://", "https://")) and (args.audio_file or args.audio_url or args.skip_whisper))
    needs_asr = not bool(args.skip_whisper)
    uses_whisper = needs_asr and args.asr == "whisper"
    uses_doubao = needs_asr and args.asr == "doubao"
    require_deps(
        skip_whisper=not uses_whisper,
        needs_parse=bool(args.link and not args.audio_url and not args.audio_file and not args.skip_whisper) or needs_page_context,
        needs_download=bool(args.audio_url),
        needs_doubao=uses_doubao,
    )

    audio_url = args.audio_url
    page_title = None
    page_context = ""
    whisper_model, whisper_draft = resolve_whisper_model(args.quality, args.model)
    draft = whisper_draft if uses_whisper else args.quality != "final"

    if source_url.startswith(("http://", "https://")) and not args.link:
        page_title, page_context = parse_page_metadata(source_url)

    workdir_ctx = Workdir(keep=args.keep_workdir, base_dir=args.work_base)
    workdir = workdir_ctx.path

    if args.audio_file:
        audio_file = args.audio_file.expanduser().resolve()
        if not audio_file.exists():
            raise SystemExit(f"本地音频文件不存在：{audio_file}")
        audio_url = audio_url or str(audio_file)
    else:
        if not args.skip_whisper and not audio_url:
            episode = parse_xiaoyuzhou(args.link)
            source_url = episode.source_url
            audio_url = episode.audio_url
            page_title = episode.page_title
            page_context = episode.page_context
        if args.asr == "doubao" and audio_url and audio_url.startswith(("http://", "https://")):
            audio_file = Path(urlparse(audio_url).path or "audio.m4a")
        else:
            audio_file = download_audio(audio_url, workdir) if audio_url else Path()

    log(f"工作目录：{workdir}")

    try:
        if args.skip_whisper:
            transcript = args.skip_whisper.read_text(encoding="utf-8")
            transcript = strip_prompt_leaks(transcript)
            transcript = apply_replacements(transcript, replacements)
            audio_url = audio_url or ""
        elif args.asr == "doubao":
            if audio_url and audio_url.startswith(("http://", "https://")):
                transcript = doubao_transcribe_url(
                    audio_url,
                    audio_file.name,
                    workdir,
                    poll_interval=args.doubao_poll_interval,
                    timeout=args.doubao_timeout,
                )
            else:
                transcript = doubao_transcribe(
                    audio_file,
                    workdir,
                    poll_interval=args.doubao_poll_interval,
                    timeout=args.doubao_timeout,
                    keep_object=args.keep_tos_object,
                )
            (workdir / "transcript.txt").write_text(transcript, encoding="utf-8")
        else:
            language = args.language.strip() or None
            initial_prompt = build_whisper_prompt(page_context, page_title)
            (workdir / "whisper_prompt.txt").write_text(initial_prompt, encoding="utf-8")
            transcript = transcribe(audio_file, whisper_model, language, args.compute_type, initial_prompt=initial_prompt)
            transcript = apply_replacements(transcript, replacements)
            (workdir / "transcript.txt").write_text(transcript, encoding="utf-8")

        if args.skip_cleaning:
            if not args.skip_whisper:
                raise SystemExit("--skip-cleaning 只能和 --skip-whisper 一起使用。")
            cleaned = transcript
            cleaned = strip_prompt_leaks(cleaned)
            (workdir / "proofread.txt").write_text(cleaned, encoding="utf-8")
        else:
            cleaned = clean_transcript(transcript, args.deepseek_model, args.chunk_chars, page_context=page_context)
            cleaned = strip_prompt_leaks(cleaned)
            cleaned = apply_replacements(cleaned, replacements)
            (workdir / "cleaned.txt").write_text(cleaned, encoding="utf-8")

            cleaned = proofread_cleaned_text(cleaned, args.deepseek_model, args.chunk_chars, page_context=page_context)
            cleaned = strip_prompt_leaks(cleaned)
            cleaned = apply_replacements(cleaned, replacements)
            (workdir / "proofread.txt").write_text(cleaned, encoding="utf-8")

        article_body = ""
        if args.output_version in {"article", "both"}:
            article_body = build_organized_article(
                cleaned,
                args.deepseek_model,
                args.chunk_chars,
                glossary=glossary,
                merge_mode=args.article_merge,
            )
            article_body = apply_replacements(article_body, replacements)
            (workdir / "organized_article.md").write_text(article_body, encoding="utf-8")

        hierarchy_body = ""
        if args.output_version in {"hierarchy", "both"}:
            hierarchy_body = build_topic_tree(cleaned, args.deepseek_model, args.chunk_chars)
            hierarchy_body = apply_replacements(hierarchy_body, replacements)
            (workdir / "hierarchy_full.md").write_text(hierarchy_body, encoding="utf-8")

        metadata_source = article_body or hierarchy_body
        metadata = generate_note_metadata(cleaned, metadata_source, page_title, args.deepseek_model)
        date_prefix = dt.date.today().isoformat()
        written_paths: List[Path] = []

        if args.output_version in {"article", "both"}:
            article_path = args.vault.expanduser() / args.output_subdir / f"{date_prefix} {metadata.title} 精读整理版.md"
            markdown_visuals: List[Tuple[str, str]] = []
            if args.visual_mode == "brief":
                visual_specs = generate_visual_briefs(article_body, args.deepseek_model, max_visuals=4)
                (workdir / "visual_briefs.json").write_text(
                    json.dumps([spec.__dict__ for spec in visual_specs], ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            elif args.visual_mode == "auto":
                generated_visuals = generate_visual_images(
                    article_body,
                    metadata,
                    article_path,
                    args.deepseek_model,
                    workdir,
                    max_visuals=4,
                )
                markdown_visuals = [(item.rel_path, item.caption) for item in generated_visuals]
            markdown = build_markdown(source_url, audio_url, metadata, article_body, visuals=markdown_visuals, draft=draft)
            atomic_write(article_path, markdown)
            written_paths.append(article_path)

        if args.output_version in {"hierarchy", "both"}:
            hierarchy_path = args.vault.expanduser() / args.output_subdir / f"{date_prefix} {metadata.title} 层级全文版.md"
            hierarchy_markdown = build_hierarchy_markdown(source_url, audio_url, metadata, hierarchy_body, draft=draft)
            atomic_write(hierarchy_path, hierarchy_markdown)
            written_paths.append(hierarchy_path)

        for path in written_paths:
            log(f"已写入 Obsidian：{path}")
            print(path)
    finally:
        if args.keep_workdir:
            log(f"保留工作目录：{workdir}")
        else:
            workdir_ctx.cleanup()


if __name__ == "__main__":
    main()
