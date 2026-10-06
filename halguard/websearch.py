"""联网搜索模块（强制联网模式）。

开关打开时，HalGuard 在把请求转发给本地 LLM 之前先执行网络搜索，
把搜索结果（标题/摘要/URL）作为[网络搜索结果]上下文注入 system 消息，
并强制模型"仅依据搜索结果作答、标注来源编号"，从源头抑制编造。

搜索源策略（面向中国大陆网络环境，按优先级回退）：
  1. 360 搜索（www.so.com）——国内可直连且对程序化请求返回真实相关结果（首选）
  2. Bing（cn.bing.com）——部分网络可用；注意：Bing 对无 cookie 爬虫可能返回
     HTTP 200 但内容被污染（标题正确、结果无关），因此必须过相关性门控
  3. DuckDuckGo（html.duckduckgo.com）——国际网络环境下的回退（国内被墙）

防污染门控：每条结果都会与查询词做内容单元（CJK bigram/拉丁词）重合检查，
与查询完全无关的结果（反爬污染）会被剔除；全部无关则视为搜索失败。

仅使用 httpx（已是项目依赖），无新增第三方依赖。
"""
from __future__ import annotations

import re
from typing import Dict, List
from urllib.parse import unquote, urlparse, parse_qs

import httpx

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# 常见功能字（用于过滤无意义的内容单元）
_STOP = set("的是了和与或也都就而及在把被个这那等有还更最着过将已多少什么怎么如何为什么哪")


def _strip(html: str) -> str:
    """去 HTML 标签并压缩空白。"""
    return _WS_RE.sub(" ", _TAG_RE.sub("", html)).strip()


# ------------------------------------------------- 相关性门控（防反爬污染）----

def content_units(text: str) -> set:
    """查询文本的内容单元：拉丁/数字词 + 中日韩相邻 bigram（过滤含功能字的 bigram）。"""
    units = set()
    for w in re.findall(r"[A-Za-z0-9]+", text):
        low = w.lower()
        if low not in _STOP and len(low) >= 2:
            units.add(low)
    cjk = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    for i in range(len(cjk) - 1):
        bg = cjk[i] + cjk[i + 1]
        if not (set(bg) & _STOP):
            units.add(bg)
    return units


def is_relevant(query: str, text: str) -> bool:
    """判断结果文本与查询是否相关：至少共享一个内容单元。"""
    q_units = content_units(query)
    if not q_units:
        return True  # 无法判断时不做过滤
    t_units = content_units(text)
    return bool(q_units & t_units)


def filter_relevant(query: str, results: List[Dict]) -> List[Dict]:
    """剔除与查询无关的结果（Bing 反爬污染的典型特征：HTTP 200 但结果无关）。"""
    return [r for r in results if is_relevant(query, f"{r.get('title', '')} {r.get('snippet', '')}")]


# ---------------------------------------------------------------- 360 搜索 ----

# <li class="res-list"> 块内：<h3><a href="URL">标题</a></h3>，摘要在 <p class="res-desc">
_SO_BLOCK_SPLIT = re.compile(r'<li[^>]*class="[^"]*\bres-list\b[^"]*"[^>]*>', re.I)
_SO_TITLE_RE = re.compile(r'<h3[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)
_SO_SNIPPET_RE = re.compile(r'<p[^>]*class="[^"]*\bres-desc\b[^"]*"[^>]*>(.*?)</p>', re.S | re.I)


def parse_so360_html(html: str, max_results: int = 5) -> List[Dict]:
    """解析 360 搜索结果页 HTML -> [{title, url, snippet}]（纯函数，可离线测试）。"""
    out: List[Dict] = []
    blocks = _SO_BLOCK_SPLIT.split(html)[1:]
    for block in blocks:
        m = _SO_TITLE_RE.search(block)
        if not m:
            continue
        url, title = m.group(1), _strip(m.group(2))
        if url.startswith("/"):
            url = "https://www.so.com" + url
        if not title or url.startswith("https://www.so.com/s?"):
            continue  # 跳过"其他人还搜了"等内部推荐
        snip = ""
        sm = _SO_SNIPPET_RE.search(block)
        if sm:
            snip = _strip(sm.group(1))
        if not url.startswith("http"):
            continue
        out.append({"title": title, "url": url, "snippet": snip})
        if len(out) >= max_results:
            break
    return out


def search_so360(query: str, max_results: int = 5, timeout: float = 10.0) -> List[Dict]:
    r = httpx.get(
        "https://www.so.com/s",
        params={"q": query, "pn": "1"},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
    )
    r.raise_for_status()
    return parse_so360_html(r.text, max_results)


# ---------------------------------------------------------------- Bing ----

_BING_BLOCK_RE = re.compile(r'<li[^>]*class="[^"]*\bb_algo\b[^"]*"[^>]*>(.*?)</li>', re.S | re.I)
_BING_URL_RE = re.compile(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)
_BING_SNIPPET_RE = re.compile(r'<p[^>]*>(.*?)</p>', re.S | re.I)


def parse_bing_html(html: str, max_results: int = 5) -> List[Dict]:
    """解析 Bing 搜索结果页 HTML -> [{title, url, snippet}]（纯函数，可离线测试）。"""
    out: List[Dict] = []
    for block in _BING_BLOCK_RE.findall(html):
        m = _BING_URL_RE.search(block)
        if not m:
            continue
        url, title = m.group(1), _strip(m.group(2))
        snip = ""
        sm = _BING_SNIPPET_RE.search(block)
        if sm:
            snip = _strip(sm.group(1))
        if not title or not url.startswith("http"):
            continue
        out.append({"title": title, "url": url, "snippet": snip})
        if len(out) >= max_results:
            break
    return out


def search_bing(query: str, max_results: int = 5, timeout: float = 10.0) -> List[Dict]:
    r = httpx.get(
        "https://cn.bing.com/search",
        params={"q": query, "mkt": "zh-CN", "count": str(max_results)},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
    )
    r.raise_for_status()
    return parse_bing_html(r.text, max_results)


# ------------------------------------------------------- DuckDuckGo ----

_DDG_TITLE_RE = re.compile(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)
_DDG_SNIPPET_RE = re.compile(r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', re.S | re.I)


def _unwrap_ddg_url(url: str) -> str:
    """DDG 结果链接常是 /l/?uddg=<urlencoded> 重定向，解出真实 URL。"""
    if "uddg=" in url:
        try:
            qs = parse_qs(urlparse(url).query)
            real = qs.get("uddg", [""])[0]
            if real:
                return unquote(real)
        except Exception:
            pass
    if url.startswith("//"):
        return "https:" + url
    return url


def parse_ddg_html(html: str, max_results: int = 5) -> List[Dict]:
    """解析 DuckDuckGo HTML 版结果页 -> [{title, url, snippet}]（纯函数，可离线测试）。"""
    out: List[Dict] = []
    titles = _DDG_TITLE_RE.findall(html)
    snippets = [_strip(s) for s in _DDG_SNIPPET_RE.findall(html)]
    for i, (url, title) in enumerate(titles):
        real = _unwrap_ddg_url(url)
        if not title.strip() or not real.startswith("http"):
            continue
        out.append({
            "title": _strip(title),
            "url": real,
            "snippet": snippets[i] if i < len(snippets) else "",
        })
        if len(out) >= max_results:
            break
    return out


def search_ddg(query: str, max_results: int = 5, timeout: float = 10.0) -> List[Dict]:
    r = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
    )
    r.raise_for_status()
    return parse_ddg_html(r.text, max_results)


# ---------------------------------------------------------------- 入口 ----

_SOURCES = ("so360", "bing", "ddg")


def web_search(query: str, max_results: int = 5, timeout: float = 10.0) -> List[Dict]:
    """执行联网搜索：so360 -> Bing -> DuckDuckGo 依次回退；全失败返回 []。

    所有结果经过相关性门控（防 Bing 式反爬污染：HTTP 200 但结果无关）。

    返回: [{"title": str, "url": str, "snippet": str}, ...]
    """
    if not query or not query.strip():
        return []
    fns = {"so360": search_so360, "bing": search_bing, "ddg": search_ddg}
    for name in _SOURCES:
        try:
            results = fns[name](query, max_results, timeout)
        except Exception:
            continue
        relevant = filter_relevant(query, results)
        if relevant:
            return relevant[:max_results]
    return []
