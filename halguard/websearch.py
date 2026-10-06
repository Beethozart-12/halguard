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


# ---------------------------------------------------------------- 360 新闻（时效性） ----

# news.so.com/ns?q=..&sort=time 按时间排序的新闻条目：
# <li class="...res-list..." data-url="URL"><a href="URL" title="标题">...
#   <div class="g-txt-inner">标题</div> ... <p class="summary ...">摘要</p>
#   <span class="g-linkinfo-txt g-c-gray time">2025-09-03 13:31 / 8小时前</span>
_NEWS_BLOCK_SPLIT = re.compile(r'<li[^>]*class="[^"]*\bres-list\b[^"]*"[^>]*>', re.I)
_NEWS_URL_RE = re.compile(r'<a[^>]+href="(https?://[^"]+)"', re.I)
_NEWS_TITLE_RE = re.compile(r'title="([^"]+)"', re.I)
_NEWS_TITLE_ALT_RE = re.compile(r'<div class="g-txt-inner[^"]*">(.*?)</div>', re.S | re.I)
_NEWS_SNIPPET_RE = re.compile(r'<p[^>]*class="[^"]*\bsummary\b[^"]*"[^>]*>(.*?)</p>', re.S | re.I)
_NEWS_TIME_RE = re.compile(r'class="[^"]*\btime\b[^"]*">([^<]+)</span>', re.I)


def parse_so360_news_html(html: str, max_results: int = 5) -> List[Dict]:
    """解析 360 新闻搜索（时间排序）结果页 -> [{title, url, snippet, time}]（纯函数，可离线测试）。"""
    out: List[Dict] = []
    blocks = _NEWS_BLOCK_SPLIT.split(html)[1:]
    for block in blocks:
        m = _NEWS_URL_RE.search(block)
        if not m:
            continue
        url = m.group(1)
        tm = _NEWS_TITLE_RE.search(block)
        if tm:
            title = _strip(tm.group(1))
        else:
            ta = _NEWS_TITLE_ALT_RE.search(block)
            title = _strip(ta.group(1)) if ta else ""
        snip = ""
        sm = _NEWS_SNIPPET_RE.search(block)
        if sm:
            snip = _strip(sm.group(1))
        time_str = ""
        tm2 = _NEWS_TIME_RE.search(block)
        if tm2:
            time_str = _strip(tm2.group(1))
        if not title or not url.startswith("http"):
            continue
        out.append({"title": title, "url": url, "snippet": snip, "time": time_str})
        if len(out) >= max_results:
            break
    return out


def search_so360_news(query: str, max_results: int = 5, timeout: float = 10.0) -> List[Dict]:
    """360 新闻垂直搜索，按时间排序（sort=time），返回最新新闻条目。"""
    r = httpx.get(
        "https://news.so.com/ns",
        params={"q": query, "pn": "1", "sort": "time"},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
    )
    r.raise_for_status()
    return parse_so360_news_html(r.text, max_results)


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

# 时效性查询的启发式关键词：命中则优先走新闻垂直源（时间排序）
TIMELY_KEYWORDS = (
    "今天", "今日", "今天有", "最新", "最近", "现在", "当前", "今年", "本月", "昨天",
    "刚刚", "新闻", "热搜", "实况", "实时", "天气", "气温", "汇率", "股价", "行情",
    "比分", "发布了", "上线", "发布", "更新", "几号", "多少号", "几点", "价格",
)


def is_timely_query(query: str) -> bool:
    """判断查询是否具有时效性（应优先返回最新信息）。"""
    return any(k in query for k in TIMELY_KEYWORDS)


# ---- 新鲜度过滤：解析发布时间，剔除陈旧内容（新闻池里常见标题含"今天"的旧闻）----

import datetime as _dt

# 统一用东八区"现在"：容器环境默认 UTC，会让日期新鲜度判断慢 8 小时
_CST = _dt.timezone(_dt.timedelta(hours=8))


def _now() -> _dt.datetime:
    return _dt.datetime.now(_CST).replace(tzinfo=None)


def is_fresh_time(t: str, max_age_days: int = 3) -> bool | None:
    """判断发布时间字符串是否在 max_age_days 天内。

    返回 True（新鲜）/ False（陈旧）/ None（无法解析，不过滤）。
    支持中文相对时间（"8小时前"/"3天前"/"昨天"）、绝对日期（"2026-10-07 13:31"）、
    纯时刻（"06:05"，视为今天）与 "10-05"/"10月5日" 形式（按当年计算）。
    """
    if not t:
        return None
    t = t.strip()
    now = _now()
    m = re.search(r"(\d+)\s*分钟前", t)
    if m:
        return True
    m = re.search(r"(\d+)\s*小时前", t)
    if m:
        return True
    m = re.search(r"(\d+)\s*天前", t)
    if m:
        return int(m.group(1)) <= max_age_days
    if "昨天" in t or "昨日" in t:
        return max_age_days >= 1
    if "前天" in t:
        return max_age_days >= 2
    m = re.search(r"(\d+)\s*周前", t)
    if m:
        return False  # 周级粒度默认超过 3 天
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        try:
            d = _dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return (now - d).days <= max_age_days
        except ValueError:
            return None
    if re.fullmatch(r"\d{1,2}:\d{2}", t):
        return True  # 纯时刻（无日期）按今天发布处理
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", t)
    if m:
        try:
            d = _dt.datetime(now.year, int(m.group(1)), int(m.group(2)))
            return abs((now - d).days) <= max_age_days
        except ValueError:
            return None
    m = re.search(r"(\d{1,2})月(\d{1,2})[日号]", t)
    if m:
        try:
            d = _dt.datetime(now.year, int(m.group(1)), int(m.group(2)))
            return abs((now - d).days) <= max_age_days
        except ValueError:
            return None
    return None


def filter_fresh(results: List[Dict], max_age_days: int = 3) -> List[Dict]:
    """保留有发布时间且时间新鲜的条目；无法判断时间的条目保留；全被剔除则原样返回。"""
    kept = [r for r in results if is_fresh_time(r.get("time", ""), max_age_days) is not False]
    return kept if kept else results


def web_search(query: str, max_results: int = 5, timeout: float = 10.0,
               timely: bool = False) -> List[Dict]:
    """执行联网搜索，所有结果经过相关性门控（防反爬污染）。

    timely=True（时效性查询）：优先 360 新闻垂直源（按时间排序，条目带发布时间），
    之后依次回退 360 网页 / Bing / DuckDuckGo；
    timely=False：360 网页 -> Bing -> DuckDuckGo。
    全失败返回 []。

    返回: [{"title": str, "url": str, "snippet": str, "time": str(可选)}, ...]
    """
    if not query or not query.strip():
        return []
    order = ["so360_news", "so360", "bing", "ddg"] if timely else ["so360", "bing", "ddg"]
    fns = {
        "so360_news": search_so360_news,
        "so360": search_so360,
        "bing": search_bing,
        "ddg": search_ddg,
    }
    for name in order:
        try:
            # 时效模式多抓一倍再筛新鲜度，避免过滤后凑不满
            results = fns[name](query, max_results * 2 if name == "so360_news" and timely else max_results, timeout)
        except Exception:
            continue
        relevant = filter_relevant(query, results)
        if not relevant:
            continue
        if name == "so360_news" and timely:
            relevant = filter_fresh(relevant)
        return relevant[:max_results]
    return []
