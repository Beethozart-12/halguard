"""联网搜索模块测试（全部离线：HTTP 均被 mock，不真实联网）。"""
import httpx
import pytest

from halguard import websearch as W


BING_HTML = """
<html><body><ol id="b_results">
<li class="b_algo"><h2><a href="https://example.com/eiffel">埃菲尔铁塔 - 维基百科</a></h2>
<div class="b_caption"><p>埃菲尔铁塔建成于1889年，高约330米。</p></div></li>
<li class="b_algo"><h2><a href="https://example.org/tower">Eiffel Tower Facts</a></h2>
<p>The tower was completed in 1889 and stands 330 meters tall.</p></li>
<li class="b_algo"><div>没有链接的块，应被跳过</div></li>
</ol></body></html>
"""

DDG_HTML = """
<html><body>
<div class="result">
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Feiffel&amp;rut=abc">埃菲尔铁塔 - 维基百科</a>
<a class="result__snippet" href="#">埃菲尔铁塔建成于1889年，高约330米。</a>
</div>
<div class="result">
<a class="result__a" href="https://example.org/tower">Eiffel Tower Facts</a>
<a class="result__snippet" href="#">Completed in 1889.</a>
</div>
</body></html>
"""


def test_parse_bing_html_extracts_results():
    out = W.parse_bing_html(BING_HTML, max_results=5)
    assert len(out) == 2
    assert out[0]["title"] == "埃菲尔铁塔 - 维基百科"
    assert out[0]["url"] == "https://example.com/eiffel"
    assert "1889" in out[0]["snippet"]
    assert out[1]["url"] == "https://example.org/tower"


def test_parse_bing_html_max_results():
    out = W.parse_bing_html(BING_HTML, max_results=1)
    assert len(out) == 1


def test_parse_ddg_html_unwraps_redirect():
    out = W.parse_ddg_html(DDG_HTML, max_results=5)
    assert len(out) == 2
    assert out[0]["url"] == "https://example.com/eiffel"
    assert out[0]["title"] == "埃菲尔铁塔 - 维基百科"
    assert "1889" in out[0]["snippet"]
    assert out[1]["url"] == "https://example.org/tower"


# ---- 360 搜索（国内首选源） ----

SO360_HTML = """
<html><body><ul class="result">
<li class="res-list"><h3 class="res-title"><a href="https://baike.so.com/doc/123.html" target="_blank"><em>埃菲尔铁塔</em>_360百科</a></h3>
<p class="res-desc">埃菲尔铁塔于1889年建成，高约330米。</p></li>
<li class="res-list"><h3><a href="https://www.so.com/link?m=abc" target="_blank">Eiffel Tower height</a></h3>
<p class="res-desc">The tower stands 330 meters.</p></li>
<li class="res-list"><h3><a href="/s?q=%E5%85%B6%E4%BB%96" target="_blank">其他人还搜了</a></h3></li>
</ul></body></html>
"""


def test_parse_so360_html_extracts_results():
    out = W.parse_so360_html(SO360_HTML, max_results=5)
    assert len(out) == 2  # 内部推荐 /s?q= 被跳过
    assert out[0]["title"] == "埃菲尔铁塔_360百科"
    assert out[0]["url"] == "https://baike.so.com/doc/123.html"
    assert "1889" in out[0]["snippet"]
    assert out[1]["url"].startswith("https://www.so.com/link")


# ---- 相关性门控（防反爬污染） ----

def test_filter_relevant_removes_polluted_results():
    """Bing 反爬污染实测：查埃菲尔铁塔返回 MacBook——必须被剔除。"""
    polluted = [
        {"title": "M3 Mac_百度百科", "url": "https://x/1", "snippet": "M3 MacBook Air 整合了相机"},
        {"title": "埃菲尔铁塔 - 维基百科", "url": "https://x/2", "snippet": "建成于1889年，高约330米。"},
    ]
    out = W.filter_relevant("埃菲尔铁塔 建成年份 高度", polluted)
    assert len(out) == 1 and out[0]["url"] == "https://x/2"


def test_filter_relevant_all_polluted_returns_empty():
    polluted = [{"title": "MacBook Pro 评测", "url": "https://x/1", "snippet": "性能强劲"}]
    assert W.filter_relevant("埃菲尔铁塔 高度", polluted) == []


def test_content_units_cjk_and_latin():
    u = W.content_units("埃菲尔铁塔 height")
    assert "铁塔" in u and "height" in u


# ---- 搜索编排与回退 ----


# ---- 360 新闻（时效性源，时间排序） ----

NEWS_HTML = """
<html><body>
<li class="full-txt res-list pure-txt" data-url="https://www.360kuai.com/pc/1">
<a href="https://www.360kuai.com/pc/1" target="_blank" title="今天全世界都在看的新闻2026.10.7">
<h3 class="g-title js-title"><div class="g-title-inner"><div class="g-txt-inner g-ellipsis">今天全世界都在看的<em>新闻</em></div></div></h3>
<div class="g-figure-caption"><p class="summary g-ellipsis3">各大媒体报道最新国际动态。</p>
<p class="g-linkinfo info b-info"><span class="g-linkinfo-txt g-c-gray time">8小时前</span></p></div></a></li>
<li class="res-list" data-url="https://example.com/ai-news">
<a href="https://example.com/ai-news" title="AI领域新闻动态">
<h3 class="g-title js-title"><div class="g-txt-inner">AI领域新闻动态</div></h3>
<p class="summary">最新模型发布。</p>
<span class="g-linkinfo-txt g-c-gray time">2026-10-06 21:00</span></a></li>
</body></html>
"""


def test_parse_so360_news_html_extracts_results_with_time():
    out = W.parse_so360_news_html(NEWS_HTML, max_results=5)
    assert len(out) == 2
    assert out[0]["title"] == "今天全世界都在看的新闻2026.10.7"
    assert out[0]["time"] == "8小时前"
    assert out[1]["time"] == "2026-10-06 21:00"
    assert "模型发布" in out[1]["snippet"]


# ---- 时效性查询识别 ----

def test_is_timely_query():
    for q in ["今天有什么新闻", "最新 AI 新闻", "现在汇率多少", "qwen3 最新版本", "今天几号"]:
        assert W.is_timely_query(q), q
    for q in ["巴黎的首都是哪里", "埃菲尔铁塔是谁设计的", "帮我写一首诗"]:
        assert not W.is_timely_query(q), q


# ---- 新鲜度过滤（时效性查询剔除旧闻） ----

def test_is_fresh_time():
    F = W.is_fresh_time
    assert F("8小时前") is True
    assert F("30分钟前") is True
    assert F("昨天 21:00") is True
    assert F("2天前") is True
    assert F("5天前") is False
    assert F("2周前") is False
    assert F("06:05") is True              # 纯时刻按今天处理
    assert F("2020-01-01 13:31") is False  # 陈旧绝对日期
    assert F("") is None                   # 未知不过滤
    assert F("乱七八糟") is None


def test_filter_fresh_removes_old_news_with_today_in_title():
    """新闻池常见陷阱：标题含'今天'的旧闻必须被剔除。"""
    results = [
        {"title": "来自山东的女民兵今天受阅", "url": "https://x/1", "snippet": "", "time": "2025-09-03 13:31"},
        {"title": "这才是今天世界最重要的新闻", "url": "https://x/2", "snippet": "", "time": "06:05"},
    ]
    out = W.filter_fresh(results)
    assert len(out) == 1 and out[0]["url"] == "https://x/2"


def test_filter_fresh_keeps_unknown_time():
    results = [{"title": "无时间戳条目", "url": "https://x/1", "snippet": "", "time": ""}]
    assert W.filter_fresh(results) == results


def test_web_search_timely_prefers_news(monkeypatch):
    """timely=True 时优先走 360 新闻（sort=time），并带回时间戳。"""
    called = []

    def fake_get(url, **kw):
        called.append(str(url))
        if "news.so.com" in url:
            params = kw.get("params") or {}
            assert params.get("sort") == "time", f"新闻源应按时间排序: {params}"
            return httpx.Response(200, text=NEWS_HTML, request=httpx.Request("GET", url))
        raise httpx.ConnectError("should not be called")

    monkeypatch.setattr(W.httpx, "get", fake_get)
    out = W.web_search("今天有什么新闻", max_results=3, timely=True)
    assert out and out[0]["time"] == "8小时前"
    assert "news.so.com" in called[0]  # 新闻源排第一


def test_web_search_timely_falls_back_to_web(monkeypatch):
    """新闻源失败时回退常规网页源。"""
    def fake_get(url, **kw):
        if "news.so.com" in url:
            raise httpx.ConnectError("news down")
        if "so.com" in url:
            return httpx.Response(200, text=SO360_HTML, request=httpx.Request("GET", url))
        raise httpx.ConnectError("blocked")

    monkeypatch.setattr(W.httpx, "get", fake_get)
    # 查询词需与 SO360_HTML 罐头结果相关（过相关性门控）；timely=True 强制走新闻优先调度
    out = W.web_search("埃菲尔铁塔 高度", max_results=3, timely=True)
    assert out and out[0]["url"] == "https://baike.so.com/doc/123.html"


def test_web_search_falls_back_to_ddg(monkeypatch):
    """so360 与 Bing 均失败时回退 DuckDuckGo。"""
    def fake_get(url, **kw):
        if "so.com" in url or "bing.com" in url:
            raise httpx.ConnectError("blocked")
        return httpx.Response(200, text=DDG_HTML, request=httpx.Request("GET", url))

    monkeypatch.setattr(W.httpx, "get", fake_get)
    out = W.web_search("埃菲尔铁塔 高度", max_results=3)
    # 第 2 条 "Eiffel Tower Facts / Completed in 1889." 与中文查询无内容单元重合，
    # 会被相关性门控剔除——这正是门控的预期行为
    assert len(out) == 1
    assert out[0]["url"] == "https://example.com/eiffel"


def test_web_search_bing_pollution_skipped(monkeypatch):
    """so360 失败、Bing 返回污染结果（无关）时应继续回退而不是采信。"""
    POLLUTED = '<html><li class="b_algo"><h2><a href="https://x/mac">M3 Mac</a></h2><p>MacBook Air 评测</p></li></html>'

    def fake_get(url, **kw):
        if "so.com" in url:
            raise httpx.ConnectError("blocked")
        if "bing.com" in url:
            return httpx.Response(200, text=POLLUTED, request=httpx.Request("GET", url))
        return httpx.Response(200, text=DDG_HTML, request=httpx.Request("GET", url))

    monkeypatch.setattr(W.httpx, "get", fake_get)
    out = W.web_search("埃菲尔铁塔 高度", max_results=3)
    assert out and all("Mac" not in r["title"] for r in out)


def test_web_search_all_fail_returns_empty(monkeypatch):
    def fake_get(url, **kw):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(W.httpx, "get", fake_get)
    assert W.web_search("query") == []


def test_web_search_empty_query():
    assert W.web_search("   ") == []
