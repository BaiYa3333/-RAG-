"""关键词增强共用模块 — 提供统一的关键词提取和匹配函数。

Tier1 retrieval (_keyword_boost_rerank) 和 Reranker (_keyword_match_score)
原先各自实现了相似的关键词提取+加权逻辑。本模块提取共用函数，
当权重或停用词调整时只需修改一处。
"""

import re
import logging

logger = logging.getLogger(__name__)

# 中文停用词（合并 retrieval 和 reranker 两处定义）
_STOP_WORDS: set[str] = set(
    "的了吗呢嗯吧啊是和在有着对之乎者也"
    "什么哪些如何怎么为什么请问可以是否"
    "一个一种这个那种这些那些其中"
    "请说明总结分析对比比较解释描述告诉"
    "与及或其和而但若虽因为所以"
    "哪些主要有什么区别差异"
)


def extract_keywords(text: str) -> list[str]:
    """从查询文本提取关键词（jieba 分词 + bigram 补充 + 英文 token）。

    同时使用 jieba 分词和字符 bigram 捕获复合词（如"基础版"被 jieba 拆为
    "基础"+"版"时，bigram "础版"可保留部分信号）。
    过滤停用词和单字词，保留 2+ 字符的实义词。

    Args:
        text: 输入查询文本

    Returns:
        去重后的关键词列表
    """
    # 提取英文/数字 token（如 CloudDesk、API、SaaS）
    en_tokens = re.findall(r"[a-zA-Z0-9]+", text)

    # 中文分词 + bigram 补充
    all_candidates = []
    try:
        import jieba
        words = jieba.lcut(text)
        all_candidates.extend(words)
    except ImportError:
        all_candidates.extend([text[i:i + 2] for i in range(len(text) - 1)])

    # 补充字符级 bigram（捕获被 jieba 拆散的复合词）
    bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
    all_candidates.extend(bigrams)

    # 过滤：保留 >=2 字符、非纯标点、非停用词
    keywords = []
    for w in all_candidates:
        w = w.strip()
        if (
            len(w) >= 2
            and w not in _STOP_WORDS
            and not all(c in "，。、；：？！""''（）《》【】" for c in w)
        ):
            keywords.append(w)

    # 英文 token 优先，去重
    seen: set[str] = set()
    result: list[str] = []
    for kw in en_tokens + keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen and len(kw) >= 2:
            seen.add(kw_lower)
            result.append(kw)

    return result


def keyword_match_score(
    query: str | None = None,
    doc_content: str = "",
    keywords: list[str] | None = None,
) -> float:
    """计算文档对查询关键词的覆盖度 (0-1)。

    对每个关键词检查是否出现在文档中，加权求和后归一化。
    长关键词权重更高（更具体），连续匹配给满分，长关键词支持部分匹配。

    Args:
        query: 查询文本（若未提供 keywords 则从此提取）
        doc_content: 文档内容
        keywords: 预提取的关键词列表（避免重复分词）

    Returns:
        关键词匹配分数 (0-1)，无法提取关键词时返回 0.5（中性）
    """
    if keywords is None:
        if query is None:
            return 0.5
        keywords = extract_keywords(query)

    if not keywords:
        return 0.5

    content_lower = doc_content.lower()
    total_weight = 0.0
    matched_weight = 0.0

    for kw in keywords:
        kw_lower = kw.lower()
        # 长关键词权重更高（更具体）
        weight = len(kw) ** 1.5
        total_weight += weight

        if kw_lower in content_lower:
            # 完整匹配给满分
            matched_weight += weight
        else:
            # 对长关键词尝试部分匹配（2-gram 子串）
            if len(kw) >= 4:
                sub_matches = 0
                sub_count = 0
                for i in range(len(kw) - 1):
                    sub = kw[i:i + 2]
                    if len(sub) >= 2 and sub.lower() in content_lower:
                        sub_matches += 1
                    sub_count += 1
                if sub_count > 0 and sub_matches / sub_count >= 0.5:
                    matched_weight += weight * 0.5  # 部分匹配给一半权重

    if total_weight == 0:
        return 0.5
    return matched_weight / total_weight
