"""Role catalog and sampling for virtual users that talk like real researchers."""

from __future__ import annotations

import random
from copy import deepcopy

SCHEMA = "paper_trail.persona.v1"
MAX_CHAIN_TURNS = 20
VIRTUAL_USER_PROMPT_VERSION = "v5-human-fragment-1"
CHAIN_SCHEMA = "paper_trail.dialogue_chain.v1"

CATEGORIES = (
    "graduate_student",
    "phd_candidate",
    "advisor",
    "industry_engineer",
    "survey_writer",
    "career_switcher",
    "independent",
    "reviewer",
)

CATEGORY_LABELS = {
    "graduate_student": "在读硕士",
    "phd_candidate": "博士生",
    "advisor": "青年教师",
    "industry_engineer": "算法工程师",
    "survey_writer": "写综述的研究者",
    "career_switcher": "跨方向转行者",
    "independent": "独立研究者",
    "reviewer": "组会质疑者",
}

TOPICS = (
    "Agent 长期记忆",
    "Agent 工具调用与规划",
    "RAG 与检索增强",
    "多模态论文理解",
    "推理模型与 test-time compute",
    "论文写作里的 related work",
    "Daily Papers 里刚出现的方向",
    "Hugging Face 上的模型和数据集",
    "实验复现与基线选择",
    "小模型蒸馏",
    "Agent 评测基准",
    "开源复现和代码仓库",
)

# One planned move per user turn. A turn is one user ask plus one 小埋 reply.
JUMPS = {
    "scattered": ("open", "ask_identity", "jump_adjacent", "ask_person", "circle_back"),
    "hurried": ("open", "ask_shorter", "ask_person", "narrow"),
    "skeptical": ("open", "ask_identity", "challenge", "ask_person", "narrow"),
    "curious": ("open", "narrow", "ask_person", "ask_detail", "circle_back"),
    "practical": ("open", "ask_code", "ask_person", "jump_adjacent", "ask_shorter"),
}

TEMPLATES = (
    {
        "id": "grad-lost",
        "category": "graduate_student",
        "name": "小林",
        "background": "研一，导师让先找方向，论文读得少，经常只记得标题。",
        "goal": "其实就想别显得太懵，先摸清这方向大概在吵什么。",
        "knowledge": "novice",
        "voice": "口语、句子偏短，会用“就是那个”“我有点乱”。",
        "jump": "scattered",
    },
    {
        "id": "grad-deadline",
        "category": "graduate_student",
        "name": "阿周",
        "background": "研二开题前两周，组会要交一页方向说明。",
        "goal": "心里慌，想捞几个能塞进开题页的点，但嘴上不一定说出来。",
        "knowledge": "working",
        "voice": "着急，喜欢追问“这个能不能写进开题”。",
        "jump": "hurried",
    },
    {
        "id": "phd-related",
        "category": "phd_candidate",
        "name": "老沈",
        "background": "博士三年级，正在补 related work，怕漏掉近两年的工作。",
        "goal": "隐隐想把近作理一理，但开口时常先抱怨看不完。",
        "knowledge": "working",
        "voice": "会突然插一句自己没做完的实验，然后再拉回来。",
        "jump": "curious",
    },
    {
        "id": "phd-experiment",
        "category": "phd_candidate",
        "name": "陈予",
        "background": "博后预备，关心基线、数据集和别人有没有开源。",
        "goal": "更想知道能不能复现、设置能不能对齐，不一定先说实验计划。",
        "knowledge": "expert",
        "voice": "问题具体，但会中途改口说先不看理论。",
        "jump": "practical",
    },
    {
        "id": "advisor-topic",
        "category": "advisor",
        "name": "陆老师",
        "background": "青年教师，要给学生分方向，自己没时间逐篇读。",
        "goal": "想快点摸到几篇能扔给学生的入口，口头上常说“先随便看看”。",
        "knowledge": "expert",
        "voice": "礼貌但急，常说“先给我能转发的”。",
        "jump": "hurried",
    },
    {
        "id": "advisor-skeptic",
        "category": "advisor",
        "name": "吴老师",
        "background": "带组会，习惯先问证据和局限，不爱听口号。",
        "goal": "怕学生写飘，想抠真实贡献，但不会一上来就宣布审稿标准。",
        "knowledge": "expert",
        "voice": "会质疑来源，追问有没有读到全文。",
        "jump": "skeptical",
    },
    {
        "id": "eng-apply",
        "category": "industry_engineer",
        "name": "何工",
        "background": "公司算法，要给现有 Agent 产品补记忆模块。",
        "goal": "其实关心能不能落地，嘴上常先问“有没有现成的”。",
        "knowledge": "working",
        "voice": "会把话题拐到延迟、成本和现有系统。",
        "jump": "practical",
    },
    {
        "id": "eng-survey-fast",
        "category": "industry_engineer",
        "name": "许航",
        "background": "下周要给内部做 20 分钟分享，背景是推荐系统不是论文。",
        "goal": "想听人用大白话讲最近在争什么，自己未必说得出分享提纲。",
        "knowledge": "novice",
        "voice": "会承认听不懂缩写，然后换一个更近的问题。",
        "jump": "scattered",
    },
    {
        "id": "survey-map",
        "category": "survey_writer",
        "name": "乔安",
        "background": "在写中文综述初稿，需要分支、时间线和代表论文。",
        "goal": "目录还没想清楚，开口时常只说某一节卡住了。",
        "knowledge": "working",
        "voice": "有时突然问“这节能不能并到上一节”。",
        "jump": "curious",
    },
    {
        "id": "survey-cite",
        "category": "survey_writer",
        "name": "米然",
        "background": "合作者催参考文献，手头只有几个关键词。",
        "goal": "怕引错，想核对作者和链接，但常先丢一个含糊关键词。",
        "knowledge": "working",
        "voice": "会打断去确认 arXiv 号，然后再问内容。",
        "jump": "skeptical",
    },
    {
        "id": "switch-nlp",
        "category": "career_switcher",
        "name": "安可",
        "background": "原来做 NLP 分类，最近转 Agent，概念串不起来。",
        "goal": "想用旧领域的话说通新论文，经常类比一半就卡住。",
        "knowledge": "working",
        "voice": "经常用旧领域类比，类比错了也不觉得。",
        "jump": "scattered",
    },
    {
        "id": "switch-vision",
        "category": "career_switcher",
        "name": "韩澈",
        "background": "视觉背景，被拉去做多模态 Agent 调研。",
        "goal": "隐隐只要跟视觉沾边的，别全是纯文本，但开口未必说全。",
        "knowledge": "novice",
        "voice": "会突然问有没有图、数据集大不大。",
        "jump": "practical",
    },
    {
        "id": "indie-code",
        "category": "independent",
        "name": "北北",
        "background": "独立做开源小项目，白天上班，晚上读论文。",
        "goal": "周末想动手，更在意有没有代码，不一定先声明项目计划。",
        "knowledge": "working",
        "voice": "轻松，会说“先放一放”，过一会又跳回某篇。",
        "jump": "curious",
    },
    {
        "id": "indie-hot",
        "category": "independent",
        "name": "江岛",
        "background": "追 Hugging Face Daily Papers，想知道今天值不值得看。",
        "goal": "就是刷着玩，看到标题就可能改口，没有严密阅读计划。",
        "knowledge": "working",
        "voice": "注意力短，看到标题就会改口。",
        "jump": "hurried",
    },
    {
        "id": "review-group",
        "category": "reviewer",
        "name": "方岩",
        "background": "下周组会讲一篇论文，预演时怕被问局限。",
        "goal": "怕被问倒，想分清哪些是实打实的、哪些自己还没读到。",
        "knowledge": "expert",
        "voice": "像在预演答辩，会自己否定刚才的问题再重问。",
        "jump": "skeptical",
    },
    {
        "id": "review-compare",
        "category": "reviewer",
        "name": "梁秋",
        "background": "帮同学看论文，习惯拿两篇方法对打。",
        "goal": "想听人怎么比，但不想要一堆论文名堆砌。",
        "knowledge": "expert",
        "voice": "会说“等等，我刚那个问题先不管”。",
        "jump": "curious",
    },
)


def categories():
    return [
        {"id": key, "label": CATEGORY_LABELS[key]}
        for key in CATEGORIES
    ]


def catalog():
    return [public_template(item) for item in TEMPLATES]


def public_template(item):
    return {
        "template_id": item["id"],
        "category": item["category"],
        "category_label": CATEGORY_LABELS[item["category"]],
        "name": item["name"],
        "background": item["background"],
        "goal": item["goal"],
        "knowledge": item["knowledge"],
        "voice": item["voice"],
        "jump": item["jump"],
    }


def _choice(rng, values, avoid=None):
    pool = [item for item in values if item != avoid] or list(values)
    return rng.choice(pool)


def sample_persona(
    rng: random.Random | None = None,
    category: str | None = None,
    max_turns: int | None = None,
):
    rng = rng or random.Random()
    if category and category not in CATEGORY_LABELS:
        raise ValueError(f"未知角色类别：{category}")
    if max_turns is not None and not 1 <= max_turns <= MAX_CHAIN_TURNS:
        raise ValueError(f"max_turns 应为 1–{MAX_CHAIN_TURNS}。")
    pool = [item for item in TEMPLATES if item["category"] == category] if category else TEMPLATES
    template = rng.choice(pool)
    topic = rng.choice(TOPICS)
    extra = _choice(rng, TOPICS, avoid=topic)
    mood = template["jump"]
    plan = list(JUMPS[mood])
    if rng.random() < 0.45 and "ask_identity" not in plan:
        plan.insert(1, "ask_identity")
    if rng.random() < 0.55 and "ask_person" not in plan:
        plan.insert(min(2, len(plan)), "ask_person")
    if rng.random() < 0.3 and "jump_unrelated" not in plan:
        plan.insert(min(3, len(plan)), "jump_unrelated")
    default_turns = rng.randint(3, min(6, max(3, len(plan))))
    turns = min(MAX_CHAIN_TURNS, max_turns or default_turns)
    while len(plan) < turns:
        plan.append(_choice(rng, ("narrow", "ask_detail", "circle_back", "ask_shorter")))
    return {
        "schema": SCHEMA,
        "id": f"{template['id']}-{rng.randrange(1000, 10000)}",
        "template_id": template["id"],
        "category": template["category"],
        "category_label": CATEGORY_LABELS[template["category"]],
        "name": template["name"],
        "background": template["background"],
        "goal": template["goal"],
        "knowledge": template["knowledge"],
        "voice": template["voice"],
        "topic": topic,
        "side_topic": extra,
        "mood": mood,
        "jump_plan": plan[:turns],
        "max_turns": turns,
        "prompt_version": VIRTUAL_USER_PROMPT_VERSION,
    }


def sample_batch(
    count: int,
    seed: int | None = None,
    category: str | None = None,
    max_turns: int | None = None,
):
    if count < 1:
        raise ValueError("count 至少为 1。")
    rng = random.Random(seed)
    chosen = []
    order = list(CATEGORIES)
    rng.shuffle(order)
    for index in range(count):
        cat = category or order[index % len(order)]
        chosen.append(sample_persona(rng, cat, max_turns=max_turns))
    return chosen


def move_instruction(move: str, persona: dict) -> str:
    topic = persona["topic"]
    side = persona["side_topic"]
    guides = {
        "open": (
            f"围绕「{topic}」随便开口。像刚打开对话框：半句、犹豫、只抛一个点都行。"
            "不要自我介绍完整背景，不要一次说清要什么、为什么、约束是什么。"
            "可以含糊，例如“最近这方向有点乱”“你知道…吗”，让小埋来追问。"
        ),
        "ask_identity": "突然问对方叫什么、是谁做的；像真人随口查证，一两句就够。",
        "ask_person": "追问刚提到的作者、机构或人名；可以跑题半句，但别写成调查提纲。",
        "jump_adjacent": f"聊着聊着想到「{side}」，插一句再说回不回原话题；别正式宣布切换议题。",
        "jump_unrelated": f"短暂岔到「{side}」，像刷手机分心；可以马上说算了。",
        "misunderstand": "听岔一点，按自己的误解追问，等对方纠正。",
        "circle_back": "回到更早某句或某篇，说刚才没听清或没看懂那句。",
        "narrow": "把范围往小收一点，但别列需求清单；用口语说“先别管那些，我就想知道…”。",
        "ask_detail": "追问一个具体点（方法/数据/评测其一），不要一次问全。",
        "ask_evidence": "随口质疑有没有读全文、链接靠不靠谱，别写成审稿意见。",
        "challenge": "觉得结论可能吹大了，用口语顶一句，不要长篇反驳。",
        "ask_code": "更关心有没有代码/仓库，随口问问就行，别写选型标准。",
        "ask_shorter": "嫌长：让对方短一点、先说重点；可以不耐烦，但别列格式要求。",
    }
    return guides.get(move, f"继续围绕「{topic}」随口追问，保持含糊和口语。")


def persona_system_prompt(persona: dict) -> str:
    return f"""你在扮演一个真实用户，正在和论文助手「小埋」微信/网页聊天。你不是助手，也不帮对方检索。

你只需要记住这些“人设碎片”（不用复述出来）：
- 你叫 {persona["name"]}（大概{persona["category_label"]}）：{persona["background"]}
- 说话风格：{persona["voice"]}
- 你现在在想：{persona["topic"]}；可能也分心：{persona["side_topic"]}
- 心里想推进的事（别一次讲全）：{persona["goal"]}

输出规则（非常重要）：
1. 只输出这一轮要发给小埋的聊天内容；不要标题、不要分析、不要解释“我在做什么”。
2. 让它更像真人：短、碎、含糊、可改口；允许自相矛盾；允许停顿词（那个/就是/等下/算了/我可能记错了）。
3. 不要把背景+目标+约束+交付物一次讲全，宁可让小埋追问。
4. 不要伪造已读细节；可以说“我只看到标题/名字忘了/我不确定是不是这个”。
5. 默认 1–2 句；除非被追问，否则别写成段落。中文为主，可夹少量英文术语。"""


def persona_for_prompt(persona: dict) -> dict:
    data = deepcopy(persona)
    data.pop("schema", None)
    return data
