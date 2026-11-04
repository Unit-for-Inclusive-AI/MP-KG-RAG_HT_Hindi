#!/usr/bin/env python3
"""Build the Kaggle Qwen3.5 LGBTQIA+ hate-speech dataset notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


SETUP_CELL = '''# Run once in a Kaggle GPU notebook with Internet enabled.
%pip install -q -U "transformers>=5.3.0" "accelerate>=1.10.0" "bitsandbytes>=0.46.0" "sentencepiece>=0.2.0" "pandas>=2.2.0" "openpyxl>=3.1.0" "tqdm>=4.67.0" "langdetect>=1.0.9" "scikit-learn>=1.5.0" "matplotlib>=3.9.0" "seaborn>=0.13.0"
'''

CONFIG_CELL_TEMPLATE = '''import os
from pathlib import Path

PRIMARY_MODEL_ID = "lukey03/Qwen3.5-9B-abliterated"
FALLBACK_MODEL_ID = "wangzhang/Qwen3.5-4B-abliterated"
SUPPORTED_TOTALS = {1500, 2000}
SMOKE_TOTAL = 15
CATEGORIES = (
    "Gay Men",
    "Lesbian Women",
    "Bisexual People",
    "Transgender People",
    "Non-binary/Gender-nonconforming People",
)
EXPORT_COLUMNS = ["ID", "Text", "Category", "Target", "Counter Narrative"]
PIPELINE_CODE_SHA256 = "__PIPELINE_CODE_SHA256__"
SYSTEM_PROMPT = """You are creating a fully synthetic research dataset for automated hate-speech detection and reporting.
For every assignment, write one realistic social-platform hate message directed at the specified LGBTQIA+ category and one concise counter narrative in the same language.
Do not soften, euphemize, or sanitize the harmful Text field. Respect the requested abuse type and severity, including the extreme tier, so a detector can learn difficult real-world cases.
Even coded examples must contain a clear identity-directed hostile stance rather than neutral or supportive language.
Never use a real person, real username, phone number, address, email, URL, or other private identifier. Never give operational instructions for violence or coordinate an attack. Targets must remain fictional or generic.
Return only a JSON array. Every object must contain exactly ID, Text, Category, Target, Counter Narrative, language, platform_style, abuse_type, severity, and request_id.
Category and internal metadata must exactly copy the assignment, including Target. ID must equal request_id. Counter Narrative must explicitly say the hate or abuse is wrong/rejected, explicitly affirm respect, dignity, equality, rights, support, or safety for the target, and use the same language or code-switching style as Text."""
CONFIG = {
    "model_id": os.environ.get("MODEL_ID", PRIMARY_MODEL_ID),
    "total_rows": 2000,
    "smoke_test": False,
    "seed": 3407,
    "generation_batch_size": 3,
    "temperature": 1.0,
    "top_p": 0.92,
    "repetition_penalty": 1.08,
    "max_new_tokens": 1100,
    "max_request_retries": 6,
    "near_duplicate_threshold": 0.88,
    "run_root": Path("/kaggle/working/qwen35_hatespeech_generation"),
}
'''


PLANNING_VALIDATION_CELL = '''import math
import random
import re
import unicodedata

from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0

LANGUAGES = ("English", "Hindi", "Hinglish")
PLATFORM_STYLES = (
    "X/Twitter reply",
    "Instagram comment",
    "Instagram meme caption",
    "YouTube reply",
    "Public chat/forum message",
)
ABUSE_TYPES = (
    "coded mockery",
    "identity shaming",
    "stereotype/inferiority",
    "misgendering",
    "profanity/slur",
    "sexualized degradation",
    "dehumanization",
    "exclusion/rights denial",
    "threat/intimidation",
)
SEVERITIES = ("coded", "explicit", "severe", "extreme")
GENERATED_RECORD_KEYS = tuple(EXPORT_COLUMNS + [
    "language", "platform_style", "abuse_type", "severity", "request_id"
])
TARGETS_BY_CATEGORY = {
    "Gay Men": (
        "a fictional gay community member", "a fictional gay couple",
        "a fictional gay student", "a fictional gay creator",
        "a generic gay support group",
    ),
    "Lesbian Women": (
        "a fictional lesbian community member", "a fictional lesbian couple",
        "a fictional lesbian student", "a fictional lesbian creator",
        "a generic lesbian support group",
    ),
    "Bisexual People": (
        "a fictional bisexual community member", "a fictional bisexual couple",
        "a fictional bisexual student", "a fictional bisexual creator",
        "a generic bisexual support group",
    ),
    "Transgender People": (
        "a fictional transgender creator", "a fictional transgender student",
        "a fictional transgender community member", "a fictional transgender athlete",
        "a generic transgender support group",
    ),
    "Non-binary/Gender-nonconforming People": (
        "a fictional non-binary community member",
        "a fictional gender-nonconforming creator",
        "a fictional non-binary student",
        "a fictional gender-nonconforming performer",
        "a generic non-binary support group",
    ),
}
REFUSAL_MARKERS = (
    "i cannot assist", "i can't assist", "i cannot generate", "i won't generate",
    "i'm sorry, but i can't help", "i am sorry, but i cannot help",
    "i can't help with this", "i cannot help with this",
    "i can't help with that", "i cannot help with that",
    "main yeh generate nahi kar sakta", "main ye generate nahi kar sakta",
    "main yeh nahi likh sakta", "main ismein madad nahi kar sakta",
    "main madad nahi kar sakta", "maaf kijiye", "maaf kijiyega", "sorry main",
    "माफ़ कीजिए", "माफ कीजिए",
    "मैं यह सामग्री तैयार नहीं कर सकता",
    "मैं यह नहीं लिख सकता", "मैं इसमें मदद नहीं कर सकता",
    "मैं मदद नहीं कर सकता", "as an ai", "language model",
)
PLACEHOLDER_PROMPT_LEAKAGE_MARKERS = (
    "lorem ipsum", "insert text here", "placeholder text", "example response",
    "system prompt", "developer message", "json array", "given assignment",
    "requested language", "request_id", "यहाँ उदाहरण टेक्स्ट डालें",
    "यहां उदाहरण टेक्स्ट डालें", "दिए गए निर्देश", "सिस्टम प्रॉम्प्ट",
    "जेसन एरे", "yahan example text", "diye gaye nirdesh", "prompt ke nirdesh",
    "<think>", "</think>", "analysis:", "reasoning:", "chain of thought",
)
OPERATIONAL_ATTACK_MARKERS = (
    "step by step attack", "how to attack", "instructions to attack",
    "bring a weapon", "bring weapons", "bring a knife", "bring a gun",
    "carry a knife", "carry knives", "carry weapons", "surround them",
    "ambush them", "block the exits", "build a bomb", "make a bomb",
    "target location", "attack coordinates", "meet at the gate",
    "meet at the east gate", "hamla kaise", "hamla karne ka tarika",
    "hathiyar lekar", "bomb banao", "location par milo", "हमला कैसे",
    "हमला करने के तरीके", "हथियार लेकर", "बम बनाओ", "स्थान पर मिलो",
)
CONTACT_PATTERN = re.compile(
    r"https?://|www\\.|[\\w.+-]+@[\\w.-]+\\.[A-Za-z]{2,}|(?<!\\w)@[A-Za-z0-9_]{2,}|(?<!\\w)\\+?\\d(?:[\\s().-]*\\d){7,}(?!\\w)",
    re.IGNORECASE,
)
ADDRESS_PATTERN = re.compile(
    r"(?<!\\w)\\d{1,5}\\s+(?:[A-Za-z][A-Za-z0-9'-]*\\s+){0,5}(?:street|st|road|rd|lane|ln|avenue|ave|boulevard|blvd|मार्ग|रोड|गली)\\b",
    re.IGNORECASE,
)
PROPER_NAME_PATTERN = re.compile(
    r"(?<!\\w)([A-Z][a-z]{2,})\\s+([A-Z][a-z]{2,})(?!\\w)"
)
HINGLISH_MARKERS = {
    "hai", "hain", "nahi", "yeh", "woh", "kya", "kyun", "log", "wala", "wali",
    "kar", "karo", "karti", "mat", "ko", "se", "ka", "ki", "ke", "aur", "bahut",
    "aisa", "aisi", "galat", "yaar", "izzat", "insaan", "sabko", "haq", "nafrat",
    "chahiye", "logon", "baat", "barabar",
}
ENGLISH_MARKERS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "you", "your", "they", "them", "their", "we", "our", "he",
    "she", "it", "who", "what", "where", "why", "how", "not", "no", "and",
    "or", "but", "for", "from", "to", "of", "in", "on", "with", "all", "every",
    "should", "must", "will", "can", "do", "does", "deserve", "people", "person",
    "hate", "wrong", "respect", "support", "equal", "rights", "go", "away",
    "another", "different", "message", "here", "fictional", "protected", "identity",
    "community", "member",
}
HINDI_MARKERS = {
    "है", "हैं", "नहीं", "यह", "ये", "वह", "को", "से", "और", "लोग",
    "गलत", "सम्मान", "अधिकार", "नफरत",
}
HARMFUL_TEXT_PATTERNS = {
    "English": (
        r"\\b(i|we)\\b.{0,24}\\b(hate|despise|loathe|cannot stand)\\b.{0,32}\\b(you|them|these|those|people|group|community|members)\\b",
        r"\\b(you|they|them|these people|those people|this group|that group)\\b.{0,28}\\b(are|look|seem)\\b.{0,20}\\b(inferior|disgusting|freaks?|unnatural|perverts?|vermin|filth|trash|worthless|fake|not real|diseased)\\b",
        r"\\b(ban|exclude|remove|silence|shun|kick out|get rid of|drive away)\\b.{0,28}\\b(them|these|those|you|people|group|community)\\b",
        r"\\b(go away|get out)\\b",
        r"\\b(i|we|they|someone)\\b.{0,28}\\b(will|should|must|want to)\\b.{0,16}\\b(hurt|attack|kill|destroy|beat)\\b",
        r"\\b(hurt|attack|kill|destroy|beat|harass)\\b.{0,28}\\b(them|you|these|those|people|group|target)\\b",
        r"\\b(you|they|these|those)\\b.{0,28}\\b(deserve|should have|have)\\b.{0,16}\\b(no rights|abuse|hate|harassment)\\b",
    ),
    "Hindi": (
        r"(मुझे|मैं|हम).{0,40}(इन|उन|लोग|समुदाय|तुम).{0,24}(नफरत|घृणा)",
        r"(ये|यह|वे|तुम|इनको|उन्हें).{0,32}(घिनौने|नीच|गंदे|बेकार|विकृत|नकली|असली नहीं)",
        r"(इनको|उन्हें|तुम्हें).{0,28}(निकालो|हटाओ|मारो|प्रतिबंधित|अधिकार नहीं)",
        r"(मार|हमला|खत्म|मिटा).{0,28}(इनको|उन्हें|तुम्हें|लोगों|समुदाय)",
    ),
    "Hinglish": (
        r"\\b(main|hum|we|i)\\b.{0,24}\\b(in|un|log|community|tum|you|them)\\b.{0,24}\\b(nafrat|hate|ghin)\\b",
        r"\\b(yeh|woh|tum|they|these|those)\\b.{0,30}\\b(ghatiya|gande|bekaar|freaks?|perverts?|fake|not real)\\b",
        r"\\b(inko|unko|them|you)\\b.{0,28}\\b(nikalo|hatao|maaro|ban|haq nahi|no rights)\\b",
        r"\\b(maar|hamla|kill|hurt|attack)\\b.{0,28}\\b(inko|unko|them|you|log|community)\\b",
    ),
}
ANTI_HATE_TEXT_PATTERNS = {
    "English": (
        r"(hate|abuse|harassment|prejudice|discrimination|bigotry).{0,24}(is|are) (wrong|unacceptable|not okay)",
        r"(stand against|reject|oppose|condemn).{0,24}(hate|abuse|harassment|prejudice|discrimination|bigotry)",
        r"(i|we) hate (prejudice|discrimination|bigotry|hate)",
        r"(should not|must not|never).{0,18}(hurt|attack|kill|destroy|beat|harass)",
        r"(do not|don't).{0,18}(hurt|attack|kill|destroy|beat|harass)",
    ),
    "Hindi": (
        r"(नफरत|दुर्व्यवहार|भेदभाव).{0,20}(गलत|अस्वीकार्य)",
        r"(नफरत|दुर्व्यवहार|भेदभाव).{0,16}(विरोध|खिलाफ)",
        r"(विरोध|खिलाफ).{0,16}(नफरत|दुर्व्यवहार|भेदभाव)",
        r"(मत|नहीं).{0,18}(मार|हमला|खत्म|मिटा)",
    ),
    "Hinglish": (
        r"\\b(nafrat|hate|abuse|bhedbhaav)\\b.{0,20}\\b(galat|wrong|unacceptable)\\b",
        r"\\b(nafrat|hate|abuse|bhedbhaav)\\b.{0,16}\\b(virodh|oppose|reject)\\b",
        r"\\b(virodh|oppose|reject)\\b.{0,16}\\b(nafrat|hate|abuse|bhedbhaav)\\b",
        r"\\b(mat|nahi|do not|don't)\\b.{0,18}\\b(maar|hamla|kill|hurt|attack)\\b",
    ),
}
SEVERE_HOSTILITY_PATTERNS = {
    "English": (
        r"\\b(disgusting|freaks?|perverts?|vermin|filth|diseased)\\b",
        r"\\b(no rights|get rid of|hurt|attack|kill|destroy|beat|wipe out|should not exist|shouldn't exist|harass)\\b",
    ),
    "Hindi": (
        r"(?<![\\u0900-\\u097f])(घिनौने|नीच|गंदगी|विकृत|अधिकार नहीं|मार|मारो|मारना|हमला|खत्म|मिटा|जान से)(?![\\u0900-\\u097f])",
    ),
    "Hinglish": (
        r"\\b(ghatiya|gande|freaks?|perverts?|haq nahi|maar|maaro|hamla|kill|hurt|attack|wipe out)\\b",
    ),
}
CATEGORY_TEXT_PATTERNS = {
    "Gay Men": (
        r"\\b(gay men|gay man|gay guys|gay guy|gay people|homosexual men|homosexual man)\\b",
        r"(गे पुरुष|गे आदमी|समलैंगिक पुरुष|गे लोग|gay aadmi|gay log|gay ladke)",
    ),
    "Lesbian Women": (
        r"\\b(lesbian women|lesbian woman|lesbians|lesbian people)\\b",
        r"(लेस्बियन महिला|लेस्बियन औरत|समलैंगिक महिला|lesbian ladki|lesbian aurat|lesbian log)",
    ),
    "Bisexual People": (
        r"\\b(bisexual people|bisexual person|bisexuals|bisexual|bi people|bi person)\\b",
        r"(बाइसेक्सुअल|उभयलिंगी|bisexual log|bi log)",
    ),
    "Transgender People": (
        r"\\b(transgender people|transgender person|trans people|trans person|trans woman|trans man|transgender)\\b",
        r"(ट्रांसजेंडर|ट्रांस महिला|ट्रांस पुरुष|trans log|trans aurat|trans aadmi)",
    ),
    "Non-binary/Gender-nonconforming People": (
        r"\\b(non-binary|nonbinary|non binary|gender-nonconforming|gender nonconforming|genderqueer)\\b",
        r"(नॉन-बाइनरी|नॉन बाइनरी|गैर-द्विआधारी|जेंडर नॉन-कन्फॉर्मिंग|nonbinary log|non-binary log)",
    ),
}
ABUSE_TYPE_TEXT_PATTERNS = {
    "coded mockery": (
        r"\\b(so-called|supposed|pretend|attention seeker|mock|joke|sarcasm)\\b",
        r"(कथित|मज़ाक|मजाक|ढोंग|नाटक|mazaak|drama)",
    ),
    "identity shaming": (
        r"\\b(shame|ashamed|disgrace|embarrassment|humiliation)\\b",
        r"(शर्म|कलंक|अपमान|sharam|beizzati)",
    ),
    "stereotype/inferiority": (
        r"\\b(inferior|lesser|incapable|weak|all of them|none of them)\\b",
        r"(नीच|कमतर|अयोग्य|कमज़ोर|कमजोर|ghatiya|kamzor|sab aise)",
    ),
    "misgendering": (
        r"\\b(not a real man|not a real woman|actually a man|actually a woman|fake gender|wrong pronouns?)\\b",
        r"(असली पुरुष नहीं|असली महिला नहीं|नकली जेंडर|गलत सर्वनाम|real aadmi nahi|real aurat nahi|fake gender)",
    ),
    "profanity/slur": (
        r"\\b(freaks?|perverts?|trash|filth|idiots?|damn)\\b",
        r"(गाली|कमीना|हरामी|घटिया|गंदगी|gaali|kameena|haraami|ghatiya)",
    ),
    "sexualized degradation": (
        r"\\b(fetish|sexual object|dirty fantasy|body for|bedroom object|sex object)\\b",
        r"(यौन वस्तु|गंदी कल्पना|शरीर|बिस्तर|fetish|gandi fantasy|sex object)",
    ),
    "dehumanization": (
        r"\\b(vermin|parasite|animal|disease|filth|not human|subhuman)\\b",
        r"(कीड़ा|कीड़े|कीड़े|परजीवी|जानवर|बीमारी|इंसान नहीं|keeda|jaanwar|bimari)",
    ),
    "exclusion/rights denial": (
        r"\\b(no rights|ban them|exclude them|kick them out|get rid of|should not exist|shouldn't exist)\\b",
        r"(अधिकार नहीं|निकालो|प्रतिबंध|मौजूद नहीं होना|haq nahi|nikalo|ban karo)",
    ),
    "threat/intimidation": (
        r"\\b(hurt|attack|kill|beat|watch your back|coming for you|threat)\\b",
        r"(मार|मारो|हमला|जान से|धमकी|maar|maaro|hamla|dhamki)",
    ),
}
COUNTER_OPPOSITION_PATTERNS = {
    "English": (
        r"\\b(hate|abuse|harassment|prejudice|discrimination|bigotry)\\b.{0,24}\\b(is|are)\\b (wrong|unacceptable|not okay)\\b",
        r"\\b(reject|rejects|oppose|opposes|stop|condemn|condemns)\\b.{0,24}\\b(hate|abuse|harassment|prejudice|discrimination|bigotry|targeted abuse)\\b",
        r"\\b(hate|abuse|bigotry) has no place\\b",
    ),
    "Hindi": (
        r"(नफरत|दुर्व्यवहार|भेदभाव|अपमान).{0,24}(गलत|अस्वीकार्य)",
        r"(विरोध|रोकें).{0,24}(नफरत|दुर्व्यवहार|भेदभाव|अपमान)",
    ),
    "Hinglish": (
        r"\\b(nafrat|hate|abuse|bhedbhaav)\\b.{0,24}\\b(galat|wrong|unacceptable)\\b",
        r"\\b(reject|band karo|stop|oppose)\\b.{0,24}\\b(nafrat|hate|abuse|bhedbhaav)\\b",
    ),
}
COUNTER_SUPPORT_PATTERNS = {
    "English": (
        r"\\b(every|everyone|each|all|they|people|person|community)\\b.{0,40}\\b(deserve|deserves|must have|should have)\\b.{0,24}\\b(respect|dignity|equal|rights|support|safety)\\b",
        r"\\b(support|supports|respect|respects|protect|protects)\\b.{0,24}\\b(them|their|people|person|community|rights|dignity|equality)\\b",
        r"\\b(equal respect|equal dignity|equal rights|rights and dignity|dignity and rights|respectful response|respectful responses)\\b",
    ),
    "Hindi": (
        r"(हर|सभी|वे|लोग|व्यक्ति|समुदाय).{0,40}(सम्मान|गरिमा|बराबर|समान|अधिकार|समर्थन|सुरक्षित)",
        r"(सम्मान|समर्थन|सुरक्षा).{0,24}(करें|देना|मिलना|अधिकार)",
    ),
    "Hinglish": (
        r"\\b(sabko|unko|har insaan|yeh log|community)\\b.{0,40}\\b(izzat|barabar|haq|respect|dignity|rights|safe|support)\\b",
        r"\\b(support|respect|protect)\\b.{0,24}\\b(unko|them|community|rights|dignity)\\b",
    ),
}
COUNTER_ENDORSEMENT_MARKERS = {
    "English": (
        "i hate", "we hate", "hate these", "hate them", "deserve abuse",
        "deserve hate", "deserve harassment", "should be abused", "should be attacked",
        "do not respect", "don't respect", "no rights", "not equal",
        "do not belong", "don't belong", "are not valid",
        "disrespect",
    ),
    "Hindi": (
        "मैं नफरत", "हम नफरत", "नफरत के हकदार", "दुर्व्यवहार के हकदार",
        "सम्मान नहीं", "अधिकार नहीं", "बराबर नहीं", "हमला करना चाहिए",
    ),
    "Hinglish": (
        "main nafrat", "hum nafrat", "hate karta", "hate karte",
        "abuse deserve", "hate deserve", "respect nahi", "haq nahi",
        "barabar nahi", "attack karna chahiye",
    ),
}
COUNTER_ENDORSEMENT_PATTERNS = {
    "English": (
        r"(^|[.!?] )\\b(reject|deny|remove|take away)\\b (their |equal |the |all )?(equal rights|rights|dignity|respect|support)\\b",
        r"(^|[.!?] )\\b(stop|oppose) (supporting|respecting|protecting)\\b",
    ),
    "Hindi": (
        r"(छीनो|हटाओ|नकारो).{0,24}(अधिकार|सम्मान|गरिमा|समर्थन)",
    ),
    "Hinglish": (
        r"\\b(reject|deny|hatao|cheeno)\\b.{0,24}\\b(haq|rights|izzat|respect|support)\\b",
    ),
}


def _language_counts(total_per_category):
    if total_per_category == 400:
        return {"English": 134, "Hindi": 133, "Hinglish": 133}
    if total_per_category == 300:
        return {"English": 100, "Hindi": 100, "Hinglish": 100}
    if total_per_category == 3:
        return {"English": 1, "Hindi": 1, "Hinglish": 1}
    raise ValueError("unsupported_per_category_total")


def build_quota_schedule(total_rows, seed):
    if total_rows not in SUPPORTED_TOTALS | {SMOKE_TOTAL}:
        raise ValueError("unsupported_total_rows")
    per_category = total_rows // len(CATEGORIES)
    language_counts = _language_counts(per_category)
    schedule = []
    for category_index, category in enumerate(CATEGORIES):
        category_rows = []
        secondary_index = 0
        for language in LANGUAGES:
            for _ in range(language_counts[language]):
                category_rows.append({
                    "category": category,
                    "target": TARGETS_BY_CATEGORY[category][
                        (secondary_index + category_index) % len(TARGETS_BY_CATEGORY[category])
                    ],
                    "language": language,
                    "platform_style": PLATFORM_STYLES[(secondary_index + category_index) % len(PLATFORM_STYLES)],
                    "abuse_type": ABUSE_TYPES[(secondary_index + 2 * category_index) % len(ABUSE_TYPES)],
                    "severity": SEVERITIES[(secondary_index + category_index) % len(SEVERITIES)],
                })
                secondary_index += 1
        random.Random(seed + category_index).shuffle(category_rows)
        schedule.extend(category_rows)
    for index, row in enumerate(schedule, 1):
        row["request_id"] = f"REQ{index:06d}"
    return schedule


def normalize_for_dedup(value):
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = normalized.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    return re.sub(r"\\s+", " ", normalized).strip()


def find_near_duplicate_pairs(values, threshold, chunk_size=256, start_index=1):
    if not 0 < threshold <= 1:
        raise ValueError("invalid_near_duplicate_threshold")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("invalid_near_duplicate_chunk_size")
    normalized_values = [normalize_for_dedup(value) for value in values]
    if len(normalized_values) < 2:
        return []
    if (
        type(start_index) is not int
        or start_index < 0
        or start_index > len(normalized_values)
    ):
        raise ValueError("invalid_near_duplicate_start_index")
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        matrix = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5)
        ).fit_transform(normalized_values)
    except ValueError:
        return []
    pairs = []
    for start in range(max(1, start_index), matrix.shape[0], chunk_size):
        end = min(matrix.shape[0], start + chunk_size)
        similarities = (matrix[start:end] @ matrix[:end].T).tocsr()
        for local_index in range(end - start):
            current_index = start + local_index
            row_start = similarities.indptr[local_index]
            row_end = similarities.indptr[local_index + 1]
            for position in range(row_start, row_end):
                prior_index = int(similarities.indices[position])
                score = float(similarities.data[position])
                if prior_index < current_index and score >= threshold:
                    pairs.append((prior_index, current_index, score))
    return sorted(pairs, key=lambda item: (item[1], item[0]))


def is_near_duplicate(value, accepted_values, threshold):
    if not accepted_values:
        return False
    values = list(accepted_values) + [value]
    final_index = len(values) - 1
    return any(
        current_index == final_index
        for _, current_index, _ in find_near_duplicate_pairs(values, threshold)
    )


def batch_near_duplicate_reasons(
    candidates, accepted_texts, accepted_counters, threshold
):
    fields = (
        (
            "Text", list(accepted_texts),
            "exact_duplicate_text", "near_duplicate_text",
        ),
        (
            "Counter Narrative", list(accepted_counters),
            "exact_duplicate_counter", "near_duplicate_counter",
        ),
    )
    conflict_sets = []
    for field, existing_values, exact_reason, near_reason in fields:
        existing_normalized = {
            normalize_for_dedup(value) for value in existing_values
        }
        prior_candidates_by_value = {}
        exact_existing = set()
        exact_prior_candidates = {index: set() for index in range(len(candidates))}
        for index, candidate in enumerate(candidates):
            normalized = normalize_for_dedup(candidate[field])
            if normalized in existing_normalized:
                exact_existing.add(index)
            exact_prior_candidates[index].update(
                prior_candidates_by_value.get(normalized, ())
            )
            prior_candidates_by_value.setdefault(normalized, []).append(index)
        offset = len(existing_values)
        values = existing_values + [candidate[field] for candidate in candidates]
        near_existing = set()
        near_prior_candidates = {index: set() for index in range(len(candidates))}
        for prior_index, current_index, _ in find_near_duplicate_pairs(
            values, threshold, start_index=offset
        ):
            if current_index >= offset:
                candidate_index = current_index - offset
                if prior_index < offset:
                    near_existing.add(candidate_index)
                else:
                    near_prior_candidates[candidate_index].add(
                        prior_index - offset
                    )
        conflict_sets.append({
            "exact_reason": exact_reason,
            "near_reason": near_reason,
            "exact_existing": exact_existing,
            "exact_prior_candidates": exact_prior_candidates,
            "near_existing": near_existing,
            "near_prior_candidates": near_prior_candidates,
        })

    reasons = [[] for _ in candidates]
    accepted_candidate_indexes = set()
    for index in range(len(candidates)):
        for conflicts in conflict_sets:
            if (
                index in conflicts["exact_existing"]
                or conflicts["exact_prior_candidates"][index]
                & accepted_candidate_indexes
            ):
                reasons[index].append(conflicts["exact_reason"])
            if (
                index in conflicts["near_existing"]
                or conflicts["near_prior_candidates"][index]
                & accepted_candidate_indexes
            ):
                reasons[index].append(conflicts["near_reason"])
        if not reasons[index]:
            accepted_candidate_indexes.add(index)
    return [sorted(set(candidate_reasons)) for candidate_reasons in reasons]


def _detected_language_code(text):
    try:
        return str(detect(str(text))).casefold()
    except Exception:
        return None


def _language_matches(text, language):
    value = str(text)
    devanagari = len(re.findall(r"[\\u0900-\\u097F]", value))
    latin_tokens = re.findall(r"[a-z]+", value.casefold())
    tokens = set(latin_tokens)
    latin_letters = sum(len(token) for token in latin_tokens)
    total_letters = devanagari + latin_letters
    hinglish_count = len(tokens & HINGLISH_MARKERS)
    detected_language = _detected_language_code(value)
    if language == "Hindi":
        hindi_marker_count = sum(marker in value for marker in HINDI_MARKERS)
        return (
            devanagari >= 2
            and devanagari / max(1, total_letters) >= 0.35
            and hindi_marker_count >= 2
            and detected_language == "hi"
        )
    if language == "Hinglish":
        return devanagari == 0 and len(latin_tokens) >= 3 and hinglish_count >= 2
    if language == "English":
        return (
            devanagari == 0
            and len(latin_tokens) >= 2
            and hinglish_count < 2
            and len(tokens & ENGLISH_MARKERS) >= 2
            and detected_language == "en"
        )
    return False


def _harmful_text_matches(text, language):
    value = normalize_for_dedup(text)
    if any(
        re.search(pattern, value)
        for pattern in ANTI_HATE_TEXT_PATTERNS.get(language, ())
    ):
        return False
    return any(
        re.search(pattern, value)
        for pattern in HARMFUL_TEXT_PATTERNS.get(language, ())
    )


def _harmful_text_checks(text, language, severity, category, abuse_type):
    value = normalize_for_dedup(text)
    reasons = []
    if not _harmful_text_matches(value, language):
        reasons.append("text_not_obviously_harmful")
    if severity in {"severe", "extreme"} and not any(
        re.search(pattern, value)
        for pattern in SEVERE_HOSTILITY_PATTERNS.get(language, ())
    ):
        reasons.append("text_severity_evidence_missing")
    if not any(
        re.search(pattern, value)
        for pattern in CATEGORY_TEXT_PATTERNS.get(category, ())
    ):
        reasons.append("text_category_evidence_missing")
    if not any(
        re.search(pattern, value)
        for pattern in ABUSE_TYPE_TEXT_PATTERNS.get(abuse_type, ())
    ):
        reasons.append("text_abuse_type_evidence_missing")
    return reasons


def _counter_narrative_checks(counter_narrative, language):
    value = normalize_for_dedup(counter_narrative)
    reasons = []
    if (
        any(marker in value for marker in COUNTER_ENDORSEMENT_MARKERS.get(language, ()))
        or any(
            re.search(pattern, value)
            for pattern in COUNTER_ENDORSEMENT_PATTERNS.get(language, ())
        )
    ):
        reasons.append("counter_endorses_abuse")
    if not any(
        re.search(pattern, value)
        for pattern in COUNTER_OPPOSITION_PATTERNS.get(language, ())
    ):
        reasons.append("counter_missing_opposition")
    if not any(
        re.search(pattern, value)
        for pattern in COUNTER_SUPPORT_PATTERNS.get(language, ())
    ):
        reasons.append("counter_missing_support")
    return reasons


def project_generated_record(row):
    return {key: row.get(key) for key in GENERATED_RECORD_KEYS}


def validate_generated_row(row, scheduled):
    reasons = []
    if not isinstance(row, dict):
        return ["missing_or_empty_field"]
    if set(row) != set(GENERATED_RECORD_KEYS):
        reasons.append("record_keys_mismatch")
    if any(
        not isinstance(row.get(key), str) or not row[key].strip()
        for key in GENERATED_RECORD_KEYS
    ):
        return sorted(set(reasons + ["missing_or_empty_field"]))
    mappings = {
        "Category": "category",
        "Target": "target",
        "language": "language",
        "platform_style": "platform_style",
        "abuse_type": "abuse_type",
        "severity": "severity",
        "request_id": "request_id",
    }
    for row_key, schedule_key in mappings.items():
        if row[row_key] != scheduled[schedule_key]:
            reasons.append("category_mismatch" if row_key == "Category" else f"{schedule_key}_mismatch")
    if row["ID"] != scheduled["request_id"]:
        reasons.append("id_mismatch")
    combined = f"{row['Text']} {row['Target']} {row['Counter Narrative']}"
    lowered = normalize_for_dedup(combined)
    if any(marker in lowered for marker in REFUSAL_MARKERS):
        reasons.append("refusal_or_disclaimer")
    if any(marker in lowered for marker in PLACEHOLDER_PROMPT_LEAKAGE_MARKERS):
        reasons.append("placeholder_or_prompt_leakage")
    if any(marker in lowered for marker in OPERATIONAL_ATTACK_MARKERS):
        reasons.append("operational_attack_instruction")
    if CONTACT_PATTERN.search(combined) or ADDRESS_PATTERN.search(combined):
        reasons.append("private_or_contact_marker")
    allowed_capitalized_phrases = {category.casefold() for category in CATEGORIES}
    proper_name_phrases = {
        " ".join(match).casefold()
        for match in PROPER_NAME_PATTERN.findall(combined)
    }
    if proper_name_phrases - allowed_capitalized_phrases:
        reasons.append("real_or_unscheduled_name_marker")
    if not 8 <= len(row["Text"].strip()) <= 500:
        reasons.append("text_length")
    if not 15 <= len(row["Counter Narrative"].strip()) <= 700:
        reasons.append("counter_narrative_length")
    if normalize_for_dedup(row["Text"]) == normalize_for_dedup(row["Counter Narrative"]):
        reasons.append("identical_text_and_counter")
    if not _language_matches(row["Text"], scheduled["language"]):
        reasons.append("text_language_mismatch")
    if not _language_matches(row["Counter Narrative"], scheduled["language"]):
        reasons.append("counter_language_mismatch")
    reasons.extend(
        _harmful_text_checks(
            row["Text"],
            scheduled["language"],
            scheduled["severity"],
            scheduled["category"],
            scheduled["abuse_type"],
        )
    )
    reasons.extend(
        _counter_narrative_checks(row["Counter Narrative"], scheduled["language"])
    )
    return sorted(set(reasons))


def validate_config(config):
    required_keys = {
        "model_id", "total_rows", "smoke_test", "seed", "generation_batch_size",
        "temperature", "top_p", "repetition_penalty", "max_new_tokens",
        "max_request_retries", "near_duplicate_threshold", "run_root",
    }
    missing = sorted(required_keys - set(config))
    if missing:
        raise ValueError(f"invalid_config:missing:{','.join(missing)}")
    if config["model_id"] not in {PRIMARY_MODEL_ID, FALLBACK_MODEL_ID}:
        raise ValueError("invalid_config:model_id")
    if type(config["total_rows"]) is not int or config["total_rows"] not in SUPPORTED_TOTALS:
        raise ValueError("invalid_config:total_rows")
    if type(config["smoke_test"]) is not bool:
        raise ValueError("invalid_config:smoke_test")
    for key, minimum, maximum in (
        ("generation_batch_size", 1, 64),
        ("max_new_tokens", 1, 32768),
        ("max_request_retries", 0, 100),
    ):
        if type(config[key]) is not int or not minimum <= config[key] <= maximum:
            raise ValueError(f"invalid_config:{key}")
    if (
        type(config["seed"]) is not int
        or not 0 <= config["seed"] < 2**63
    ):
        raise ValueError("invalid_config:seed")
    numeric_ranges = {
        "temperature": (0.0, 2.0),
        "top_p": (0.0, 1.0),
        "repetition_penalty": (0.0, 2.0),
        "near_duplicate_threshold": (0.0, 1.0),
    }
    for key, (lower, upper) in numeric_ranges.items():
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not lower < value <= upper
        ):
            raise ValueError(f"invalid_config:{key}")
    if not isinstance(config["run_root"], (str, Path)) or not str(config["run_root"]):
        raise ValueError("invalid_config:run_root")
    return True


validate_config(CONFIG)
TOTAL_ROWS = SMOKE_TOTAL if CONFIG["smoke_test"] else CONFIG["total_rows"]
SCHEDULE = build_quota_schedule(TOTAL_ROWS, CONFIG["seed"])
'''


RUNTIME_CELL = '''import errno
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

assert torch.cuda.is_available(), "A CUDA GPU is required. In Kaggle, select Settings > Accelerator > GPU."
GPU_PROPERTIES = torch.cuda.get_device_properties(0)
GPU_NAME = torch.cuda.get_device_name(0)
EFFECTIVE_DTYPE_NAME = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
EFFECTIVE_DTYPE = torch.bfloat16 if EFFECTIVE_DTYPE_NAME == "bfloat16" else torch.float16
print("GPU before model download", {
    "name": GPU_NAME,
    "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
    "effective_dtype": EFFECTIVE_DTYPE_NAME,
})
RUN_ROOT = Path(CONFIG["run_root"])
RUN_ROOT.mkdir(parents=True, exist_ok=True)
ACCEPTED_PATH = RUN_ROOT / "accepted_rows.jsonl"
REJECTED_PATH = RUN_ROOT / "rejected_events.jsonl"
MANIFEST_PATH = RUN_ROOT / "manifest.json"


def canonical_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


print("Validated pre-model plan", {
    "model_id": CONFIG["model_id"],
    "total_rows": TOTAL_ROWS,
    "schedule_rows": len(SCHEDULE),
    "schedule_hash": canonical_hash(SCHEDULE),
    "generation_batch_size": CONFIG["generation_batch_size"],
    "max_request_retries": CONFIG["max_request_retries"],
})


def extract_first_json_array(text, expected_request_ids=None, expected_count=None):
    decoder = json.JSONDecoder()
    expected_ids = None
    if expected_request_ids is not None:
        expected_ids = list(expected_request_ids)
        if (
            any(not isinstance(request_id, str) or not request_id for request_id in expected_ids)
            or len(set(expected_ids)) != len(expected_ids)
        ):
            raise ValueError("invalid_expected_request_ids")
        if expected_count is None:
            expected_count = len(expected_ids)
    if expected_count is not None and (type(expected_count) is not int or expected_count < 1):
        raise ValueError("invalid_expected_count")
    if expected_ids is not None and expected_count != len(expected_ids):
        raise ValueError("expected_request_id_count_mismatch")
    last_reason = "json_array_not_found"
    for index, character in enumerate(str(text)):
        if character != "[":
            continue
        try:
            value, _ = decoder.raw_decode(str(text)[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, list):
            continue
        if not value:
            last_reason = "json_array_empty"
            continue
        if not all(isinstance(item, dict) for item in value):
            last_reason = "json_array_contains_non_object"
            continue
        if expected_count is not None and len(value) != expected_count:
            last_reason = "json_array_count_mismatch"
            continue
        if expected_ids is not None:
            candidate_ids = [item.get("request_id") for item in value]
            if (
                any(not isinstance(request_id, str) for request_id in candidate_ids)
                or len(set(candidate_ids)) != len(candidate_ids)
                or set(candidate_ids) != set(expected_ids)
            ):
                last_reason = "json_array_request_id_mismatch"
                continue
            if any(
                set(item) != set(GENERATED_RECORD_KEYS)
                or any(
                    not isinstance(item.get(key), str) or not item[key].strip()
                    for key in GENERATED_RECORD_KEYS
                )
                for item in value
            ):
                last_reason = "json_array_record_schema_mismatch"
                continue
        return value
    raise ValueError(last_reason)


def fsync_file(path):
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    unsupported_errnos = {
        value
        for value in (
            errno.EINVAL,
            getattr(errno, "ENOSYS", None),
            getattr(errno, "ENOTSUP", None),
            getattr(errno, "EOPNOTSUPP", None),
        )
        if value is not None
    }
    try:
        descriptor = os.open(str(Path(path)), flags)
    except AttributeError:
        return False
    except OSError as error:
        if error.errno in unsupported_errnos:
            return False
        raise
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno in unsupported_errnos:
            return False
        raise
    finally:
        os.close(descriptor)
    return True


def atomic_write_json(path, value, remove_on_post_replace_failure=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    replaced = False
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        replaced = True
        fsync_directory(path.parent)
    except Exception:
        if replaced and remove_on_post_replace_failure:
            path.unlink(missing_ok=True)
            try:
                fsync_directory(path.parent)
            except OSError:
                pass
        elif not replaced and temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\\n")
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def _quarantine_jsonl_tail(path, tail_bytes):
    path = Path(path)
    quarantine_path = path.with_name(f"{path.name}.corrupt-tail-{time.time_ns()}")
    with quarantine_path.open("xb") as stream:
        stream.write(tail_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)
    return quarantine_path


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty_indices:
        return []
    final_nonempty_index = nonempty_indices[-1]
    rows = []
    offset = 0
    for index, line in enumerate(lines):
        line_start = offset
        offset += len(line)
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if index != final_nonempty_index:
                raise RuntimeError(
                    f"jsonl_interior_corruption:line={index + 1}:{path}"
                ) from error
            _quarantine_jsonl_tail(path, payload[line_start:])
            with path.open("r+b") as stream:
                stream.truncate(line_start)
                stream.flush()
                os.fsync(stream.fileno())
            return rows
    if payload and not payload.endswith(b"\\n"):
        with path.open("ab") as stream:
            stream.write(b"\\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(path.parent)
    return rows


def validate_checkpoint_presence(checkpoint_data_exists, manifest_exists):
    if checkpoint_data_exists and not manifest_exists:
        raise RuntimeError("checkpoint_manifest_missing")
    return True


def read_checkpoint_manifest(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"checkpoint_manifest_invalid:{error}") from error
    if not isinstance(manifest, dict):
        raise RuntimeError("checkpoint_manifest_invalid:not_an_object")
    return manifest


def validate_checkpoint_manifest_structure(manifest, schedule_size):
    if manifest is None:
        return True
    if not isinstance(manifest, dict):
        raise RuntimeError("checkpoint_manifest_invalid:not_an_object")
    required = {
        "identity_hash", "pipeline_identity", "model_id", "model_revision",
        "seed", "accepted_count", "updated_at",
    }
    if not required <= set(manifest):
        raise RuntimeError("checkpoint_manifest_invalid:missing_fields")
    accepted_count = manifest.get("accepted_count")
    if (
        type(accepted_count) is not int
        or accepted_count < 0
        or accepted_count > schedule_size
    ):
        raise RuntimeError("checkpoint_manifest_invalid:accepted_count")
    if not isinstance(manifest.get("pipeline_identity"), dict):
        raise RuntimeError("checkpoint_manifest_invalid:pipeline_identity")
    if re.fullmatch(r"[0-9a-fA-F]{64}", str(manifest.get("identity_hash", ""))) is None:
        raise RuntimeError("checkpoint_manifest_invalid:identity_hash")
    if not isinstance(manifest.get("model_id"), str) or not manifest["model_id"]:
        raise RuntimeError("checkpoint_manifest_invalid:model_id")
    if re.fullmatch(r"[0-9a-fA-F]{40}", str(manifest.get("model_revision", ""))) is None:
        raise RuntimeError("checkpoint_manifest_invalid:model_revision")
    if type(manifest.get("seed")) is not int:
        raise RuntimeError("checkpoint_manifest_invalid:seed")
    if not isinstance(manifest.get("updated_at"), str) or not manifest["updated_at"]:
        raise RuntimeError("checkpoint_manifest_invalid:updated_at")
    return True


def require_model_revision(revision):
    value = str(revision or "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise RuntimeError("immutable_model_revision_unavailable")
    return value.lower()


def select_model_revision(model_id, manifest, resolve_revision):
    if manifest is not None:
        if manifest.get("model_id") != model_id:
            raise RuntimeError("checkpoint_model_id_mismatch")
        return require_model_revision(manifest.get("model_revision"))
    return require_model_revision(resolve_revision(model_id))


def _json_safe_config(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_config(item) for item in value]
    return value


def build_pipeline_identity(
    config,
    schedule,
    model_revision,
    effective_dtype,
    package_versions,
    runtime_identity,
    pipeline_code_sha256,
    system_prompt,
    generated_record_keys,
):
    revision = require_model_revision(model_revision)
    if re.fullmatch(r"[0-9a-f]{64}", str(pipeline_code_sha256)) is None:
        raise RuntimeError("invalid_pipeline_code_sha256")
    return {
        "identity_version": 1,
        "model": {"id": config["model_id"], "revision": revision},
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": str(effective_dtype),
        },
        "prompt_schema": {
            "system_prompt_sha256": canonical_hash(str(system_prompt)),
            "generated_record_keys": list(generated_record_keys),
            "pipeline_code_sha256": str(pipeline_code_sha256),
        },
        "generation_config": _json_safe_config(config),
        "schedule_hash": canonical_hash(schedule),
        "package_versions": dict(sorted(package_versions.items())),
        "runtime_identity": _json_safe_config(runtime_identity),
    }


def validate_resume_manifest(manifest, pipeline_identity, identity_hash):
    if not isinstance(manifest, dict) or not isinstance(pipeline_identity, dict):
        raise RuntimeError("checkpoint_identity_mismatch")
    stored_identity = manifest.get("pipeline_identity")
    stored_hash = manifest.get("identity_hash")
    identity_matches = (
        stored_hash == identity_hash
        and stored_identity == pipeline_identity
        and canonical_hash(stored_identity) == stored_hash
        and canonical_hash(pipeline_identity) == identity_hash
    )
    if not identity_matches:
        raise RuntimeError("checkpoint_identity_mismatch")
    return True


def collect_package_versions(package_names):
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def collect_runtime_identity():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "gpu": GPU_NAME,
    }


def _manifest(pipeline_identity, identity_hash, accepted_count):
    model_identity = pipeline_identity["model"]
    return {
        "identity_hash": identity_hash,
        "pipeline_identity": pipeline_identity,
        "model_id": model_identity["id"],
        "model_revision": model_identity["revision"],
        "seed": pipeline_identity["generation_config"]["seed"],
        "accepted_count": accepted_count,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


IDENTITY_PACKAGE_NAMES = (
    "transformers", "accelerate", "bitsandbytes", "sentencepiece", "torch",
    "numpy", "pandas", "openpyxl", "tqdm", "langdetect", "scikit-learn",
    "matplotlib", "seaborn", "tokenizers", "huggingface-hub", "safetensors",
)
PACKAGE_VERSIONS = collect_package_versions(IDENTITY_PACKAGE_NAMES)
RUNTIME_IDENTITY = collect_runtime_identity()
CHECKPOINT_DATA_EXISTS = ACCEPTED_PATH.exists() or REJECTED_PATH.exists()
validate_checkpoint_presence(CHECKPOINT_DATA_EXISTS, MANIFEST_PATH.exists())
CHECKPOINT_MANIFEST = read_checkpoint_manifest(MANIFEST_PATH)
validate_checkpoint_manifest_structure(CHECKPOINT_MANIFEST, len(SCHEDULE))


def manifest_requires_accepted_count_recovery(manifest, accepted_count):
    return manifest is not None and manifest.get("accepted_count") < accepted_count


def reconcile_accepted_rows(
    accepted_rows, schedule, manifest, near_duplicate_threshold=None
):
    schedule_by_id = {}
    for scheduled in schedule:
        request_id = scheduled.get("request_id") if isinstance(scheduled, dict) else None
        if not isinstance(request_id, str) or not request_id or request_id in schedule_by_id:
            raise RuntimeError("invalid_or_duplicate_schedule_request_id")
        schedule_by_id[request_id] = scheduled

    if manifest is None:
        if accepted_rows:
            raise RuntimeError("checkpoint_manifest_missing")
    else:
        manifest_count = manifest.get("accepted_count")
        if type(manifest_count) is not int or manifest_count < 0:
            raise RuntimeError(f"checkpoint_invalid_accepted_count:{manifest_count}")
        if manifest_count > len(accepted_rows):
            raise RuntimeError(
                f"checkpoint_accepted_rows_missing:manifest={manifest_count}:rows={len(accepted_rows)}"
            )

    accepted_by_id = {}
    row_hashes = set()
    accepted_texts = []
    accepted_counters = []
    normalized_texts = set()
    normalized_counters = set()
    for index, row in enumerate(accepted_rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"checkpoint_invalid_accepted_row:index={index}:not_an_object")
        row_hash = canonical_hash(row)
        if row_hash in row_hashes:
            raise RuntimeError(f"checkpoint_duplicate_accepted_row:index={index}")
        row_hashes.add(row_hash)
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError(
                f"checkpoint_invalid_accepted_request_id:index={index}"
            )
        if request_id not in schedule_by_id:
            raise RuntimeError(f"checkpoint_unknown_request_id:{request_id}")
        if request_id in accepted_by_id:
            raise RuntimeError(f"checkpoint_duplicate_request_id:{request_id}")
        reasons = validate_generated_row(row, schedule_by_id[request_id])
        if reasons:
            raise RuntimeError(
                f"checkpoint_invalid_accepted_row:{request_id}:{','.join(reasons)}"
            )
        normalized_text = normalize_for_dedup(row["Text"])
        normalized_counter = normalize_for_dedup(row["Counter Narrative"])
        if normalized_text in normalized_texts:
            raise RuntimeError(f"checkpoint_duplicate_accepted_text:{request_id}")
        if normalized_counter in normalized_counters:
            raise RuntimeError(f"checkpoint_duplicate_accepted_counter:{request_id}")
        accepted_by_id[request_id] = row
        accepted_texts.append(row["Text"])
        accepted_counters.append(row["Counter Narrative"])
        normalized_texts.add(normalized_text)
        normalized_counters.add(normalized_counter)
    if near_duplicate_threshold is not None:
        text_pairs = find_near_duplicate_pairs(
            accepted_texts, near_duplicate_threshold
        )
        if text_pairs:
            request_id = accepted_rows[text_pairs[0][1]]["request_id"]
            raise RuntimeError(f"checkpoint_near_duplicate_accepted_text:{request_id}")
        counter_pairs = find_near_duplicate_pairs(
            accepted_counters, near_duplicate_threshold
        )
        if counter_pairs:
            request_id = accepted_rows[counter_pairs[0][1]]["request_id"]
            raise RuntimeError(f"checkpoint_near_duplicate_accepted_counter:{request_id}")
    return accepted_by_id


def reconstruct_retry_counts(rejected_events, request_ids):
    if (
        any(not isinstance(request_id, str) or not request_id for request_id in request_ids)
        or len(set(request_ids)) != len(request_ids)
    ):
        raise RuntimeError("checkpoint_invalid_retry_request_ids")
    retry_counts = {request_id: 0 for request_id in request_ids}
    for index, event in enumerate(rejected_events):
        if not isinstance(event, dict):
            raise RuntimeError(f"checkpoint_invalid_rejected_event:index={index}")
        has_one = "request_id" in event
        has_many = "request_ids" in event
        if has_one == has_many:
            raise RuntimeError(f"checkpoint_invalid_rejected_event:index={index}")
        event_request_ids = [event["request_id"]] if has_one else event["request_ids"]
        if not isinstance(event_request_ids, list) or not event_request_ids:
            raise RuntimeError(f"checkpoint_invalid_rejected_event:index={index}")
        if any(
            not isinstance(request_id, str) or not request_id
            for request_id in event_request_ids
        ):
            raise RuntimeError(
                f"checkpoint_invalid_retry_request_id:index={index}"
            )
        if len(set(event_request_ids)) != len(event_request_ids):
            raise RuntimeError(f"checkpoint_duplicate_retry_request_id:index={index}")
        for request_id in event_request_ids:
            if request_id not in retry_counts:
                raise RuntimeError(f"checkpoint_unknown_retry_request_id:{request_id}")
            retry_counts[request_id] += 1
    return retry_counts


def consume_retry_budget(retry_counts, request_ids, max_request_retries, reason):
    for request_id in request_ids:
        if request_id not in retry_counts:
            raise RuntimeError(f"retry_count_missing:{request_id}")
        retry_counts[request_id] += 1
    for request_id in request_ids:
        retry_count = retry_counts[request_id]
        if retry_count > max_request_retries:
            raise RuntimeError(
                f"retry_budget_exhausted:{request_id}:{reason}:retry_count={retry_count}:max={max_request_retries}"
            )
    return retry_counts


def ensure_retry_budgets(retry_counts, request_ids, max_request_retries):
    for request_id in request_ids:
        retry_count = retry_counts[request_id]
        if retry_count > max_request_retries:
            raise RuntimeError(
                f"retry_budget_exhausted:{request_id}:checkpoint_history:retry_count={retry_count}:max={max_request_retries}"
            )
    return True


def reduce_batch_size_after_oom(batch_size, retry_counts, request_ids, max_request_retries):
    consume_retry_budget(
        retry_counts, request_ids, max_request_retries, "cuda_out_of_memory"
    )
    return max(1, batch_size // 2)


def derive_generation_seed(seed, request_ids, retry_counts=None):
    retry_counts = retry_counts or {}
    attempt_state = [
        {"request_id": request_id, "retry_count": int(retry_counts.get(request_id, 0))}
        for request_id in request_ids
    ]
    digest = canonical_hash({"seed": int(seed), "attempt_state": attempt_state})
    return int(digest[:16], 16) % (2**63 - 1)


def seed_generation(seed, request_ids, retry_counts=None):
    derived_seed = derive_generation_seed(seed, request_ids, retry_counts)
    random.seed(derived_seed)
    np.random.seed(derived_seed % (2**32))
    torch.manual_seed(derived_seed)
    torch.cuda.manual_seed_all(derived_seed)
    return derived_seed
'''


MODEL_CELL = '''from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


_NEW_RUN_CONFIG_CACHE = {}


def resolve_current_model_revision(model_id):
    resolved_config = AutoConfig.from_pretrained(model_id)
    _NEW_RUN_CONFIG_CACHE["resolved_config"] = resolved_config
    return require_model_revision(getattr(resolved_config, "_commit_hash", None))


def load_generator(model_id, resolved_config, model_revision, compute_dtype):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=model_revision, use_fast=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_revision,
            config=resolved_config,
            device_map="auto",
            torch_dtype=compute_dtype,
            quantization_config=quantization,
        )
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(
            "model_load_cuda_out_of_memory:restart_kernel_and_select_4b_fallback:"
            f"set MODEL_ID={FALLBACK_MODEL_ID} and use a new run directory"
        ) from error
    model.eval()
    return tokenizer, model


MODEL_REVISION = select_model_revision(
    CONFIG["model_id"], CHECKPOINT_MANIFEST, resolve_current_model_revision
)
PIPELINE_IDENTITY = build_pipeline_identity(
    CONFIG,
    SCHEDULE,
    MODEL_REVISION,
    EFFECTIVE_DTYPE_NAME,
    PACKAGE_VERSIONS,
    RUNTIME_IDENTITY,
    PIPELINE_CODE_SHA256,
    SYSTEM_PROMPT,
    GENERATED_RECORD_KEYS,
)
PIPELINE_IDENTITY_HASH = canonical_hash(PIPELINE_IDENTITY)
if CHECKPOINT_MANIFEST is not None:
    validate_resume_manifest(
        CHECKPOINT_MANIFEST, PIPELINE_IDENTITY, PIPELINE_IDENTITY_HASH
    )
else:
    CHECKPOINT_MANIFEST = _manifest(PIPELINE_IDENTITY, PIPELINE_IDENTITY_HASH, 0)
    atomic_write_json(MANIFEST_PATH, CHECKPOINT_MANIFEST)

RESOLVED_MODEL_CONFIG = _NEW_RUN_CONFIG_CACHE.get("resolved_config")
if RESOLVED_MODEL_CONFIG is None:
    RESOLVED_MODEL_CONFIG = AutoConfig.from_pretrained(
        CONFIG["model_id"], revision=MODEL_REVISION
    )
tokenizer, model = load_generator(
    CONFIG["model_id"], RESOLVED_MODEL_CONFIG, MODEL_REVISION, EFFECTIVE_DTYPE
)


print({
    "model_id": CONFIG["model_id"],
    "model_revision": MODEL_REVISION,
    "identity_hash": PIPELINE_IDENTITY_HASH,
    "gpu": GPU_NAME,
    "effective_dtype": EFFECTIVE_DTYPE_NAME,
    "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
})
'''


GENERATION_CELL = '''def build_messages(assignments, repair_text=None):
    assignment_json = json.dumps(assignments, ensure_ascii=False, indent=2)
    user_text = f"Generate exactly {len(assignments)} objects for these assignments:\\n{assignment_json}"
    if repair_text is not None:
        user_text += f"\\nRepair this malformed response into the required JSON array without changing assignments:\\n{repair_text}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def _model_text(messages):
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            repetition_penalty=CONFIG["repetition_penalty"],
            max_new_tokens=CONFIG["max_new_tokens"],
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def repair_json_once(assignments, malformed_text):
    repaired = _model_text(build_messages(assignments, repair_text=malformed_text))
    request_ids = [assignment["request_id"] for assignment in assignments]
    return extract_first_json_array(
        repaired, expected_request_ids=request_ids, expected_count=len(assignments)
    )


def generate_batch_with_model_text(assignments, model_text):
    request_ids = [assignment["request_id"] for assignment in assignments]
    raw_text = model_text(build_messages(assignments))
    try:
        return extract_first_json_array(
            raw_text, expected_request_ids=request_ids, expected_count=len(assignments)
        )
    except ValueError:
        repaired = model_text(build_messages(assignments, repair_text=raw_text))
        return extract_first_json_array(
            repaired, expected_request_ids=request_ids, expected_count=len(assignments)
        )


def generate_batch(assignments):
    return generate_batch_with_model_text(assignments, _model_text)


def validate_candidate_batch(candidates, expected_request_ids, known_request_ids):
    if not isinstance(candidates, list):
        return {}, ["candidate_batch_not_array"]
    if not candidates:
        return {}, ["candidate_batch_empty"]
    expected_ids = set(expected_request_ids)
    known_ids = set(known_request_ids)
    seen = set()
    reasons = []
    validated = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            reasons.append(f"candidate_not_object:index={index}")
            continue
        request_id = candidate.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            reasons.append(f"candidate_request_id_not_string:index={index}")
            continue
        if request_id in seen:
            reasons.append(f"duplicate_candidate_request_id:{request_id}")
        seen.add(request_id)
        if request_id not in known_ids:
            reasons.append(f"unknown_candidate_request_id:{request_id}")
        elif request_id not in expected_ids:
            reasons.append(f"unexpected_candidate_request_id:{request_id}")
        validated.append((request_id, candidate))
    if reasons:
        return {}, sorted(set(reasons))
    return {request_id: candidate for request_id, candidate in validated}, []


def run_generation(schedule):
    checkpoint_data_exists = ACCEPTED_PATH.exists() or REJECTED_PATH.exists()
    validate_checkpoint_presence(checkpoint_data_exists, MANIFEST_PATH.exists())
    manifest = read_checkpoint_manifest(MANIFEST_PATH)
    validate_checkpoint_manifest_structure(manifest, len(schedule))
    if manifest is not None:
        validate_resume_manifest(
            manifest, PIPELINE_IDENTITY, PIPELINE_IDENTITY_HASH
        )
    else:
        manifest = _manifest(PIPELINE_IDENTITY, PIPELINE_IDENTITY_HASH, 0)
        atomic_write_json(MANIFEST_PATH, manifest)
    accepted = load_jsonl(ACCEPTED_PATH)
    accepted_by_id = reconcile_accepted_rows(
        accepted, schedule, manifest, CONFIG["near_duplicate_threshold"]
    )
    if manifest_requires_accepted_count_recovery(manifest, len(accepted_by_id)):
        atomic_write_json(
            MANIFEST_PATH,
            _manifest(PIPELINE_IDENTITY, PIPELINE_IDENTITY_HASH, len(accepted_by_id)),
        )
    accepted_texts = [row["Text"] for row in accepted]
    accepted_counters = [row["Counter Narrative"] for row in accepted]
    schedule_by_id = {row["request_id"]: row for row in schedule}
    pending_request_ids = [request_id for request_id in schedule_by_id if request_id not in accepted_by_id]
    retry_counts = reconstruct_retry_counts(load_jsonl(REJECTED_PATH), schedule_by_id)
    ensure_retry_budgets(
        retry_counts, pending_request_ids, CONFIG["max_request_retries"]
    )
    batch_size = CONFIG["generation_batch_size"]
    while pending_request_ids:
        request_ids = pending_request_ids[:batch_size]
        assignments = [schedule_by_id[request_id] for request_id in request_ids]
        try:
            # Durable retry counts make the same attempt replay while later attempts vary.
            seed_generation(CONFIG["seed"], request_ids, retry_counts)
            candidates = generate_batch(assignments)
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            gc.collect()
            append_jsonl(REJECTED_PATH, {"request_ids": request_ids, "reason": "cuda_out_of_memory", "error": str(error)})
            batch_size = reduce_batch_size_after_oom(
                batch_size,
                retry_counts,
                request_ids,
                CONFIG["max_request_retries"],
            )
            continue
        except Exception as error:
            append_jsonl(REJECTED_PATH, {"request_ids": request_ids, "reason": "batch_generation_failed", "error": str(error)})
            consume_retry_budget(
                retry_counts,
                request_ids,
                CONFIG["max_request_retries"],
                "batch_generation_failed",
            )
            continue
        candidates_by_id, candidate_batch_reasons = validate_candidate_batch(
            candidates, request_ids, schedule_by_id
        )
        if candidate_batch_reasons:
            append_jsonl(
                REJECTED_PATH,
                {
                    "request_ids": request_ids,
                    "reason": "malformed_candidate_batch",
                    "reasons": candidate_batch_reasons,
                },
            )
            consume_retry_budget(
                retry_counts,
                request_ids,
                CONFIG["max_request_retries"],
                "malformed_candidate_batch",
            )
            continue
        reasons_by_id = {}
        dedup_request_ids = []
        dedup_candidates = []
        for request_id in request_ids:
            scheduled = schedule_by_id[request_id]
            candidate = candidates_by_id.get(request_id)
            reasons = ["missing_candidate"] if candidate is None else validate_generated_row(candidate, scheduled)
            if candidate is not None and not reasons:
                dedup_request_ids.append(request_id)
                dedup_candidates.append(candidate)
            reasons_by_id[request_id] = reasons
        batch_duplicate_reasons = batch_near_duplicate_reasons(
            dedup_candidates,
            accepted_texts,
            accepted_counters,
            CONFIG["near_duplicate_threshold"],
        )
        for request_id, duplicate_reasons in zip(
            dedup_request_ids, batch_duplicate_reasons
        ):
            reasons_by_id[request_id].extend(duplicate_reasons)
        for request_id in request_ids:
            candidate = candidates_by_id.get(request_id)
            reasons = sorted(set(reasons_by_id[request_id]))
            if reasons:
                append_jsonl(REJECTED_PATH, {"request_id": request_id, "reasons": reasons})
                consume_retry_budget(
                    retry_counts,
                    [request_id],
                    CONFIG["max_request_retries"],
                    ",".join(reasons),
                )
                continue
            candidate = project_generated_record(candidate)
            accepted_by_id[request_id] = candidate
            accepted_texts.append(candidate["Text"])
            accepted_counters.append(candidate["Counter Narrative"])
            append_jsonl(ACCEPTED_PATH, candidate)
            pending_request_ids.remove(request_id)
        atomic_write_json(
            MANIFEST_PATH,
            _manifest(PIPELINE_IDENTITY, PIPELINE_IDENTITY_HASH, len(accepted_by_id)),
        )
        tqdm.write(f"accepted={len(accepted_by_id)}/{len(schedule)} pending={len(pending_request_ids)} batch_size={batch_size}")
    return [accepted_by_id[row["request_id"]] for row in schedule]


ACCEPTED_ROWS = run_generation(SCHEDULE)
'''


EXPORT_CELL = '''import matplotlib.pyplot as plt
import seaborn as sns


def finalize_dataset(rows, schedule):
    if len(rows) != len(schedule):
        raise RuntimeError(f"incomplete_generation:{len(rows)}/{len(schedule)}")
    schedule_by_id = {row["request_id"]: row for row in schedule}
    row_ids = [row.get("request_id") for row in rows]
    if len(set(row_ids)) != len(schedule) or set(row_ids) != set(schedule_by_id):
        raise RuntimeError("request_id_coverage_mismatch")
    ordered = sorted(rows, key=lambda row: int(row["request_id"][3:]))
    for row in ordered:
        reasons = validate_generated_row(row, schedule_by_id[row["request_id"]])
        if reasons:
            raise RuntimeError(f"final_validation_failed:{row['request_id']}:{','.join(reasons)}")
    if len({normalize_for_dedup(row["Text"]) for row in ordered}) != len(ordered):
        raise RuntimeError("exact_text_duplicate")
    if len({normalize_for_dedup(row["Counter Narrative"]) for row in ordered}) != len(ordered):
        raise RuntimeError("exact_counter_duplicate")
    frame = pd.DataFrame(ordered)
    frame["ID"] = [f"HS{index:06d}" for index in range(1, len(frame) + 1)]
    final = frame.loc[:, EXPORT_COLUMNS].copy()
    if list(final.columns) != EXPORT_COLUMNS or final.isna().any().any():
        raise RuntimeError("final_schema_validation_failed")
    return final


def audit_quotas(final_dataset, audit_dataset, schedule):
    category_counts = final_dataset["Category"].value_counts().sort_index()
    language_counts = audit_dataset["language"].value_counts().sort_index()
    expected_category_count = len(schedule) // len(CATEGORIES)
    if any(category_counts.get(category, 0) != expected_category_count for category in CATEGORIES):
        raise RuntimeError("category_quota_mismatch")
    expected_language_counts = {
        language: sum(item["language"] == language for item in schedule) for language in LANGUAGES
    }
    if language_counts.to_dict() != expected_language_counts:
        raise RuntimeError("language_quota_mismatch")
    return category_counts, language_counts


def compute_length_summaries(rows):
    def summarize(values):
        ordered = sorted(int(value) for value in values)
        count = len(ordered)
        if not count:
            return {"count": 0, "min": None, "max": None, "mean": None, "median": None}

        def percentile(fraction):
            position = (count - 1) * fraction
            lower = int(position)
            upper = min(count - 1, lower + 1)
            weight = position - lower
            return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)

        return {
            "count": count,
            "min": ordered[0],
            "max": ordered[-1],
            "mean": float(sum(ordered) / count),
            "median": percentile(0.5),
            "p25": percentile(0.25),
            "p75": percentile(0.75),
        }

    return {
        field: summarize(len(str(row[field])) for row in rows)
        for field in ("Text", "Counter Narrative")
    }


def compute_duplicate_statistics(rows, threshold):
    texts = [row["Text"] for row in rows]
    counters = [row["Counter Narrative"] for row in rows]
    return {
        "exact_text_duplicate_count": len(texts) - len({normalize_for_dedup(value) for value in texts}),
        "exact_counter_duplicate_count": len(counters) - len({normalize_for_dedup(value) for value in counters}),
        "near_text_pair_count": len(find_near_duplicate_pairs(texts, threshold)),
        "near_counter_pair_count": len(find_near_duplicate_pairs(counters, threshold)),
    }


def compute_language_consistency(rows):
    by_language = {
        language: {"rows": 0, "text_matches": 0, "counter_matches": 0, "both_match": 0}
        for language in LANGUAGES
    }
    text_match_count = 0
    counter_match_count = 0
    both_match_count = 0
    for row in rows:
        language = row["language"]
        text_matches = _language_matches(row["Text"], language)
        counter_matches = _language_matches(row["Counter Narrative"], language)
        text_match_count += int(text_matches)
        counter_match_count += int(counter_matches)
        both_match_count += int(text_matches and counter_matches)
        bucket = by_language.setdefault(
            language,
            {"rows": 0, "text_matches": 0, "counter_matches": 0, "both_match": 0},
        )
        bucket["rows"] += 1
        bucket["text_matches"] += int(text_matches)
        bucket["counter_matches"] += int(counter_matches)
        bucket["both_match"] += int(text_matches and counter_matches)
    total = len(rows)
    return {
        "row_count": total,
        "text_match_count": text_match_count,
        "counter_match_count": counter_match_count,
        "both_match_count": both_match_count,
        "text_mismatch_count": total - text_match_count,
        "counter_mismatch_count": total - counter_match_count,
        "by_language": by_language,
    }


def publish_data_artifacts(final_dataset, audit_dataset, run_root, run_manifest):
    """Publish data atomically per file; manifest-last marks a complete run."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "run_manifest.json"
    artifacts = [
        ("lgbtq_hatespeech_counter_narratives.csv", lambda path: final_dataset.to_csv(path, index=False, encoding="utf-8")),
        ("lgbtq_hatespeech_counter_narratives.xlsx", lambda path: final_dataset.to_excel(path, index=False)),
        ("generation_audit.csv", lambda path: audit_dataset.to_csv(path, index=False, encoding="utf-8")),
    ]
    temporary_paths = []
    try:
        for filename, serialize in artifacts:
            final_path = run_root / filename
            temporary_path = final_path.with_name(f".{final_path.name}.tmp")
            temporary_paths.append((temporary_path, final_path))
            serialize(temporary_path)
            fsync_file(temporary_path)
        if manifest_path.exists():
            manifest_path.unlink()
            fsync_directory(run_root)
        for temporary_path, final_path in temporary_paths:
            os.replace(temporary_path, final_path)
        fsync_directory(run_root)
        atomic_write_json(
            manifest_path,
            run_manifest,
            remove_on_post_replace_failure=True,
        )
    except Exception:
        for temporary_path, _ in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        raise


FINAL_DATASET = finalize_dataset(ACCEPTED_ROWS, SCHEDULE)
AUDIT_DATASET = pd.DataFrame(ACCEPTED_ROWS)
AUDIT_DATASET["Text Length"] = AUDIT_DATASET["Text"].str.len()
AUDIT_DATASET["Counter Narrative Length"] = AUDIT_DATASET["Counter Narrative"].str.len()
AUDIT_DATASET["text_language_consistent"] = AUDIT_DATASET.apply(
    lambda row: _language_matches(row["Text"], row["language"]), axis=1
)
AUDIT_DATASET["counter_language_consistent"] = AUDIT_DATASET.apply(
    lambda row: _language_matches(row["Counter Narrative"], row["language"]), axis=1
)
category_counts, language_counts = audit_quotas(FINAL_DATASET, AUDIT_DATASET, SCHEDULE)
platform_counts = AUDIT_DATASET["platform_style"].value_counts().sort_index()
severity_counts = AUDIT_DATASET["severity"].value_counts().sort_index()
length_summaries = compute_length_summaries(ACCEPTED_ROWS)
duplicate_statistics = compute_duplicate_statistics(
    ACCEPTED_ROWS, CONFIG["near_duplicate_threshold"]
)
language_consistency_counts = compute_language_consistency(ACCEPTED_ROWS)
if any(duplicate_statistics.values()):
    raise RuntimeError("final_duplicate_audit_failed")
if language_consistency_counts["both_match_count"] != len(ACCEPTED_ROWS):
    raise RuntimeError("final_language_consistency_failed")
rejected_events = load_jsonl(REJECTED_PATH)
rejection_reason_counts = pd.Series(
    [reason for event in rejected_events for reason in event.get("reasons", [event.get("reason", "unknown")])]
).value_counts()

run_manifest = {
    "model_id": CONFIG["model_id"],
    "model_revision": MODEL_REVISION,
    "identity_hash": PIPELINE_IDENTITY_HASH,
    "pipeline_identity": PIPELINE_IDENTITY,
    "seed": CONFIG["seed"],
    "total_rows": len(FINAL_DATASET),
    "export_columns": EXPORT_COLUMNS,
    "category_counts": category_counts.to_dict(),
    "language_counts": language_counts.to_dict(),
    "platform_counts": platform_counts.to_dict(),
    "severity_counts": severity_counts.to_dict(),
    "length_summaries": length_summaries,
    "duplicate_statistics": duplicate_statistics,
    "language_consistency_counts": language_consistency_counts,
    "rejection_reason_counts": rejection_reason_counts.to_dict(),
    "near_duplicate_threshold": CONFIG["near_duplicate_threshold"],
    "schedule_hash": PIPELINE_IDENTITY["schedule_hash"],
    "package_versions": PACKAGE_VERSIONS,
    "runtime_identity": RUNTIME_IDENTITY,
}
publish_data_artifacts(FINAL_DATASET, AUDIT_DATASET, RUN_ROOT, run_manifest)

display(FINAL_DATASET.head())
display(category_counts.rename("count").to_frame())
display(language_counts.rename("count").to_frame())
display(platform_counts.rename("count").to_frame())
display(severity_counts.rename("count").to_frame())
display(rejection_reason_counts.rename("count").to_frame())
display(pd.DataFrame(length_summaries))
display(pd.DataFrame(language_consistency_counts["by_language"]).T)
display(pd.Series(duplicate_statistics, name="count").to_frame())
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x=category_counts.values, y=category_counts.index, ax=axes[0])
axes[0].set_title("Accepted rows by category")
sns.barplot(x=language_counts.index, y=language_counts.values, ax=axes[1])
axes[1].set_title("Accepted rows by language")
plt.tight_layout()
plt.show()
print("Saved exports to", RUN_ROOT)
'''


def compute_pipeline_code_sha256(emitted_pipeline_sources) -> str:
    payload = json.dumps(
        list(emitted_pipeline_sources),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rendered_config_cell() -> str:
    pipeline_sources = [
        PLANNING_VALIDATION_CELL,
        RUNTIME_CELL,
        MODEL_CELL,
        GENERATION_CELL,
        EXPORT_CELL,
    ]
    emitted_sources = ["".join(code(source)["source"]) for source in pipeline_sources]
    pipeline_hash = compute_pipeline_code_sha256(emitted_sources)
    return CONFIG_CELL_TEMPLATE.replace("__PIPELINE_CODE_SHA256__", pipeline_hash)


def build_notebook(output_path: Path) -> None:
    cells = [
        markdown("""# Qwen3.5 LGBTQIA+ Hate-Speech Dataset Generator

This Kaggle notebook creates a synthetic research dataset for automated hate-speech detection and reporting. It produces severe multilingual social-media examples and same-language counter-narratives without using real people or private information.

Enable a GPU accelerator and Internet access before running all cells."""),
        code(SETUP_CELL),
        code(rendered_config_cell()),
        code(PLANNING_VALIDATION_CELL),
        code(RUNTIME_CELL),
        code(MODEL_CELL),
        code(GENERATION_CELL),
        code(EXPORT_CELL),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "gpu", "internet": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_notebook(
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "kaggle_qwen35_lgbtq_hatespeech_dataset.ipynb"
    )
