from typing import List, Set, Tuple

# Comprehensive Hindi Stop Words (Devanagari)
HINDI_STOPWORDS: Set[str] = {
    "का", "के", "की", "को", "में", "से", "पर", "और", "या", "तथा", "एवं",
    "है", "हैं", "था", "थी", "थे", "होगा", "होगी", "होंगे", "होता", "होती", "होते",
    "यह", "वह", "ये", "वे", "इस", "उस", "इन", "उन", "इसे", "उसे", "इन्हें", "उन्हें",
    "ने", "तो", "भी", "ही", "तक", "मात्र", "भर",
    "एक", "दो", "तीन", "चार", "सब", "सभी", "कुछ", "कई",
    "कर", "किया", "किए", "की", "करना", "करने", "करती", "करता", "करते",
    "रहा", "रही", "रहे", "गया", "गई", "गए", "जाना", "जाता", "जाती", "जाते",
    "अपने", "अपनी", "अपना", "हमारे", "हमारी", "हमारा", "उनके", "उनकी", "उनका",
    "द्वारा", "लिए", "बारे", "साथ", "बिना", "अनुसार", "कारण", "जब", "तब", "यहाँ", "वहाँ",
    "कि", "जो", "जिस", "जिन", "जिसने", "जिन्होंने", "कहा", "कहते", "होने"
}

# Common Romanized Hindi Stop Words
ROMANIZED_HINDI_STOPWORDS: Set[str] = {
    "ka", "ke", "ki", "ko", "mein", "men", "se", "par", "aur", "ya",
    "hai", "hain", "tha", "thi", "the", "hoga", "hogi", "honge", "hota", "hoti", "hote",
    "yeh", "woh", "voh", "is", "us", "in", "un", "ise", "use",
    "ne", "to", "toh", "bhi", "hi", "tak", "matra",
    "ek", "do", "teen", "chaar", "sab", "sabhi", "kuch", "kai",
    "kar", "kiya", "kiye", "karna", "karne", "karta", "karti", "karte",
    "raha", "rahi", "rahe", "gaya", "gayi", "gaye", "jaana", "jata", "jati",
    "apne", "apni", "apna", "hamara", "hamari", "hamare",
    "dwara", "liye", "ke-liye", "saath", "bina", "anusar", "karan", "jab", "tab", "yahan", "wahan",
    "ki", "jo", "jis", "jin", "jisne", "kaha", "kaise", "hona"
}

# Standard English Stop Words
ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "isn't", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours"
}

ALL_STOPWORDS: Set[str] = HINDI_STOPWORDS | ROMANIZED_HINDI_STOPWORDS | ENGLISH_STOPWORDS


def is_stopword(token: str) -> bool:
    """Check if token is a stop word in any supported language/script."""
    return token.lower().strip() in ALL_STOPWORDS


def remove_stopwords(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """
    Filter tokens and return (retained_tokens, removed_stopwords).
    Preserves token order.
    """
    retained: List[str] = []
    removed: List[str] = []

    for token in tokens:
        clean = token.strip()
        if not clean:
            continue
        if is_stopword(clean):
            removed.append(clean)
        else:
            retained.append(clean)

    # In case all tokens were removed, retain original to avoid empty query
    if not retained and tokens:
        return tokens, []

    return retained, removed
