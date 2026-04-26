"""
Extended training dataset for fine-tuning a Gemini model on mood classification.

Each example is {"text_input": ..., "output": ...} where output is one of:
  positive | negative | neutral | mixed

The dataset covers: direct emotion, sarcasm, negation, slang, emojis,
ambiguous phrasing, and mixed feelings — designed to be harder than the
base SAMPLE_POSTS so the fine-tuned model generalizes better.
"""

FINETUNE_EXAMPLES = [
    # ── Clearly positive ──────────────────────────────────────────────
    {"text_input": "I love this class so much", "output": "positive"},
    {"text_input": "So excited for the weekend!", "output": "positive"},
    {"text_input": "I can't wait for the concert tonight! 🎉", "output": "positive"},
    {"text_input": "Best day ever, everything went perfectly", "output": "positive"},
    {"text_input": "Just got the job offer!! I'm so happy", "output": "positive"},
    {"text_input": "This song is absolutely amazing 🔥", "output": "positive"},
    {"text_input": "Finally finished my project, feeling great about it", "output": "positive"},
    {"text_input": "My dog is the cutest thing in the world 😊", "output": "positive"},
    {"text_input": "I'm cooking on this project rn", "output": "positive"},
    {"text_input": "Lebron James my GOAT 🐐", "output": "positive"},
    {"text_input": "no cap this was the best meal I've ever had", "output": "positive"},
    {"text_input": "Lowkey proud of myself for getting through today", "output": "positive"},
    {"text_input": "That movie was an absolute banger", "output": "positive"},
    {"text_input": "Woke up feeling blessed and grateful today", "output": "positive"},
    {"text_input": "The vibes at this party are immaculate fr", "output": "positive"},

    # ── Clearly negative ──────────────────────────────────────────────
    {"text_input": "Today was a terrible day", "output": "negative"},
    {"text_input": "I'm gonna get cooked on this math test!", "output": "negative"},
    {"text_input": "Legend of Zelda: Breath of the Wild is so buns", "output": "negative"},
    {"text_input": "Stop trolling me bruh", "output": "negative"},
    {"text_input": "I am not happy about this at all", "output": "negative"},
    {"text_input": "This is the worst thing I've ever experienced", "output": "negative"},
    {"text_input": "I feel so overwhelmed and lost right now", "output": "negative"},
    {"text_input": "Failed my exam again, I hate this", "output": "negative"},
    {"text_input": "Can't believe they cancelled my favorite show 😭", "output": "negative"},
    {"text_input": "Traffic is making me lose my mind rn", "output": "negative"},
    {"text_input": "My wifi keeps cutting out and I have a deadline tomorrow", "output": "negative"},
    {"text_input": "Just found out I didn't make the team", "output": "negative"},
    {"text_input": "Feeling super drained and burnt out", "output": "negative"},
    {"text_input": "Nobody showed up to my event, that hurts", "output": "negative"},
    {"text_input": "I hate how things turned out 💔", "output": "negative"},

    # ── Sarcasm (positive words, negative meaning) ───────────────────
    {"text_input": "I absolutely love getting stuck in traffic 🙄", "output": "negative"},
    {"text_input": "Can't wait to waste my time on some gardening.", "output": "negative"},
    {"text_input": "Oh wow, another Monday. Just what I needed.", "output": "negative"},
    {"text_input": "Great, my laptop crashed right before the deadline. Awesome.", "output": "negative"},
    {"text_input": "Yay, another all-nighter. Living the dream.", "output": "negative"},
    {"text_input": "Love when my alarm doesn't go off. Really helpful.", "output": "negative"},
    {"text_input": "So fun sitting in a 3-hour lecture that could've been an email", "output": "negative"},
    {"text_input": "Oh brilliant, they moved the deadline again.", "output": "negative"},

    # ── Negation ─────────────────────────────────────────────────────
    {"text_input": "I'm not happy about this decision", "output": "negative"},
    {"text_input": "Not feeling great today honestly", "output": "negative"},
    {"text_input": "I don't hate it, it's actually pretty good", "output": "positive"},
    {"text_input": "It's not bad at all, I kind of liked it", "output": "positive"},
    {"text_input": "Can't say I'm upset about the outcome", "output": "positive"},
    {"text_input": "Not the worst day I've had", "output": "neutral"},
    {"text_input": "I wouldn't say I'm thrilled but it's fine", "output": "neutral"},

    # ── Neutral / Ambiguous ───────────────────────────────────────────
    {"text_input": "This is fine", "output": "neutral"},
    {"text_input": "i'm fine. totally fine. everything is fine.", "output": "negative"},
    {"text_input": "I just ate but I'm lowkey hungry.", "output": "neutral"},
    {"text_input": "Just got home from work", "output": "neutral"},
    {"text_input": "Today happened. That's about all I can say.", "output": "neutral"},
    {"text_input": "Went to the store. Got what I needed.", "output": "neutral"},
    {"text_input": "It is what it is", "output": "neutral"},
    {"text_input": "I guess today was okay", "output": "neutral"},
    {"text_input": "Not sure how to feel about this yet", "output": "neutral"},

    # ── Mixed feelings ────────────────────────────────────────────────
    {"text_input": "Feeling tired but kind of hopeful", "output": "mixed"},
    {"text_input": "Happy it's over but sad it ended", "output": "mixed"},
    {"text_input": "Excited for the new job but scared of the change", "output": "mixed"},
    {"text_input": "Proud of how far I've come but exhausted from the journey", "output": "mixed"},
    {"text_input": "Lowkey stressed but kind of proud of myself", "output": "mixed"},
    {"text_input": "I love my family but they drive me crazy", "output": "mixed"},
    {"text_input": "This bittersweet feeling is hard to describe", "output": "mixed"},
    {"text_input": "Relieved it's done but a little sad it's over", "output": "mixed"},
    {"text_input": "I'm having fun but also kind of wishing I stayed home", "output": "mixed"},
    {"text_input": "Grateful for the opportunity even though it was really hard", "output": "mixed"},

    # ── Emoji-heavy ───────────────────────────────────────────────────
    {"text_input": "😭😭😭 why does this keep happening to me", "output": "negative"},
    {"text_input": "LFG!!! 🔥🔥🔥 we finally won!!", "output": "positive"},
    {"text_input": "💀💀 I literally cannot right now", "output": "negative"},
    {"text_input": "Everything is great 😊❤️", "output": "positive"},
    {"text_input": "This weather is 😤😤 absolutely ruining my day", "output": "negative"},

    # ── Internet slang ────────────────────────────────────────────────
    {"text_input": "bro said that and I'm dead 💀", "output": "positive"},
    {"text_input": "this whole situation is cooked", "output": "negative"},
    {"text_input": "ngl that slapped hard", "output": "positive"},
    {"text_input": "on god this is mid", "output": "negative"},
    {"text_input": "it's giving main character energy today", "output": "positive"},
    {"text_input": "that's sus and I'm not about it", "output": "negative"},
    {"text_input": "W behavior honestly", "output": "positive"},
    {"text_input": "biggest L of my life rn", "output": "negative"},
]
