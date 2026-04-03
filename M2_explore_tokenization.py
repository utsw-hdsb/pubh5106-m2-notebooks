# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exploration: How LLMs See Text
#
# **PUBH 5106 — AI in Health Applications | Module 2, Segment B**
#
# ## What This Notebook Is For
#
# This is a **pre-class exploration** — a short, self-guided activity designed
# to build your intuition about tokenization before the lecture. There is
# nothing to submit. Experiment freely.
#
# ## What You Will Do
#
# Large language models cannot read text. They process **tokens** — small
# numerical chunks produced by a tokenizer. The way text is split into tokens
# has real consequences for clinical AI: it affects what the model can
# "see," how much text fits in a single request, and where errors creep in.
#
# In this notebook you will:
#
# 1. Tokenize everyday English and compare it to clinical text
# 2. See how a tokenizer splits medical abbreviations and drug names
# 3. Measure token counts and think about context window limits
# 4. Search the tokenizer's vocabulary for clinical terms
#
# **Time:** ~20 minutes
#
# **Platform:** BinderHub

# %% [markdown]
# ---
# ## Setup

# %%
import tiktoken

# Load the cl100k_base tokenizer (used by GPT-4, GPT-3.5-turbo)
enc = tiktoken.get_encoding("cl100k_base")

# Quick test
tokens = enc.encode("Hello, world!")
print(f"Text:   'Hello, world!'")
print(f"Tokens: {tokens}")
print(f"Count:  {len(tokens)}")

# %% [markdown]
# ---
# ## Part 1: Everyday English vs. Clinical Text
#
# Let's compare how the tokenizer handles ordinary language versus the kind
# of text you find in clinical notes.

# %%
def show_tokens(text, encoder=enc):
    """Tokenize text and display each token with its ID."""
    token_ids = encoder.encode(text)
    token_strings = [encoder.decode([tid]) for tid in token_ids]
    print(f"Text:    {text}")
    print(f"Tokens:  {token_strings}")
    print(f"IDs:     {token_ids}")
    print(f"Count:   {len(token_ids)}")
    print()
    return token_ids

# %%
# Everyday English
show_tokens("The patient is feeling much better today.")

# Clinical note
show_tokens("Pt c/o SOB, denies CP. Hx of HTN, T2DM, CKD stage 3.")

# %% [markdown]
# **Look at the output above.**
#
# - Which words stayed whole?
# - Which got split into pieces?
# - Did the abbreviations (SOB, CP, HTN, T2DM, CKD) survive as single tokens
#   or get broken apart?

# %% [markdown]
# ---
# ## Part 2: How Does the Tokenizer Split Drug Names?
#
# Drug names are often long, rare words that the tokenizer has never seen as
# a unit. Let's see what happens.

# %%
drugs = [
    "aspirin",
    "metformin",
    "atorvastatin",
    "pembrolizumab",
    "empagliflozin",
    "hydroxychloroquine",
    "trimethoprim-sulfamethoxazole",
]

print("Drug Name Tokenization")
print("=" * 60)
for drug in drugs:
    ids = enc.encode(drug)
    pieces = [enc.decode([tid]) for tid in ids]
    print(f"  {drug:<35} → {pieces}")

# %% [markdown]
# **Notice the pattern:**
#
# - Common, short words ("aspirin") tend to be single tokens.
# - Longer or rarer names get split into subword pieces that may have
#   no medical meaning on their own.
#
# This is **byte-pair encoding (BPE)** — the tokenizer learns common
# character sequences from its training data. Medical terminology is
# underrepresented in general-purpose training corpora, so clinical
# words get fragmented more aggressively.

# %% [markdown]
# ---
# ## Part 3: Token Counts and Context Windows
#
# Every LLM has a **context window** — the maximum number of tokens it can
# process in a single request. For example:
#
# - GPT-3.5-turbo: 4,096 tokens
# - GPT-4: 8,192 or 128,000 tokens
# - Llama 3.2 (3B): 8,192 tokens
#
# Clinical notes can be long. Let's see how quickly tokens add up.

# %%
# A short clinical note
short_note = """
ASSESSMENT AND PLAN:
1. Acute exacerbation of COPD - Start prednisone 40mg daily x5 days,
   albuterol nebulizer q4h PRN, azithromycin 500mg x1 then 250mg daily x4.
   Continue home tiotropium. Check sputum culture.
2. Type 2 diabetes mellitus - Hold metformin given acute illness.
   Sliding scale insulin. Monitor BG QID.
3. Hypertension - Continue amlodipine 10mg daily. Hold lisinopril given
   elevated creatinine. Recheck BMP in AM.
"""

# A plain English version of the same content
plain_version = """
ASSESSMENT AND PLAN:
1. The patient's chronic obstructive pulmonary disease has gotten worse.
   Start prednisone 40 milligrams daily for five days, albuterol breathing
   treatments every four hours as needed, azithromycin 500 milligrams once
   then 250 milligrams daily for four days. Continue the home tiotropium
   inhaler. Check a sputum culture to look for bacteria.
2. Type 2 diabetes - Stop the metformin because of the acute illness.
   Use a sliding scale for insulin dosing. Check blood sugar four times
   a day.
3. High blood pressure - Continue amlodipine 10 milligrams daily. Stop
   the lisinopril because the creatinine level is elevated. Recheck the
   basic metabolic panel in the morning.
"""

clinical_tokens = enc.encode(short_note)
plain_tokens = enc.encode(plain_version)

print(f"Clinical note:     {len(clinical_tokens):>4} tokens")
print(f"Plain English:     {len(plain_tokens):>4} tokens")
print(f"Difference:        {len(plain_tokens) - len(clinical_tokens):>+4} tokens")
print(f"Ratio:             {len(plain_tokens) / len(clinical_tokens):.2f}x")

# %% [markdown]
# **Think about this:**
#
# - Clinical abbreviations are more token-efficient — the same information
#   in fewer tokens.
# - But those abbreviations may be split into meaningless subwords.
# - There is a tension: abbreviations *save* context window space but may
#   *degrade* the model's ability to understand the content.

# %% [markdown]
# ---
# ## Part 4: What's in the Vocabulary?
#
# The cl100k_base tokenizer has about 100,000 tokens in its vocabulary.
# Let's search it for clinical terms.

# %%
# Build a reverse lookup: token string → token ID
vocab = {}
for i in range(enc.n_vocab):
    try:
        token_str = enc.decode([i])
        vocab[token_str] = i
    except Exception:
        pass

print(f"Vocabulary size: {len(vocab):,} tokens")

# %%
# Search for clinical terms in the vocabulary
search_terms = [
    "hypertension",
    "diabetes",
    "atorvastatin",
    "metformin",
    "pneumonia",
    "COPD",
    "eGFR",
    "HbA1c",
    "PCSK9",
    "dyslipidemia",
]

print("Is this an exact token in the vocabulary?")
print("=" * 50)
for term in search_terms:
    if term in vocab:
        print(f"  {term:<20} YES (token ID {vocab[term]})")
    else:
        ids = enc.encode(term)
        pieces = [enc.decode([tid]) for tid in ids]
        print(f"  {term:<20} NO  → split into {pieces}")

# %% [markdown]
# **The takeaway:** Common English words and some frequent medical terms
# exist as whole tokens. Rarer clinical terms are broken into subword
# pieces. This means the model has to *reconstruct* meaning from fragments
# rather than processing the term as a single unit.

# %% [markdown]
# ---
# ## Part 5: Your Turn — Experiment
#
# Try tokenizing text from your own domain. Some ideas:
#
# - A medication list from a discharge summary
# - A radiology report impression
# - Text in another language (Spanish medical terms?)
# - ICD-10 codes (e.g., "E11.65", "I10", "J44.1")
# - A clinical question you might ask an AI assistant

# %%
# Your experiments here
your_text = "Replace this with your own text"
show_tokens(your_text)

# %% [markdown]
# ---
# ## Reflection
#
# Before moving on to the lecture, think about:
#
# 1. If a clinical abbreviation like "SOB" (shortness of breath) is a
#    single token, does the model "know" it means shortness of breath?
#    Or could it confuse it with the everyday English word?
#
# 2. A discharge summary might be 2,000 tokens. A full hospital stay
#    (admission note + daily progress notes + labs + imaging reports)
#    could easily exceed 50,000 tokens. What are the implications for
#    AI systems that try to summarize a patient's entire chart?
#
# 3. If you were building a tokenizer specifically for clinical text,
#    what would you do differently?
