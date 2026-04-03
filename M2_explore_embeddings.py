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
# # Exploration: Word Embeddings and Distributional Semantics
#
# **PUBH 5106 — AI in Health Applications | Module 2, Segment B**
#
# ## What This Notebook Is For
#
# This is a **pre-class exploration** designed to build your intuition about
# word embeddings before the lecture on how LLMs represent text. There is
# nothing to submit. Experiment freely.
#
# ## The Big Idea
#
# In 1957, the British linguist J.R. Firth wrote:
#
# > *"You shall know a word by the company it keeps."*
#
# This principle — **distributional semantics** — is the theoretical
# foundation for word embeddings. Words that appear in similar contexts
# end up with similar numerical representations. In this notebook, you
# will see this principle in action.
#
# ## What You Will Do
#
# 1. Load pre-trained word embeddings (GloVe)
# 2. Find the "neighbors" of clinical terms — testing Firth's principle
# 3. Try vector arithmetic with words (the famous "king - man + woman" trick)
# 4. Visualize clusters of medical terms in 2D
# 5. Probe for gaps and biases in the embeddings
#
# **Time:** ~25 minutes
#
# **Platform:** BinderHub

# %% [markdown]
# ---
# ## Setup
#
# We use GloVe embeddings (Global Vectors for Word Representation),
# pre-trained on 6 billion tokens of text from Wikipedia and news articles.
# Each word is represented as a vector of 100 numbers.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api

# Download GloVe embeddings (this takes 1-2 minutes on Colab)
print("Loading GloVe embeddings (this may take a minute)...")
glove = api.load("glove-wiki-gigaword-100")
print(f"Loaded {len(glove.key_to_index):,} word vectors, each with {glove.vector_size} dimensions.")

# %% [markdown]
# ---
# ## Part 1: Testing Firth's Principle
#
# If Firth is right, words that appear in similar contexts should have
# similar embeddings. Let's check: what are the nearest neighbors of
# some clinical terms?

# %%
def show_neighbors(word, model=glove, n=10):
    """Show the n most similar words to a given word."""
    if word not in model:
        print(f"'{word}' is not in the vocabulary.")
        return
    neighbors = model.most_similar(word, topn=n)
    print(f"Nearest neighbors of '{word}':")
    for neighbor, score in neighbors:
        print(f"  {neighbor:<20} similarity: {score:.3f}")
    print()

# %%
show_neighbors("diabetes")

# %%
show_neighbors("hypertension")

# %%
show_neighbors("aspirin")

# %% [markdown]
# **Look at the results:**
#
# - Are the neighbors medically sensible?
# - Do they reflect the *meaning* of the word or just the *context* in which
#   it typically appears?
# - Firth's principle says these are the same thing. Do you agree?

# %% [markdown]
# ---
# ## Part 2: Vector Arithmetic
#
# One of the most striking properties of word embeddings is that
# **relationships between words are encoded as directions in vector space.**
#
# The classic example:
#
# > king − man + woman ≈ queen
#
# This works because the "gender direction" (man → woman) is consistent
# across related word pairs. Let's try it.

# %%
def analogy(a, b, c, model=glove, n=5):
    """Solve: a is to b as c is to ___"""
    if any(w not in model for w in [a, b, c]):
        missing = [w for w in [a, b, c] if w not in model]
        print(f"Words not in vocabulary: {missing}")
        return
    result = model.most_similar(positive=[b, c], negative=[a], topn=n)
    print(f"{a} is to {b} as {c} is to ___")
    for word, score in result:
        print(f"  {word:<20} ({score:.3f})")
    print()

# %%
# The classic
analogy("man", "woman", "king")

# %%
# Clinical analogies — do these work?
analogy("heart", "cardiologist", "brain")

# %%
analogy("diabetes", "insulin", "hypertension")

# %%
analogy("doctor", "hospital", "teacher")

# %% [markdown]
# **Try your own analogies.** Some will work, some won't. When they fail,
# think about why — what does the failure tell you about what the
# embeddings actually captured from the training text?

# %%
# Your analogy experiments here
analogy("___", "___", "___")

# %% [markdown]
# ---
# ## Part 3: Visualizing Clinical Term Clusters
#
# Word embeddings live in 100-dimensional space. We can use PCA
# (Principal Component Analysis) to project them down to 2D for
# visualization. If Firth's principle holds, related terms should
# cluster together.

# %%
# Define groups of clinical terms
term_groups = {
    "Diseases": [
        "diabetes", "hypertension", "pneumonia", "asthma",
        "cancer", "stroke", "epilepsy", "arthritis",
    ],
    "Drugs": [
        "aspirin", "insulin", "metformin", "penicillin",
        "morphine", "warfarin", "prednisone", "ibuprofen",
    ],
    "Anatomy": [
        "heart", "lung", "kidney", "liver",
        "brain", "bone", "skin", "stomach",
    ],
    "Procedures": [
        "surgery", "biopsy", "transplant", "dialysis",
        "chemotherapy", "radiation", "intubation", "catheter",
    ],
}

# Collect vectors and labels
words = []
vectors = []
colors = []
color_map = {"Diseases": "red", "Drugs": "blue", "Anatomy": "green", "Procedures": "orange"}

for group, term_list in term_groups.items():
    for term in term_list:
        if term in glove:
            words.append(term)
            vectors.append(glove[term])
            colors.append(color_map[group])

vectors = np.array(vectors)

# %%
# Project to 2D with PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)

plt.figure(figsize=(12, 8))
for group, color in color_map.items():
    mask = [c == color for c in colors]
    group_coords = coords[np.array(mask)]
    group_words = [w for w, m in zip(words, mask) if m]
    plt.scatter(group_coords[:, 0], group_coords[:, 1],
                c=color, label=group, s=80, alpha=0.7)
    for i, word in enumerate(group_words):
        plt.annotate(word, (group_coords[i, 0] + 0.02, group_coords[i, 1] + 0.02),
                     fontsize=9)

plt.legend(fontsize=12)
plt.title("Clinical Terms in Embedding Space (PCA projection to 2D)", fontsize=14)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Questions to consider:**
#
# - Do the four groups form distinct clusters?
# - Which terms are "between" groups? Why might that be?
# - Are any terms surprisingly close to or far from where you would expect?

# %% [markdown]
# ---
# ## Part 4: What's Missing?
#
# These embeddings were trained on Wikipedia and news text — not clinical
# notes. Let's probe for gaps.

# %%
# Which clinical terms are in the vocabulary?
clinical_terms = [
    "hypertension", "diabetes", "pneumonia",       # common
    "atorvastatin", "metformin", "lisinopril",      # drugs
    "dyslipidemia", "hyperlipidemia",               # clinical terms
    "eGFR", "HbA1c", "BNP",                        # lab values / biomarkers
    "COPD", "CHF", "DVT",                           # abbreviations
    "comorbidity", "polypharmacy",                   # clinical concepts
    "sepsis", "tachycardia", "bradycardia",         # clinical findings
]

print("Clinical Term Coverage in GloVe")
print("=" * 50)
found = 0
for term in clinical_terms:
    present = term in glove
    if present:
        found += 1
    status = "FOUND" if present else "MISSING"
    print(f"  {term:<25} {status}")

print(f"\n{found}/{len(clinical_terms)} terms found ({found/len(clinical_terms):.0%})")

# %% [markdown]
# **This is a real problem.** General-purpose embeddings have gaps in
# clinical vocabulary — especially for:
#
# - Abbreviations (COPD, CHF, DVT)
# - Lab values and biomarkers (eGFR, HbA1c, BNP)
# - Specialized clinical terms
#
# This is why projects like **BioBERT** and **ClinicalBERT** exist —
# they retrain embeddings on biomedical and clinical text so that
# medical vocabulary is properly represented.

# %% [markdown]
# ---
# ## Part 5: Probing for Bias
#
# Embeddings encode the patterns of their training data — including
# social biases present in that data. Let's look.

# %%
# Similarity between professions and gendered terms
professions = ["doctor", "nurse", "surgeon", "therapist",
               "pharmacist", "dentist", "psychiatrist", "midwife"]

print("Profession similarity to 'man' vs 'woman'")
print("=" * 55)
print(f"  {'Profession':<15} {'sim(man)':<12} {'sim(woman)':<12} {'Difference':<12}")
print("-" * 55)
for prof in professions:
    if prof in glove:
        sim_man = glove.similarity(prof, "man")
        sim_woman = glove.similarity(prof, "woman")
        diff = sim_woman - sim_man
        marker = " ←" if abs(diff) > 0.05 else ""
        print(f"  {prof:<15} {sim_man:<12.3f} {sim_woman:<12.3f} {diff:<+12.3f}{marker}")

# %% [markdown]
# **Think about this:**
#
# - Are any professions notably more associated with one gender?
# - These biases come from the training data (Wikipedia, news).
# - If an LLM's embeddings encode "nurse ≈ woman" and "surgeon ≈ man,"
#   how might that affect clinical AI applications?
# - This is not a hypothetical concern — it has been documented in
#   real deployed systems.

# %% [markdown]
# ---
# ## Part 6: Your Turn — Explore
#
# Some ideas for further exploration:
#
# - Find neighbors of a disease you're interested in
# - Try analogies from your own clinical or public health experience
# - Compare terms across languages (e.g., "hospital" vs. its cognates)
# - Look for terms that are similar in embedding space but different
#   in meaning (false friends)

# %%
# Your experiments here


# %% [markdown]
# ---
# ## Reflection
#
# Before the lecture, think about:
#
# 1. Firth said "you shall know a word by the company it keeps."
#    Based on what you've seen, what does an embedding actually capture —
#    meaning, usage patterns, or something else?
#
# 2. These embeddings were trained on general text. If you retrained
#    them on 10 million clinical notes, how would the neighbors of
#    "diabetes" change?
#
# 3. The bias probe showed that embeddings encode social patterns from
#    their training data. Should we "fix" these biases in clinical
#    embeddings, or is it more important to be aware of them? What
#    are the risks of each approach?
