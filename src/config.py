"""Central configuration: paths, file names and data-loading options."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
IMAGES_DIR = PROJECT_ROOT / "images"
DOCS_DIR = PROJECT_ROOT / "docs"

TRANSACTIONS_FILE = RAW_DIR / "transactions_train.csv"
ARTICLES_FILE = RAW_DIR / "articles.csv"
CUSTOMERS_FILE = RAW_DIR / "customers.csv"
ARTICLE_IMAGES_DIR = RAW_DIR / "images"  # optional: H&M product photos

# Part A of the notebook streams the FULL transactions file in chunks (low memory).
CHUNK_SIZE = 2_000_000

# Working set for in-memory analysis (Part C of the notebook and later weeks).
# None loads every transaction (about 31.8M rows; needs roughly 4 to 8 GB of RAM
# with the compact dtypes used here). Set e.g. "2020-08-01" to keep only a recent
# time window, or MAX_ROWS for a quick smoke test.
START_DATE = None
MAX_ROWS = None

# Minimum data size stated in the DSA 4060XA project brief.
BRIEF_MIN_INTERACTIONS = 10_000
BRIEF_MIN_USERS = 500
BRIEF_MIN_ITEMS = 500

RANDOM_SEED = 42

# ---- Week 2: data preparation, splitting and baseline defaults -----------------------
# These are starting values. Notebook 02 prints a sensitivity table so the final choice
# can be justified from the data; change them here and re-run the notebook.
WINDOW_DAYS = 365            # keep the most recent N days of transactions (None = all)
MIN_USER_INTERACTIONS = 5    # k-core: distinct items per user in the TRAINING period
MIN_ITEM_INTERACTIONS = 20   # k-core: distinct buyers per item in the TRAINING period
SAMPLE_USERS = 50_000        # seeded random sample of eligible users (None = all)
VAL_DAYS = 14                # most recent days before the test block used for validation
TEST_DAYS = 14               # most recent days held out for the final test (Week 4)
TOP_K = 10                   # brief asks for Top 5 or Top 10
HALF_LIVES_DAYS = (1, 3, 5, 7, 14, 28, 56)  # recency half-lives compared for the popularity baseline
LIFT_BASKET_SIZES = (2, 6)   # basket sizes used for outfit-like co-purchase analysis
LIFT_MIN_SUPPORT = 2_000     # minimum co-occurring baskets for a pair to be reported

# ---- Week 3: item-based collaborative filtering --------------------------------------
KNN_NEIGHBOURS = (20, 50, 100, 200)   # neighbours kept per item (tuned on validation)
KNN_SHRINK = (0.0, 20.0)              # similarity shrinkage: dot / (norm_i * norm_j + shrink)
KNN_HISTORY_HALF_LIVES = (None, 14, 28, 90)  # recency weighting of a user's own purchases
N_CANDIDATES = 50                     # candidates generated before product de-duplication

# ---- Week 4: latent factors, content, repurchase and hybrid fusion --------------------
SVD_FACTORS = (32, 64, 128)           # latent dimensions tried (nested: one SVD per setting below)
SVD_ITEM_DAMPING = (0.0, 0.5)         # columns scaled by count ** -damping (reduces popularity bias)
SVD_HISTORY_HALF_LIVES = (None, 28)   # recency weighting of the matrix that is factorised
CONTENT_NEIGHBOURS = (20, 50)         # content-similar items kept per item
CONTENT_CATEGORICAL_WEIGHTS = (0.5, 1.0, 2.0)  # weight of categorical attributes vs text in item features
REPURCHASE_HALF_LIVES = (7, 28, 90)   # recency decay when ranking a user's own past items
RRF_C = 20                            # reciprocal-rank-fusion constant: contribution = 1 / (c + rank)
FUSION_GRID = {                       # component weights searched on validation
    "popularity": (0.0, 1.0, 2.0),
    "item_knn": (0.0, 1.0, 2.0),
    "svd": (0.0, 1.0, 2.0),
    "content": (0.0, 0.5, 1.0),
    "repurchase": (0.0, 1.0, 2.0, 4.0),
}
OUTFIT_GAMMAS = (0.0, 0.1, 0.2, 0.4)  # strength of the product-type affinity re-rank (0 = off)
OUTFIT_MIN_SUPPORT = 30               # baskets needed before a product-type pair gets a lift value
BOOTSTRAP_SAMPLES = 2000
