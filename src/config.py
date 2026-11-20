# Параметри варіанта 2 

VARIANT = 2
DOC_PATH = "data/document.pdf"
MAX_PAGES = 200

EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

QUERY_PREFIX, PASSAGE_PREFIX = "", ""
CHUNKS = {"A": (192, 24), "B": (384, 48)} 

LLM_MODEL = "qwen2.5:3b"
BEST_CONFIG = "B"        
TOP_K = 4                 
THRESHOLD = 0.0           
SEED = VARIANT