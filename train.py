"""
Grand Bible research — current query.

Only ENDPOINT and QUERY are rewritten each iteration.
All logic lives in prepare.py.
"""
import prepare

# ---------------------------------------------------------------------------
# Rewrite these two lines each iteration
ENDPOINT = "/api/search"
QUERY    = "flood myth"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prepare.run_loop(ENDPOINT, QUERY)
