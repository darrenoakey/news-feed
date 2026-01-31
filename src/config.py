from pathlib import Path

# ##################################################################
# frequency bounds
# min and max polling intervals for RSS feeds in seconds
MIN_FREQUENCY_SECONDS = 300  # 5 minutes
MAX_FREQUENCY_SECONDS = 14400  # 4 hours
DEFAULT_FREQUENCY_SECONDS = 3600  # 1 hour starting point

# ##################################################################
# frequency adjustment
# how much to adjust frequency when new entries found or not
FREQUENCY_ADJUSTMENT_SECONDS = 60

# ##################################################################
# background worker sleep
# how long to sleep when no feeds need checking
WORKER_SLEEP_SECONDS = 60

# ##################################################################
# paths
# database and project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATABASE_PATH = PROJECT_ROOT / "local" / "feeds.db"

# ##################################################################
# server settings
# host and port for the FastAPI server
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 19100

# ##################################################################
# scoring api settings
# host and port for the news-score API
SCORING_API_URL = "http://10.0.0.46:19091"
SCORING_WORKER_SLEEP_SECONDS = 60
SCORING_API_TIMEOUT_SECONDS = 120

# ##################################################################
# discord publishing settings
# threshold and timing for publishing to discord
DISCORD_MIN_SCORE = 8.0
DISCORD_WORKER_SLEEP_SECONDS = 60
DISCORD_RATE_LIMIT_BACKOFF_SECONDS = 300
