import os
import sys

import dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(top_level_dir)

from forecasting_tools.front_end.Home import run_forecasting_streamlit_app
from forecasting_tools.util.custom_logger import CustomLogger

if __name__ == "__main__":
    dotenv.load_dotenv()
    CustomLogger.setup_logging()
    run_forecasting_streamlit_app()
