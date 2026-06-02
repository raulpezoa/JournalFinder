"""Back-compat entry point.

The app lives in streamlit_app.py, which Streamlit Cloud runs by default. This
wrapper exists so an older deployment configured to run app.py still works.
"""

from streamlit_app import main

main()
