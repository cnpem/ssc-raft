from ._version import __version__
import logging

try:
    from .rafttypes   import *
    from .geometries  import *
    from .processing  import *
    from .filters     import *
    from .pipelines   import *

except OSError:
    logging.error("Could not load sscRaft shared libraries")

try:
    import atexit
    from sscRaft.lib.ssccommons_wrapper import (
        log_event, log_start, log_stop, event_start, event_stop
    )
    log_start(project="sscRaft",
            version=__version__,
            level="error",
            telem_key="https://0486f9be2e157d5e4dd2bbb9a17353da@o1066143.ingest.sentry.io/4506825999712256")
    atexit.register(log_stop)
except:
    logging.error("Could not load sscCommons shared libraries")