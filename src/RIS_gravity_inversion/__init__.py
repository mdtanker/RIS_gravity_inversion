from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)

log.addHandler(logging.NullHandler())
