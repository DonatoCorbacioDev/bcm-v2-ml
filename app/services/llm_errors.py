# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

"""What the user sees when a call to Ollama fails.

The exception's text stays in the log: it can carry the internal address of
Ollama (OWASP API8/API10), which means nothing to the user and should not
reach them.
"""

import logging

AI_UNAVAILABLE = "Servizio AI non disponibile, riprova più tardi."

logger = logging.getLogger(__name__)


def unavailable(exc: Exception) -> str:
    logger.warning("Ollama call failed: %s: %s", type(exc).__name__, exc)
    return AI_UNAVAILABLE
