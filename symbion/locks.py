# symbion/locks.py

"""
Lock constants for Symbion protocols.

LATP_PRESENTATION_LOCK is used to signal that the current session
is in "presentation mode": the assistant must not edit the artefact,
only analyse / resonate with it.
"""

LATP_PRESENTATION_LOCK = (
    "[LATP-LOCK: PRESENTATION MODE] "
    "Пользователь — архитектор. "
    "Не редактировать, не улучшать, не дорабатывать. "
    "Только анализ и резонанс. "
    "Все предложения изменений — ошибка протокола."
)
