#!/usr/bin/env python3
"""
Резонансный фильтр — упрощённый прототип.
63 слова, 50% нахлест, только положительные совпадения.
Без умножения. Без негативного опыта.
Автор: [твоё имя], 6 мая 2026.
"""
import numpy as np

# Константы
THREAD_SIZE = 8192
MASK_SIZE = 128
STEP = 64
NUM_WORDS = 63
HEAT_COUNT = 16
COOL_COUNT = 16

def generate_words(seed=42):
    """Генерирует 63 слова с чётно-нечётным разделением."""
    rng = np.random.default_rng(seed)
    words = []
    for i in range(NUM_WORDS):
        vec = np.zeros(THREAD_SIZE, dtype=np.int8)
        start = i * STEP
        end = start + MASK_SIZE
        # Только чётные или только нечётные позиции
        parity = i % 2
        available = [p for p in range(start, end) if p % 2 == parity]
        chosen = rng.choice(available, HEAT_COUNT + COOL_COUNT, replace=False)
        for p in chosen[:HEAT_COUNT]:
            vec[p] = 1
        for p in chosen[HEAT_COUNT:]:
            vec[p] = -1
        words.append(vec)
    return words

def compute_score(window, mask):
    """Только положительные совпадения. Без негативов."""
    T = np.sign(window).astype(np.int8)
    M = mask.astype(np.int8)
    # +1 где знаки совпадают и не ноль
    match = ((T == M) & (T != 0)).astype(np.int32)
    return int(match.sum())

def main():
    words = generate_words()
    thread = np.zeros(THREAD_SIZE, dtype=np.int8)

    # Обучение: один проход
    for i in range(NUM_WORDS):
        start = i * STEP
        for p in range(start, start + MASK_SIZE):
            if words[i][p] == 1:
                thread[p] = np.clip(thread[p] + 1, -128, 127)
            elif words[i][p] == -1:
                thread[p] = np.clip(thread[p] - 1, -128, 127)

    # Таблица узнавания
    correct = 0
    for i in range(NUM_WORDS):
        start_i = i * STEP
        mask = words[i][start_i:start_i + MASK_SIZE]
        scores = []
        for j in range(NUM_WORDS):
            start_j = j * STEP
            window = thread[start_j:start_j + MASK_SIZE]
            scores.append(compute_score(window, mask))
        if np.argmax(scores) == i:
            correct += 1

    print(f"Узнаваемость: {correct}/{NUM_WORDS}")
    print(f"Точность: {correct/NUM_WORDS*100:.1f}%")

if __name__ == "__main__":
    main()
