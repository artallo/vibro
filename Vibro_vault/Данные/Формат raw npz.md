---
tags: [vibro, данные, формат]
---

# Формат raw npz

Файл `<дата>_<время>_ODR<частота>_runNN_raw.npz` пишет [[main.py]] в конце
полной записи. Это архив numpy.

| Поле | Тип и размер | Смысл |
|---|---|---|
| `format_version` | int, сейчас 1 | версия формата |
| `created_at` | строка | время начала записи, ISO |
| `requested_odr_hz` | float | заказанная частота опроса |
| `packet_count` | int | число пакетов, обычно 256 |
| `samples_per_packet` | int | отсчётов в пакете, 1024 |
| `packets_per_session` | int | пакетов в сессии при записи |
| `target_sessions` | int | заказанное число сессий |
| `x`, `y`, `z` | float, пакеты × отсчёты | ускорение в g |
| `packet_fs_hz` | float, по пакету | частота опроса пакета: `(N − 1) / elapsed` |

Особенности:

- Z содержит 1 g от силы тяжести, X и Y — несколько mg от наклона.
- Первый отсчёт первого пакета равен нулю, это старт датчика.
- Фактическая частота при ODR 250 около 252,77 Гц.
- Номер пакета из прошивки пока не сохраняется, см. [[Открытые задачи]].

Прочитать файл:

```python
import numpy as np
d = np.load("tumen_results/20260916_193930_ODR250_run01_raw.npz")
print(d.files, d["z"].shape, d["packet_fs_hz"].mean())
```

Все скрипты анализа принимают этот формат: [[stable_spectrum.py]],
[[overview_figure.py]], [[repeatability_check.py]], `main.py --replay`.
