---
tags: [vibro, конфигурация]
---

# Конфигурация config.toml

Файл лежит в корне репозитория. Его читают все скрипты.

| Секция | Параметр | Значение | Кто использует и зачем |
|---|---|---|---|
| `[serial]` | `port` | `COM8` | [[main.py]]: порт ESP32 |
| | `baud` | 115200 | скорость UART |
| | `timeout_seconds` | 2.0 | таймаут чтения |
| `[session]` | `packets_per_session` | 8 | пакетов в одной сессии старого детектора |
| | `min_recommended_sessions` | 5 | сколько сессий записать, если не задано в командной строке |
| `[sensor]` | `odr_hz` | 250 | частота опроса по умолчанию |
| `[welch]` | `nperseg` | 1024 | длина отрезка спектра в старом детекторе и в [[family_analysis.py]]; [[stable_spectrum.py]] перебирает свои значения, см. [[Разрешение nperseg]] |
| | `noverlap` | 512 | перекрытие отрезков |
| `[visualization.trusted_frequency]` | `min_support_fraction` | 0.50 | порог старого детектора «доверенных» частот |
| | `min_median_prominence_db` | 1.55 | то же |
| | `background_weight`, `min_band_contrast_db`, `weak_trusted_weight` | 0.5, 1.0, 0.7 | оформление старых графиков |
| `[analysis.frequency_clustering]` | `frequency_tolerance_hz_250/125/62p5` | 0.40 / 0.35 / 0.25 | допуск склейки частот между сессиями |
| `[analysis.frequency_cluster_consolidation]` | `median_frequency_tolerance_hz` | 0.20 | склейка кластеров по пику медианного спектра |
| `[[analysis.bands]]` | `Low frequency` | 0,5–10 Гц | полосы; [[stable_spectrum.py]] берёт из них границы анализа, 0,5–15 Гц |
| | `High frequency` | 10–15 Гц | |
| | `prominence_db`, `min_stability` и др. | 1.8, 4.1 … | пороги старого детектора |

> [!warning] Фиксированные пороги в дБ
> `prominence_db`, `min_median_prominence_db` и похожие параметры — это
> фиксированные пороги. Подбирать их на вечерних записях бесполезно:
> пики там на уровне шума оценки. См. [[Почему результаты скачут]].

Чтобы [[stable_spectrum.py]] и [[overview_figure.py]] смотрели выше 15 Гц,
не нужно менять конфиг: у обоих есть ключ `--band`.
