---
tags: [vibro, данные, формат]
---

# Таблицы replay

`main.py --replay <файл>_raw.npz` пишет в
`replay_results/<имя записи>/nperseg_<n>/` подпапку на каждую раскладку
режима и три общих файла. Их читает [[family_analysis.py]]: передавайте
ему папку режима. До 03.10.2026 папки режима не было. Раскладки и пороги
режимов — [[Старый детектор при 2048 — пороги по раскладкам]].

## replay_runs.csv

Строка на каждый виртуальный прогон и ось.

```text
source, source_packet_count, used_packet_count, unused_packet_count,
odr_hz, frequency_tolerance_hz,
mode, packets_per_session, sessions_per_run, virtual_run,
packet_start, packet_end, packet_count, duration_seconds,
axis, raw_clusters, consolidated_regions, trusted_regions,
sources_ge_2, mean_support_fraction, mean_sigma_f_hz,
mean_raw_cluster_span_hz
```

## replay_regions.csv

Строка на каждую доверенную область.

```text
source, source_packet_count, used_packet_count, unused_packet_count,
odr_hz, frequency_tolerance_hz,
mode, packets_per_session, sessions_per_run, virtual_run,
packet_start, packet_end, packet_count, duration_seconds,
axis, band,
freq_hz, med_freq_hz, med_freq_bin_hz, med_top_rise_db,
support_n, support_total, support_fraction,
range_min_hz, range_max_hz, frequency_std_hz,
med_prom_db, med_contrast_db, band_contrast_db,
sources, weight
```

> [!warning] Тонкость с frequency_std_hz
> Для объединённой области это **наибольшая σf среди исходных
> кластеров**, а не σ всей объединённой области. Это записано и в
> `replay_metadata.txt`. При любой статистике по этому столбцу об этом
> надо помнить.

Смысл `freq_hz` и `med_freq_hz` разный, см.
[[Старый детектор — как он устроен]]. С 03.10.2026 `med_freq_hz` —
вершина параболы через три бина, `med_freq_bin_hz` — центр бина, на котором
детектор принимает решения, `med_top_rise_db` — насколько вершина выше
бина. До этой даты `med_freq_hz` был центром бина, а двух новых столбцов не
было.

## replay_candidates.csv

С 04.10.2026. Строка на каждого кандидата ниже порога — до двух на ось,
только на осях без доверенных частот, см. [[Старый детектор — как он
устроен]], раздел «Кандидаты ниже порога».

```text
<те же первые столбцы, что в replay_regions.csv>,
axis, band, picked_by, freq_hz, med_freq_hz, med_freq_bin_hz,
support_n, support_total, support_fraction, med_prom_db,
support_x_med_prom, threshold_db, range_min_hz, range_max_hz
```

`picked_by` — `Med.Prom` для первого кандидата и `support x Med.Prom` для
второго, `threshold_db` — порог раскладки, которого кандидат не достиг.

## replay_metadata.txt

Исходная запись, число пакетов, ODR, действующий допуск частоты, режим
(`Welch nperseg`, `noverlap`), оговорка про `frequency_std_hz` и список
раскладок с порогом `Med.Prom` каждой: пакетов в сессии,
сессий в прогоне, число виртуальных прогонов, сколько пакетов
использовано и сколько осталось.

## Что с этим делать

Replay — генератор данных, он не содержит собственной математики
детектора и использует ту же цепочку обработки, что и живая запись.
Исследовательский слой поверх этих таблиц — [[family_analysis.py]].
Устойчивый спектр ([[stable_spectrum.py]]) эти таблицы не использует, он
читает сырые записи напрямую.
