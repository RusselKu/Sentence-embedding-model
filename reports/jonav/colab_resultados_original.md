# Resultados supervisados de Jonav

Configuración: {"selected_using": "dev_only", "best_run": "jonav_sup_hnon_lr3e-05_tau0.05_bs64_ep3_seed42", "lr": 3e-05, "temperature": 0.05, "batch_size": 64, "epochs": 3, "trial_runs": ["jonav_sup_hnon_lr3e-05_tau0.03_bs64_ep3_seed42", "jonav_sup_hnon_lr3e-05_tau0.05_bs64_ep3_seed42", "jonav_sup_hnon_lr3e-05_tau0.1_bs64_ep3_seed42", "jonav_sup_hnon_lr5e-05_tau0.03_bs64_ep3_seed42", "jonav_sup_hnon_lr5e-05_tau0.05_bs64_ep3_seed42", "jonav_sup_hnon_lr5e-05_tau0.1_bs64_ep3_seed42"]}

| Negativos | Split | Spearman x100 | Alignment | Uniformity |
|---|---|---:|---:|---:|
| on | dev | 81.67 | 0.1857 | -3.1374 |
| on | test | 79.02 | 0.1903 | -3.0460 |
| off | dev | 81.01 | 0.1799 | -3.0665 |
| off | test | 77.11 | 0.1914 | -2.9656 |

Delta ON - OFF: dev +0.66; test +1.91 puntos.
Media de deltas dev entre semillas: +0.53; desviación: 0.38.

Alignment: pares STS-B con puntuación humana >=4/5. Uniformity: todas las oraciones de cada split.

Pendiente de redactar: interpretación, limitaciones y distancia al artículo.