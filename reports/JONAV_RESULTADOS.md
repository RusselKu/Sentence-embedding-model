# SimCSE supervisado: resultados de Jonav

El modelo seleccionado obtuvo **81.67 en dev y 79.02 en test** de STS-B (Spearman ×100). Incluir contradicciones como negativos difíciles mejoró **+0.66 puntos en dev y +1.91 en test**, frente al mismo entrenamiento sin esos negativos.

## Configuración y selección

Se entrenó `bert-base-uncased` con CLS y una MLP lineal + tanh, conservada durante la inferencia. Se utilizaron 33,351 pares de entailment del subconjunto SNLI de 100,000 registros; 9,488 pares (28.45%) tienen una contradicción asociada. Se mantienen todos los pares positivos en ambos modos. Cuando falta una contradicción, no se inventa ni se sustituye por el positivo.

Se ejecutaron seis configuraciones con semilla 42 y se seleccionó exclusivamente por dev. Los mejores checkpoints se guardaron tras las evaluaciones cada 250 pasos y al final de cada época. La búsqueda y las semillas adicionales no evaluaron test. Se cerró la selección antes de evaluar una vez el par ON/OFF de semilla 42; los JSON conservan esos resultados.

| Parámetro | Valor |
|---|---|
| Learning rate seleccionado | 3e-5 |
| Temperatura | 0.05 |
| Batch contrastivo | 64 |
| Épocas | 3 |
| Longitud máxima | 64 tokens |
| Semilla final | 42 |
| Warmup / weight decay | 0.05 / 0.0 |
| Dropout | Valor predeterminado del backbone |
| Hardware | Google Colab, Tesla T4 |
| Entorno | Python 3.13.15; PyTorch 2.11.0+cu130 |

La MLP se conserva al evaluar SimCSE supervisado, siguiendo la [documentación oficial](https://github.com/princeton-nlp/SimCSE#evaluation).

## Búsqueda de hiperparámetros

| Learning rate | Temperatura | Mejor dev |
|---:|---:|---:|
| 3e-5 | 0.03 | 81.45 |
| **3e-5** | **0.05** | **81.67** |
| 3e-5 | 0.10 | 80.49 |
| 5e-5 | 0.03 | 81.04 |
| 5e-5 | 0.05 | 81.23 |
| 5e-5 | 0.10 | 80.27 |

La temperatura 0.05 dio el mayor dev dentro de esta búsqueda. No se exploraron otras cantidades de épocas, batches ni longitudes. Los resultados no permiten afirmar que sea la mejor configuración fuera de estos seis ensayos.

## Resultado final y ablación

| Negativos difíciles | Split | Spearman ×100 | Alignment (α=2) | Uniformity (t=2) |
|---|---|---:|---:|---:|
| ON | Dev | 81.67 | 0.1857 | -3.1374 |
| ON | Test | 79.02 | 0.1903 | -3.0460 |
| OFF | Dev | 81.01 | 0.1799 | -3.0665 |
| OFF | Test | 77.11 | 0.1914 | -2.9656 |

Alignment usa pares con puntuación humana ≥4/5 y embeddings normalizados. Uniformity usa todas las ocurrencias de oraciones de cada split, incluyendo repeticiones. Valores menores corresponden a menor distancia positiva y a una distribución más uniforme, respectivamente.

Los negativos difíciles mejoran el orden de similitudes medido por Spearman. La uniformidad también mejora con ON en ambos splits. Alignment empeora ligeramente en dev y mejora ligeramente en test: no hay una mejora uniforme de todas las métricas. La explicación plausible es que las contradicciones aportan información para separar oraciones con vocabulario parecido y significados incompatibles; estos resultados agregados no prueban por sí solos ese mecanismo ni sustituyen el análisis de casos de recuperación.

Con batch 64 hay 63 negativos procedentes de otros positivos por ancla. En ON se añaden K contradicciones reales disponibles en ese batch: el total es 63+K, con K variable, en lugar de asumir siempre 127 negativos. El paso de prueba tuvo K=34 y logits de tamaño 64×98.

## Variación entre semillas

En cada pareja ON/OFF se mantiene la misma configuración y semilla, cambiando únicamente la inclusión de contradicciones.

| Semilla | Dev ON | Dev OFF | Delta ON−OFF |
|---:|---:|---:|---:|
| 42 | 81.67 | 81.01 | +0.66 |
| 123 | 82.05 | 81.22 | +0.83 |
| 456 | 82.15 | 82.04 | +0.11 |

La media de las diferencias es **+0.53** y su desviación estándar muestral es **0.38**. La desviación entre semillas es **0.25 para ON** y **0.54 para OFF**. La media de la mejora es del mismo orden que la variación de OFF y supera la desviación de las diferencias, pero el efecto varía de +0.11 a +0.83. Las tres semillas aportan evidencia descriptiva favorable; no demuestran significación estadística. Solo el par de semilla 42 tiene test, por lo que no se ha estimado la variación entre semillas de test ni se compara estadísticamente su delta de +1.91 con la desviación de dev.

Las semillas 123 y 456 estiman variación; no sustituyen al checkpoint final de semilla 42 aunque tengan mayor dev. El plan de selección ya estaba fijado antes de calcular test.

## Comparación con referencias

Las dos primeras referencias proceden de los JSON de baseline existentes del equipo, no de nuevas ejecuciones en esta integración.

| Referencia | Dev | Test | Delta de nuestro modelo ON en test |
|---|---:|---:|---:|
| BERT sin entrenar, mean pooling | 59.31 | 47.29 | +31.73 |
| SBERT-2019, `bert-base-nli-mean-tokens` | 80.77 | 76.98 | +2.04 |
| SimCSE supervisado de Jonav | 81.67 | 79.02 | — |
| SimCSE supervisado BERT-base, artículo | 86.20 | 84.25 | -5.23 |

El [artículo de Gao et al. (2021)](https://aclanthology.org/2021.emnlp-main.552.pdf), tablas 4 y 5, informa 86.2 en dev con negativos difíciles y 84.25 en STS-B test. **81.57 es el promedio de siete tareas STS**, y 84.9 en dev corresponde a entailment sin negativos difíciles. Las brechas correctas son **4.53 en dev y 5.23 en test**. Se usan referencias publicadas, no una reproducción local del modelo del artículo.

El artículo emplea SNLI+MNLI (aproximadamente 314k positivos), con mayor cobertura de contradicciones, y batch 512 para BERT-base supervisado (apéndice A). Aquí se usan 33,351 positivos de SNLI, cobertura de contradicciones del 28.45% y batch 64. Son diferencias que pueden contribuir a la brecha; estos experimentos no aíslan sus efectos. El artículo también señala que la sensibilidad al batch depende del ajuste del learning rate, por lo que un batch menor no demuestra por sí solo la causa de la diferencia.

## Evidencia y entrega al equipo

- [Tabla CSV](jonav/supervised_benchmark.csv), [variación entre semillas](jonav/ablation_dev_noise.json) y [selección de dev](jonav/selection_dev.json).
- [Selección cerrada](jonav/selection_locked.json), [historial original de Colab](jonav/colab_run_history.json), [versiones reales del entorno](jonav/requirements-runtime.txt) y [verificación de importación](jonav/import_verification.json).
- Los JSON individuales están en `runs/jonav_*.json`; el historial común integra los 11 entrenamientos supervisados y conserva los dos registros previos del equipo. `test: null` significa que ese ensayo no se evaluó en test.
- El código ejecutado coincide con los hashes del manifiesto. El comentario de `requirements-colab.txt` difiere; sus versiones coinciden y se conserva el archivo ejecutado exacto con su hash verificado.

El checkpoint final ON está en `Mi unidad/SimCSE_jonav/checkpoints/jonav_sup_hnon_lr3e-05_tau0.05_bs64_ep3_seed42/best_checkpoint`. El OFF está en la carpeta equivalente `jonav_sup_hnoff_lr3e-05_tau0.05_bs64_ep3_seed42/best_checkpoint`. Los pesos permanecen en Drive y no están incluidos en el ZIP de resultados ni en Git.

Rivaldo puede cargar ambos mediante `SimCSEModel.from_checkpoint(ruta, device=...)` y usar `get_sentence_embeddings` para el análisis de recuperación. Russel debe conservar la MLP al exportar el modelo supervisado y verificar que la recarga produce los mismos embeddings: el exportador actual de `src/publish.py` requiere adaptación para este checkpoint completo. La publicación supervisada y los casos de recuperación aún corresponden a esas tareas del equipo; este informe documenta el entrenamiento, selección y ablación terminados de Jonav.
