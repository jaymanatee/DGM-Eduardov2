# Fase 1 · Cómo reproducirla

Los comandos, en orden, desde un clon limpio hasta el endpoint `/reasoning` con el modelo final.
Por qué se eligió cada cosa está en [`EXPERIMENTS.md`](../EXPERIMENTS.md); aquí solo está la
receta. Hay dos caminos:

- **Corto** (minutos): descargar los adaptadores ya entrenados y levantar la API.
- **Completo** (unas 80 horas de GPU): regenerar todo desde los datos.

Todo se lanza desde la raíz del repositorio. En la DGX, `source smoke/dgx_env.sh` antes de nada
(ver [`dgx.md`](dgx.md)). Los scripts ya desactivan la atención de cuDNN, que en la DGX hacía la
generación 19 veces más lenta (entrada del 29/10).

```bash
uv sync --extra train
uv run pytest            # todo verde
```

## Camino corto: adaptadores de Hugging Face

Los pesos no están en git (`rlm/weights/` está en `.gitignore`):

```bash
hf auth login            # solo si los repositorios son privados
hf download enriquefbrt/arca-sft-lora   --local-dir rlm/weights/sft_lora
hf download enriquefbrt/arca-grpo-lora  --local-dir rlm/weights/final_rlm_lora
hf download enriquefbrt/arca-grpo2-lora --local-dir rlm/weights/grpo_pps3
```

Solo `grpo_pps3` hace falta para el endpoint; los otros dos, para repetir la evaluación y las
curvas. Con eso se puede saltar directamente a [Evaluación](#5-evaluación) o a
[Endpoint](#7-endpoint).

## Camino completo

### 1. Datos (segundos, sin GPU)

El generador es determinista: con la misma semilla escribe los mismos ficheros, byte a byte.

```bash
uv run python -m rlm.generate_problems --n 4000 --split train --seed 0 --out rlm/data/train.jsonl
uv run python -m rlm.generate_problems --n 200  --split test  --seed 0 --out rlm/data/test.jsonl
uv run python -m rlm.generate_problems --n 100  --split ood   --seed 0 --out rlm/data/test_ood.jsonl
uv run python -m rlm.generate_problems --n 500  --split train --seed 0 --out rlm/data/sft_problemas.jsonl
```

Comprobación (son los hashes de la tabla de ficheros de `EXPERIMENTS.md`):

```bash
sha256sum -c <<'EOF'
b61dcfc8012aea2d7343a48eed8549f0a07b7a6c6c91bb4526605839cbf4e3d2  rlm/data/train.jsonl
9e66110dd87cd5bc9d3970ae38795c76d4b1aad649194f6e6c47f0f7d95840be  rlm/data/test.jsonl
58787a033551ba6f68a0c92d9375985752f5855ba4de37b7b38223200cee08c8  rlm/data/test_ood.jsonl
3327deb2e5d4a831c458cc7731bddeafc61b70f80b9c6821e9427f1e766ca17d  rlm/data/sft_problemas.jsonl
EOF
```

### 2. Destilación (unos 3 minutos por problema en `MIG 1g.18gb`)

Profesor `Qwen/Qwen3-4B`, 4 trazas por problema, temperatura 0,6; el verificador de origen tira
las incorrectas.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup uv run python -m rlm.distill \
    --data rlm/data/sft_problemas.jsonl --output rlm/data/sft_traces.jsonl \
    --max-new-tokens 3072 --batch-size 2 > runs/distill.log 2>&1 &
```

Es muestreo, así que una tirada nueva no da las mismas trazas. Las que usamos están en git
(`rlm/data/sft_traces.jsonl`, 1.282 válidas de 2.000): para reproducir el SFT, partid de ese
fichero y saltad este paso.

### 3. SFT (unos 60 minutos en `MIG 1g.18gb`)

LoRA de rango 16 sobre `Qwen/Qwen3-1.7B`, 2 épocas, learning rate 2e-4, secuencias de 4.096 tokens.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup uv run --extra train \
    python -m rlm.train_sft --data rlm/data/sft_traces.jsonl \
    --output rlm/weights/sft_lora > runs/sft.log 2>&1 &
```

### 4. GRPO (unas 45 horas en `MIG 1g.18gb`, en sesiones de 8 horas)

El modelo final es el GRPO 2: 270 pasos de 3 problemas con 8 respuestas cada uno, sobre el SFT.
`--max-hours 7.5` guarda un checkpoint y para antes de que acabe la sesión;
`--resume-from-checkpoint last` continúa desde el último. **Se lanza el mismo comando en cada
sesión** hasta que llega a 270 pasos.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup uv run --extra train \
    python -m rlm.train_grpo --steps 270 --prompts-per-step 3 --learning-rate 7.5e-6 \
    --save-steps 20 --resume-from-checkpoint last --max-hours 7.5 \
    --output rlm/weights/grpo_pps3 >> runs/grpo_pps3.log 2>&1 &
```

El GRPO 1 (1 problema por paso, 400 pasos, unas 10 horas) solo hace falta para la comparación:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup uv run --extra train \
    python -m rlm.train_grpo --steps 400 --save-steps 20 \
    --resume-from-checkpoint last --max-hours 7.5 \
    --output rlm/weights/final_rlm_lora >> runs/grpo_run01.log 2>&1 &
```

### 5. Evaluación (1 h 40 min por cada dos modelos, con lote 50 en `MIG 3g.71gb`)

Greedy, 3.072 tokens. `evaluate.py` añade cada modelo al JSON de salida sin borrar los que ya
había.

```bash
for data in test test_ood; do
  uv run --extra train python -m rlm.evaluate --data rlm/data/$data.jsonl \
      --adapters base=none sft=rlm/weights/sft_lora grpo=rlm/weights/final_rlm_lora --batch-size 16
done
uv run --extra train python -m rlm.evaluate --data rlm/data/test.jsonl \
    --adapters grpo2=rlm/weights/grpo_pps3 --batch-size 16 --out reports/grpo2_test.json
uv run --extra train python -m rlm.evaluate --data rlm/data/test_ood.jsonl \
    --adapters grpo2=rlm/weights/grpo_pps3 --batch-size 16 --out reports/grpo2_ood.json
```

En los JSON del repositorio, el `sft` de `phase1_eval_*.json` se evaluó con lote 50 en
`MIG 3g.71gb`, y `reports/sft_rerun_*.json` es el mismo adaptador repetido con lote 16 en
`MIG 1g.18gb`. Al repetirlo cambió de veredicto en 78 de 300 problemas: la generación greedy
no es reproducible entre particiones y tamaños de lote (entrada del 03/11). Se puede esperar,
pues, una diferencia de unos pocos puntos con las cifras del cuaderno; las diferencias entre
modelos que dimos por significativas se calcularon con esa variación en mente.

```bash
uv run --extra train python -m rlm.evaluate --data rlm/data/test.jsonl \
    --adapters sft=rlm/weights/sft_lora --batch-size 16 --out reports/sft_rerun_test.json
uv run --extra train python -m rlm.evaluate --data rlm/data/test_ood.jsonl \
    --adapters sft=rlm/weights/sft_lora --batch-size 16 --out reports/sft_rerun_ood.json
```

### 6. Análisis y curvas (segundos, sin GPU)

```bash
uv run python -m rlm.analyze_eval reports/phase1_eval_test.json
uv run python -m rlm.analyze_eval reports/phase1_eval_test_ood.json --no-teacher
uv run python -m rlm.analyze_eval reports/phase1_eval_test.json reports/sft_rerun_test.json \
    reports/grpo2_test.json --compare sft grpo2 --out reports/phase1_analysis_grpo2_test.md
uv run python -m rlm.analyze_eval reports/phase1_eval_test_ood.json reports/sft_rerun_ood.json \
    reports/grpo2_ood.json --compare sft grpo2 --out reports/phase1_analysis_grpo2_test_ood.md
uv run python -m rlm.evaluate --history rlm/weights/sft_lora/log_history.json
uv run python -m rlm.evaluate --history rlm/weights/grpo_pps3/log_history.json
uv run python -m rlm.plot_grpo_curves      # reports/phase1_grpo_curves.png, los dos GRPO
```

En los dos `phase1_analysis_grpo2_*.md` del repositorio, la línea «Modelos:» se completó a mano
con qué es cada modelo; el resto sale tal cual de estos comandos.

### 7. Endpoint

```bash
cp -n .env.example .env  # si no tenéis .env; ya apunta ARCA_RLM_ADAPTER a rlm/weights/grpo_pps3
uv run arca-api          # o: docker compose up api
```

Prueba con el primer problema de `test`:

```bash
python3 - <<'EOF' | curl -s -X POST localhost:8000/reasoning -H 'Content-Type: application/json' -d @- | python3 -m json.tool
import json
row = json.loads(open("rlm/data/test.jsonl").readline())
print(json.dumps({"question": row["question"], "expected_answer": row["answer"]}))
EOF
```

La respuesta tiene que traer `thinking`, `answer`, `has_valid_format: true` y el veredicto del
verificador. El endpoint usa el mismo prompt, la misma generación greedy y el mismo verificador
que `evaluate.py`; aun así, en otra GPU algún veredicto puede cambiar, por la misma razón que en
la sección 5. En una prueba con 45 problemas de `test` servidos desde un portátil (RTX 5070),
el endpoint acertó 30 y `reports/grpo2_test.json` también 30, con 4 problemas que solo acertaba
uno y 4 que solo acertaba el otro.

## Lo que no está en la cadena principal

- `rlm/pilot_subset_deprecated.py` elige los 20 problemas «dirigidos» de los pilotos E3 a E5.
  Solo hace falta para repetir esos pilotos (entradas del 12/10).
- `rlm/try_model.py` deja ver el razonamiento de un modelo token a token sobre un problema;
  sirve para leer fallos, no genera nada que se use después.
