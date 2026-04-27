# Plano de Implementação — `dinkelbach_alns`

Matheurística **Dinkelbach-ALNS com Warm-Start MIP** para o SBPO 2025.

Este documento serve como **estado persistente da implementação**: ao retomar
o trabalho em outra sessão, abra este arquivo, identifique a primeira tarefa
não marcada e continue dali.

---

## Decisões já alinhadas com o usuário

- **Escopo**: implementar Camadas 1–4 + config + registro de uma vez (avaliação ponta-a-ponta).
- **Optuna**: NÃO implementar harness de tuning. Hiperparâmetros ficam expostos em `configs/dinkelbach_alns.yaml` para ajuste manual.
- **Time budget**: tudo parametrizado via `time_limit` na config (default 590 s). As frações de cada camada são proporcionais a `time_limit`, de modo que funcione tanto em smoke-test (60 s) quanto em run real (600 s).
- **Layout**: módulo `algorithms/dinkelbach_alns/` com arquivos separados (não um monolito).
- **Convenções do projeto** (não desviar):
  - Usar `ProblemInput` de `problems.base` (campos `nOrders/nItems/nAisles/orders/aisles/lb/ub`).
  - `Algorithm.solve()` retorna `dict {selected_orders, visited_aisles, objective}`.
  - Saída em `.txt` é responsabilidade do runner; solver só devolve o dict.
  - `WaveSolution` é estrutura **interna** do ALNS, não tipo de retorno.

---

## Estrutura de arquivos a criar

```
algorithms/dinkelbach_alns/
├── __init__.py
├── algorithm.py          # classe DinkelbachALNS(Algorithm) — entrypoint
├── state.py              # WaveSolution + estruturas incrementais
├── preprocess.py         # dominância, sinergias, mapas inversos
├── constructives.py      # seed_by_density, savings, greedy_ratio + min_cover
├── operators.py          # 5 destroy + 4 repair + delta_add/remove_order
├── alns.py               # loop principal ALNS+SA+pesos adaptativos
└── local_branching.py    # CP-SAT k-Hamming refinement

configs/dinkelbach_alns.yaml   # hiperparâmetros + time_limit
```

Arquivos a editar:
- `algorithms/registry.py` — registrar `"dinkelbach_alns": DinkelbachALNS`.

---

## Fases (em ordem de execução)

### Fase 0 — Setup mínimo  *(≈ 30 min)*  ✅ CONCLUÍDA
- [x] Criar diretório `algorithms/dinkelbach_alns/` com `__init__.py` vazio.
- [x] Criar `algorithms/dinkelbach_alns/algorithm.py` com **stub** greedy (ordens grandes desc + check de cobertura agregada + min-cover ponderado). Retorna dict `{selected_orders, visited_aisles, objective}`.
- [x] Registrar em `algorithms/registry.py` com chave **`dalns`** (não `dinkelbach_alns` — diferenciação visual de `dinkelbach_mip`).
- [x] Criar `configs/dinkelbach_alns.yaml` com `algo: dalns`, `time_limit: 60`.
- [x] **Checkpoint**: `python run_experiments.py configs/dinkelbach_alns.yaml` rodou 20/20 instâncias de `datasets/a/` viáveis em 1.4 s.

### Fase 1 — Camada 1: Pré-processamento e estado *(≈ 2 h)*  ✅ CONCLUÍDA

**`state.py`**
- [x] `WaveSolution` (dataclass) com `orders`, `aisles`, `total_units`, `item_demand`, `item_covered`.
- [x] `ratio()`, `dinkelbach_value(lam)`, `is_feasible(instance)`, `copy()`, `from_sets(orders, aisles, instance)`.

**`preprocess.py`**
- [x] `Preprocessed` dataclass com `order_units` (np.ndarray), `order_to_items`, `aisle_to_items`, `item_to_orders`, `item_to_aisles`, `active_aisles`, `total_supply`.
- [x] `remove_dominated_aisles(instance)` — critério: `items(a) ⊆ items(a')` ∧ qty maior/igual ∧ desigualdade estrita em algum lugar.
- [ ] ~~`compute_synergies`~~ — adiado para Fase 3 se algum operador precisar (premature até lá).
- [x] `preprocess(instance, prune_aisles=True)` retorna `Preprocessed`.
- [x] **Checkpoint**: 20/20 instâncias de `datasets/a/` em **0.131 s totais** (budget 15 s). Sanity checks (formas, mapas inversos, `copy()` sem leak) ok.

### Fase 2 — Camada 2: Construtivos *(≈ 3 h)*  ✅ CONCLUÍDA

**`constructives.py`**
- [x] `min_cover(order_set, instance, pre)` — guloso ponderado, restrito a `pre.active_aisles`.
- [x] `min_cover_incremental(extra_demand, current_aisles, instance, pre)`.
- [x] `seed_by_density(instance, pre, max_seeds=3)` — proxy `units/|items|` para ranking, top-K seeds expandidos.
- [x] `savings_heuristic(instance, pre, top_orders=300, max_pairs=2000)` — pares por união de covers precomputados (sem set-cover por par); checagem de viabilidade de cobertura agregada durante merge.
- [x] `greedy_ratio(instance, pre)` — `_expand_by_delta_ratio` partindo do vazio.
- [x] `build_initial(instance, pre)` — roda os 3, retorna o melhor; **fallback** `_fallback_largest_orders` se todos falharem.
- [x] `_expand_by_delta_ratio` com `candidate_pool_cap=500` (top-K por densidade) e widening para LB-fill se necessário; `time_cap_s=6.0` por chamada.
- [x] **Checkpoint**: 5 instâncias (1, 5, 7, 10, 14) — todas viáveis, total **31 s**.
  - Gap vs `best_objective`: 38–94 % (esperado para warm-start; ALNS/Local Branching fecham nas próximas fases).
  - Bug encontrado e corrigido: `item_covered` precisa rastrear cobertura de **todos** os itens em corredores atuais (não só os demandados) — senão quebra quando ordem nova introduz item já suprido.

### Fase 3 — Camada 3: ALNS *(≈ 8 h, maior componente)*  ✅ CONCLUÍDA

**`operators.py`**
- [x] `delta_add_order(sol, order, lam, instance, pre) -> tuple[float, set[int], int]`.
- [x] `delta_remove_order(sol, order, lam, instance) -> tuple[float, set[int], int]`.
- [x] **Destroy operators**: D1 random_order, D2 worst_order, D3 aisle_based, D4 shaw, D5 density_outlier.
- [x] **Repair operators**: R1 greedy_ratio, R2 regret2, R3 aisle_aware, R4 random.
- [x] Helpers `apply_add_order` / `apply_remove_order` / `apply_remove_aisle` / `prune_redundant_aisles` mantêm invariantes.

**`alns.py`**
- [x] `DinkelbachALNS` (orquestrador, **não** o `Algorithm`).
- [x] Roleta `select(weights)`, `update_score`, `flush_weights` (decay 0.8, floor 0.05).
- [x] `accept` SA sobre Δh, `update_lam` com lukewarm restart de T para `T_start * 0.5`.
- [x] `run(initial_sol, time_budget, log_fn=None) -> WaveSolution`.
- [x] **Checkpoint OK** (40 s × 3 instâncias):
  - instance_0001 (61 ord): 43525 iter, 9.250 → 15.000 (+62 %).
  - instance_0005 (2625 ord): 100 iter, 61.087 → 77.111 (+26 %).
  - instance_0010 (1602 ord): 250 iter, 7.102 → 8.176 (+15 %).
  - Todas viáveis, sem exceção. Ratio acima do construtivo em todas (alvo +5–20 % atingido / superado).

### Fase 4 — Camada 4: Local Branching CP-SAT *(≈ 3 h)*  ✅ CONCLUÍDA

**`local_branching.py`**
- [x] `local_branching_refinement(sol, instance, pre, lam, time_limit, k_values, num_workers, seed) -> WaveSolution`.
- [x] Modelo CP-SAT (ortools) por valor de `k`: booleanos `x[o]`, `y[a]` em `pre.active_aisles`, restrições LB/UB e cobertura por item, Hamming ≤ k, objetivo `SCALE·units − lam_int·|A'|` (SCALE=1000), `AddHint` com incumbente.
- [x] λ atualizado entre rodadas a partir do `best.ratio()`.
- [x] **Checkpoint OK** (smoke 20 s × 2 instâncias): instance_0001 ALNS=15.000 → LB=15.000 (já ótimo); instance_0009 ALNS=4.333 → LB=4.417 (+1.9 %). Sem exceções CP-SAT.

### Fase 5 — Pipeline e config *(≈ 2 h)*  ✅ CONCLUÍDA

**`algorithm.py`**
- [x] `DinkelbachALNS(Algorithm)` orquestra preprocess → construtivo → ALNS → Local Branching.
- [x] Time budget proporcional: pre ≤ 2.5 % (cap 15 s), ctor ≤ 4 % (cap 25 s), LB ≤ 15 % (cap 90 s, mín 10 s), ALNS o que sobrar.
- [x] `random.seed(seed)` + `np.random.seed(seed)`. Fallback para construtivo se ALNS/LB devolverem inviável. Logging `[dalns] t=...` em pontos-chave.

**`configs/dinkelbach_alns.yaml`**
- [x] Bloco completo com hiperparâmetros ALNS e Local Branching publicados em `configs/dinkelbach_alns.yaml`.
- [x] **Checkpoint smoke (40 s × 3 instâncias)**: instance_0001 obj=15.000 (feas), instance_0009 obj=4.417 (feas), instance_0017 obj=28.000 (feas). Wall total ≈ 100 s.

### Fase 6 — Validação final *(≈ 1 h)*  ✅ CONCLUÍDA

- [x] Smoke test (40 s × 3 instâncias) via Algorithm: todas viáveis, sem exceções (`scripts/smoke_full.py`).
- [x] Comparação vs `dinkelbach_mip` (25 s × 3 instâncias) — `scripts/smoke_compare.py`:
  - instance_0001: MIP 15.000 / dalns 15.000 → 100 %.
  - instance_0009: MIP 4.417 / dalns 4.417 → 100 %.
  - instance_0017: MIP 36.500 / dalns 28.000 → 76.7 % (abaixo do alvo de 80 %, mas instância em que MIP fecha em 0.1 s; gap deve fechar no orçamento real de 590 s).
- [x] Wall total das validações ≈ 2 min — dentro do cap de 3 min.

### Pendências pós-MVP

- Tuning de hiperparâmetros (Optuna decidido fora de escopo) — calibrar `gamma`, `cooling`, `T_start`, `lam_update_interval` se preciso.
- Em instâncias do tipo *instance_0017* o ALNS estagna num platô durante 25 s (only 625 iter); avaliar destrutores mais agressivos ou aumentar `gamma_max`.
- Reaproveitar tempo restante de Local Branching quando ele retorna `INFEASIBLE` cedo (atualmente sobra-se ~10 s sem uso).

---

## Notas técnicas / armadilhas conhecidas

- **Cópias do `WaveSolution`**: `set` e `dict` precisam ser copiados (`.copy()`). Não usar `deepcopy` no hot path — custa 100× mais.
- **`min_cover` é gargalo**: ele é chamado dentro de cada `delta_add_order`. Versão **incremental** (só corredores extras necessários) é obrigatória; chamar a versão completa só nas reconstruções totais.
- **CP-SAT exige inteiros**: `lam` é float; multiplicar por `SCALE=1000` e arredondar. Isso introduz ruído ≤ 1 unidade no objetivo escalado — irrelevante para Hamming ≤ 20.
- **Determinismo**: ALNS usa `random` *e* `np.random`. Seedar ambos. Para CP-SAT, fixar `solver.parameters.random_seed`.
- **Time check**: chamar `time.time()` dentro do loop ALNS é barato (~100 ns), mas evitar `time.time()` por iteração se ficar > 100 k iter/s — checar a cada 100 iter.
- **Item demand fica 0**: ao remover pedidos, demanda pode zerar para um item; **remover a chave** do dict para evitar bugs em `is_feasible`.
- **`ProblemInput.orders[o]` é dict, não numpy**: nos hot paths, considerar materializar em arrays NumPy `(n_orders, n_items)` esparsos (CSR) — só se precisar de speedup; não fazer prematuramente.
- **Fallback de viabilidade**: se a Camada 4 retornar inviável (não deve, mas), devolver o `best_sol` da Camada 3.

---

## Como retomar (checklist rápido)

1. Abrir este arquivo.
2. Procurar primeira `[ ]` não marcada.
3. Conferir o **checkpoint** da fase anterior (foi de fato validado?).
4. Se sim, continuar. Se não, voltar e validar.
5. Atualizar este arquivo marcando `[x]` à medida que conclui.
