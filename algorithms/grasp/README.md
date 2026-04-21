# GraspHeuristic

## Visão geral

`GraspHeuristic` implementa uma metaheurística **GRASP** (Greedy Randomized Adaptive Search Procedure) para o problema de picking em ondas. Combina duas fases clássicas, repetidas por múltiplas iterações:

1. **Construção gulosa randomizada** — constrói uma solução inicial pegando ordens uma a uma a partir de uma Restricted Candidate List (RCL) controlada por `alpha`.
2. **Busca local** — aplica movimentos de vizinhança (swap / drop / add) em first-improvement até nenhum movimento melhorar o objetivo.

A cada iteração, mantém a melhor solução vista (`best`). Ao final de `max_iterations`, retorna esse melhor global.

- **Código**: `algorithms/grasp/grasp_heuristic.py`
- **Nome registrado**: `grasp`
- **Base**: `algorithms/base.py` (`Algorithm`)

## Ideia central

O objetivo do problema é maximizar `total_units / num_aisles`. Heurísticas gulosas puras (como `simple` e `seed`) ficam presas em ótimos locais determinísticos — uma mesma instância sempre produz o mesmo resultado, bom ou ruim. GRASP resolve isso combinando:

- **Diversificação** via construção randomizada: `alpha` interpola entre guloso puro (`alpha=0`) e aleatório puro (`alpha=1`). Valores intermediários geram soluções iniciais distintas a cada iteração, explorando regiões diferentes do espaço.
- **Intensificação** via busca local: cada solução construída é refinada por movimentos locais até convergir para um ótimo local.

Como o objetivo é uma razão, a busca local pode tanto adicionar ordens (mais unidades) quanto remover ordens (menos corredores) para melhorar o ratio — ambos os movimentos são considerados.

## Fluxo da implementação

```
melhor ← solução vazia (objetivo = 0)

para k em 1..max_iterations:
    x ← randomized_construction(...)
    se x é inviável (total_units < lb):
        continuar
    se local_search != "none":
        x ← local_search_improve(x, ...)
    se x.objetivo > melhor.objetivo:
        melhor ← x

retornar melhor
```

### Construção randomizada (RCL baseada em recompensa)

```
selected ← []; demand ← {}; total ← 0; stock ← estoque agregado
remaining ← {0, ..., n_orders-1}

enquanto remaining:
    feasible ← { i in remaining :
                 total + size[i] ≤ ub
                 e orders[i] cabe no stock restante }
    se feasible vazio: break

    scores ← _score_candidates(feasible, demand, ...)   # maior = melhor

    g_max ← max(scores.values())
    g_min ← min(scores.values())
    threshold ← g_max − alpha * (g_max − g_min)
    rcl ← { i in feasible : scores[i] ≥ threshold }

    pick ← rng.choice(rcl)
    selected ← selected ∪ {pick}
    total ← total + size[pick]
    demand, stock atualizados

se total < lb: retornar inviável

se greedy == "multi":
    visited_aisles ← multi_greedy_aisle_select(demand, aisles)
senão:
    visited_aisles ← greedy_aisle_select(demand, aisles)

retornar (selected, visited_aisles, total/|visited_aisles|, demand, total)
```

O estado completo (incluindo `demand` e `total_units` como chaves privadas `_demand`/`_total_units`) é propagado para a busca local para evitar recomputações.

### Funções de score (`construction_score`)

Todas retornam **maior = melhor**:

| `construction_score` | Definição | Custo |
|---|---|---|
| `"size"` | `order_sizes[i]` | O(1) por candidato |
| `"synergy"` | `similarity(demand, orders[i], weighted=similarity_weighted)`; cai para `size` quando `demand` está vazio | O(\|items\|) por candidato |
| `"aisle_cost"` | `−\|new_aisles\|` onde `new_aisles` são corredores extras que seriam necessários ao adicionar `i` (executa `greedy_aisle_select` a cada candidato); cai para `size` quando `demand` está vazio | O(candidatos × aisles × \|demand\|) por passo |
| `"aisle_cost_fast"` | Proxy barato para `aisle_cost`: pré-computa `best_aisle_for_item[item]` = corredor com maior estoque do item; mantém `current_aisles_set` incrementalmente (cresce ao comprometer `best_aisle_for_item` dos itens dos pedidos selecionados); score = `−Σ qty_i` para itens de `orders[i]` cujo `best_aisle_for_item` **não** está em `current_aisles_set`. Cai para `size` quando `demand` está vazio | O(\|items\|) por candidato; O(n_aisles × \|items por aisle\|) uma vez por `solve` |

### Busca local (first-improvement)

A cada passo, varre os movimentos habilitados; ao encontrar o primeiro estritamente melhor, aceita e reinicia o loop. Termina quando uma varredura completa não encontra melhoria.

```
x ← solução
repetir:
    melhorou ← False

    swap: para cada s in x.selected, u not in x.selected:
        se troca(s → u) é viável (lb ≤ total' ≤ ub, dentro do stock global):
            x' ← (selected − {s}) ∪ {u}, recompute visited_aisles e objetivo
            se x'.obj > x.obj: x ← x'; melhorou ← True; break

    se local_search == "full" e não melhorou:
        drop: para cada s in x.selected:
            se total − size[s] ≥ lb:
                x' ← selected − {s}, recompute
                se x'.obj > x.obj: x ← x'; melhorou ← True; break

    se local_search == "full" e não melhorou:
        add: para cada u not in x.selected:
            se total + size[u] ≤ ub e cabe no stock global:
                x' ← selected ∪ {u}, recompute
                se x'.obj > x.obj: x ← x'; melhorou ← True; break
até não melhorar
```

**Observações:**
- A cada movimento aceito, `visited_aisles` é recalculado pelo mesmo primitivo indicado em `greedy`.
- Viabilidade de `swap`/`add` verifica apenas o **estoque global** (`stock_total`), não o remanescente — porque o estoque consumido por outras ordens já está refletido no `demand` novo e será conferido implicitamente. A feasibilidade final é confirmada pelo runner via `is_solution_feasible`.
- `drop` pode melhorar o objetivo quando remover uma ordem elimina um corredor inteiro, compensando a perda em unidades.

## Parâmetros

Validados em `__init__`; `ValueError` é levantado na construção, não durante `solve`.

| Parâmetro | Valores aceitos | Obrigatório | Efeito |
|---|---|---|---|
| `alpha` | `float` em `[0.0, 1.0]` | sim | Controla o tamanho da RCL. `0.0` = guloso puro (RCL = {arg max}); `1.0` = aleatório puro (RCL = todos os candidatos). |
| `construction_score` | `"size"` / `"synergy"` / `"aisle_cost"` / `"aisle_cost_fast"` | sim | Função de score para a RCL. Ver tabela acima. |
| `max_iterations` | `int > 0` | sim | Número de iterações do laço externo GRASP. |
| `greedy` | `"simple"` / `"multi"` | sim | Primitivo de seleção de corredores (`greedy_aisle_select` vs `multi_greedy_aisle_select`). Usado tanto na construção quanto no recálculo da busca local. |
| `local_search` | `"none"` / `"swap"` / `"full"` | sim | `"none"` = apenas multi-start randomizado; `"swap"` = só 1-por-1 swap; `"full"` = swap + drop + add. |
| `similarity_weighted` | `bool` (default `False`) | não | Somente para `construction_score: synergy`. `False` usa Jaccard sobre conjuntos de itens; `True` usa Jaccard ponderado por quantidades. |
| `seed` | `int` / ausente | não | Semente do RNG usado na escolha dentro da RCL. Ausente = Python `random.Random()` sem semente. |

## Variantes

Combinando `alpha` × `construction_score` × `local_search` × `greedy`, o mesmo algoritmo cobre desde multi-start puro até busca local intensiva. Algumas combinações úteis registradas em `configs/grasp.yaml`:

| Nome sugerido | `params` | Abordagem |
|---|---|---|
| `grasp_size_a03_nols` | `alpha: 0.3, construction_score: size, max_iterations: 100, greedy: simple, local_search: none` | Ablação: apenas multi-start randomizado — serve como baseline para medir o ganho da busca local |
| `grasp_size_a03_swap` | `alpha: 0.3, construction_score: size, max_iterations: 20, greedy: multi, local_search: swap` | Construção barata + busca local leve; melhor razão custo/ganho |
| `grasp_synergy_a03_full` | `alpha: 0.3, construction_score: synergy, max_iterations: 10, greedy: multi, local_search: full, similarity_weighted: false` | Construção orientada a compatibilidade entre ordens + busca local completa |
| `grasp_size_a05_swap` | `alpha: 0.5, construction_score: size, max_iterations: 20, greedy: multi, local_search: swap` | Maior diversificação (`alpha=0.5`) com swap rápido |
| `grasp_aisle_a02_full` | `alpha: 0.2, construction_score: aisle_cost, max_iterations: 5, greedy: multi, local_search: full` | Construção informada pelo custo de corredores (cara) + busca local completa; poucas iterações |
| `grasp_aisle_fast_a02_full` | `alpha: 0.2, construction_score: aisle_cost_fast, max_iterations: 20, greedy: multi, local_search: full` | Mesma ideia do `aisle_cost`, mas com proxy O(\|items\|) por candidato — libera ordens de grandeza a mais em `max_iterations` dentro do mesmo time budget |
| `grasp_aisle_fast_a03_swap` | `alpha: 0.3, construction_score: aisle_cost_fast, max_iterations: 30, greedy: multi, local_search: swap` | Scoring barato informado por corredor + swap leve; alto throughput de iterações |

Como a construção é cara quando `construction_score="aisle_cost"` e a busca local `"full"` domina o tempo, `max_iterations` foi calibrado para caber no `time_limit` de 20s do runner. `"aisle_cost_fast"` relaxa essa restrição: troca exatidão por velocidade, permitindo dezenas de iterações mesmo em instâncias grandes. Veja `configs/grasp.yaml` para a grade completa.

## Pontos fortes

- **Combina diversificação e intensificação** em um mesmo algoritmo, evitando os ótimos locais determinísticos de `simple`/`seed`.
- **Tuning único (`alpha`)** interpola continuamente entre greedy e aleatório, facilitando a varredura de parâmetros.
- **Implementação totalmente independente**: não herda nem delega a `SimpleHeuristic`/`SeedHeuristic`. Reutiliza apenas primitivos puros (`similarity`, `greedy_aisle_select`, `multi_greedy_aisle_select`).
- **Busca local configurável**: `"none"` / `"swap"` / `"full"` permite ablações limpas para quantificar o efeito da busca local.
- **Reprodutibilidade**: com `seed` fixado, duas execuções na mesma instância produzem resultados idênticos.

## Limitações

- **Tempo de construção escala** com `aisle_cost`: O(candidatos × aisles × \|demand\|) por passo de construção, vs O(1) para `size`. Use `aisle_cost_fast` como alternativa O(\|items\|) por candidato se o orçamento de tempo for apertado.
- **`aisle_cost_fast` é um proxy** — aproxima `current_aisles_set` por "corredor de maior estoque de cada item". Pode superestimar o custo de novos corredores, mas a ordem relativa entre candidatos costuma preservar-se o bastante para o corte pela RCL.
- **Busca local first-improvement** é sensível à ordem de varredura — um best-improvement daria soluções mais estáveis ao custo de mais tempo por passo.
- **Sem controle interno de tempo**: GRASP não checa o `time_limit` do runner. Se o SIGALRM disparar no meio da busca local, a solução parcial é perdida (o runner retorna vazio). `max_iterations` precisa ser calibrado manualmente por variante.
- **Viabilidade de estoque** nos movimentos da busca local é verificada apenas contra o estoque global. A feasibilidade fim-a-fim é garantida pelo `is_solution_feasible` do runner.
- **Nenhum path relinking / memória adaptativa** — é GRASP padrão, não uma variante avançada (Reactive GRASP, GRASP+PR, etc.).

## Dependências internas

- `algorithms/utils/greedy_aisle_select.py` — cobertura gulosa padrão (usado para `greedy: simple` e para o score `aisle_cost`).
- `algorithms/utils/multi_greedy_aisle_select.py` — variante multi-greedy (usado para `greedy: multi`).
- `algorithms/utils/similarity.py` — Jaccard (ponderado ou não) para `construction_score: synergy`.
- `algorithms/base.py` — classe base `Algorithm`.
- `problems/base.py` — tipo `ProblemInput`.
