# SimulatedAnnealing

## Visão geral

`SimulatedAnnealing` implementa a metaheurística **Simulated Annealing (SA)** para o problema de picking em ondas. A ideia é partir de uma solução inicial viável e caminhar pelo espaço aplicando movimentos de vizinhança aleatórios; diferente da busca local clássica, o SA **aceita movimentos de piora com probabilidade decrescente ao longo do tempo**, controlada por uma temperatura que resfria gradualmente. Isso permite escapar de ótimos locais no início e convergir para um ótimo estável no final.

1. **Solução inicial** — greedy determinístico ou construção randomizada estilo GRASP.
2. **Loop SA** — a cada iteração sorteia um movimento de vizinhança, calcula `Δ = obj(vizinho) − obj(atual)` e aceita:
   - sempre, se `Δ > 0`;
   - com probabilidade `exp(Δ / T)`, se `Δ ≤ 0`.
3. **Resfriamento** — a temperatura `T` é atualizada a cada iteração (geométrico ou linear). O loop para quando `T < min_temp` ou `max_iterations` é atingido.

Mantém a melhor solução já vista (`best`) e retorna-a ao final.

- **Código**: `algorithms/sa/sa_heuristic.py`
- **Nome registrado**: `sa`
- **Base**: `algorithms/base.py` (`Algorithm`)

## Ideia central

O objetivo `total_units / num_aisles` é uma razão — melhorar pode significar **adicionar unidades** (mais ordens) ou **reduzir corredores** (menos ordens que compartilham menos itens). Heurísticas gulosas convergem rapidamente para um ótimo local e travam.

O SA contorna isso aceitando pioras **probabilisticamente**:

- **Temperatura alta (início)** — `exp(Δ / T) ≈ 1` mesmo para `Δ < 0` grande → praticamente qualquer vizinho é aceito, permitindo explorar regiões distantes.
- **Temperatura baixa (fim)** — `exp(Δ / T) ≈ 0` para `Δ < 0` → comporta-se como hill-climbing, intensificando em torno do ótimo local atual.

O resfriamento gradual faz o algoritmo transitar suavemente de diversificação para intensificação. Com parâmetros bem calibrados, SA tipicamente supera busca local pura em instâncias com muitos ótimos locais próximos.

Diferenças em relação ao GRASP deste repositório:

| Aspecto | GRASP | SA |
|---|---|---|
| Exploração | Múltiplas partidas independentes (multi-start) | Uma trajetória longa com saltos probabilísticos |
| Busca local | First-improvement determinística até convergir | Aceita pioras controladas por temperatura |
| Seleção de vizinho | Varre até achar melhora | Sorteia **um** vizinho por iteração |
| Parâmetro-chave | `alpha` (RCL) | `initial_temp` + `cooling_rate` |

## Fluxo da implementação

```
current ← build_initial(...)
se current é inviável: retornar vazio
best ← current
T ← initial_temp

para k em 1..max_iterations:
    se T < min_temp: break

    move ← rng.choice(moves)            # swap | drop | add
    neighbor ← generate_neighbor(current, move, ...)
    se neighbor é None: continuar       # movimento inviável no estado atual

    Δ ← neighbor.objetivo − current.objetivo
    se Δ > 0 ou rng.random() < exp(Δ / T):
        current ← neighbor
        se current.objetivo > best.objetivo:
            best ← current

    T ← cool(T)

retornar best
```

### Construção inicial (`init`)

Duas estratégias disponíveis, selecionadas por `init`:

- **`init: greedy`** — ordena as ordens por `order_sizes` decrescente e adiciona enquanto `total_units ≤ ub` e o estoque global permitir. Totalmente determinístico (não depende de `seed`).
- **`init: grasp`** — reusa a construção randomizada do GRASP com `init_alpha` e `init_construction` (`size` / `synergy` / `aisle_cost`). Cada execução com `seed` diferente começa de um ponto distinto no espaço — útil quando combinado com `max_iterations` menor, pois o SA já começa próximo de um bom ótimo local.

Se a solução inicial é inviável (`total_units < lb` ou não cobre com corredores), o solver retorna resultado vazio imediatamente — SA não tenta repetir a construção.

### Movimentos de vizinhança (`moves`)

Todos os movimentos escolhem **um candidato aleatório** e verificam viabilidade uma única vez (diferente do GRASP, que varre o espaço inteiro de vizinhos procurando melhora).

| Movimento | Ação | Viabilidade |
|---|---|---|
| `swap` | Sorteia `s ∈ selected`, `u ∉ selected`; troca `s` por `u` | `lb ≤ total' ≤ ub` e `new_demand` cabe no estoque global |
| `drop` | Sorteia `s ∈ selected` (requer `|selected| > 1`); remove `s` | `total' ≥ lb` |
| `add` | Sorteia `u ∉ selected`; adiciona `u` | `total' ≤ ub` e `new_demand` cabe no estoque global |

Após qualquer movimento, `visited_aisles` é recomputado via `greedy_aisle_select` ou `multi_greedy_aisle_select` (conforme `greedy`). Se o movimento sorteado for inviável no estado atual, o SA simplesmente "pula" essa iteração (`neighbor = None` → nenhuma mudança, mas `T` ainda resfria).

A lista `moves` em `params` é **livre**: pode conter qualquer subconjunto não-vazio de `{swap, drop, add}`. A cada passo o algoritmo sorteia **uniformemente** um elemento dessa lista.

### Schedules de resfriamento (`cooling`)

| `cooling` | Regra | `cooling_rate` | Comportamento |
|---|---|---|---|
| `"geometric"` | `T ← T × cooling_rate` | `(0, 1)` | Decaimento exponencial; `T_k = T₀ × α^k`. Padrão clássico. |
| `"linear"` | `T ← max(min_temp, T − cooling_rate)` | `> 0` | Decaimento linear com piso em `min_temp`. Mais lento no início, mais rápido no fim. |

`cooling_rate` é validado conforme o modo: em `"geometric"` deve estar em `(0, 1)`; em `"linear"` qualquer positivo é aceito (a interpretação é o decremento absoluto por iteração).

### Critério de parada

O loop termina no primeiro de:
- `T < min_temp` (resfriamento completo);
- `max_iterations` atingido.

Movimentos inviáveis consomem iteração mas não alteram estado nem `best`.

## Parâmetros

Validados em `__init__`; `ValueError` é levantado na construção, não durante `solve`.

| Parâmetro | Valores aceitos | Obrigatório | Efeito |
|---|---|---|---|
| `init` | `"greedy"` / `"grasp"` | sim | Estratégia de construção inicial. |
| `init_alpha` | `float` em `[0.0, 1.0]` (default `0.3`) | só se `init="grasp"` | Alpha da RCL da construção GRASP; ignorado quando `init="greedy"`. |
| `init_construction` | `"size"` / `"synergy"` / `"aisle_cost"` (default `"size"`) | só se `init="grasp"` | Score da RCL (veja README do GRASP); ignorado quando `init="greedy"`. |
| `moves` | lista não-vazia ⊆ `{"swap", "drop", "add"}` | sim | Repertório de vizinhança; cada iteração sorteia um deles uniformemente. |
| `cooling` | `"geometric"` / `"linear"` | sim | Schedule de resfriamento. |
| `initial_temp` | `float > 0` | sim | `T₀`. Para problema normalizado, valores típicos estão entre 0.1 e 10 — como `Δ` é delta de razão `total/|aisles|`, escolha `T₀` na mesma ordem de grandeza do objetivo esperado. |
| `cooling_rate` | `float > 0` | sim | Em `geometric`: fator multiplicativo em `(0, 1)`, tipicamente `0.99–0.999`. Em `linear`: decremento absoluto por iteração. |
| `min_temp` | `float > 0` (default `1e-3`) | não | Piso de temperatura; atinge = parada. |
| `max_iterations` | `int > 0` | sim | Teto de iterações do loop SA. |
| `greedy` | `"simple"` / `"multi"` | sim | Primitivo de seleção de corredores (`greedy_aisle_select` vs `multi_greedy_aisle_select`). |
| `similarity_weighted` | `bool` (default `False`) | não | Somente para `init_construction: synergy`. `False` usa Jaccard sobre conjuntos de itens; `True` usa Jaccard ponderado por quantidades. |
| `seed` | `int` / ausente | não | Semente do RNG (usado na construção GRASP, no sorteio de movimentos e no teste de aceitação). Ausente = `random.Random()` sem semente. |

## Variantes

Combinando `init` × `moves` × `cooling` × cooling schedule, é possível cobrir desde SA clássico (swap + geométrico) até SA com vizinhança rica e resfriamento linear. Algumas combinações registradas em `configs/sa.yaml`:

| Nome sugerido | `params` | Abordagem |
|---|---|---|
| `sa_greedy_swap_geo` | `init: greedy, moves: [swap], cooling: geometric, initial_temp: 1.0, cooling_rate: 0.995, max_iterations: 2000` | Baseline: init determinístico + vizinhança mínima + resfriamento clássico |
| `sa_grasp_full_geo` | `init: grasp, init_alpha: 0.3, init_construction: size, moves: [swap, drop, add], cooling: geometric, initial_temp: 2.0, cooling_rate: 0.995, max_iterations: 3000` | Partida randomizada + repertório completo; temperatura inicial maior para compensar a diversificação já embutida |
| `sa_greedy_full_linear` | `init: greedy, moves: [swap, drop, add], cooling: linear, initial_temp: 1.0, cooling_rate: 0.0005, max_iterations: 2000` | Resfriamento linear; decai mais devagar no começo |
| `sa_greedy_full_fast` | `init: greedy, moves: [swap, drop, add], cooling: geometric, initial_temp: 0.5, cooling_rate: 0.99, max_iterations: 1500` | Resfriamento agressivo + poucas iterações; favorece intensificação rápida |
| `sa_grasp_synergy_full_geo` | `init: grasp, init_construction: synergy, moves: [swap, drop, add], cooling: geometric, initial_temp: 2.0, cooling_rate: 0.995, max_iterations: 2000` | Partida orientada por compatibilidade entre ordens + vizinhança completa |

Veja `configs/sa.yaml` para a grade completa.

## Pontos fortes

- **Escape de ótimos locais** — aceitação probabilística de pioras permite visitar regiões que a busca local determinística nunca alcançaria.
- **Controle contínuo diversificação/intensificação** via `initial_temp` e `cooling_rate`, sem recomeçar do zero como o GRASP.
- **Implementação totalmente independente** — reusa apenas primitivos puros (`similarity`, `greedy_aisle_select`, `multi_greedy_aisle_select`); não herda nem delega a outros solvers.
- **Vizinhança configurável** — `moves` permite ablações limpas (ex.: SA só com `swap` vs SA com `swap+drop+add`) para quantificar o ganho de cada operador.
- **Init flexível** — `greedy` entrega determinismo para depuração; `grasp` entrega diversidade útil em instâncias com muitos ótimos locais.
- **Reprodutibilidade** — com `seed` fixado, duas execuções na mesma instância produzem resultados idênticos.

## Limitações

- **Sensibilidade a `initial_temp`** — temperatura mal calibrada vira aleatória pura (`T₀` muito alto) ou hill-climbing puro (`T₀` muito baixo). Como a unidade de `Δ` é a mesma do objetivo (`total/|aisles|`, tipicamente 1–20), `T₀ ≈ 1–2` é um chute razoável, mas convém afinar por instância.
- **Um vizinho por iteração** — diferente do GRASP que varre vizinhos em busca de melhora, o SA aqui testa **apenas 1** por iteração. Movimentos inviáveis desperdiçam a iteração (o estado não muda, mas `T` resfria). Em instâncias muito restritas, a maioria dos movimentos pode falhar.
- **Sem reaquecimento (reheating)** — se a busca estabilizar cedo, não há mecanismo para reinjetar energia. Uma extensão natural seria adicionar reheat quando `best` não melhora por N iterações.
- **Sem controle interno de tempo** — o loop para por `max_iterations` ou `min_temp`, não pelo `time_limit` do runner. Se o SIGALRM disparar no meio, a solução parcial é perdida.
- **Viabilidade de estoque** nos movimentos é verificada apenas contra o estoque global; a feasibilidade fim-a-fim é garantida pelo `is_solution_feasible` do runner.
- **Aceitação uniforme de movimentos** — cada item de `moves` é sorteado com probabilidade igual. Não há amostragem adaptativa (ex.: pesar movimentos por taxa de sucesso histórica).

## Dependências internas

- `algorithms/utils/greedy_aisle_select.py` — cobertura gulosa padrão (usado para `greedy: simple` e para o score `aisle_cost` do init GRASP).
- `algorithms/utils/multi_greedy_aisle_select.py` — variante multi-greedy (usado para `greedy: multi`).
- `algorithms/utils/similarity.py` — Jaccard (ponderado ou não) para `init_construction: synergy`.
- `algorithms/base.py` — classe base `Algorithm`.
- `problems/base.py` — tipo `ProblemInput`.
