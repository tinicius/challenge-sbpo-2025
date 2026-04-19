# AisleFirstHeuristic

## Visão geral

`AisleFirstHeuristic` é uma heurística gulosa **orientada a corredores** (aisle-first). Dado um problema `ProblemInput`, ela:

1. pontua cada corredor por uma métrica de densidade (variedade e/ou quantidade de itens),
2. expande incrementalmente o conjunto de corredores do mais denso ao menos denso, agregando seu inventário,
3. em cada tamanho `k` de conjunto, empacota gulosamente pedidos que caibam no inventário disponível e no limite superior da onda (`ub`), mantendo o melhor par (pedidos, corredores) encontrado.

- **Código**: `algorithms/aisle_first/aisle_first_heuristic.py`
- **Nome registrado**: `aisle_first`
- **Base**: `algorithms/base.py` (`Algorithm`)

## Ideia central

O objetivo do problema é maximizar `total de unidades coletadas / número de corredores visitados`. Enquanto a `SimpleHeuristic` ataca o **numerador** (escolhe pedidos primeiro e depois cobre sua demanda com corredores), esta heurística ataca o **denominador**: parte do princípio de que um número pequeno de corredores muito densos já é capaz de satisfazer uma onda viável, e procura explicitamente esse conjunto mínimo denso antes de olhar para os pedidos.

Isso é especialmente interessante quando:

- há corredores com alta concentração de itens úteis (alta variedade **ou** alta quantidade),
- as instâncias têm `lb`/`ub` relativamente baixos, de forma que poucos corredores bastam para fechar uma onda válida.

A heurística também serve como **ferramenta de análise**: alterando `score` e `prune`, é possível estudar quais corredores a lógica aisle-first considera "densos" e como essa escolha se compara com o conjunto final após a poda por `greedy_aisle_select` / `multi_greedy_aisle_select`.

## Fluxo da implementação

```
total_demand ← { item: soma de qty em todos os pedidos }

ranked_aisles ← sort desc corredores por score(corredor)
    score depende de params.score:
        "useful"  → Σ min(aisle[i], total_demand[i])      # interseção útil com a demanda
        "units"   → Σ aisle.values()                      # quantidade total no corredor
        "variety" → |aisle|                               # nº de itens únicos
        "mixed"   → Σ aisle.values() × |aisle|            # variedade × quantidade

order_sequence ← _build_order_sequence(params.order, order_sizes, params.seed)
    params.order ausente → shuffle aleatório (usa params.seed se fornecido)
    params.order "asc"  → pedidos menores primeiro
    params.order "desc" → pedidos maiores primeiro

inventory ← {}
best_obj ← 0
best_orders, best_aisles, best_units ← [], [], 0

para cada k, aisle_idx em enumerate(ranked_aisles, start=1):
    inventory += aisles[aisle_idx]                     # acumula itens do corredor k

    se ub / k ≤ best_obj:                              # parada antecipada: não dá mais para melhorar
        break

    selected, total_units ← _pack_orders(order_sequence, orders, inventory, ub)
        para cada idx em order_sequence:
            se total + order_sizes[idx] > ub: pular
            se inventory remanescente não cobre a ordem: pular
            aceitar ordem, consumir do inventory remanescente, somar em total

    se total_units < lb: continuar                     # onda inviável com este k
    obj ← total_units / k
    se obj > best_obj: atualizar (best_orders, best_aisles, best_units, best_obj)

se params.prune é None:
    retornar (best_orders, best_aisles, best_obj)

senão:                                                 # poda opcional
    demand ← demanda agregada dos best_orders
    se params.prune == "multi":
        visited ← multi_greedy_aisle_select(demand, aisles)
    senão:
        visited ← greedy_aisle_select(demand, aisles)
    retornar (best_orders, visited, best_units / |visited|)
```

**Nota sobre a parada antecipada**: como `total_units ≤ ub`, vale `obj_futuro ≤ ub / (k+1)`. Se `ub / k ≤ best_obj`, nenhum `k′ ≥ k` pode superar o melhor atual e podemos encerrar a busca.

**Nota sobre `_pack_orders`**: trabalha contra uma cópia do `inventory` para não mutar o acumulador compartilhado entre iterações de `k`. O guard `total + size > ub` é avaliado antes da checagem de inventário, evitando a consulta ao dict para pedidos que já ultrapassariam a onda.

## Parâmetros

Validados em `__init__`; `ValueError` é levantado na construção, não durante `solve`.

| Parâmetro | Valores aceitos | Obrigatório | Efeito |
|---|---|---|---|
| `score` | `"useful"` / `"units"` / `"variety"` / `"mixed"` | não (default `"useful"`) | Métrica usada para ordenar os corredores em ordem decrescente de densidade. |
| `order` | `"asc"` / `"desc"` / ausente | não | Sequência de empacotamento dos pedidos dentro do inventário. Ausente = ordem aleatória. |
| `prune` | `"simple"` / `"multi"` / ausente | não | Se definido, aplica poda dos corredores sobre a demanda dos pedidos escolhidos. Ausente = mantém exatamente os `k` corredores mais densos. |
| `seed` | `int` / ausente | não | Semente para o shuffle quando `order` é ausente. |

### Comportamento de `score`

- `"useful"` → soma de `min(qty_no_corredor, total_demand[item])` — igual à métrica do warm-start do `DinkelbachMIP`. Prioriza corredores cuja oferta é útil em relação à demanda global.
- `"units"` → `sum(aisle.values())` — prioriza corredores com muitas unidades, ignorando se essas unidades são demandadas.
- `"variety"` → `len(aisle)` — prioriza corredores com mais tipos diferentes de itens.
- `"mixed"` → `sum(aisle.values()) * len(aisle)` — combina variedade **e** quantidade simultaneamente (interpretação literal de "mais densos = com maior variedade e quantidade de itens").

### Comportamento de `prune`

- ausente → `visited_aisles` é exatamente o prefixo de `ranked_aisles` que atingiu o melhor `obj`. Útil para **estudar o conjunto denso escolhido** pela heurística.
- `"simple"` → aplica `greedy_aisle_select` sobre a demanda dos pedidos escolhidos; costuma reduzir drasticamente o número de corredores quando o prefixo denso carregava muito inventário sobrando.
- `"multi"` → aplica `multi_greedy_aisle_select`; variante que reexamina corredores disponíveis a cada rodada.

Quando `prune` é aplicado, o conjunto final pode divergir completamente do prefixo denso — inclusive trocando corredores pelos que melhor cobrem a demanda real — e o `objective` é recalculado com `|visited_aisles|` pós-poda.

## Variantes sugeridas

| Nome sugerido | `params` | Abordagem |
|---|---|---|
| `aisle_useful` | `score: useful` | Densidade por interseção com demanda global, pedidos em ordem aleatória |
| `aisle_useful_desc` | `score: useful, order: desc` | Densidade por interseção com demanda + pedidos maiores primeiro |
| `aisle_units` | `score: units` | Densidade por quantidade pura |
| `aisle_variety` | `score: variety` | Densidade por variedade pura |
| `aisle_mixed` | `score: mixed, order: desc` | Variedade × quantidade + pedidos maiores primeiro |
| `aisle_useful_pruned` | `score: useful, prune: simple` | Denso + poda gulosa simples |
| `aisle_mixed_pruned_multi` | `score: mixed, order: desc, prune: multi` | Denso misto + poda multi-greedy |
| `aisle_variety_pruned` | `score: variety, prune: simple` | Variedade + poda simples — estuda se a variedade sozinha sobrevive à poda |

## Pontos fortes

- Ataca explicitamente o denominador da objetivo, complementando as variantes order-first de `SimpleHeuristic`.
- Determinístico quando `score` e `order` estão definidos (e `seed` não é relevante para esses modos).
- Parada antecipada `ub / k ≤ best_obj` evita iterações inúteis em instâncias grandes.
- Parametrização de `prune` permite usar o mesmo solver como **ferramenta de análise** — comparar quais corredores a escolha densa preserva antes e depois da poda.
- Totalmente desacoplado: `score`, `order` e `prune` são ortogonais.

## Limitações

- Guloso puro: nenhuma troca, *swap* ou reordenação pós-loop.
- A pontuação dos corredores não depende de quais pedidos **serão efetivamente** aceitos; corredores densos podem conter muitos itens que nenhum pedido viável usa.
- `"mixed"` pode superponderar corredores grandes porém irrelevantes para a demanda — para instâncias assimétricas, `"useful"` tende a ser mais robusto.
- Sem `prune`, o conjunto final pode conter corredores com inventário ocioso (bom para análise, ruim para a objetivo).
- Com `prune`, o conjunto final pode divergir do prefixo denso e mudar significativamente o objetivo — o que é intencional, mas precisa ser considerado ao interpretar os resultados.

## Dependências internas

- `algorithms/utils/greedy_aisle_select.py` — cobertura gulosa simples (`prune: simple`).
- `algorithms/utils/multi_greedy_aisle_select.py` — variante multi-greedy (`prune: multi`).
- `algorithms/base.py` — classe base `Algorithm`.
- `problems/base.py` — tipo `ProblemInput`.
