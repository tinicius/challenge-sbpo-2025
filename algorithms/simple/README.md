# SimpleHeuristic

## Visão geral

`SimpleHeuristic` é o algoritmo guloso mais básico do projeto. Dado um problema `ProblemInput`, ele:

1. percorre as ordens em uma sequência (aleatória, ordenada por tamanho ou ordenada por similaridade a uma ordem de referência),
2. aceita cada ordem enquanto houver estoque e o limite superior da onda (`ub`) não for violado,
3. fecha a solução escolhendo corredores suficientes para cobrir a demanda agregada.

- **Código**: `algorithms/simple/simple_heuristic.py`
- **Nome registrado**: `simple`
- **Base**: `algorithms/base.py` (`Algorithm`)

## Ideia central

O objetivo do problema é maximizar `total de unidades coletadas / número de corredores visitados`. Essa heurística separa o problema em duas etapas independentes e gulosas:

- **Seleção de ordens**: tenta encher a onda sob `lb ≤ total_units ≤ ub` seguindo uma sequência (aleatória, por tamanho ou por similaridade).
- **Seleção de corredores**: dada a demanda resultante, delega a cobertura a um dos utilitários existentes (`greedy_aisle_select` ou `multi_greedy_aisle_select`).

Por ser desacoplada, a heurística é rápida e serve como **baseline** e como **provedor de seed** para algoritmos mais elaborados.

## Fluxo da implementação

```
order_sizes ← [soma de unidades de cada ordem]

indices ← _build_traversal(n_orders, order_sizes, orders)
    se params.order ausente:
        shuffle aleatório (usa params.seed se fornecido)
    se params.order in {asc, desc}:
        sort por order_sizes (asc ou desc)
    se params.order in {similar, diff}:
        reference ← _pick_first_order(...)        # smaller / bigger / random
        sort por similarity(reference, orders[i], weighted=params.similarity_weighted)
        reverse=True para "similar", reverse=False para "diff"

stock ← { item: soma de qty em todos os corredores }

selected_orders ← []
demand ← {}
total_units ← 0

para cada idx em indices:
    se total_units + order_sizes[idx] > ub:
        pular
    se order não cabe no stock atual:
        pular
    adicionar idx a selected_orders
    total_units += order_sizes[idx]
    stock -= unidades consumidas pela order
    demand += unidades consumidas pela order

se total_units < lb:
    retornar solução vazia (objetivo = 0)

se params.greedy == "multi":
    visited_aisles ← multi_greedy_aisle_select(demand, aisles)
senão:
    visited_aisles ← greedy_aisle_select(demand, aisles)

retornar (selected_orders, visited_aisles, total_units / |visited_aisles|)
```

**Nota de desempenho**: o guard `total + size > ub` é avaliado antes do scan de estoque, evitando a consulta ao dict para ordens que ultrapassariam o limite da onda. O `demand` é acumulado incrementalmente no mesmo loop, eliminando uma passagem extra sobre `selected_orders`.

## Parâmetros

Validados em `__init__`; `ValueError` é levantado na construção, não durante `solve`.

| Parâmetro | Valores aceitos | Obrigatório | Efeito |
|---|---|---|---|
| `greedy` | `"simple"` / `"multi"` | sim | `"simple"` usa `greedy_aisle_select`; `"multi"` usa `multi_greedy_aisle_select`. |
| `order` | `"asc"` / `"desc"` / `"similar"` / `"diff"` / ausente | não | Critério de ordenação das ordens. Ausente = ordem aleatória. |
| `seed` | `int` / ausente | não | Semente para o shuffle. Aplica-se quando `order` é ausente, ou na escolha aleatória de `first_order` quando `first_order` é ausente. |
| `first_order` | `"smaller"` / `"bigger"` / ausente | não | Apenas para `order ∈ {similar, diff}`. Define a ordem de referência usada no cálculo de similaridade. Ausente = primeira ordem de um shuffle. |
| `similarity_weighted` | `bool` (default `False`) | não | Apenas para `order ∈ {similar, diff}`. `False` usa Jaccard sobre o conjunto de itens. `True` usa Jaccard ponderado pelas quantidades. |

### Comportamento de `order`

- `"asc"` → percorre ordens **do menor para o maior** — tende a maximizar quantidade de ordens aceitas.
- `"desc"` → percorre ordens **do maior para o menor** — tende a maximizar unidades por ordem aceita.
- `"similar"` → percorre ordens da **mais similar para a menos similar** em relação à `first_order`.
- `"diff"` → percorre ordens da **menos similar para a mais similar** em relação à `first_order`.
- ausente → shuffle aleatório via `random.Random(seed)` se `seed` fornecido, ou `random.shuffle` global caso contrário.

Quando `order` é fornecido, nenhum shuffle do conjunto principal é realizado.

## Variantes

Combinando `order` × `greedy` × `first_order` × `similarity_weighted`, o mesmo algoritmo cobre dezenas de estratégias. Algumas combinações úteis:

| Nome sugerido | `params` | Abordagem |
|---|---|---|
| `random` | `greedy: simple` | Seed aleatória + cobertura gulosa simples |
| `random_multi` | `greedy: multi` | Seed aleatória + cobertura multi-greedy |
| `smaller` | `order: asc, greedy: simple` | Ordens menores primeiro + gulosa simples |
| `bigger` | `order: desc, greedy: multi` | Ordens maiores primeiro + multi-greedy |
| `similar` | `order: similar, greedy: simple` | Mais similares à primeira ordem (aleatória) primeiro |
| `similar_bigger` | `order: similar, first_order: bigger, greedy: simple` | Mais similares à maior ordem primeiro |
| `similar_smaller_multi` | `order: similar, first_order: smaller, greedy: multi` | Mais similares à menor ordem + multi-greedy |
| `diff` | `order: diff, greedy: simple` | Menos similares à primeira ordem (aleatória) primeiro |
| `diff_bigger_multi` | `order: diff, first_order: bigger, greedy: multi` | Menos similares à maior ordem + multi-greedy |
| `similar_weighted` | `order: similar, greedy: simple, similarity_weighted: true` | Similaridade ponderada por quantidades |
| `diff_smaller_weighted` | `order: diff, first_order: smaller, similarity_weighted: true` | Diferença ponderada em relação à menor ordem |

Veja `config.yaml` para o conjunto completo registrado nos benchmarks.

## Pontos fortes

- Extremamente simples e rápido; serve como baseline e gerador de warm-start.
- Determinístico quando `order` ou `seed` é fornecido (e quando `first_order` é fornecido para os modos de similaridade).
- Totalmente desacoplado: trocar o seletor de corredores não afeta a lógica de seleção de ordens.
- Critério de ordenação configurável: tamanho ou similaridade (ponderada ou não), em ambas as direções.

## Limitações

- Greedy puro: nenhuma troca ou reordenação pós-loop.
- Garante feasibilidade apenas contra o estoque global; não considera a interação entre ordens e corredores.
- A escolha de `first_order` no modo `similar`/`diff` afeta fortemente o resultado e é heurística — não há garantia de que a "melhor" ordem de referência seja a maior, a menor ou uma aleatória.

## Dependências internas

- `algorithms/utils/greedy_aisle_select.py` — cobertura gulosa padrão (`greedy: simple`).
- `algorithms/utils/multi_greedy_aisle_select.py` — variante multi-greedy (`greedy: multi`).
- `algorithms/utils/similarity.py` — Jaccard (ponderado ou não) entre ordens.
- `algorithms/base.py` — classe base `Algorithm`.
- `problems/base.py` — tipo `ProblemInput`.
