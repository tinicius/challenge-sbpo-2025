# SimpleHeuristic

## Visão geral

`SimpleHeuristic` é o algoritmo guloso mais básico do projeto. Dado um problema `ProblemInput`, ele:

1. percorre as ordens em uma sequência (aleatória ou ordenada por tamanho),
2. aceita cada ordem enquanto houver estoque e o limite superior da onda (`ub`) não for violado,
3. fecha a solução escolhendo corredores suficientes para cobrir a demanda agregada.

- **Código**: `algorithms/simple/simple_heuristic.py`
- **Nome registrado**: `simple`
- **Base**: `algorithms/base.py` (`Algorithm`)

## Ideia central

O objetivo do problema é maximizar `total de unidades coletadas / número de corredores visitados`. Essa heurística separa o problema em duas etapas independentes e gulosas:

- **Seleção de ordens**: tenta encher a onda sob `lb ≤ total_units ≤ ub` seguindo uma sequência (aleatória ou ordenada).
- **Seleção de corredores**: dada a demanda resultante, delega a cobertura a um dos utilitários existentes (`greedy_aisle_select` ou `multi_greedy_aisle_select`).

Por ser desacoplada, a heurística é rápida e serve como **baseline** e como **provedor de seed** para algoritmos mais elaborados.

## Fluxo da implementação

```
order_sizes ← [soma de unidades de cada ordem]

se params.order está definido:
    indices ← sort([0..n_orders-1], por order_sizes, asc ou desc)
senão:
    indices ← shuffled([0..n_orders-1])   # usa params.seed se fornecido

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
| `order` | `"asc"` / `"desc"` / ausente | não | Ordena por soma de unidades (crescente ou decrescente). Ausente = ordem aleatória. |
| `seed` | `int` / ausente | não | Semente para o shuffle aleatório. Só tem efeito quando `order` não está definido. |

### Comportamento de `order`

- `"asc"` → percorre ordens **do menor para o maior** — tende a maximizar quantidade de ordens aceitas.
- `"desc"` → percorre ordens **do maior para o menor** — tende a maximizar unidades por ordem aceita.
- ausente → shuffle aleatório via `random.Random(seed)` se `seed` fornecido, ou `random.shuffle` global caso contrário.

Quando `order` é fornecido, nenhum shuffle é realizado.

## Variantes

Combinando `order` × `greedy`, o mesmo algoritmo cobre várias estratégias:

| Nome sugerido | `params` | Abordagem |
|---|---|---|
| `random` | `greedy: simple` | Seed aleatória + cobertura gulosa simples |
| `random_multi` | `greedy: multi` | Seed aleatória + cobertura multi-greedy |
| `smaller` | `order: asc, greedy: simple` | Ordens menores primeiro + gulosa simples |
| `smaller_multi` | `order: asc, greedy: multi` | Ordens menores primeiro + multi-greedy |
| `bigger` | `order: desc, greedy: simple` | Ordens maiores primeiro + gulosa simples |
| `bigger_multi` | `order: desc, greedy: multi` | Ordens maiores primeiro + multi-greedy |

## Pontos fortes

- Extremamente simples e rápido; serve como baseline e gerador de warm-start.
- Determinístico quando `order` ou `seed` é fornecido.
- Totalmente desacoplado: trocar o seletor de corredores não afeta a lógica de seleção de ordens.

## Limitações

- Não leva em conta similaridade entre ordens nem entre corredores.
- Garante feasibilidade apenas contra o estoque global; não realiza trocas ou reordenações pós-loop.
- Aleatoriedade sem `seed` produz variância entre execuções quando `order` não é fornecido.

## Dependências internas

- `algorithms/utils/greedy_aisle_select.py` — cobertura gulosa padrão (`greedy: simple`).
- `algorithms/utils/multi_greedy_aisle_select.py` — variante multi-greedy (`greedy: multi`).
- `algorithms/base.py` — classe base `Algorithm`.
- `problems/base.py` — tipo `ProblemInput`.
