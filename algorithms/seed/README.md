# SeedHeuristic

## Visão geral

`SeedHeuristic` é uma heurística construtiva orientada por **ordem-semente**. Ela começa escolhendo uma ordem inicial (`seed`) por uma estratégia configurável e, a partir dela, vai adicionando novas ordens de forma gulosa enquanto respeita:

- limite superior da wave (`ub`),
- disponibilidade de estoque global,
- e, ao final, o limite inferior (`lb`).

A escolha do próximo pedido é guiada por um critério de sinergia:

1. **maximizar similaridade** com a demanda já acumulada, ou
2. **minimizar novos corredores** necessários para cobrir a demanda após a adição.

- **Código**: `algorithms/seed/seed_heuristic.py`
- **Nome no registry**: `seed`
- **Nome retornado por `name`**: `seed_heuristic`
- **Base**: `algorithms/base.py` (`Algorithm`)

## Ideia central

Diferente de um greedy puramente ordenado por tamanho (`simple`), aqui a construção é **ancorada em uma seed explícita**. A intuição é:

- uma seed bem escolhida define um "núcleo" de itens promissor,
- ordens adicionadas com alta compatibilidade (ou baixo custo marginal de corredores) tendem a manter uma boa razão `total_units / |aisles|`.

Isso torna o algoritmo rápido, simples de parametrizar e útil como baseline intermediário entre heurísticas muito simples e metaheurísticas mais caras.

## Fluxo da implementação

```text
orders, aisles, lb, ub ← instância

se n_orders == 0 ou n_aisles == 0:
    retornar vazio

order_sizes ← soma de unidades por ordem
stock_total ← soma de estoque por item em todos os corredores

seed_idx ← _pick_seed(...)
se seed inexistente, seed_size > ub, ou seed inviável no estoque global:
    retornar vazio

selected ← [seed_idx]
demand ← cópia da seed
total_units ← size(seed)
stock_remaining ← stock_total - demand(seed)
remaining ← todas as ordens exceto seed

enquanto houver remaining:
    candidates ← ordens que
        (total_units + size <= ub)
        e (cabem no stock_remaining)

    se candidates vazio: break

    se synergy == "max_similarity":
        best ← argmax similarity(demand, order_i) desempate por maior size
    senão ("min_new_aisles"):
        aisles_now ← greedy_aisle_select(demand)
        para cada candidato i:
            combined ← demand + order_i
            after ← greedy_aisle_select(combined)
            custo_i ← |after - aisles_now|
        best ← argmin custo_i desempate por maior size

    adicionar best em selected
    atualizar demand, total_units, stock_remaining
    remover best de remaining

se total_units < lb:
    retornar vazio

visited_aisles ←
    multi_greedy_aisle_select(demand)  se greedy == "multi"
    greedy_aisle_select(demand)        se greedy == "simple"

se visited_aisles vazio:
    retornar vazio

retornar selected, visited_aisles, objective = total_units / |visited_aisles|
```

### Nota de implementação

No modo `min_new_aisles`, o conjunto de corredores da demanda atual (`aisles_now`) é cacheado dentro da iteração para evitar recomputação desnecessária ao avaliar múltiplos candidatos no mesmo estado.

## Estratégias de seed (`seed_strategy`)

| Estratégia | Regra |
|---|---|
| `biggest` | Escolhe a ordem não-vazia com maior número de unidades. |
| `smallest` | Escolhe a ordem não-vazia com menor número de unidades. |
| `random` | Escolhe ordem não-vazia aleatória (`seed` controla reprodutibilidade quando fornecido). |
| `most_shared` | Escolhe a ordem cujos itens aparecem em mais corredores (soma de frequências por item), com desempate por maior tamanho. |

## Estratégias de sinergia (`synergy`)

| Estratégia | Regra |
|---|---|
| `max_similarity` | Escolhe o candidato mais similar à `demand` atual (Jaccard simples ou ponderado). |
| `min_new_aisles` | Escolhe o candidato que adiciona menos corredores novos na cobertura greedy da demanda. |

## Parâmetros

Validados no `__init__`; erros de configuração geram `ValueError` antes da execução.

| Parâmetro | Valores aceitos | Obrigatório | Efeito |
|---|---|---|---|
| `seed_strategy` | `biggest` / `smallest` / `most_shared` / `random` | sim | Define qual ordem inicia a construção. |
| `synergy` | `min_new_aisles` / `max_similarity` | sim | Regra para escolher a próxima ordem candidata. |
| `greedy` | `simple` / `multi` | sim | Método de seleção final de corredores (`greedy_aisle_select` ou `multi_greedy_aisle_select`). |
| `similarity_weighted` | `bool` (default `False`) | não | Apenas para `synergy=max_similarity`: ativa Jaccard ponderado por quantidade. |
| `seed` | `int` / ausente | não | Semente para sorteio quando `seed_strategy=random`. |

## Variantes no projeto

O arquivo `configs/seed.yaml` registra a grade principal combinando:

- `seed_strategy ∈ {biggest, smallest, most_shared, random}`
- `synergy ∈ {min_new_aisles, max_similarity}`
- `greedy ∈ {simple, multi}`

Total: **16 variantes**.

Exemplos de nomes configurados:

| Nome | `params` | Intuição |
|---|---|---|
| `seed_biggest_minaisles_simple` | `biggest + min_new_aisles + simple` | Seed grande e expansão que evita abrir corredores novos. |
| `seed_smallest_similarity_multi` | `smallest + max_similarity + multi` | Seed pequena e agregação por compatibilidade com cobertura multi-greedy. |
| `seed_shared_similarity_simple` | `most_shared + max_similarity + simple` | Começa por itens amplamente distribuídos e prioriza pedidos parecidos. |
| `seed_random_minaisles_multi` | `random + min_new_aisles + multi` | Maior diversificação entre repetições com custo marginal de corredor. |

## Pontos fortes

- Simples e rápido, com custo computacional moderado.
- Parametrização pequena e interpretável (`seed_strategy`, `synergy`, `greedy`).
- Boa ponte entre baselines determinísticos e metaheurísticas mais caras.
- Pode capturar estruturas de compatibilidade entre pedidos sem busca local explícita.

## Limitações

- Não realiza melhoria pós-construção (sem swap/drop/add), então pode parar cedo em ótimo local.
- A viabilidade durante a construção é checada contra estoque agregado, sem otimização conjunta pedido-corredor no processo de seleção.
- `min_new_aisles` usa `greedy_aisle_select` para estimar custo marginal, mesmo quando a seleção final usa `greedy=multi`.
- Resultado pode ser sensível à escolha da seed inicial, especialmente em instâncias heterogêneas.

## Dependências internas

- `algorithms/utils/greedy_aisle_select.py` — cobertura gulosa padrão.
- `algorithms/utils/multi_greedy_aisle_select.py` — cobertura multi-greedy.
- `algorithms/utils/similarity.py` — similaridade entre demandas/pedidos.
- `algorithms/base.py` — classe base `Algorithm`.
- `problems/base.py` — tipo `ProblemInput`.
