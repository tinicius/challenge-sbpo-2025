# Aisle-First — Revisão e melhorias pendentes

Este documento lista as sugestões de melhoria para `AisleFirstHeuristic` e
`AisleFirstExactOrders` que **não foram implementadas** ainda. Os fixes #1
(loop prune-aware) e #2 (local search pós-construção) já estão integrados.

Validação dos fixes já implementados (dataset `a`, prune=multi, order=desc):

| Instância      | Baseline obj | LS obj    | Δ        |
| -------------- | ------------ | --------- | -------- |
| instance_0003  | 7.20         | 12.00     | +66.7%   |
| instance_0008  | 146.63       | 162.06    | +10.5%   |

A configuração padrão de LS (`first_improvement`, 200 iter, `neighbor_cap=50`)
adiciona ~50% no tempo de execução das instâncias maiores e mantém-se bem
abaixo do orçamento de 60s.

---

## #3 — Multi-start / multi-config

**Problema.** A heurística atual é determinística para um dado `(score, order,
prune, seed)`. Toda a exploração depende do ranqueamento *global* de aisles
por score: se a configuração escolhida classifica mal um aisle-chave, não há
mecanismo para se recuperar dentro de um único run.

**Proposta.** Em uma única chamada a `solve`, executar o loop principal
com várias configurações e ficar com a melhor solução encontrada.
Configurações naturais:

- `score ∈ {"useful", "units", "variety", "mixed"}`
- `order ∈ {"desc", "asc", None}`
- Sementes diferentes para `build_order_sequence` quando `order is None`
- (Opcional) Variar o critério de desempate em `rank_aisles`

Implementação sugerida:

```python
self._starts = params.get("starts", [
    {"score": "useful", "order": "desc"},
    {"score": "mixed",  "order": "desc"},
    {"score": "useful", "order": "asc"},
])
```

E o `solve` faz `for cfg in self._starts: run_inner(cfg)` retornando o melhor.
Como cada start ainda termina em <1s nas instâncias pequenas, dá para rodar
4–8 starts dentro do orçamento atual. Combina muito bem com o LS pós-construção:
o LS limpa cada start individualmente.

**Esforço.** Pequeno — refactor o corpo de `solve` em um método interno
parametrizado por `(score, order, seed)` e iterar.

---

## #4 — Re-ranqueamento dinâmico de aisles

**Problema.** O score atual é estático (calculado uma vez sobre `total_demand`
agregada de **todos** os pedidos). Mas a partir do momento em que `pack_orders`
começa a selecionar pedidos no estado parcial, a "demanda residual" muda — e
o melhor próximo aisle não é necessariamente o de maior score global.

Exemplo: se os 3 primeiros aisles já cobrem todos os itens dos pedidos
"top-K", o quarto aisle ideal não é o de maior `useful_score` global, mas o que
melhor cobre a *demanda ainda não atendida*.

**Proposta.** Recalcular o score após cada inserção (ou a cada bloco de N
inserções) usando `score_aisles(aisles, score_mode, residual_demand)`, onde
`residual_demand = total_demand - (demanda já coberta por aisles ativos)`.

Pseudocódigo:

```python
remaining = set(range(n_aisles))
chosen = []
inventory = {}
residual = dict(total_demand)
while remaining and not stop_condition(...):
    scores = score_aisles(aisles, mode, residual, restrict=remaining)
    next_aisle = max(remaining, key=scores.__getitem__)
    chosen.append(next_aisle)
    remaining.discard(next_aisle)
    update(inventory, aisles[next_aisle])
    update_residual(residual, aisles[next_aisle])
    # ... pack_orders + tracking de melhor
```

**Custo.** O re-ranqueamento custa `O(nAisles × nItems)` por passo — para
nAisles=200, nItems=1000 isso é 200k ops, ~1ms. Em uma execução com 50 passos,
fica em 50ms. Aceitável.

**Esforço.** Médio — exige novo helper `score_aisles_residual`, ou adicionar
parâmetro `restrict_to: set[int] | None` em `score_aisles`/`rank_aisles`.

---

## #5 — Tie-break por complementaridade

**Problema.** Quando dois aisles têm o mesmo score (ou scores muito próximos),
o ranqueamento atual desempata pelo `aisle_idx` natural. Ideal seria preferir
o aisle que **menos sobrepõe** com o já selecionado, maximizando cobertura
incremental.

**Proposta.** Para o k-ésimo aisle a ser inserido, dentre candidatos com
score ~empatado, preferir o de maior **complementaridade**:

```
complement(a, S) = sum(qty for item, qty in aisles[a].items()
                       if item in residual_demand
                       and item not in covered_items_in(S))
```

Pode ser usado como score secundário (um epsilon-ball ao redor do score
máximo) ou como score primário com escala híbrida.

Já existe `algorithms/utils/similarity.py` que pode ser usado para definir
distância entre aisles.

**Esforço.** Pequeno-médio — convive com #4 (mesma estrutura de loop com
estado dinâmico).

---

## #6 — `pack_orders` priorizando pedidos recém-desbloqueados

**Problema.** `pack_orders` recebe uma sequência *fixa* de pedidos
(`order_sequence`) e tenta cada um em ordem. Quando um aisle novo é adicionado
e desbloqueia pedidos antes inviáveis, esses pedidos são tentados na mesma
ordem global — não são priorizados pela "novidade".

**Proposta.** Ao invés de só re-rodar `pack_orders` do zero a cada k, manter
o estado parcial e processar primeiro os pedidos **que ficaram inviáveis em
k-1** e podem ter virado viáveis em k. Isso poda o trabalho em iterações
posteriores (early-out se nenhum pedido foi desbloqueado) e tende a empacotar
melhor quando ub é restritivo.

Variante mais simples: re-ordenar `order_sequence` a cada k para colocar na
frente os pedidos que demandam pelo menos um item do aisle recém-adicionado.

**Esforço.** Médio — exige refactor de `pack_orders` para aceitar estado
incremental, OU é trivial se ficar só na re-ordenação per-k.

**Cuidado.** Esse hot-path é chamado nAisles × por solve — qualquer aumento
no custo de `pack_orders` é multiplicado por ~200. Medir antes de mergir.

---

## #7 — ILP set-cover substituindo `multi_greedy_aisle_select` no prune

**Problema.** `multi_greedy_aisle_select` é uma heurística gulosa para o
problema de Set Multicover. Para a maioria das instâncias do dataset,
o ótimo dessa subproblema é alcançável em <1s pelo ILP (CP-SAT).

**Proposta.** Já existe `algorithms/utils/ilp_aisle_select.solve_min_aisle_cover`
implementado. Substituir o `prune_fn` em duas situações:

1. Pós-construção (após o loop principal), quando há tempo de orçamento sobrando
2. Dentro do loop quando `prune` está ativo (com time_limit pequeno, ex: 0.5s)

Esquema sugerido:

```python
prune_mode in {"simple", "multi", "ilp"}
```

Quando `prune="ilp"`, usar `solve_min_aisle_cover(demand, aisles,
time_limit_seconds=cfg["time_limit_per_prune"])` com fallback para `multi`
em caso de timeout.

**Custo.** O ILP de cobertura tipicamente roda em 50-300ms para nAisles<200,
nItems<1000. Em afe_* com `time_limit_per_k=2s`, somar 0.3s por k não é
proibitivo — mas multiplicado por ~200 k's é. Recomendo ativar o ILP só
**após** o loop principal (single shot), em conjunto com o LS.

**Esforço.** Pequeno — função já existe; basta incorporar em
`_local_search.py` (ou em ambos os solvers como prune_mode adicional).

---

## #8 — Aproveitar o orçamento de tempo restante

**Problema.** `AisleFirstHeuristic` retorna em ~0.5–10s mesmo com orçamento
de 60–600s, deixando o resto do tempo inativo. Já há infra para deadline
em `_local_search.py` (`ls_config["time_limit"]`), mas o solver ainda não
**aproveita** esse tempo restante de forma agressiva.

**Proposta.** Ao final do `solve`, com `remaining = total_budget - elapsed`,
reinvestir esse tempo em:

1. **Multi-start adicional** (sugestão #3): rodar `min(remaining // 1s, 20)`
   starts extras com seeds e configs aleatórias, manter o melhor.
2. **LS profundo**: chamar `apply_local_search` com `time_limit=remaining,
   neighbor_cap=200, max_iterations=2000`.
3. **ILP refinement** (sugestão #7): aplicar set-cover ILP no resultado final
   com time_limit=remaining.

A combinação `multi-start curtos + LS pesado no melhor` costuma ser o melhor
custo-benefício.

**Implementação.** Aceitar param `time_limit` no construtor da heurística e
medir `time.monotonic()` ao começar o solve. Atualmente o orçamento total é
gerenciado externamente (no main.py via SIGALRM); precisa ser exposto para
a heurística para ela auto-distribuir.

**Esforço.** Médio — exige passar o time_limit como param e bookkeeping de
elapsed dentro de solve. Casa naturalmente com #3 e #7.

---

## Ordem de implementação sugerida

1. ✅ #1 — loop prune-aware
2. ✅ #2 — local search pós-construção
3. **#3 — Multi-start** (maior bang/buck, baixo risco, casa com #2 já implementado)
4. **#7 — ILP set-cover pós-construção** (rápido de adicionar, ortogonal aos outros)
5. **#8 — Reinvestir tempo restante** (precisa de #3 e #7 para ter o que rodar)
6. **#4 — Re-ranqueamento dinâmico** (mais ambicioso, pode quebrar baselines — testar com cuidado)
7. **#5 — Tie-break por complementaridade** (refinamento de #4)
8. **#6 — `pack_orders` priorizando recém-desbloqueados** (otimização de hot-path; medir antes)
