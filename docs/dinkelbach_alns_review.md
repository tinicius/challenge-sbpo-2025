# Dinkelbach-ALNS — Avaliação em datasets B e X

Configuração: `time_limit=60s`, 1 repetição, seed=42, 12 workers paralelos
(config completa em `configs/dinkelbach_alns_b.yaml` e `_x.yaml`).
Resultados em `results/b_20260428_110940/` e `results/x_20260428_111211/`.

## Resumo

| dataset | n  | matched | ≤1% | ≤5% | mean gap | worst gap | timed out |
|---------|----|---------|-----|-----|----------|-----------|-----------|
| b       | 15 | 3       | 3   | 8   | 5.51 %   | 16.02 %   | 6         |
| x       | 15 | 1       | 5   | 9   | 5.20 %   | 21.75 %   | 6         |

Tabela completa por instância: ver saída de `python analyze_dalns.py`.

## Padrão observado: três regimes de instância

1. **Muito grande (≥28k orders)** — `b_0011–0014`, `x_0009–0014`.
   Gap **0.86–2.11 %**, todas com TIMEOUT. O construtivo `aisle_first_sweep`
   resolve quase tudo; ALNS só polish. Tempo extra ajudaria pouco.

2. **Médio-grande (8–15k orders, ~400 aisles)** — `b_0008,0015`, `x_0007,0008`.
   Gap **5.9–16 %**. Várias **não** atingiram timeout (`b_0015` em 54s,
   `x_0007/0008` em ~59s) — convergiram para ótimos locais ruins.

3. **Pequeno-médio (1.8–6k orders)** — `x_0001/0002/0004`, `b_0009`.
   Gap **5.9–21.75 %**. **Pior caso geral** (`x_0002`: 21.75 %).
   Mesmo cenário da classe 2 mas com mais tempo "desperdiçado".

## Diagnóstico

Inspeção das instâncias com pior gap (`x_0002`, `b_0015`, `b_0009`, `x_0001`):

| inst   | n_orders | avg u/order | avg u/aisle | best ratio | aisles na sol |
|--------|----------|-------------|-------------|------------|---------------|
| x_0002 | 2942     | 1.4         | 516         | 217.67     | 3             |
| b_0015 | 11541    | 1.3         | 240         | 166.14     | 17            |
| b_0009 | 5581     | 1.3         | 119         | 82.50      | 16            |
| x_0001 | 1949     | 3.8         | 771         | 70.85      | 16            |

Característica comum: **orders pequenos (~1 item, 1–4 unidades) + aisles
largos**. A escolha de **quais aisles** entrar é o que define o ótimo.
O ALNS atualmente:

- **Destrói**: 5 operadores. Quatro deles (`random_order`, `worst_order`,
  `shaw`, `density_outlier`) operam em *orders* — só `aisle_based` toca em
  aisles, e remove aleatoriamente 1..|A|/3 (e todos os orders dependentes).
  Para soluções com 3 aisles isso equivale a recomeçar.
- **Repara**: 4 operadores. Todos enxergam aisles como "consequência" de
  adicionar orders. **Nenhum operador troca um aisle por outro**.

Resultado: uma vez fixado o conjunto de aisles pelo construtivo, o ALNS
explora `C(orders, k)` mas quase nunca o espaço de aisles. Para `x_0002`
(168 aisles, ótimo com 3) o espaço de combinações de aisles é
`C(168,3) ≈ 768k` — totalmente inexplorado.

Sinais corroborando o diagnóstico:

- `x_0002` selecionou exatamente 3 aisles (mesmo número que o ótimo) mas
  com 511 unidades em vez de ~653 → **escolheu os aisles errados**.
- `b_0015` finalizou em 54s (não foi timeout) com `r=139.5` vs `r*=166.1`
  → SA esfriou, sem incentivo a deslocar aisles.
- `prune_redundant_aisles` é single-pass e ordenado por contribuição
  ascendente — pode deixar pares de aisles redundantes que só ficam
  redundantes após remoção mútua.

## Direções de melhoria (priorizadas)

### Alta prioridade — atacam o pior gap

1. **Operador destroy `d_aisle_swap`**: remover 1..k aisles e re-construir
   coverage com aisles **diferentes** (sample dirigido por
   `item_to_aisles`). Compara com o conjunto removido e mantém só se melhora
   ratio. Resolve diretamente x_0002 e similares.

2. **Operador destroy `d_aisle_compress`**: para soluções com `k > k_min`,
   tentar reduzir `k` em 1 forçando os orders a se acomodarem em `k-1`
   aisles via re-cobertura. Diretamente alvo do objetivo (que penaliza
   aisles).

3. **Restart com construtivo diverso ao estagnar**: detectar
   `iters_since_improve > N` (e.g. 500), reiniciar `current_sol` a partir
   de um construtivo **com aisles iniciais diferentes** dos já vistos —
   não basta resetar T (já é feito parcialmente em `update_lam`).

4. **Construtivo `aisle_first_fixed_k`**: variante de `aisle_first_sweep`
   que testa **explicitamente** k pequenos (3, 5, 7, 10) com ranking
   alternativo (ex.: aisles com maior cobertura de demanda dos orders mais
   densos). O sweep atual aborta cedo via `instance.ub / k <= best_obj` e
   pode pular k baixos quando uma instância tem ratio máximo lá.

### Média prioridade — robustez

5. **`prune_redundant_aisles` iterativo**: reescrever para iterar até
   ponto-fixo (após remover um aisle, reavaliar os outros). Atual é greedy
   single-pass.

6. **CP-SAT LB com k adaptativo**: `k_values=[5,10,20]` é minúsculo para
   soluções com 1k+ orders. Escalonar
   `k_eff = max(20, int(0.05 * len(sol.orders)))` e adicionar um round
   final com `k = ceil(0.1 * n_selected)`. Para instâncias médias o LB
   atualmente termina em poucos segundos sem mover.

7. **Multi-start com diversidade de aisle, não só de order**: hoje os
   candidatos são deduplicados por `frozenset(orders)`. Adicionar
   deduplicação por `frozenset(aisles)` e forçar starts com aisle-sets
   distintos. Atualmente as 4 sementes podem partir de prefixos do mesmo
   ranking de aisles.

### Baixa prioridade — perfilagem / micro-tuning

8. **Profile do `_candidate_orders`**: cap=400 pode ser estreito demais
   para B0011 (45k orders). Validar via experimento se `cap=1000` muda
   gap nas instâncias que NÃO estouraram tempo.

9. **Reset completo de T**: após `update_lam` o código faz
   `T = max(T, T_start * 0.5)` (meio-restart). Em casos com 500+ iters
   sem improve, fazer `T = T_start` total.

10. **Eliminar `time_check_interval` desnecessário**: hoje é 1 (checa todo
    iter). Em instâncias pequenas onde cada iter é <1ms, isso não dói. Em
    instâncias grandes onde cada iter pode ser >1s, está ok. Provavelmente
    não é gargalo, mas vale confirmar com profiling.

## Próximos passos sugeridos

1. Implementar (1) e (2) — operadores aisle-swap/compress. Esperado:
   maior impacto nas instâncias com 5–20 aisles na solução.
2. Adicionar (3) — restart por estagnação. Esperado: melhora as instâncias
   classe 3 que finalizam antes do timeout.
3. Re-rodar B+X (mesma config, 60s) e comparar gap mean/worst contra esta
   baseline (B: 5.51 %/16 %, X: 5.20 %/21.75 %).

Critério de sucesso razoável após (1)+(2)+(3):

- Reduzir worst gap em B+X para ≤ 10 %.
- Reduzir mean gap em B+X para ≤ 3 %.
- Aumentar matched count para ≥ 8/30.
