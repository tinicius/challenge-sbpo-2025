# Contexto do Problema: Desafio de Otimização SBPO 2025 / Mercado Livre
**Título:** O Problema da Seleção de Pedidos Ótima
**Data:** Janeiro de 2025

## 1. Visão Geral do Problema
O objetivo deste desafio é otimizar o processo de coleta (picking) de itens em um armazém do Mercado Livre. Para maximizar a produtividade, os pedidos individuais são agrupados em subconjuntos chamados **waves**. 
O problema consiste em encontrar uma *wave* de pedidos ótima, selecionando um subconjunto de pedidos de um *backlog* e os corredores correspondentes onde os itens serão coletados, de forma a **maximizar a quantidade de itens coletados por corredor visitado**, respeitando limites de capacidade da wave e a disponibilidade de estoque nos corredores.

## 2. Definições Básicas
* **Item:** Produto que pode ser solicitado.
* **Pedido:** Lista de itens solicitados por um cliente, com suas respectivas quantidades.
* **Coleta:** Processo de recuperar produtos do armazém.
* **Backlog:** Conjunto de pedidos cujos itens ainda não foram coletados.
* **Wave:** Subconjunto de pedidos do *backlog* selecionados para processamento conjunto.
* **Corredor:** Espaço de circulação entre estantes onde os itens estão armazenados.

## 3. Formulação Matemática

### 3.1 Conjuntos
* $\mathcal{O}$: Conjunto de pedidos no *backlog*.
* $\mathcal{I}_o$: Subconjunto de itens solicitados pelo pedido $o \in \mathcal{O}$.
* $\mathcal{I}$: Conjunto total de itens, onde $\mathcal{I} = \bigcup_{o \in \mathcal{O}} \mathcal{I}_o$.
* $\mathcal{A}_i$: Subconjunto de corredores contendo pelo menos uma unidade do item $i$.
* $\mathcal{A}$: Conjunto total de corredores, onde $\mathcal{A} = \bigcup_{i \in \mathcal{I}} \mathcal{A}_i$.

### 3.2 Constantes e Parâmetros
* $u_{oi}$: Número de unidades do item $i \in \mathcal{I}$ solicitado pelo pedido $o \in \mathcal{O}$.
* $u_{ai}$: Número de unidades do item $i \in \mathcal{I}$ disponíveis no corredor $a \in \mathcal{A}$.
* $LB$: Limite inferior (Lower Bound) do tamanho da wave (total de unidades).
* $UB$: Limite superior (Upper Bound) do tamanho da wave (total de unidades).

### 3.3 Variáveis de Decisão (Implícitas)
Para fins de modelagem algorítmica, o problema busca definir:
* $\mathcal{O}' \subset \mathcal{O}$: O subconjunto de pedidos selecionados para compor a *wave*.
* $\mathcal{A}' \subset \mathcal{A}$: O subconjunto de corredores selecionados para visitar.

### 3.4 Função Objetivo
Maximizar o número de itens coletados por corredor visitado:
$$ \max \frac{\sum_{o \in \mathcal{O}'} \sum_{i \in \mathcal{I}_o} u_{oi}}{|\mathcal{A}'|} $$

### 3.5 Restrições
1. **Tamanho Mínimo da Wave:**
   $$ \sum_{o \in \mathcal{O}'} \sum_{i \in \mathcal{I}_o} u_{oi} \ge LB $$

2. **Tamanho Máximo da Wave:**
   $$ \sum_{o \in \mathcal{O}'} \sum_{i \in \mathcal{I}_o} u_{oi} \le UB $$

3. **Garantia de Estoque:** Os corredores selecionados devem ter estoque suficiente para todos os itens demandados pelos pedidos da wave:
   $$ \sum_{o \in \mathcal{O}'} u_{oi} \le \sum_{a \in \mathcal{A}'} u_{ai}, \quad \forall i \in \mathcal{I}_o \text{ com } o \in \mathcal{O}' $$

---

## 4. Exemplo/Instância de Teste (Unit Test Data)

### Parâmetros da Instância
* $LB = 5$
* $UB = 12$
* 5 pedidos ($\mathcal{O}$), 5 itens ($\mathcal{I}$), 5 corredores ($\mathcal{A}$)

### Matriz de Pedidos x Itens ($u_{oi}$)
| Pedido | Item 0 | Item 1 | Item 2 | Item 3 | Item 4 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | 3 | 0 | 1 | 0 | 0 |
| **1** | 0 | 1 | 0 | 1 | 0 |
| **2** | 0 | 0 | 1 | 0 | 2 |
| **3** | 1 | 0 | 2 | 1 | 1 |
| **4** | 0 | 1 | 0 | 0 | 0 |

### Matriz de Corredores x Itens ($u_{ai}$)
| Corredor | Item 0 | Item 1 | Item 2 | Item 3 | Item 4 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | 2 | 1 | 1 | 0 | 1 |
| **1** | 2 | 1 | 2 | 0 | 1 |
| **2** | 0 | 2 | 0 | 1 | 2 |
| **3** | 2 | 1 | 0 | 1 | 1 |
| **4** | 0 | 1 | 2 | 1 | 2 |

### Soluções Possíveis (Para validação de código)

1. **Solução Viável 1:**
   * Pedidos: `0, 4`
   * Corredores: `0, 1`
   * Total de unidades: 5
   * Número de corredores: 2
   * Valor Objetivo: $5 / 2 = 2.5$

2. **Solução Viável 2:**
   * Pedidos: `0, 2, 3`
   * Corredores: `1, 3, 4`
   * Total de unidades: 12
   * Número de corredores: 3
   * Valor Objetivo: $12 / 3 = 4.0$

3. **Solução Ótima:**
   * Pedidos: `0, 1, 2, 4`
   * Corredores: `1, 3`
   * Total de unidades: 10
   * Número de corredores: 2
   * Valor Objetivo: $10 / 2 = 5.0$

---

## 5. Notas para o Desenvolvedor / Assistente de IA (Implementation Hints)

Ao desenvolver algoritmos para este problema, a IA deve considerar o seguinte:

1. **Natureza do Problema:** É um problema de otimização combinatória com uma função objetivo **fracionária** (Non-linear fractional programming). Trata-se de um misto de problema de seleção (Knapsack multidimensional com limites inferior e superior) e cobertura de conjuntos (Set Covering para os corredores).
