Boxplot de gap_mean por instância, comparando apenas os algoritmos da fronteira de Pareto para ver estabilidade e variabilidade em cada problema.

Boxplot de exec_time_mean por instância, para identificar quais algoritmos são mais consistentes e quais têm maior dispersão de tempo.

Boxplot conjunto por dataset (a, b, x), separando gap_mean e exec_time_mean, para ver se a fronteira muda de comportamento entre os grupos.

Gráfico de barras do ranking médio por instância, usando uma métrica composta como gap_mean e exec_time_mean, para destacar os melhores algoritmos de forma agregada.

Gráfico de dispersão gap_mean vs exec_time_mean com um ponto por instância e cor por algoritmo, para visualizar em quais instâncias cada solução domina.

Heatmap de desempenho algoritmo × instância, com cores para gap_mean ou objetivo, para revelar padrões de sucesso/fraqueza por cenário.

Tabela resumo por algoritmo com média, mediana, desvio padrão, mínimo e máximo de gap_mean, exec_time_mean e objective_mean, para comparar robustez.

Tabela por dataset com médias e desvios dos algoritmos da fronteira, permitindo comparar performance em a, b e x separadamente.

Gráfico de linhas por instância ordenada, mostrando a evolução de gap_mean e exec_time_mean para cada algoritmo, útil para perceber tendências e cruzamentos.

Análise de dominância intra-fronteira: contar em quantas instâncias cada algoritmo é o melhor em gap, em tempo ou em uma combinação ponderada.

Gráfico de violino ou boxen plot por dataset, para enxergar melhor a distribuição das métricas e identificar assimetrias e outliers.

Scatter com tamanho do ponto proporcional ao tamanho da instância ou à diferença para o melhor baseline, para destacar onde a solução ganha mais.

Tabela de “vitórias por instância”, listando o melhor algoritmo da fronteira em cada dataset/instância para gap, tempo e objetivo.

Gráfico de correlação entre gap_mean, exec_time_mean, items_mean e aisles_mean, para entender quais características operacionais explicam o desempenho.

Análise de sensibilidade por classe de instância, agrupando instâncias similares e comparando o comportamento dos algoritmos da fronteira em cada grupo.