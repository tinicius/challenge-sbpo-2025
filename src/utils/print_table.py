def print_table(nItems, nOrders, orders):
    data = []

    header = ["Pedido"]

    for i in range(nItems):
        header.append(f"Item {i}")

    data.append(header)

    for p in range(nOrders):
        row = [f"Pedido {p}"]

        for i in range(nItems):
            row.append(orders[p].get(i, 0))

        data.append(row)

    print(tabulate(data, headers="firstrow", tablefmt="grid"))
