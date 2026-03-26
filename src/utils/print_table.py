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

    col_widths = [max(len(str(row[c])) for row in data) for c in range(len(data[0]))]

    for row_idx, row in enumerate(data):
        rendered = " | ".join(
            str(cell).ljust(col_widths[col_idx]) for col_idx, cell in enumerate(row)
        )
        print(rendered)
        if row_idx == 0:
            print("-+-".join("-" * width for width in col_widths))
