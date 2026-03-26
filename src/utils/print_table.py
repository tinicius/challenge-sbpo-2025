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

    if not data:
        return

    n_cols = max((len(row) for row in data), default=0)
    if n_cols == 0:
        return
    col_widths = [
        max(len(str(row[c])) if c < len(row) else 0 for row in data) for c in range(n_cols)
    ]

    for row_idx, row in enumerate(data):
        rendered = " | ".join(
            (str(row[col_idx]) if col_idx < len(row) else "").ljust(col_widths[col_idx])
            for col_idx in range(n_cols)
        )
        print(rendered)
        if row_idx == 0:
            print("-+-".join("-" * width for width in col_widths))
