def read_input(filename: str):
    with open(filename, 'r') as file:
        data = file.read()

    lines = data.splitlines()

    nOrders = int(lines[0].split(' ')[0])
    nItems = int(lines[0].split(' ')[1])
    nAisles = int(lines[0].split(' ')[2])

    orders = []

    for o_i in range(nOrders):
        order_line = lines[o_i + 1].split(' ')

        n_order_items = int(order_line[0])

        details = {}

        for k in range(n_order_items):
            item_idx = int(order_line[2 * k + 1])
            item_quantity = int(order_line[2 * k + 2])

            details[item_idx] = item_quantity

        orders.append(details)

    aisles = []

    for o_i in range(nAisles):
        aisle_line = lines[o_i + 1 + nOrders].split(' ')

        n_aisle_items = int(aisle_line[0])

        details = {}

        for k in range(n_aisle_items):
            item_idx = int(aisle_line[2 * k + 1])
            item_quantity = int(aisle_line[2 * k + 2])

            details[item_idx] = item_quantity

        aisles.append(details)

    wave_size_line = lines[nOrders + nAisles + 1].split(' ')

    lb = int(wave_size_line[0])
    ub = int(wave_size_line[1])

    return nOrders, nItems, nAisles, orders, aisles, lb, ub
