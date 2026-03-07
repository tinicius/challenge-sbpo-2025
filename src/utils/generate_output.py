
def generate_output(output_file, n_orders, n_aisles, orders: list, aisles: list):
    with open(output_file, 'w') as file:
        file.write(f"{str(n_orders)}\n")

        for o in orders:
            file.write(f"{str(o)}\n")

        file.write(f"{str(n_aisles)}\n")

        for a in aisles:
            file.write(f"{str(a)}\n")
