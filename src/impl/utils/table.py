def write_dict_table_to_file(
    dict_1: dict, dict_2: dict, filename: str, missing_val="-"
):
    # 1. Get the union of all keys from both dictionaries and sort them
    all_keys = sorted(set(dict_1.keys()) | set(dict_2.keys()))

    rows = []
    for key in all_keys:
        # String representations for display
        val_1_str = str(dict_1.get(key, missing_val))
        val_2_str = str(dict_2.get(key, missing_val))

        # 2. Evaluate dict_1 >= dict_2 safely
        if key in dict_1 and key in dict_2:
            try:
                # Compare the actual values (not the strings)
                cmp_val = str(dict_1[key] >= dict_2[key])
            except TypeError:
                # Fallback if values are incomparable
                cmp_val = "Type Error"
        else:
            # If the key isn't in both dictionaries, comparison is not applicable
            cmp_val = missing_val

        rows.append((str(key), val_1_str, val_2_str, cmp_val))

    # 3. Determine the maximum width for each column
    key_width = max([len("key")] + [len(r[0]) for r in rows])
    d1_width = max([len("dict_1")] + [len(r[1]) for r in rows])
    d2_width = max([len("dict_2")] + [len(r[2]) for r in rows])
    cmp_width = max([len("dict_1 >= dict_2")] + [len(r[3]) for r in rows])

    # 4. Open the file in write mode
    with open(filename, "w", encoding="utf-8") as file:

        # 5. Build and write the header and separator
        header = (
            f"{'key'.ljust(key_width)} | {'dict_1'.ljust(d1_width)} | "
            f"{'dict_2'.ljust(d2_width)} | {'dict_1 >= dict_2'.ljust(cmp_width)}"
        )

        separator = (
            f"{'-' * key_width}-+-{'-' * d1_width}-+-{'-' * d2_width}-+-"
            f"{'-' * cmp_width}"
        )

        file.write(header + "\n")
        file.write(separator + "\n")

        # 6. Build and write the rows
        for r_key, r_v1, r_v2, r_cmp in rows:
            row_str = (
                f"{r_key.ljust(key_width)} | {r_v1.ljust(d1_width)} | "
                f"{r_v2.ljust(d2_width)} | {r_cmp.ljust(cmp_width)}"
            )
            file.write(row_str + "\n")


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    d1 = {"apple": 10, "banana": 12, "cherry": 7, "date": 9, "fig": 20}
    d2 = {"apple": 8, "banana": 12, "cherry": 15, "elderberry": 3}

    # This will create a file named "comparison_table.txt" in your current directory
    write_dict_table_to_file(d1, d2, filename="comparison_table.txt", missing_val="N/A")
