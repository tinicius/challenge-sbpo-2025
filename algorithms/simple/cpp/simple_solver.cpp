// Simple heuristic solver for the SBPO 2025 wave order-picking problem.
//
// CLI:
//   simple_solver <instance_path>
//       --greedy=simple|multi
//       [--order=none|asc|desc|similar|diff]
//       [--first-order=none|smaller|bigger]
//       [--similarity-weighted=0|1]
//       [--seed=N]
//
// Output (stdout): JSON lines compatible with Python wrappers.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

enum class OrderMode { None, Asc, Desc, Similar, Diff };
enum class GreedyMode { Simple, Multi };
enum class FirstOrderMode { None, Smaller, Bigger };

struct Options {
    std::string instance_path;
    GreedyMode greedy = GreedyMode::Simple;
    bool greedy_set = false;
    OrderMode order = OrderMode::None;
    FirstOrderMode first_order = FirstOrderMode::None;
    bool similarity_weighted = false;
    bool seed_set = false;
    uint64_t seed = 0;
};

struct Problem {
    int n_orders = 0;
    int n_items = 0;
    int n_aisles = 0;
    int lb = 0;
    int ub = 0;
    std::vector<int> order_off;
    std::vector<int> order_item;
    std::vector<int> order_qty;
    std::vector<int> order_size;
    std::vector<int> aisle_off;
    std::vector<int> aisle_item;
    std::vector<int> aisle_qty;
    std::vector<int> stock_total;
};

struct Solution {
    std::vector<int> selected_orders;
    std::vector<int> visited_aisles;
    double objective = 0.0;
};

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "simple_solver: " << msg << "\n";
    std::exit(2);
}

bool parse_bool(const std::string& s) {
    if (s == "1" || s == "true" || s == "True") return true;
    if (s == "0" || s == "false" || s == "False") return false;
    die("invalid boolean: " + s);
}

Options parse_args(int argc, char** argv) {
    Options opt;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) != 0) {
            if (!opt.instance_path.empty()) die("multiple positional args");
            opt.instance_path = a;
            continue;
        }
        auto eq = a.find('=');
        std::string key = a.substr(2, eq == std::string::npos ? std::string::npos : eq - 2);
        std::string val = eq == std::string::npos ? "" : a.substr(eq + 1);

        if (key == "greedy") {
            if (val == "simple") opt.greedy = GreedyMode::Simple;
            else if (val == "multi") opt.greedy = GreedyMode::Multi;
            else die("invalid greedy: " + val);
            opt.greedy_set = true;
        } else if (key == "order") {
            if (val.empty() || val == "none") opt.order = OrderMode::None;
            else if (val == "asc") opt.order = OrderMode::Asc;
            else if (val == "desc") opt.order = OrderMode::Desc;
            else if (val == "similar") opt.order = OrderMode::Similar;
            else if (val == "diff") opt.order = OrderMode::Diff;
            else die("invalid order: " + val);
        } else if (key == "first-order") {
            if (val.empty() || val == "none") opt.first_order = FirstOrderMode::None;
            else if (val == "smaller") opt.first_order = FirstOrderMode::Smaller;
            else if (val == "bigger") opt.first_order = FirstOrderMode::Bigger;
            else die("invalid first-order: " + val);
        } else if (key == "similarity-weighted") {
            opt.similarity_weighted = parse_bool(val);
        } else if (key == "seed") {
            opt.seed = static_cast<uint64_t>(std::stoll(val));
            opt.seed_set = true;
        } else {
            die("unknown option: " + key);
        }
    }

    if (opt.instance_path.empty()) die("missing instance path");
    if (!opt.greedy_set) die("missing required option --greedy");

    return opt;
}

Problem load_instance(const std::string& path) {
    std::ifstream f(path);
    if (!f) die("cannot open instance: " + path);

    Problem p;
    f >> p.n_orders >> p.n_items >> p.n_aisles;
    if (!f) die("malformed header");

    p.order_off.assign(p.n_orders + 1, 0);
    p.order_size.assign(p.n_orders, 0);
    p.order_item.reserve(p.n_orders * 4);
    p.order_qty.reserve(p.n_orders * 4);
    for (int o = 0; o < p.n_orders; ++o) {
        int k;
        f >> k;
        p.order_off[o + 1] = p.order_off[o] + k;
        int size = 0;
        for (int j = 0; j < k; ++j) {
            int it, q;
            f >> it >> q;
            p.order_item.push_back(it);
            p.order_qty.push_back(q);
            size += q;
        }
        p.order_size[o] = size;
    }

    p.aisle_off.assign(p.n_aisles + 1, 0);
    p.stock_total.assign(p.n_items, 0);
    p.aisle_item.reserve(p.n_aisles * 4);
    p.aisle_qty.reserve(p.n_aisles * 4);
    for (int a = 0; a < p.n_aisles; ++a) {
        int k;
        f >> k;
        p.aisle_off[a + 1] = p.aisle_off[a] + k;
        for (int j = 0; j < k; ++j) {
            int it, q;
            f >> it >> q;
            p.aisle_item.push_back(it);
            p.aisle_qty.push_back(q);
            p.stock_total[it] += q;
        }
    }

    f >> p.lb >> p.ub;
    if (!f) die("malformed lb/ub");

    return p;
}

double order_similarity(const Problem& p, int a, int b, bool weighted) {
    const int a0 = p.order_off[a], a1 = p.order_off[a + 1];
    const int b0 = p.order_off[b], b1 = p.order_off[b + 1];

    if (!weighted) {
        int i = a0;
        int j = b0;
        int inter = 0;
        int uni = 0;
        while (i < a1 && j < b1) {
            int ita = p.order_item[i];
            int itb = p.order_item[j];
            if (ita == itb) {
                ++inter;
                ++uni;
                ++i;
                ++j;
            } else if (ita < itb) {
                ++uni;
                ++i;
            } else {
                ++uni;
                ++j;
            }
        }
        uni += (a1 - i) + (b1 - j);
        if (uni == 0) return 0.0;
        return static_cast<double>(inter) / static_cast<double>(uni);
    }

    int i = a0;
    int j = b0;
    long long num = 0;
    long long den = 0;
    while (i < a1 && j < b1) {
        int ita = p.order_item[i];
        int itb = p.order_item[j];
        if (ita == itb) {
            int qa = p.order_qty[i];
            int qb = p.order_qty[j];
            num += std::min(qa, qb);
            den += std::max(qa, qb);
            ++i;
            ++j;
        } else if (ita < itb) {
            den += p.order_qty[i];
            ++i;
        } else {
            den += p.order_qty[j];
            ++j;
        }
    }
    while (i < a1) {
        den += p.order_qty[i];
        ++i;
    }
    while (j < b1) {
        den += p.order_qty[j];
        ++j;
    }
    if (den == 0) return 0.0;
    return static_cast<double>(num) / static_cast<double>(den);
}

int pick_first_order_index(const Options& opt, const Problem& p, std::mt19937& rng) {
    if (p.n_orders <= 0) return -1;

    if (opt.first_order == FirstOrderMode::Bigger) {
        return static_cast<int>(
            std::max_element(p.order_size.begin(), p.order_size.end()) - p.order_size.begin()
        );
    }
    if (opt.first_order == FirstOrderMode::Smaller) {
        return static_cast<int>(
            std::min_element(p.order_size.begin(), p.order_size.end()) - p.order_size.begin()
        );
    }

    std::vector<int> indices(p.n_orders);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    return indices.front();
}

std::vector<int> build_traversal(const Options& opt, const Problem& p, std::mt19937& rng) {
    std::vector<int> indices(p.n_orders);
    std::iota(indices.begin(), indices.end(), 0);

    if (opt.order == OrderMode::None) {
        std::shuffle(indices.begin(), indices.end(), rng);
        return indices;
    }

    if (opt.order == OrderMode::Asc || opt.order == OrderMode::Desc) {
        std::stable_sort(indices.begin(), indices.end(), [&](int x, int y) {
            if (opt.order == OrderMode::Asc) return p.order_size[x] < p.order_size[y];
            return p.order_size[x] > p.order_size[y];
        });
        return indices;
    }

    int reference = pick_first_order_index(opt, p, rng);
    std::vector<double> sims(p.n_orders, 0.0);
    for (int i = 0; i < p.n_orders; ++i) {
        sims[i] = order_similarity(p, reference, i, opt.similarity_weighted);
    }

    std::stable_sort(indices.begin(), indices.end(), [&](int x, int y) {
        if (opt.order == OrderMode::Similar) return sims[x] > sims[y];
        return sims[x] < sims[y];
    });
    return indices;
}

std::vector<int> simple_greedy_aisles(const Problem& p, std::vector<int> demand) {
    std::vector<int> order(p.n_aisles);
    std::vector<int> score(p.n_aisles, 0);
    for (int a = 0; a < p.n_aisles; ++a) {
        int s = 0;
        for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j) {
            int d = demand[p.aisle_item[j]];
            if (d > 0) s += std::min(d, p.aisle_qty[j]);
        }
        score[a] = s;
        order[a] = a;
    }
    std::sort(order.begin(), order.end(),
              [&](int x, int y) { return score[x] > score[y]; });

    int remaining = 0;
    for (int v : demand) {
        if (v > 0) remaining += v;
    }

    std::vector<int> selected;
    for (int a : order) {
        if (remaining == 0) break;

        int real = 0;
        for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j) {
            int d = demand[p.aisle_item[j]];
            if (d > 0) real += std::min(d, p.aisle_qty[j]);
        }
        if (real == 0) continue;

        selected.push_back(a);
        for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j) {
            int it = p.aisle_item[j];
            int d = demand[it];
            if (d <= 0) continue;
            int take = std::min(d, p.aisle_qty[j]);
            demand[it] = d - take;
            remaining -= take;
        }
    }

    return selected;
}

std::vector<int> multi_greedy_aisles(const Problem& p, std::vector<int> demand) {
    std::vector<char> used(p.n_aisles, 0);
    std::vector<int> selected;

    int remaining = 0;
    for (int v : demand) {
        if (v > 0) remaining += v;
    }

    while (remaining > 0) {
        int best = -1;
        int best_score = 0;
        for (int a = 0; a < p.n_aisles; ++a) {
            if (used[a]) continue;
            int s = 0;
            for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j) {
                int d = demand[p.aisle_item[j]];
                if (d > 0) s += std::min(d, p.aisle_qty[j]);
            }
            if (s > best_score) {
                best = a;
                best_score = s;
            }
        }

        if (best_score == 0) break;

        used[best] = 1;
        selected.push_back(best);
        for (int j = p.aisle_off[best]; j < p.aisle_off[best + 1]; ++j) {
            int it = p.aisle_item[j];
            int d = demand[it];
            if (d <= 0) continue;
            int take = std::min(d, p.aisle_qty[j]);
            demand[it] = d - take;
            remaining -= take;
        }
    }

    return selected;
}

std::vector<int> select_aisles(const Options& opt, const Problem& p, const std::vector<int>& demand) {
    if (opt.greedy == GreedyMode::Multi) return multi_greedy_aisles(p, demand);
    return simple_greedy_aisles(p, demand);
}

bool solve_simple(const Options& opt, const Problem& p, std::mt19937& rng, Solution& out) {
    if (p.n_orders == 0 || p.n_aisles == 0) return false;

    std::vector<int> indices = build_traversal(opt, p, rng);
    std::vector<int> stock_rem = p.stock_total;
    std::vector<int> demand(p.n_items, 0);

    int total_units = 0;
    std::vector<int> selected;
    selected.reserve(p.n_orders);

    for (int idx : indices) {
        int size = p.order_size[idx];
        if (total_units + size > p.ub) continue;

        bool feasible = true;
        for (int j = p.order_off[idx]; j < p.order_off[idx + 1]; ++j) {
            int it = p.order_item[j];
            int q = p.order_qty[j];
            if (stock_rem[it] < q) {
                feasible = false;
                break;
            }
        }
        if (!feasible) continue;

        selected.push_back(idx);
        total_units += size;
        for (int j = p.order_off[idx]; j < p.order_off[idx + 1]; ++j) {
            int it = p.order_item[j];
            int q = p.order_qty[j];
            stock_rem[it] -= q;
            demand[it] += q;
        }
    }

    if (total_units < p.lb) return false;

    std::vector<int> visited = select_aisles(opt, p, demand);
    if (visited.empty()) return false;

    out.selected_orders = std::move(selected);
    out.visited_aisles = std::move(visited);
    out.objective = static_cast<double>(total_units) / static_cast<double>(out.visited_aisles.size());
    return true;
}

void emit(const Solution& s) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\"objective\":" << s.objective << ",\"selected_orders\":[";
    for (size_t i = 0; i < s.selected_orders.size(); ++i) {
        if (i) out << ',';
        out << s.selected_orders[i];
    }
    out << "],\"visited_aisles\":[";
    for (size_t i = 0; i < s.visited_aisles.size(); ++i) {
        if (i) out << ',';
        out << s.visited_aisles[i];
    }
    out << "]}\n";
    std::cout << out.str();
    std::cout.flush();
}

int run(const Options& opt) {
    const Problem p = load_instance(opt.instance_path);

    std::mt19937 rng;
    if (opt.seed_set) rng.seed(static_cast<uint32_t>(opt.seed));
    else rng.seed(std::random_device{}());

    Solution best;
    emit(best);

    Solution candidate;
    if (solve_simple(opt, p, rng, candidate)) {
        best = std::move(candidate);
        emit(best);
    }

    emit(best);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << "simple_solver exception: " << e.what() << "\n";
        return 3;
    }
}
