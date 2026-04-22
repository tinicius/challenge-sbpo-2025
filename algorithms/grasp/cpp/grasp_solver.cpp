// GRASP solver for the SBPO 2025 wave order-picking problem.
//
// CLI:
//   grasp_solver <instance_path>
//       --alpha=F
//       --construction=size|synergy|aisle_cost|aisle_cost_fast
//       --max-iterations=N
//       --greedy=simple|multi
//       --local-search=none|swap|full
//       [--similarity-weighted=0|1]
//       [--seed=N]
//       [--time-limit=F]            seconds; 0 or omitted = unlimited
//
// Output (stdout): one JSON object per line. Each time the best-known
// solution improves, a line is emitted immediately (and flushed) with:
//   {"objective":F,"selected_orders":[...],"visited_aisles":[...]}
// The first line is an initial empty incumbent; the final line repeats
// the terminal incumbent so the consumer always has a valid record.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

enum class Construction { Size, Synergy, AisleCost, AisleCostFast };
enum class Greedy { Simple, Multi };
enum class LocalSearch { None, Swap, Full };

struct Options {
    std::string instance_path;
    double alpha = 0.0;
    Construction construction = Construction::Size;
    int max_iterations = 0;
    Greedy greedy = Greedy::Simple;
    LocalSearch local_search = LocalSearch::None;
    bool similarity_weighted = false;
    bool seed_set = false;
    uint64_t seed = 0;
    double time_limit = 0.0;
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
    std::vector<int> best_aisle_for_item;
};

struct Solution {
    std::vector<int> selected_orders;
    std::vector<int> visited_aisles;
    double objective = 0.0;
    std::vector<int> demand;
    int total_units = 0;
};

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "grasp_solver: " << msg << "\n";
    std::exit(2);
}

bool parse_bool(const std::string& s) {
    if (s == "1" || s == "true" || s == "True") return true;
    if (s == "0" || s == "false" || s == "False") return false;
    die("invalid boolean: " + s);
}

Options parse_args(int argc, char** argv) {
    Options opt;
    bool got_alpha = false, got_constr = false, got_iters = false;
    bool got_greedy = false, got_ls = false;
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
        if (key == "alpha") { opt.alpha = std::stod(val); got_alpha = true; }
        else if (key == "construction") {
            if (val == "size") opt.construction = Construction::Size;
            else if (val == "synergy") opt.construction = Construction::Synergy;
            else if (val == "aisle_cost") opt.construction = Construction::AisleCost;
            else if (val == "aisle_cost_fast") opt.construction = Construction::AisleCostFast;
            else die("invalid construction: " + val);
            got_constr = true;
        }
        else if (key == "max-iterations") { opt.max_iterations = std::stoi(val); got_iters = true; }
        else if (key == "greedy") {
            if (val == "simple") opt.greedy = Greedy::Simple;
            else if (val == "multi") opt.greedy = Greedy::Multi;
            else die("invalid greedy: " + val);
            got_greedy = true;
        }
        else if (key == "local-search") {
            if (val == "none") opt.local_search = LocalSearch::None;
            else if (val == "swap") opt.local_search = LocalSearch::Swap;
            else if (val == "full") opt.local_search = LocalSearch::Full;
            else die("invalid local-search: " + val);
            got_ls = true;
        }
        else if (key == "similarity-weighted") opt.similarity_weighted = parse_bool(val);
        else if (key == "seed") { opt.seed = static_cast<uint64_t>(std::stoll(val)); opt.seed_set = true; }
        else if (key == "time-limit") opt.time_limit = std::stod(val);
        else die("unknown option: " + key);
    }
    if (opt.instance_path.empty()) die("missing instance path");
    if (!got_alpha || !got_constr || !got_iters || !got_greedy || !got_ls)
        die("missing required option");
    if (opt.alpha < 0.0 || opt.alpha > 1.0) die("alpha out of range");
    if (opt.max_iterations <= 0) die("max-iterations must be positive");
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
        int k; f >> k;
        p.order_off[o + 1] = p.order_off[o] + k;
        int s = 0;
        for (int j = 0; j < k; ++j) {
            int it, q; f >> it >> q;
            p.order_item.push_back(it);
            p.order_qty.push_back(q);
            s += q;
        }
        p.order_size[o] = s;
    }

    p.aisle_off.assign(p.n_aisles + 1, 0);
    p.stock_total.assign(p.n_items, 0);
    p.best_aisle_for_item.assign(p.n_items, -1);
    std::vector<int> best_qty(p.n_items, -1);
    p.aisle_item.reserve(p.n_aisles * 4);
    p.aisle_qty.reserve(p.n_aisles * 4);
    for (int a = 0; a < p.n_aisles; ++a) {
        int k; f >> k;
        p.aisle_off[a + 1] = p.aisle_off[a] + k;
        for (int j = 0; j < k; ++j) {
            int it, q; f >> it >> q;
            p.aisle_item.push_back(it);
            p.aisle_qty.push_back(q);
            p.stock_total[it] += q;
            if (q > best_qty[it]) { best_qty[it] = q; p.best_aisle_for_item[it] = a; }
        }
    }
    f >> p.lb >> p.ub;
    return p;
}

// ---- Aisle selection ---------------------------------------------------

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
    for (int v : demand) if (v > 0) remaining += v;
    std::vector<int> selected;
    selected.reserve(16);
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
            if (d > 0) {
                int take = std::min(d, p.aisle_qty[j]);
                demand[it] = d - take;
                remaining -= take;
            }
        }
    }
    return selected;
}

std::vector<int> multi_greedy_aisles(const Problem& p, std::vector<int> demand) {
    std::vector<char> used(p.n_aisles, 0);
    std::vector<int> selected;
    int remaining = 0;
    for (int v : demand) if (v > 0) remaining += v;
    while (remaining > 0) {
        int best = -1, best_score = 0;
        for (int a = 0; a < p.n_aisles; ++a) {
            if (used[a]) continue;
            int s = 0;
            for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j) {
                int d = demand[p.aisle_item[j]];
                if (d > 0) s += std::min(d, p.aisle_qty[j]);
            }
            if (s > best_score) { best_score = s; best = a; }
        }
        if (best_score == 0) break;
        used[best] = 1;
        selected.push_back(best);
        for (int j = p.aisle_off[best]; j < p.aisle_off[best + 1]; ++j) {
            int it = p.aisle_item[j];
            int d = demand[it];
            if (d > 0) {
                int take = std::min(d, p.aisle_qty[j]);
                demand[it] = d - take;
                remaining -= take;
            }
        }
    }
    return selected;
}

std::vector<int> select_aisles(const Options& opt, const Problem& p,
                               const std::vector<int>& demand) {
    if (opt.greedy == Greedy::Multi) return multi_greedy_aisles(p, demand);
    return simple_greedy_aisles(p, demand);
}

// ---- Emission ----------------------------------------------------------

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

// ---- Construction phase -----------------------------------------------

struct ConstructState {
    std::vector<int> demand;
    std::vector<int> stock_rem;
    std::vector<char> demand_mask;
    int demand_keys = 0;
    std::vector<char> aisle_used;
    int total_units = 0;
    std::vector<int> selected;
    std::vector<int> remaining;
};

struct Scratch {
    std::vector<int> feasible;
    std::vector<double> scores;
    std::vector<int> rcl;
    explicit Scratch(int n_orders) {
        feasible.reserve(n_orders);
        scores.reserve(n_orders);
        rcl.reserve(n_orders);
    }
};

bool order_feasible(const Problem& p, int idx, const ConstructState& st, int ub) {
    if (p.order_size[idx] == 0) return false;
    if (st.total_units + p.order_size[idx] > ub) return false;
    for (int j = p.order_off[idx]; j < p.order_off[idx + 1]; ++j) {
        if (st.stock_rem[p.order_item[j]] < p.order_qty[j]) return false;
    }
    return true;
}

void score_candidates(const Options& opt, const Problem& p,
                      const ConstructState& st, Scratch& sc) {
    sc.scores.assign(sc.feasible.size(), 0.0);
    const bool demand_empty = (st.demand_keys == 0);

    if (opt.construction == Construction::Size || demand_empty) {
        for (size_t k = 0; k < sc.feasible.size(); ++k)
            sc.scores[k] = static_cast<double>(p.order_size[sc.feasible[k]]);
        return;
    }

    if (opt.construction == Construction::Synergy) {
        if (!opt.similarity_weighted) {
            int keys_a = st.demand_keys;
            for (size_t k = 0; k < sc.feasible.size(); ++k) {
                int idx = sc.feasible[k];
                int inter = 0;
                int keys_b = p.order_off[idx + 1] - p.order_off[idx];
                for (int j = p.order_off[idx]; j < p.order_off[idx + 1]; ++j)
                    if (st.demand_mask[p.order_item[j]]) ++inter;
                int uni = keys_a + keys_b - inter;
                sc.scores[k] = uni == 0 ? 0.0 : static_cast<double>(inter) / uni;
            }
        } else {
            long long a_total = st.total_units;
            for (size_t k = 0; k < sc.feasible.size(); ++k) {
                int idx = sc.feasible[k];
                long long num = 0, den = 0, a_in_b = 0;
                for (int j = p.order_off[idx]; j < p.order_off[idx + 1]; ++j) {
                    int qb = p.order_qty[j];
                    int qa = st.demand[p.order_item[j]];
                    num += std::min(qa, qb);
                    den += std::max(qa, qb);
                    a_in_b += qa;
                }
                den += (a_total - a_in_b);  // items in a\b contribute qa to den
                sc.scores[k] = den == 0 ? 0.0 : static_cast<double>(num) / static_cast<double>(den);
            }
        }
        return;
    }

    if (opt.construction == Construction::AisleCostFast) {
        for (size_t k = 0; k < sc.feasible.size(); ++k) {
            int idx = sc.feasible[k];
            long long shortfall = 0;
            for (int j = p.order_off[idx]; j < p.order_off[idx + 1]; ++j) {
                int it = p.order_item[j];
                int best_a = p.best_aisle_for_item[it];
                if (best_a < 0 || !st.aisle_used[best_a]) shortfall += p.order_qty[j];
            }
            sc.scores[k] = -static_cast<double>(shortfall);
        }
        return;
    }

    // AisleCost: exact but expensive — two greedy_aisle_select calls per candidate.
    auto current = select_aisles(opt, p, st.demand);
    std::vector<char> cur_set(p.n_aisles, 0);
    for (int a : current) cur_set[a] = 1;
    std::vector<int> combined = st.demand;
    for (size_t k = 0; k < sc.feasible.size(); ++k) {
        int idx = sc.feasible[k];
        for (int j = p.order_off[idx]; j < p.order_off[idx + 1]; ++j)
            combined[p.order_item[j]] += p.order_qty[j];
        auto after = select_aisles(opt, p, combined);
        int new_aisles = 0;
        for (int a : after) if (!cur_set[a]) ++new_aisles;
        sc.scores[k] = -static_cast<double>(new_aisles);
        for (int j = p.order_off[idx]; j < p.order_off[idx + 1]; ++j)
            combined[p.order_item[j]] -= p.order_qty[j];
    }
}

bool construct(const Options& opt, const Problem& p, std::mt19937& rng,
               Scratch& sc, Solution& out) {
    ConstructState st;
    st.demand.assign(p.n_items, 0);
    st.stock_rem = p.stock_total;
    st.demand_mask.assign(p.n_items, 0);
    st.aisle_used.assign(p.n_aisles, 0);
    st.selected.reserve(p.n_orders);
    st.remaining.resize(p.n_orders);
    for (int i = 0; i < p.n_orders; ++i) st.remaining[i] = i;

    while (!st.remaining.empty()) {
        // Compact remaining in place to the currently-feasible set. Since stock
        // and total_units are monotone in construction, any order that fails
        // here will keep failing — safe to drop permanently.
        sc.feasible.clear();
        size_t w = 0;
        for (size_t r = 0; r < st.remaining.size(); ++r) {
            int idx = st.remaining[r];
            if (order_feasible(p, idx, st, p.ub)) {
                st.remaining[w++] = idx;
                sc.feasible.push_back(idx);
            }
        }
        st.remaining.resize(w);
        if (sc.feasible.empty()) break;

        score_candidates(opt, p, st, sc);

        double g_max = sc.scores[0], g_min = sc.scores[0];
        for (double s : sc.scores) { if (s > g_max) g_max = s; if (s < g_min) g_min = s; }
        double threshold = g_max - opt.alpha * (g_max - g_min);
        sc.rcl.clear();
        for (size_t k = 0; k < sc.feasible.size(); ++k)
            if (sc.scores[k] >= threshold) sc.rcl.push_back(sc.feasible[k]);
        std::uniform_int_distribution<size_t> pick_dist(0, sc.rcl.size() - 1);
        int pick = sc.rcl[pick_dist(rng)];

        st.selected.push_back(pick);
        st.total_units += p.order_size[pick];
        for (int j = p.order_off[pick]; j < p.order_off[pick + 1]; ++j) {
            int it = p.order_item[j];
            int q = p.order_qty[j];
            if (st.demand[it] == 0) { st.demand_mask[it] = 1; ++st.demand_keys; }
            st.demand[it] += q;
            st.stock_rem[it] -= q;
            int best_a = p.best_aisle_for_item[it];
            if (best_a >= 0) st.aisle_used[best_a] = 1;
        }
        // Remove pick from remaining.
        for (size_t r = 0; r < st.remaining.size(); ++r) {
            if (st.remaining[r] == pick) {
                st.remaining[r] = st.remaining.back();
                st.remaining.pop_back();
                break;
            }
        }
    }

    if (st.total_units < p.lb) return false;
    auto visited = select_aisles(opt, p, st.demand);
    if (visited.empty()) return false;
    out.selected_orders = std::move(st.selected);
    out.visited_aisles = std::move(visited);
    out.demand = std::move(st.demand);
    out.total_units = st.total_units;
    out.objective = static_cast<double>(out.total_units) / out.visited_aisles.size();
    return true;
}

// ---- Local search ------------------------------------------------------

bool demand_within_stock(const std::vector<int>& demand,
                         const std::vector<int>& stock_total) {
    for (size_t i = 0; i < demand.size(); ++i)
        if (demand[i] > stock_total[i]) return false;
    return true;
}

bool try_swap(const Options& opt, const Problem& p, Solution& cur) {
    auto& selected = cur.selected_orders;
    std::vector<char> in_sel(p.n_orders, 0);
    for (int i : selected) in_sel[i] = 1;

    for (size_t si = 0; si < selected.size(); ++si) {
        int s_idx = selected[si];
        int s_size = p.order_size[s_idx];
        std::vector<int> demand_minus = cur.demand;
        for (int j = p.order_off[s_idx]; j < p.order_off[s_idx + 1]; ++j)
            demand_minus[p.order_item[j]] -= p.order_qty[j];

        for (int u_idx = 0; u_idx < p.n_orders; ++u_idx) {
            if (in_sel[u_idx]) continue;
            int u_size = p.order_size[u_idx];
            if (u_size == 0) continue;
            int new_total = cur.total_units - s_size + u_size;
            if (new_total > p.ub || new_total < p.lb) continue;

            for (int j = p.order_off[u_idx]; j < p.order_off[u_idx + 1]; ++j)
                demand_minus[p.order_item[j]] += p.order_qty[j];

            bool ok = demand_within_stock(demand_minus, p.stock_total);
            if (ok) {
                auto new_visited = select_aisles(opt, p, demand_minus);
                if (!new_visited.empty()) {
                    double new_obj = static_cast<double>(new_total) / new_visited.size();
                    if (new_obj > cur.objective) {
                        selected[si] = u_idx;
                        cur.total_units = new_total;
                        cur.demand = std::move(demand_minus);
                        cur.visited_aisles = std::move(new_visited);
                        cur.objective = new_obj;
                        return true;
                    }
                }
            }

            for (int j = p.order_off[u_idx]; j < p.order_off[u_idx + 1]; ++j)
                demand_minus[p.order_item[j]] -= p.order_qty[j];
        }
    }
    return false;
}

bool try_drop(const Options& opt, const Problem& p, Solution& cur) {
    auto& selected = cur.selected_orders;
    if (selected.size() <= 1) return false;
    for (size_t si = 0; si < selected.size(); ++si) {
        int s_idx = selected[si];
        int s_size = p.order_size[s_idx];
        int new_total = cur.total_units - s_size;
        if (new_total < p.lb) continue;

        std::vector<int> new_demand = cur.demand;
        for (int j = p.order_off[s_idx]; j < p.order_off[s_idx + 1]; ++j)
            new_demand[p.order_item[j]] -= p.order_qty[j];

        auto new_visited = select_aisles(opt, p, new_demand);
        if (new_visited.empty()) continue;
        double new_obj = static_cast<double>(new_total) / new_visited.size();
        if (new_obj > cur.objective) {
            selected.erase(selected.begin() + si);
            cur.total_units = new_total;
            cur.demand = std::move(new_demand);
            cur.visited_aisles = std::move(new_visited);
            cur.objective = new_obj;
            return true;
        }
    }
    return false;
}

bool try_add(const Options& opt, const Problem& p, Solution& cur) {
    std::vector<char> in_sel(p.n_orders, 0);
    for (int i : cur.selected_orders) in_sel[i] = 1;
    for (int u_idx = 0; u_idx < p.n_orders; ++u_idx) {
        if (in_sel[u_idx]) continue;
        int u_size = p.order_size[u_idx];
        if (u_size == 0) continue;
        int new_total = cur.total_units + u_size;
        if (new_total > p.ub) continue;

        std::vector<int> new_demand = cur.demand;
        for (int j = p.order_off[u_idx]; j < p.order_off[u_idx + 1]; ++j)
            new_demand[p.order_item[j]] += p.order_qty[j];

        if (!demand_within_stock(new_demand, p.stock_total)) continue;

        auto new_visited = select_aisles(opt, p, new_demand);
        if (new_visited.empty()) continue;
        double new_obj = static_cast<double>(new_total) / new_visited.size();
        if (new_obj > cur.objective) {
            cur.selected_orders.push_back(u_idx);
            cur.total_units = new_total;
            cur.demand = std::move(new_demand);
            cur.visited_aisles = std::move(new_visited);
            cur.objective = new_obj;
            return true;
        }
    }
    return false;
}

void local_search_improve(const Options& opt, const Problem& p, Solution& cur) {
    bool improved = true;
    while (improved) {
        improved = false;
        if (try_swap(opt, p, cur)) { improved = true; continue; }
        if (opt.local_search == LocalSearch::Full) {
            if (try_drop(opt, p, cur)) { improved = true; continue; }
            if (try_add(opt, p, cur))  { improved = true; continue; }
        }
    }
}

// ---- Main --------------------------------------------------------------

int run(const Options& opt) {
    Problem p = load_instance(opt.instance_path);
    Solution best;
    best.objective = 0.0;
    emit(best);
    if (p.n_orders == 0 || p.n_aisles == 0) {
        emit(best);
        return 0;
    }

    std::mt19937 rng;
    if (opt.seed_set) rng.seed(static_cast<uint32_t>(opt.seed));
    else rng.seed(std::random_device{}());

    Scratch sc(p.n_orders);

    auto t_start = Clock::now();
    auto time_expired = [&]() {
        if (opt.time_limit <= 0) return false;
        std::chrono::duration<double> d = Clock::now() - t_start;
        return d.count() >= opt.time_limit;
    };

    for (int it = 0; it < opt.max_iterations; ++it) {
        if (time_expired()) break;
        Solution built;
        if (!construct(opt, p, rng, sc, built)) continue;

        if (built.objective > best.objective) {
            best = built;
            emit(best);
        }

        if (opt.local_search != LocalSearch::None) {
            if (time_expired()) break;
            local_search_improve(opt, p, built);
            if (built.objective > best.objective) {
                best = built;
                emit(best);
            }
        }
    }

    emit(best);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << "grasp_solver exception: " << e.what() << "\n";
        return 3;
    }
}
